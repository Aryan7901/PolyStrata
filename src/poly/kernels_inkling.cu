// src/poly/kernels_inkling.cu - Inkling 2's kernels (kernels.hpp).  The reference for every formula is llama.cpp's
// inkling2 graph (src/models/inkling2.cpp) and its relative positions (llama_kv_cache::set_input_relative_pos).
// Where a kernel takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int MIX_THREADS = 256;   // a warp a state: MIX_STATES warps
constexpr int MIX_STATES = 8;
constexpr int MIX_QUERIES = 4;
constexpr int MIX_FEW = 16;      // up to this many tokens: ink_mix spreads a token over blocks
constexpr int ATT_DIM = 128;      // the heads ink_attend takes
constexpr int ATT_THREADS = 128;  // a thread a value of the head
constexpr int ATT_KEYS = 32;      // keys a block holds at a time
constexpr int ATT_SPLIT = 512;    // keys a block takes before its sums go to the combining pass
constexpr int ATT_REL = 16;       // the relative bias's terms a head
constexpr int ATT_FEW = 8;        // up to this many tokens a call: a block a token
constexpr int ATT_TILE = 16;      // above it: a block 16 tokens
constexpr int ATT_PART = ATT_DIM + 2;   // a head's partial sums: its values, then its max and its sum

cudaStream_t cs(void* s) { return (cudaStream_t) s; }
int grid(int64_t n, int b) { return (int) ((n + b - 1) / b); }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// 16 (or 8) values of a cache row from offset `at` (in values): fp16, or Q8_0 codes times their block's scale
template <bool Q8>
__device__ __forceinline__ void load16(const void* cache, const uint16_t* sc, size_t at, float* f) {
    if (Q8) {
        const uint4 c = *(const uint4*) ((const int8_t*) cache + at);
        const float d = __half2float(*(const __half*) (sc + at / 32));
        const int8_t* b = (const int8_t*) &c;
#pragma unroll
        for (int j = 0; j < 16; ++j) f[j] = (float) b[j] * d;
    } else {
        const uint4* r = (const uint4*) ((const uint16_t*) cache + at);
        const uint4 k0 = r[0], k1 = r[1];
        const __half2* h0 = (const __half2*) &k0;
        const __half2* h1 = (const __half2*) &k1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(h0[j]), c = __half22float2(h1[j]);
            f[2 * j] = a.x;
            f[2 * j + 1] = a.y;
            f[8 + 2 * j] = c.x;
            f[8 + 2 * j + 1] = c.y;
        }
    }
}
template <bool Q8>
__device__ __forceinline__ uint4 load8h(const void* cache, const uint16_t* sc, size_t at) {
    if (!Q8) return *(const uint4*) ((const uint16_t*) cache + at);
    const uint2 c = *(const uint2*) ((const int8_t*) cache + at);
    const float d = __half2float(*(const __half*) (sc + at / 32));
    const int8_t* b = (const int8_t*) &c;
    uint4 out;
    uint32_t* o = (uint32_t*) &out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2 h = __floats2half2_rn((float) b[2 * j] * d, (float) b[2 * j + 1] * d);
        o[j] = *(const uint32_t*) &h;
    }
    return out;
}

// ---- attention over the depth

__global__ void mix_kernel(const float* __restrict__ r, int r_stride, InkStates s, const float* __restrict__ query,
                           int nq, float* __restrict__ out, size_t q_stride, int n) {
    // a warp a state: its sum of squares and its products with the queries
    __shared__ float wsum[MIX_STATES * (1 + MIX_QUERIES)];
    const float* row = r + (size_t) blockIdx.x * r_stride;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (warp < s.n) {
        const float* x = row + s.off[warp];
        float ss = 0.f, d[MIX_QUERIES] = {0.f, 0.f, 0.f, 0.f};
        for (int i = lane; i < n; i += WARP) {
            const float v = x[i];
            ss += v * v;
#pragma unroll
            for (int q = 0; q < MIX_QUERIES; ++q)
                if (q < nq) d[q] += v * query[(size_t) q * n + i];
        }
        ss = warp_sum(ss);
#pragma unroll
        for (int q = 0; q < MIX_QUERIES; ++q) d[q] = q < nq ? warp_sum(d[q]) : 0.f;
        if (lane == 0) {
            wsum[warp] = ss;
#pragma unroll
            for (int q = 0; q < MIX_QUERIES; ++q) wsum[MIX_STATES + warp * MIX_QUERIES + q] = d[q];
        }
    }
    __syncthreads();
    // each state's norm, then for each query the softmax of its products with the normed states
    __shared__ float inv[MIX_STATES], weight[MIX_QUERIES][MIX_STATES];
    if (threadIdx.x < s.n) inv[threadIdx.x] = rsqrtf(wsum[threadIdx.x] / (float) n + 1e-6f);
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        const float sc = rsqrtf((float) n);
        float mx = -INFINITY, sc_j[MIX_STATES];
        for (int j = 0; j < s.n; ++j) {
            sc_j[j] = wsum[MIX_STATES + j * MIX_QUERIES + q] * inv[j] * sc;
            mx = fmaxf(mx, sc_j[j]);
        }
        float sum = 0.f;
        for (int j = 0; j < s.n; ++j) sum += sc_j[j] = expf(sc_j[j] - mx);
        for (int j = 0; j < s.n; ++j) weight[q][j] = sc_j[j] / sum * inv[j];
    }
    __syncthreads();
    float* o = out + (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x[MIX_STATES];
#pragma unroll
        for (int j = 0; j < MIX_STATES; ++j) x[j] = j < s.n ? row[s.off[j] + i] : 0.f;
        for (int q = 0; q < nq; ++q) {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < MIX_STATES; ++j) v += j < s.n ? weight[q][j] * x[j] : 0.f;
            o[(size_t) q * q_stride + i] = v;
        }
    }
}

// The same for a few tokens, spread over MIX_CHUNKS blocks a token: each block sums its part of the vectors into
// `part` (MIX_PART floats a block); mix_out_kernel adds the parts, takes the softmax and writes its part.
constexpr int MIX_CHUNKS = 8;
constexpr int MIX_PART = MIX_STATES * (1 + MIX_QUERIES);

__global__ void mix_part_kernel(const float* __restrict__ r, int r_stride, InkStates s, const float* __restrict__ query,
                                int nq, float* __restrict__ part, int n) {
    const float* row = r + (size_t) blockIdx.y * r_stride;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int per = (n + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    float* pp = part + ((size_t) blockIdx.y * gridDim.x + blockIdx.x) * MIX_PART;
    if (warp >= s.n) {
        if (warp < MIX_STATES && lane < 1 + MIX_QUERIES) pp[lane == 0 ? warp : MIX_STATES + warp * MIX_QUERIES + lane - 1] = 0.f;
        return;
    }
    const float* x = row + s.off[warp];
    float ss = 0.f, d[MIX_QUERIES] = {0.f, 0.f, 0.f, 0.f};
    for (int i = i0 + lane; i < i1; i += WARP) {
        const float v = x[i];
        ss += v * v;
#pragma unroll
        for (int q = 0; q < MIX_QUERIES; ++q)
            if (q < nq) d[q] += v * query[(size_t) q * n + i];
    }
    ss = warp_sum(ss);
#pragma unroll
    for (int q = 0; q < MIX_QUERIES; ++q) d[q] = warp_sum(d[q]);
    if (lane == 0) {
        pp[warp] = ss;
#pragma unroll
        for (int q = 0; q < MIX_QUERIES; ++q) pp[MIX_STATES + warp * MIX_QUERIES + q] = q < nq ? d[q] : 0.f;
    }
}

__global__ void mix_out_kernel(const float* __restrict__ r, int r_stride, InkStates s, int nq,
                               const float* __restrict__ part, float* __restrict__ out, size_t q_stride, int n) {
    __shared__ float wsum[MIX_PART], inv[MIX_STATES], weight[MIX_QUERIES][MIX_STATES];
    const int t = blockIdx.y;
    if (threadIdx.x < MIX_PART) {
        float v = 0.f;
        for (int c = 0; c < (int) gridDim.x; ++c) v += part[((size_t) t * gridDim.x + c) * MIX_PART + threadIdx.x];
        wsum[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < s.n) inv[threadIdx.x] = rsqrtf(wsum[threadIdx.x] / (float) n + 1e-6f);
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        const float sc = rsqrtf((float) n);
        float mx = -INFINITY, sc_j[MIX_STATES];
        for (int j = 0; j < s.n; ++j) {
            sc_j[j] = wsum[MIX_STATES + j * MIX_QUERIES + q] * inv[j] * sc;
            mx = fmaxf(mx, sc_j[j]);
        }
        float sum = 0.f;
        for (int j = 0; j < s.n; ++j) sum += sc_j[j] = expf(sc_j[j] - mx);
        for (int j = 0; j < s.n; ++j) weight[q][j] = sc_j[j] / sum * inv[j];
    }
    __syncthreads();
    const float* row = r + (size_t) t * r_stride;
    float* o = out + (size_t) t * n;
    const int per = (n + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        float x[MIX_STATES];
#pragma unroll
        for (int j = 0; j < MIX_STATES; ++j) x[j] = j < s.n ? row[s.off[j] + i] : 0.f;
        for (int q = 0; q < nq; ++q) {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < MIX_STATES; ++j) v += j < s.n ? weight[q][j] * x[j] : 0.f;
            o[(size_t) q * q_stride + i] = v;
        }
    }
}

// ---- F32 products of a few tokens with narrow matrices (the router, the loop's controllers): a block an output
// when its row is long, a warp an output when it is short; the weights read once for all of the call's tokens.
constexpr int GEMV_ROWS = 8;

template <int R>
__global__ void gemv_wide_kernel(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                 int n_in, int n_out) {
    __shared__ float red[R][8];
    const int o = blockIdx.x;
    const float* wr = w + (size_t) o * n_in;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int i = threadIdx.x * 4; i < n_in; i += blockDim.x * 4) {
        const float4 wv = *(const float4*) (wr + i);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float4 xv = *(const float4*) (x + (size_t) r * n_in + i);
            acc[r] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    }
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) red[r][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < R) {
        float v = 0.f;
        for (int k = 0; k < (int) (blockDim.x / WARP); ++k) v += red[threadIdx.x][k];
        y[(size_t) threadIdx.x * n_out + o] = v;
    }
}

template <int R>
__global__ void gemv_short_kernel(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                  int n_in, int n_out) {
    const int o = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (o >= n_out) return;
    const float* wr = w + (size_t) o * n_in;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int i = lane; i < n_in; i += WARP) {
        const float wv = wr[i];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += wv * x[(size_t) r * n_in + i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) y[(size_t) r * n_out + o] = v;
    }
}

__global__ void gate_kernel(const float* __restrict__ cur, int cur_stride, const float* __restrict__ msg,
                            int msg_stride, const float* __restrict__ g, const float* __restrict__ f, int gf_stride,
                            float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = blockIdx.y;
    const float c = cur[t * cur_stride + i], m = msg[t * msg_stride + i];
    out[t * n + i] = (1.f - tanhf(f[t * gf_stride + i])) * c + tanhf(g[t * gf_stride + i]) * m;
}

// ---- the queries', keys' and relative terms' clip, a warp a head

__global__ void clip_kernel(float* __restrict__ x, int x_stride, const float* __restrict__ scale, float limit, int dim,
                            int heads, int rows, float alpha, float floor_, int64_t pos, const long long* at) {
    const int head = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int t = blockIdx.y;
    if (head >= heads || t >= rows) return;
    float* v = x + (size_t) t * x_stride + (size_t) head * dim;
    const float* s = scale ? scale + (size_t) head * dim : nullptr;
    float ss = 0.f;
    for (int i = lane; i < dim; i += WARP) {
        const float y = s ? v[i] * s[i] : v[i];
        ss += y * y;
    }
    ss = warp_sum(ss);
    const float norm = sqrtf(ss) + 1e-6f;
    float factor = fminf(fmaxf(limit / norm, 0.f), 1.f);
    if (alpha > 0.f) {
        const int64_t p = (at ? (int64_t) at[0] : pos) + t;
        factor *= 1.f + alpha * logf(fmaxf((float) (p + 1) / floor_, 1.f));
    }
    for (int i = lane; i < dim; i += WARP) v[i] = (s ? v[i] * s[i] : v[i]) * factor;
}

// Up to three clips (ink_clip's) in one launch: blockIdx.z picks which.
struct ClipJob {
    float* x;
    const float* scale;
    float limit, alpha;
    int x_stride, dim, heads;
};
struct ClipJobs {
    ClipJob j[3];
};

__global__ void clip3_kernel(ClipJobs jobs, int rows, float floor_, int64_t pos, const long long* at) {
    const ClipJob& J = jobs.j[blockIdx.z];
    const int head = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int t = blockIdx.y;
    if (head >= J.heads || t >= rows) return;
    float* v = J.x + (size_t) t * J.x_stride + (size_t) head * J.dim;
    const float* s = J.scale ? J.scale + (size_t) head * J.dim : nullptr;
    float ss = 0.f;
    for (int i = lane; i < J.dim; i += WARP) {
        const float y = s ? v[i] * s[i] : v[i];
        ss += y * y;
    }
    ss = warp_sum(ss);
    float factor = fminf(fmaxf(J.limit / (sqrtf(ss) + 1e-6f), 0.f), 1.f);
    if (J.alpha > 0.f) {
        const int64_t p = (at ? (int64_t) at[0] : pos) + t;
        factor *= 1.f + J.alpha * logf(fmaxf((float) (p + 1) / floor_, 1.f));
    }
    for (int i = lane; i < J.dim; i += WARP) v[i] = (s ? v[i] * s[i] : v[i]) * factor;
}

// rms of two vectors a token (no weights) into the halves of a 2n row: the loop controller's input
__global__ void rms2_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n,
                            float eps) {
    __shared__ float red[2][8];
    const float* src = blockIdx.y ? b : a;
    const float* x = src + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    ss = warp_sum(ss);
    if (lane == 0) red[0][warp] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int k = 0; k < (int) (blockDim.x / WARP); ++k) tot += red[0][k];
    const float sc = rsqrtf(tot / (float) n + eps);
    float* o = out + (size_t) blockIdx.x * 2 * n + (size_t) blockIdx.y * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = x[i] * sc;
}

// ---- the short convolution: a thread an output (token, channel); the inputs it reads before the call come from the
// state, which a second launch then advances (out never aliases x, so no output overwrites an input it needs)

__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ state, const float* __restrict__ w,
                            const float* __restrict__ dyn, int heads, float* __restrict__ out, int channels, int taps,
                            int rows, float scale, int xs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= channels) return;
    const int keep = taps - 1;
    const float* wc = w + (size_t) c * taps;
    // input lag tokens back: from x, or from the state's keep inputs (oldest first) before the call
    const auto in = [&](int lag) {
        const int u = t - lag;
        return u >= 0 ? x[(size_t) u * xs + c] : state[(size_t) c * keep + keep + u];
    };
    const float xt = x[(size_t) t * xs + c];
    float acc = wc[keep] * xt;
    for (int lag = 1; lag < taps; ++lag) acc += wc[keep - lag] * in(lag);
    if (dyn) {
        const float* d = dyn + ((size_t) t * heads + c / (channels / heads)) * taps;
        acc += d[0] * xt;
        for (int lag = 1; lag < taps; ++lag) acc += d[lag] * in(lag);
    }
    out[(size_t) t * channels + c] = (xt + acc) * scale;
}

// One token: its output and the state's advance in one thread a channel (no other thread reads the channel).
__global__ void conv1_kernel(const float* __restrict__ x, float* __restrict__ state, const float* __restrict__ w,
                             const float* __restrict__ dyn, int heads, float* __restrict__ out, int channels, int taps,
                             float scale, float* __restrict__ snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int keep = taps - 1;
    float* st = state + (size_t) c * keep;
    const float* wc = w + (size_t) c * taps;
    const float xt = x[c];
    float past[8];
    for (int j = 0; j < keep; ++j) past[j] = st[j];
    float acc = wc[keep] * xt;
    for (int k = 0; k < keep; ++k) acc += wc[k] * past[k];
    if (dyn) {
        const float* d = dyn + (size_t) (c / (channels / heads)) * taps;
        acc += d[0] * xt;
        for (int lag = 1; lag < taps; ++lag) acc += d[lag] * past[keep - lag];
    }
    out[c] = (xt + acc) * scale;
    for (int j = 0; j + 1 < keep; ++j) st[j] = past[j + 1];
    st[keep - 1] = xt;
    if (snap)
        for (int j = 0; j < keep; ++j) snap[(size_t) c * keep + j] = st[j];
}

__global__ void conv_state_kernel(const float* __restrict__ x, float* __restrict__ state, int channels, int taps,
                                  int rows, float* __restrict__ snap, size_t stride, unsigned snaps, int xs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int keep = taps - 1;
    float past[8];
    for (int j = 0; j < keep; ++j) past[j] = state[(size_t) c * keep + j];
    // a window's snapshots: the state after each token whose bit is set, in the tokens' order, stride floats apart
    if (snap)
        for (int t = 0, k = 0; t < rows && t < 32; ++t) {
            if (!(snaps >> t & 1u)) continue;
            float* d = snap + (size_t) k++ * stride + (size_t) c * keep;
            for (int j = 0; j < keep; ++j) {
                const int u = t + 1 - keep + j;   // the input j of the keep before the state after token t
                d[j] = u >= 0 ? x[(size_t) u * xs + c] : past[keep + u];
            }
        }
    // the keep inputs before the call's end, oldest first
    for (int j = 0; j < keep; ++j) {
        const int u = rows - keep + j;
        state[(size_t) c * keep + j] = u >= 0 ? x[(size_t) u * xs + c] : past[j + rows];
    }
}

// Rows of different sequences, a token each: row t's state is at state + off[t] (floats).
__global__ void conv_rows_kernel(const float* __restrict__ x, int xs, float* state, const long long* __restrict__ off,
                                 const float* __restrict__ w, const float* __restrict__ dyn, int heads,
                                 float* __restrict__ out, int channels, int taps, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= channels) return;
    const int keep = taps - 1;
    float* st = state + off[t] + (size_t) c * keep;
    const float* wc = w + (size_t) c * taps;
    const float xt = x[(size_t) t * xs + c];
    float past[8];
    for (int j = 0; j < keep; ++j) past[j] = st[j];
    float acc = wc[keep] * xt;
    for (int k = 0; k < keep; ++k) acc += wc[k] * past[k];
    if (dyn) {
        const float* d = dyn + ((size_t) t * heads + c / (channels / heads)) * taps;
        acc += d[0] * xt;
        for (int lag = 1; lag < taps; ++lag) acc += d[lag] * past[keep - lag];
    }
    out[(size_t) t * channels + c] = (xt + acc) * scale;
    for (int j = 0; j + 1 < keep; ++j) st[j] = past[j + 1];
    st[keep - 1] = xt;
}

struct Offsets {
    long long v[8];
};
__global__ void set_offsets_kernel(long long* dst, Offsets o, int n) {
    if (threadIdx.x < n) dst[threadIdx.x] = o.v[threadIdx.x];
}

// ---- the attention
//
// A block takes TQ tokens of one key head and its G query heads: TQ * G query rows, a thread a value of the head.
// It reads the positions of its split a tile of ATT_KEYS at a time into shared memory, scores them for every query
// row (bias and sink in), keeps a running max and sum a row, and adds the tile's values in.  Its partial sums go to
// `part`; combine_kernel joins a token's splits and the zero logit.

template <int TQ, int G>
__global__ void __launch_bounds__(ATT_THREADS)
attend_kernel(const float* __restrict__ q, int q_stride, const uint16_t* __restrict__ kcache,
              const uint16_t* __restrict__ vcache, const float* __restrict__ rel, int rel_stride,
              const float* __restrict__ table, int n_rel, int extent,
              const float* __restrict__ sink, float* __restrict__ part, int heads, int kv_heads, float scale,
              int64_t pos, int rows, int window, int ring, const long long* at, int split_len) {
    constexpr int QR = TQ * G;
    constexpr int QS = ATT_DIM + 1;   // a query row in shared memory, padded off the banks
    constexpr int KS = ATT_DIM + 2;   // a key or value row (halves)
    extern __shared__ float smem[];
    float* qs = smem;                                  // QR * QS
    float* ss = qs + QR * QS;                          // QR * ATT_KEYS: scores, then weights
    float* rs = ss + QR * ATT_KEYS;                    // QR * ATT_REL
    float* sk = rs + QR * ATT_REL;                     // QR: the sinks
    float* mx = sk + QR;                               // QR: running max
    float* sm = mx + QR;                               // QR: running sum
    float* cr = sm + QR;                               // QR: this tile's correction
    __half* ks = (__half*) (cr + QR);                  // ATT_KEYS * KS
    __half* vs = ks + ATT_KEYS * KS;                   // ATT_KEYS * KS

    const int kvh = blockIdx.x, t0 = blockIdx.y * TQ, split = blockIdx.z, d = threadIdx.x;
    const int64_t p0 = at ? (int64_t) at[0] : pos;
    const int t_end = min(rows, t0 + TQ);
    const int64_t hi = p0 + t_end;   // past the last position the tile's tokens read
    const int64_t lo = window > 0 ? max((int64_t) 0, p0 + t0 - window + 1) : 0;
    const int64_t s0 = lo + (int64_t) split * split_len, s1 = min(hi, s0 + split_len);
    const size_t kv_row = (size_t) kv_heads * ATT_DIM;

    for (int i = d; i < QR * QS; i += ATT_THREADS) {
        const int qr = i / QS, j = i % QS, t = t0 + qr / G, h = kvh * G + qr % G;
        qs[i] = (j < ATT_DIM && t < rows) ? q[(size_t) t * q_stride + (size_t) h * ATT_DIM + j] * scale : 0.f;
    }
    for (int i = d; i < QR * ATT_REL; i += ATT_THREADS) {
        const int qr = i / ATT_REL, j = i % ATT_REL, t = t0 + qr / G, h = kvh * G + qr % G;
        rs[i] = (j < n_rel && t < rows) ? rel[(size_t) t * rel_stride + (size_t) h * n_rel + j] : 0.f;
    }
    if (d < QR) {
        const int t = t0 + d / G, h = kvh * G + d % G;
        sk[d] = t < rows ? sink[(size_t) t * heads + h] : 0.f;
        mx[d] = -INFINITY;
        sm[d] = 0.f;
    }
    float acc[QR];
#pragma unroll
    for (int i = 0; i < QR; ++i) acc[i] = 0.f;
    __syncthreads();

    for (int64_t kb = s0; kb < s1; kb += ATT_KEYS) {
        for (int k = 0; k < ATT_KEYS; ++k) {
            const int64_t kp = kb + k;
            __half kv = __float2half(0.f), vv = __float2half(0.f);
            if (kp < s1) {
                const size_t row = (size_t) (ring > 0 ? kp % ring : kp);
                kv = ((const __half*) kcache)[row * kv_row + (size_t) kvh * ATT_DIM + d];
                vv = ((const __half*) vcache)[row * kv_row + (size_t) kvh * ATT_DIM + d];
            }
            ks[k * KS + d] = kv;
            vs[k * KS + d] = vv;
        }
        __syncthreads();
        for (int i = d; i < QR * ATT_KEYS; i += ATT_THREADS) {
            const int qr = i / ATT_KEYS, k = i % ATT_KEYS, t = t0 + qr / G;
            const int64_t kp = kb + k, p = p0 + t;
            float s = -INFINITY;
            if (t < rows && kp < s1 && kp <= p && (window <= 0 || p - kp < window)) {
                const float* qv = qs + qr * QS;
                const __half* kv = ks + k * KS;
                float dot = 0.f;
#pragma unroll 16
                for (int j = 0; j < ATT_DIM; ++j) dot += qv[j] * __half2float(kv[j]);
                const int64_t dist = p - kp;
                float b = 0.f;
                if (dist < extent) {
                    const float* tb = table + (size_t) dist * n_rel;
                    const float* rv = rs + qr * ATT_REL;
                    for (int j = 0; j < n_rel; ++j) b += rv[j] * tb[j];
                }
                s = dot + b - sk[qr];
            }
            ss[i] = s;
        }
        __syncthreads();
        if (d < QR) {
            float m = mx[d];
            for (int k = 0; k < ATT_KEYS; ++k) m = fmaxf(m, ss[d * ATT_KEYS + k]);
            float sum = 0.f;
            for (int k = 0; k < ATT_KEYS; ++k) {
                const float s = ss[d * ATT_KEYS + k];
                const float p = s == -INFINITY ? 0.f : __expf(s - m);
                ss[d * ATT_KEYS + k] = p;
                sum += p;
            }
            const float c = mx[d] == -INFINITY ? 0.f : __expf(mx[d] - m);
            cr[d] = m == -INFINITY ? 1.f : c;
            sm[d] = sm[d] * cr[d] + sum;
            mx[d] = m;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < QR; ++i) acc[i] *= cr[i];
        for (int k = 0; k < ATT_KEYS; ++k) {
            const float v = __half2float(vs[k * KS + d]);
#pragma unroll
            for (int i = 0; i < QR; ++i) acc[i] += ss[i * ATT_KEYS + k] * v;
        }
        __syncthreads();
    }
    // the partial sums: [split][token][head][values, max, sum]
#pragma unroll
    for (int i = 0; i < QR; ++i) {
        const int t = t0 + i / G, h = kvh * G + i % G;
        if (t >= rows) continue;
        float* pp = part + (((size_t) split * rows + t) * heads + h) * ATT_PART;
        pp[d] = acc[i];
        if (d == 0) {
            pp[ATT_DIM] = mx[i];
            pp[ATT_DIM + 1] = sm[i];
        }
    }
}

// A token's attention when a call has a few: a block a (token, key head, split of DEC_SPLIT positions), a warp a
// position at a time, a lane four values of the head.  Each warp keeps its own running max, sum and values for the
// G query heads; the block joins its warps and leaves the split's partial sums as attend_kernel does.
constexpr int DEC_WARPS = 4;
constexpr int DEC_SPLIT = 128;
// decode_long_kernel's blocks in all, a SM (INK_ATT_BLOCKS changes it): enough for the card to hide the reads' latency
const int kDecLongBlocksPerSm = [] {
    const char* e = std::getenv("INK_ATT_BLOCKS");
    return e && std::atoi(e) > 0 ? std::atoi(e) : 8;
}();

// Groups of 8 lanes a position: a lane 16 values of the head, so a score is 3 shuffles, not 5 a head.  The
// bias of each of the block's positions (minus the sink) is worked out once into shared memory first.
template <int G, bool Q8>
__global__ void __launch_bounds__(DEC_WARPS * WARP)
decode_kernel(const float* __restrict__ q, int q_stride, const void* __restrict__ kcache,
              const void* __restrict__ vcache, const uint16_t* __restrict__ ksc, const uint16_t* __restrict__ vsc,
              const float* __restrict__ rel, int rel_stride,
              const float* __restrict__ table, int n_rel, int extent,
              const float* __restrict__ sink, float* __restrict__ part, int heads, int kv_heads, float scale,
              int64_t pos, int rows, int window, int ring, const long long* at) {
    constexpr int GROUPS = DEC_WARPS * 4;   // groups of 8 lanes in the block
    __shared__ float bias[G][DEC_SPLIT];
    __shared__ float gm[GROUPS][G], gl[GROUPS][G];
    __shared__ float gacc[GROUPS][G][ATT_DIM];
    const int kvh = blockIdx.x, t = blockIdx.y, split = blockIdx.z;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP, grp = warp * 4 + lane / 8, sub = lane % 8;
    const int64_t p = (at ? (int64_t) at[0] : pos) + t;
    const int64_t lo = window > 0 ? max((int64_t) 0, p - window + 1) : 0;
    const int64_t s0 = lo + (int64_t) split * DEC_SPLIT, s1 = min(p + 1, s0 + DEC_SPLIT);
    const size_t kv_row = (size_t) kv_heads * ATT_DIM;
    for (int i = threadIdx.x; i < G * DEC_SPLIT; i += blockDim.x) {
        const int g = i / DEC_SPLIT, k = i % DEC_SPLIT, h = kvh * G + g;
        const int64_t kp = s0 + k, dist = p - kp;
        float b = 0.f;
        if (kp < s1 && dist < extent) {
            const float* tb = table + (size_t) dist * n_rel;
            const float* rv = rel + (size_t) t * rel_stride + (size_t) h * n_rel;
            for (int r = 0; r < n_rel; ++r) b += rv[r] * tb[r];
        }
        bias[g][k] = b - sink[(size_t) t * heads + h];
    }
    float qv[G][16], acc[G][16], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float* qh = q + (size_t) t * q_stride + (size_t) (kvh * G + g) * ATT_DIM + sub * 16;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            qv[g][j] = qh[j] * scale;
            acc[g][j] = 0.f;
        }
        m[g] = -INFINITY;
        l[g] = 0.f;
    }
    __syncthreads();
    // the warp goes round together (its shuffles take every lane); a group past the split's end weighs nothing
    const int n_keys = (int) max((int64_t) 0, s1 - s0);
    for (int base = warp * 4; base < n_keys; base += GROUPS) {
        const int k = base + lane / 8;
        const bool live = k < n_keys;
        const int64_t kp = s0 + (live ? k : 0);
        const size_t row = (size_t) (ring > 0 ? kp % ring : kp) * kv_row + (size_t) kvh * ATT_DIM + sub * 16;
        float kf[16];
        load16<Q8>(kcache, ksc, row, kf);
        float sc[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) d += qv[g][j] * kf[j];
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 4);
            sc[g] = live ? d + bias[g][k] : -INFINITY;
        }
        float vf[16];
        load16<Q8>(vcache, vsc, row, vf);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (!live) continue;
            const float mn = fmaxf(m[g], sc[g]);
            const float c = __expf(m[g] - mn), e = __expf(sc[g] - mn);
            l[g] = l[g] * c + e;
#pragma unroll
            for (int j = 0; j < 16; ++j) acc[g][j] = acc[g][j] * c + e * vf[j];
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (sub == 0) {
            gm[grp][g] = m[g];
            gl[grp][g] = l[g];
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) gacc[grp][g][sub * 16 + j] = acc[g][j];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * ATT_DIM; i += blockDim.x) {
        const int g = i / ATT_DIM, d = i % ATT_DIM;
        float mx = -INFINITY;
        for (int w = 0; w < GROUPS; ++w) mx = fmaxf(mx, gm[w][g]);
        float v = 0.f, sum = 0.f;
        if (mx != -INFINITY)
            for (int w = 0; w < GROUPS; ++w) {
                if (gm[w][g] == -INFINITY) continue;
                const float c = __expf(gm[w][g] - mx);
                v += c * gacc[w][g][d];
                sum += c * gl[w][g];
            }
        float* pp = part + (((size_t) split * rows + t) * heads + kvh * G + g) * ATT_PART;
        pp[d] = v;
        if (d == 0) {
            pp[ATT_DIM] = mx;
            pp[ATT_DIM + 1] = sum;
        }
    }
}

// A cache row's 16 values, as read (prefetched a position ahead in decode_long_kernel), then as floats
template <bool Q8>
struct Raw16 {
    uint4 a, b;
    float d;
};
template <bool Q8>
__device__ __forceinline__ void fetch16(const void* cache, const uint16_t* sc, size_t at, Raw16<Q8>& r) {
    if (Q8) {
        r.a = *(const uint4*) ((const int8_t*) cache + at);
        r.d = __half2float(*(const __half*) (sc + at / 32));
    } else {
        const uint4* p = (const uint4*) ((const uint16_t*) cache + at);
        r.a = p[0];
        r.b = p[1];
    }
}
template <bool Q8>
__device__ __forceinline__ void unpack16(const Raw16<Q8>& r, float* f) {
    if (Q8) {
        const int8_t* b = (const int8_t*) &r.a;
#pragma unroll
        for (int j = 0; j < 16; ++j) f[j] = (float) b[j] * r.d;
    } else {
        const __half2* h0 = (const __half2*) &r.a;
        const __half2* h1 = (const __half2*) &r.b;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(h0[j]), c = __half22float2(h1[j]);
            f[2 * j] = a.x;
            f[2 * j + 1] = a.y;
            f[8 + 2 * j] = c.x;
            f[8 + 2 * j + 1] = c.y;
        }
    }
}

// decode_kernel for a long span (a full-attention layer deep into a conversation): a block takes split_len
// positions, not DEC_SPLIT, so the card holds a few blocks a SM for the whole span instead of one per 128 positions
// (at 100K positions those spent more on their setup, their joining and combine_kernel's partial sums than on reading
// the cache).  A position's keys and values are read together, and the next ones a group takes are fetched while it
// scores these.  The bias is worked out only for the positions within `extent` of the token (at most DEC_NEAR);
// the others have none - as in decode_kernel, where it is 0 for them - and only the sink is taken off.
constexpr int DEC_NEAR = 1024;
template <int G, bool Q8>
__global__ void __launch_bounds__(DEC_WARPS * WARP)
decode_long_kernel(const float* __restrict__ q, int q_stride, const void* __restrict__ kcache,
                   const void* __restrict__ vcache, const uint16_t* __restrict__ ksc, const uint16_t* __restrict__ vsc,
                   const float* __restrict__ rel, int rel_stride, const float* __restrict__ table, int n_rel,
                   int extent, const float* __restrict__ sink, float* __restrict__ part, int heads, int kv_heads,
                   float scale, int64_t pos, int rows, int window, int ring, const long long* at, int split_len) {
    constexpr int GROUPS = DEC_WARPS * 4;
    // the bias of the near positions while the block reads; afterwards the groups' sums (the bias is spent by then)
    __shared__ union {
        float bias[G][DEC_NEAR];
        float gacc[GROUPS][G][ATT_DIM];
    } sm;
    __shared__ float gm[GROUPS][G], gl[GROUPS][G];
    const int kvh = blockIdx.x, t = blockIdx.y, split = blockIdx.z;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP, grp = warp * 4 + lane / 8, sub = lane % 8;
    const int64_t p = (at ? (int64_t) at[0] : pos) + t;
    const int64_t lo = window > 0 ? max((int64_t) 0, p - window + 1) : 0;
    const int64_t s0 = lo + (int64_t) split * split_len, s1 = min(p + 1, s0 + split_len);
    const int64_t near0 = max(s0, p - extent + 1);   // the first position with a bias term
    const int n_near = (int) max((int64_t) 0, s1 - near0);
    const size_t kv_row = (size_t) kv_heads * ATT_DIM;
    for (int i = threadIdx.x; i < G * n_near; i += blockDim.x) {
        const int g = i / n_near, k = i % n_near, h = kvh * G + g;
        const int64_t dist = p - (near0 + k);
        const float* tb = table + (size_t) dist * n_rel;
        const float* rv = rel + (size_t) t * rel_stride + (size_t) h * n_rel;
        float b = 0.f;
        for (int r = 0; r < n_rel; ++r) b += rv[r] * tb[r];
        sm.bias[g][k] = b;
    }
    float qv[G][16], acc[G][16], m[G], l[G], sk[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float* qh = q + (size_t) t * q_stride + (size_t) (kvh * G + g) * ATT_DIM + sub * 16;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            qv[g][j] = qh[j] * scale;
            acc[g][j] = 0.f;
        }
        m[g] = -INFINITY;
        l[g] = 0.f;
        sk[g] = sink[(size_t) t * heads + kvh * G + g];
    }
    __syncthreads();
    const int n_keys = (int) max((int64_t) 0, s1 - s0);
    const auto row_of = [&](int k) {
        const int64_t kp = s0 + k;
        return (size_t) (ring > 0 ? kp % ring : kp) * kv_row + (size_t) kvh * ATT_DIM + sub * 16;
    };
    Raw16<Q8> kr, vr, kn, vn;
    int base = warp * 4;
    if (base < n_keys) {
        const int k = base + lane / 8;
        const size_t row = row_of(k < n_keys ? k : 0);
        fetch16<Q8>(kcache, ksc, row, kr);
        fetch16<Q8>(vcache, vsc, row, vr);
    }
    // the warp goes round together (its shuffles take every lane); a group past the split's end weighs nothing
    for (; base < n_keys; base += GROUPS) {
        const int k = base + lane / 8;
        const bool live = k < n_keys;
        if (base + GROUPS < n_keys) {   // the group's next position, read while this one is scored
            const int kk = k + GROUPS;
            const size_t row = row_of(kk < n_keys ? kk : 0);
            fetch16<Q8>(kcache, ksc, row, kn);
            fetch16<Q8>(vcache, vsc, row, vn);
        }
        float kf[16];
        unpack16<Q8>(kr, kf);
        const int64_t kp = s0 + (live ? k : 0);
        const int ni = (int) (kp - near0);
        float sc[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) d += qv[g][j] * kf[j];
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 4);
            const float b = kp >= near0 ? sm.bias[g][ni] : 0.f;
            sc[g] = live ? d + (b - sk[g]) : -INFINITY;
        }
        float vf[16];
        unpack16<Q8>(vr, vf);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (!live) continue;
            const float mn = fmaxf(m[g], sc[g]);
            const float c = __expf(m[g] - mn), e = __expf(sc[g] - mn);
            l[g] = l[g] * c + e;
#pragma unroll
            for (int j = 0; j < 16; ++j) acc[g][j] = acc[g][j] * c + e * vf[j];
            m[g] = mn;
        }
        kr = kn;
        vr = vn;
    }
    __syncthreads();   // the bias is read no more: its room takes the groups' sums
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (sub == 0) {
            gm[grp][g] = m[g];
            gl[grp][g] = l[g];
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) sm.gacc[grp][g][sub * 16 + j] = acc[g][j];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * ATT_DIM; i += blockDim.x) {
        const int g = i / ATT_DIM, d = i % ATT_DIM;
        float mx = -INFINITY;
        for (int w = 0; w < GROUPS; ++w) mx = fmaxf(mx, gm[w][g]);
        float v = 0.f, sum = 0.f;
        if (mx != -INFINITY)
            for (int w = 0; w < GROUPS; ++w) {
                if (gm[w][g] == -INFINITY) continue;
                const float c = __expf(gm[w][g] - mx);
                v += c * sm.gacc[w][g][d];
                sum += c * gl[w][g];
            }
        float* pp = part + (((size_t) split * rows + t) * heads + kvh * G + g) * ATT_PART;
        pp[d] = v;
        if (d == 0) {
            pp[ATT_DIM] = mx;
            pp[ATT_DIM + 1] = sum;
        }
    }
}

// INK_KV_SIM (a quality experiment): a cache row as a 4-bit cache would give it back - rotated in blocks of B
// values by the orthonormal Walsh-Hadamard matrix (llama.cpp's KV rotation: 64 for V, the head for K), rounded to
// ggml's Q4_0 (32 a block, scale max/-8 kept as a half), rotated back - written to y for the Q8_0 store to take.
template <int B>
__global__ void kv_sim_kernel(const float* __restrict__ x, float* __restrict__ y, int kvw, int rot) {
    __shared__ float s[B];
    __shared__ float d32[B / 32];
    const int r = blockIdx.x, c0 = blockIdx.y * B, i = threadIdx.x;
    s[i] = x[(size_t) r * kvw + c0 + i];
    const auto fwht = [&] {
        for (int h = 1; h < B; h <<= 1) {
            __syncthreads();
            const float a = s[i], b = s[i ^ h];
            __syncthreads();
            s[i] = (i & h) ? b - a : a + b;
        }
        __syncthreads();
        s[i] *= rsqrtf((float) B);
        __syncthreads();
    };
    __syncthreads();
    if (rot) fwht();
    if (i % 32 == 0) {
        float amax = 0.f, mx = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float v = s[i + j];
            if (fabsf(v) > amax) amax = fabsf(v), mx = v;
        }
        d32[i / 32] = mx / -8.f;
    }
    __syncthreads();
    const float d = d32[i / 32], id = d != 0.f ? 1.f / d : 0.f;
    const int q = min(15, (int) (s[i] * id + 8.5f));
    const float v = (float) (q - 8) * __half2float(__float2half_rn(d));
    __syncthreads();
    s[i] = v;
    __syncthreads();
    if (rot) fwht();
    y[(size_t) r * kvw + c0 + i] = s[i];
}

// ---- a prompt batch's attention on the tensor cores (mma.sync m16n8k16, fp16 in, fp32 sums): a block 32 tokens
// of a key head, a warp 16 query rows (16 tokens of one query head), PF_KEYS positions at a time.  The scores stay
// in registers: their fragments are the softmax's rows and then the A operand of the values' product.

constexpr int PF_TOKENS = 32;
constexpr int PF_KEYS = 64;
constexpr int PF_LD = ATT_DIM + 8;   // a row in shared memory, in halves: off the banks for ldmatrix

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t& r0, uint32_t& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t*) &h;
}

template <int G, bool Q8>
__global__ void __launch_bounds__(2 * G * WARP)
prefill_kernel(const float* __restrict__ q, int q_stride, const void* __restrict__ kcache,
               const void* __restrict__ vcache, const uint16_t* __restrict__ ksc, const uint16_t* __restrict__ vsc,
               const float* __restrict__ rel, int rel_stride,
               const float* __restrict__ table, int n_rel, int extent, const float* __restrict__ sink,
               float* __restrict__ out, int heads, int kv_heads, float scale, int64_t pos, int rows, int window,
               int ring) {
    constexpr int R = PF_TOKENS * G;   // query rows: head-major, a head's 32 tokens together
    extern __shared__ __align__(16) unsigned char pf_smem[];
    __half* qs = (__half*) pf_smem;                 // R * PF_LD
    __half* ks = qs + R * PF_LD;                    // PF_KEYS * PF_LD
    __half* vs = ks + PF_KEYS * PF_LD;              // PF_KEYS * PF_LD
    const int kvh = blockIdx.x, t0 = blockIdx.y * PF_TOKENS;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP, nthreads = blockDim.x;
    const size_t kv_row = (size_t) kv_heads * ATT_DIM;
    const int t_end = min(rows, t0 + PF_TOKENS);
    const int64_t hi = pos + t_end;
    const int64_t lo = window > 0 ? max((int64_t) 0, pos + t0 - window + 1) : 0;

    for (int i = threadIdx.x; i < R * ATT_DIM; i += nthreads) {
        const int r = i / ATT_DIM, d = i % ATT_DIM, g = r / PF_TOKENS, t = t0 + r % PF_TOKENS, h = kvh * G + g;
        const float v = t < rows ? q[(size_t) t * q_stride + (size_t) h * ATT_DIM + d] * scale : 0.f;
        qs[r * PF_LD + d] = __float2half(v);
    }
    __syncthreads();
    const int row0 = warp * 16, g = row0 / PF_TOKENS, h = kvh * G + g;
    const int tt0 = row0 % PF_TOKENS + lane / 4, tt1 = tt0 + 8;
    const int ta = t0 + tt0, tb = t0 + tt1;
    const int64_t pa = pos + ta, pb = pos + tb;
    uint32_t qa[ATT_DIM / 16][4];
#pragma unroll
    for (int j = 0; j < ATT_DIM / 16; ++j)
        ldsm_x4(qa[j][0], qa[j][1], qa[j][2], qa[j][3], qs + (row0 + lane % 16) * PF_LD + j * 16 + (lane / 16) * 8);
    float rva[ATT_REL], rvb[ATT_REL];
#pragma unroll
    for (int r = 0; r < ATT_REL; ++r) {
        rva[r] = (r < n_rel && ta < rows) ? rel[(size_t) ta * rel_stride + (size_t) h * n_rel + r] : 0.f;
        rvb[r] = (r < n_rel && tb < rows) ? rel[(size_t) tb * rel_stride + (size_t) h * n_rel + r] : 0.f;
    }
    const float ska = ta < rows ? sink[(size_t) ta * heads + h] : 0.f;
    const float skb = tb < rows ? sink[(size_t) tb * heads + h] : 0.f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
    float o[ATT_DIM / 8][4];
#pragma unroll
    for (int dn = 0; dn < ATT_DIM / 8; ++dn) o[dn][0] = o[dn][1] = o[dn][2] = o[dn][3] = 0.f;

    for (int64_t kb = lo; kb < hi; kb += PF_KEYS) {
        __syncthreads();
        for (int i = threadIdx.x; i < PF_KEYS * (ATT_DIM / 8); i += nthreads) {
            const int k = i / (ATT_DIM / 8), c = (i % (ATT_DIM / 8)) * 8;
            const int64_t kp = kb + k;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kp < hi) {
                const size_t at = (size_t) (ring > 0 ? kp % ring : kp) * kv_row + (size_t) kvh * ATT_DIM + c;
                kv = load8h<Q8>(kcache, ksc, at);
                vv = load8h<Q8>(vcache, vsc, at);
            }
            *(uint4*) (ks + k * PF_LD + c) = kv;
            *(uint4*) (vs + k * PF_LD + c) = vv;
        }
        __syncthreads();
        float s[PF_KEYS / 8][4];
#pragma unroll
        for (int nt = 0; nt < PF_KEYS / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int j = 0; j < ATT_DIM / 16; ++j) {
                uint32_t b0, b1;
                ldsm_x2(b0, b1, ks + (nt * 8 + lane % 8) * PF_LD + j * 16 + ((lane / 8) % 2) * 8);
                mma16816(s[nt], qa[j], b0, b1);
            }
        }
        const bool far = (pos + t0) - (kb + PF_KEYS - 1) >= extent;
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int nt = 0; nt < PF_KEYS / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool second = e >= 2;
                const int64_t p = second ? pb : pa;
                const bool tok = (second ? tb : ta) < rows;
                const int64_t kp = kb + nt * 8 + (lane % 4) * 2 + (e & 1);
                float v = -INFINITY;
                if (tok && kp < hi && kp <= p && (window <= 0 || p - kp < window)) {
                    float b = 0.f;
                    const int64_t dist = p - kp;
                    if (!far && dist < extent) {
                        const float* tb_ = table + (size_t) dist * n_rel;
                        const float* rv = second ? rvb : rva;
#pragma unroll
                        for (int r = 0; r < ATT_REL; ++r) b += r < n_rel ? rv[r] * tb_[r] : 0.f;
                    }
                    v = s[nt][e] + b - (second ? skb : ska);
                }
                s[nt][e] = v;
                if (second) mxb = fmaxf(mxb, v);
                else mxa = fmaxf(mxa, v);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float ca = mxa == -INFINITY ? 1.f : __expf(ma - mxa), cb = mxb == -INFINITY ? 1.f : __expf(mb - mxb);
        float suma = 0.f, sumb = 0.f;
#pragma unroll
        for (int nt = 0; nt < PF_KEYS / 8; ++nt) {
            s[nt][0] = s[nt][0] == -INFINITY ? 0.f : __expf(s[nt][0] - mxa);
            s[nt][1] = s[nt][1] == -INFINITY ? 0.f : __expf(s[nt][1] - mxa);
            s[nt][2] = s[nt][2] == -INFINITY ? 0.f : __expf(s[nt][2] - mxb);
            s[nt][3] = s[nt][3] == -INFINITY ? 0.f : __expf(s[nt][3] - mxb);
            suma += s[nt][0] + s[nt][1];
            sumb += s[nt][2] + s[nt][3];
        }
        suma += __shfl_xor_sync(0xffffffffu, suma, 1);
        suma += __shfl_xor_sync(0xffffffffu, suma, 2);
        sumb += __shfl_xor_sync(0xffffffffu, sumb, 1);
        sumb += __shfl_xor_sync(0xffffffffu, sumb, 2);
        la = la * ca + suma;
        lb = lb * cb + sumb;
        ma = mxa;
        mb = mxb;
#pragma unroll
        for (int dn = 0; dn < ATT_DIM / 8; ++dn) {
            o[dn][0] *= ca;
            o[dn][1] *= ca;
            o[dn][2] *= cb;
            o[dn][3] *= cb;
        }
#pragma unroll
        for (int j = 0; j < PF_KEYS / 16; ++j) {
            const uint32_t pa_[4] = {pack_h2(s[2 * j][0], s[2 * j][1]), pack_h2(s[2 * j][2], s[2 * j][3]),
                                     pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]), pack_h2(s[2 * j + 1][2], s[2 * j + 1][3])};
#pragma unroll
            for (int dn = 0; dn < ATT_DIM / 8; ++dn) {
                uint32_t b0, b1;
                ldsm_x2_t(b0, b1, vs + (j * 16 + lane % 16) * PF_LD + dn * 8);
                mma16816(o[dn], pa_, b0, b1);
            }
        }
    }
    const float da = la + (ma == -INFINITY ? 1.f : __expf(-ma)), db = lb + (mb == -INFINITY ? 1.f : __expf(-mb));
#pragma unroll
    for (int dn = 0; dn < ATT_DIM / 8; ++dn) {
        const int d = dn * 8 + (lane % 4) * 2;
        if (ta < rows) {
            float* oa = out + ((size_t) ta * heads + h) * ATT_DIM + d;
            oa[0] = o[dn][0] / da;
            oa[1] = o[dn][1] / da;
        }
        if (tb < rows) {
            float* ob = out + ((size_t) tb * heads + h) * ATT_DIM + d;
            ob[0] = o[dn][2] / db;
            ob[1] = o[dn][3] / db;
        }
    }
}

__global__ void combine_kernel(const float* __restrict__ part, float* __restrict__ out, int heads, int rows,
                               int splits) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float m = 0.f;   // the zero logit
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part[(((size_t) s * rows + t) * heads + h) * ATT_PART + ATT_DIM]);
    float den = __expf(-m), v = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* pp = part + (((size_t) s * rows + t) * heads + h) * ATT_PART;
        if (pp[ATT_DIM] == -INFINITY) continue;
        const float w = __expf(pp[ATT_DIM] - m);
        den += w * pp[ATT_DIM + 1];
        v += w * pp[d];
    }
    out[((size_t) t * heads + h) * ATT_DIM + d] = v / den;
}

__global__ void rows_kernel(const float* __restrict__ src, int src_stride, float* __restrict__ dst, int dst_stride,
                            int n, bool add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = src[(size_t) blockIdx.y * src_stride + i];
    float* d = dst + (size_t) blockIdx.y * dst_stride + i;
    *d = add ? *d + v : v;
}

// ---- the output's shortlist: the rows whose approximate logit is within `margin` of the row's best, and their exact
// product with the Q8_0 output matrix

// two passes over HEAD_CHUNKS blocks a row: each block's largest, then each block keeps its part of the shortlist
constexpr int HEAD_CHUNKS = 64;

__global__ void head_max_kernel(const float* __restrict__ approx, int n, int stride, float* __restrict__ bmax,
                                int* __restrict__ count) {
    __shared__ float red[32];
    const float* a = approx + (size_t) blockIdx.y * stride;
    const int per = (n + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    float m = -INFINITY;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) m = fmaxf(m, a[i]);
    for (int o = WARP / 2; o > 0; o /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x % WARP == 0) red[threadIdx.x / WARP] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = -INFINITY;
        for (int w = 0; w < (int) (blockDim.x / WARP); ++w) v = fmaxf(v, red[w]);
        bmax[(size_t) blockIdx.y * gridDim.x + blockIdx.x] = v;
        if (blockIdx.x == 0) count[blockIdx.y] = 0;
    }
}

__global__ void shortlist_kernel(const float* __restrict__ approx, int n, int stride, float margin,
                                 const float* __restrict__ bmax, int* __restrict__ ids, int* __restrict__ count, int cap) {
    __shared__ float thr;
    if (threadIdx.x == 0) {
        float v = -INFINITY;
        for (int c = 0; c < (int) gridDim.x; ++c) v = fmaxf(v, bmax[(size_t) blockIdx.y * gridDim.x + c]);
        thr = v - margin;
    }
    __syncthreads();
    const float* a = approx + (size_t) blockIdx.y * stride;
    int* row_ids = ids + (size_t) blockIdx.y * cap;
    const int per = (n + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x)
        if (a[i] >= thr) {
            const int k = atomicAdd(count + blockIdx.y, 1);
            if (k < cap) row_ids[k] = i;
        }
}

struct BlockQ8 {
    uint16_t d;
    int8_t qs[32];
};

// a warp a shortlisted row: its Q8_0 row times h, over the approximation
__global__ void rescore_kernel(const BlockQ8* __restrict__ w, const float* __restrict__ h, int n_embd, const int* __restrict__ ids,
                               const int* __restrict__ count, int cap, float* __restrict__ logits, int stride) {
    const int r = blockIdx.y;
    const int k = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (k >= min(count[r], cap)) return;
    const int row = ids[(size_t) r * cap + k];
    const int nb = n_embd / 32;
    const BlockQ8* wr = w + (size_t) row * nb;
    const float* hr = h + (size_t) r * n_embd;
    float acc = 0.f;
    for (int b = lane; b < nb; b += WARP) {
        const BlockQ8 blk = wr[b];
        const float d = __half2float(*(const __half*) &blk.d);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 32; ++j) s += (float) blk.qs[j] * hr[b * 32 + j];
        acc += d * s;
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[(size_t) r * stride + row] = acc;
}

__global__ void fill_tail_kernel(float* x, int from, int n) {
    const int i = from + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(size_t) blockIdx.y * n + i] = -INFINITY;
}

template <int TQ, int G>
size_t attend_smem() {
    constexpr int QR = TQ * G;
    return sizeof(float) * (QR * (ATT_DIM + 1) + QR * ATT_KEYS + QR * ATT_REL + 4 * QR) +
           sizeof(__half) * 2 * ATT_KEYS * (ATT_DIM + 2);
}

template <int TQ, int G>
void launch_attend(dim3 g, const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, const float* rel,
                   int rel_stride, const float* table, int n_rel, int extent, const float* sink, float* part, int heads, int kv_heads,
                   float scale, int64_t pos, int rows, int window, int ring, const long long* at, int split_len,
                   cudaStream_t st) {
    static bool raised = false;
    const size_t bytes = attend_smem<TQ, G>();
    if (!raised && bytes > 48 * 1024) {
        cudaFuncSetAttribute(attend_kernel<TQ, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes);
        raised = true;
    }
    attend_kernel<TQ, G><<<g, ATT_THREADS, bytes, st>>>(q, q_stride, kc, vc, rel, rel_stride, table, n_rel, extent, sink, part, heads,
                                                        kv_heads, scale, pos, rows, window, ring, at, split_len);
}

// How many splits a call's positions go into: a few tokens spread over the card by their keys; a prompt batch's
// tiles are many blocks already, and splits beyond a few would only multiply the partial sums.
int attend_splits(int rows, int kv_heads, int64_t span) {
    const int by_keys = std::max(1, grid(span, ATT_SPLIT));
    if (rows <= ATT_FEW) return by_keys;
    const int blocks = kv_heads * grid(rows, ATT_TILE);
    return std::clamp(grid(256, blocks), 1, std::min(16, by_keys));   // the split's length grows to cover the span
}

}  // namespace

size_t ink_mix_floats(int rows) { return (size_t) std::min(rows, MIX_FEW) * MIX_CHUNKS * MIX_PART; }

void ink_mix(const float* r, int r_stride, const InkStates& s, const float* query, int nq, float* out,
             size_t q_stride, int n_embd, int rows, void* stream, float* scratch) {
    if (rows <= 0) return;
    if (s.n < 1 || s.n > MIX_STATES || nq < 1 || nq > MIX_QUERIES)
        throw std::invalid_argument("ink_mix: 1 to 8 states and 1 to 4 queries");
    if (scratch && rows <= MIX_FEW) {   // a few tokens: MIX_CHUNKS blocks each
        const dim3 g(MIX_CHUNKS, rows);
        mix_part_kernel<<<g, MIX_THREADS, 0, cs(stream)>>>(r, r_stride, s, query, nq, scratch, n_embd);
        mix_out_kernel<<<g, MIX_THREADS, 0, cs(stream)>>>(r, r_stride, s, nq, scratch, out, q_stride, n_embd);
        ck("ink_mix");
        return;
    }
    mix_kernel<<<rows, MIX_THREADS, 0, cs(stream)>>>(r, r_stride, s, query, nq, out, q_stride, n_embd);
    ck("ink_mix");
}

void ink_gate(const float* cur, int cur_stride, const float* msg, int msg_stride, const float* g, const float* f,
              int gf_stride, float* out, int n, int rows, void* stream) {
    if (rows <= 0) return;
    gate_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(cur, cur_stride, msg, msg_stride, g, f, gf_stride,
                                                                  out, n);
}

void ink_clip(float* x, int x_stride, const float* scale, float limit, int dim, int heads, int rows, float alpha,
              float floor_, int64_t pos, const long long* at, void* stream) {
    if (rows <= 0) return;
    constexpr int per = 4;   // heads a block, a warp each
    clip_kernel<<<dim3(grid(heads, per), rows), per * WARP, 0, cs(stream)>>>(x, x_stride, scale, limit, dim, heads,
                                                                            rows, alpha, floor_, pos, at);
}

void ink_clip3(float* x0, int s0, const float* sc0, float lim0, int dim0, int heads0, float alpha0, float* x1, int s1,
               const float* sc1, float lim1, int dim1, int heads1, float alpha1, float* x2, int s2, const float* sc2,
               float lim2, int dim2, int heads2, float alpha2, int rows, float floor_, int64_t pos,
               const long long* at, void* stream) {
    if (rows <= 0) return;
    ClipJobs jobs{{{x0, sc0, lim0, alpha0, s0, dim0, heads0}, {x1, sc1, lim1, alpha1, s1, dim1, heads1},
                   {x2, sc2, lim2, alpha2, s2, dim2, heads2}}};
    const int heads = std::max({heads0, heads1, heads2});
    constexpr int per = 4;
    clip3_kernel<<<dim3(grid(heads, per), rows, 3), per * WARP, 0, cs(stream)>>>(jobs, rows, floor_, pos, at);
}

void ink_rms2(const float* a, const float* b, float* out, int n, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    rms2_kernel<<<dim3(rows, 2), 256, 0, cs(stream)>>>(a, b, out, n, eps);
}

void ink_conv_rows(const float* x, int x_stride, float* state, const long long* off, const float* w, const float* dyn,
                   int heads, float* out, int channels, int taps, int rows, float scale, void* stream) {
    if (rows <= 0) return;
    conv_rows_kernel<<<dim3(grid(channels, 128), rows), 128, 0, cs(stream)>>>(x, x_stride > 0 ? x_stride : channels, state,
                                                                              off, w, dyn, heads, out, channels, taps,
                                                                              scale);
}

void ink_set_offsets(long long* dst, const long long* v, int n, void* stream) {
    if (n < 1 || n > 8) throw std::invalid_argument("ink_set_offsets: 1 to 8 values");
    Offsets o{};
    for (int i = 0; i < n; ++i) o.v[i] = v[i];
    set_offsets_kernel<<<1, 32, 0, cs(stream)>>>(dst, o, n);
}

void ink_conv(const float* x, float* state, const float* w, const float* dyn, int heads, float* out, int channels,
              int taps, int rows, float scale, void* stream, float* snap, size_t stride, unsigned snaps, int x_stride) {
    const int xs = x_stride > 0 ? x_stride : channels;
    if (rows <= 0) return;
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv: 2 to 8 taps");
    if (dyn && (heads < 1 || channels % heads != 0)) throw std::invalid_argument("ink_conv: heads do not divide the channels");
    if (out == x) throw std::invalid_argument("ink_conv: out must not be x");
    if (rows == 1) {
        conv1_kernel<<<grid(channels, 128), 128, 0, cs(stream)>>>(x, state, w, dyn, heads, out, channels, taps, scale,
                                                                  snap && (snaps & 1u) ? snap : nullptr);
        return;
    }
    conv_kernel<<<dim3(grid(channels, 128), rows), 128, 0, cs(stream)>>>(x, state, w, dyn, heads, out, channels, taps,
                                                                         rows, scale, xs);
    conv_state_kernel<<<grid(channels, 128), 128, 0, cs(stream)>>>(x, state, channels, taps, rows, snap, stride, snaps, xs);
}

size_t ink_attend_floats(int rows, int heads, int max_ctx) {
    const size_t few = (size_t) std::min(rows, ATT_FEW) * (size_t) (grid(max_ctx + ATT_TILE, DEC_SPLIT) + 1);
    const size_t many = (size_t) std::max(rows, 2048);
    return std::max(few, many) * (size_t) heads * ATT_PART;
}

void ink_attend(const float* q, int q_stride, const void* kcache, const void* vcache, const float* rel,
                int rel_stride, const float* table, int n_rel, int extent, const float* sink, float* out, float* part, int dim, int heads, int kv_heads,
                float scale, int64_t pos, int rows, int window, int ring, void* stream, const long long* at,
                int room, const uint16_t* kscales, const uint16_t* vscales) {
    const bool q8 = kscales != nullptr;
    if (rows <= 0) return;
    if (dim != ATT_DIM) throw std::invalid_argument("ink_attend: heads of 128 values only");
    if (n_rel < 0 || n_rel > ATT_REL) throw std::invalid_argument("ink_attend: at most 16 relative terms a head");
    if (kv_heads <= 0 || heads % kv_heads != 0) throw std::invalid_argument("ink_attend: bad key heads");
    const int g = heads / kv_heads;
    // the positions a tile reads, at the most: for a recorded launch, what `room` allows
    int64_t span = at ? (int64_t) room : pos + rows;
    if (window > 0) span = std::min<int64_t>(span, window + std::max(rows <= ATT_FEW ? 1 : ATT_TILE, 1) - 1);
    const int splits = attend_splits(rows, kv_heads, span);
    const int split_len = (int) (grid(grid(span, splits), ATT_KEYS) * ATT_KEYS);   // whole tiles, every position
    const cudaStream_t st = cs(stream);
    if (rows <= ATT_FEW) {   // a block a token and split of positions, a warp a position
        int dsplits = std::max(1, grid(span, DEC_SPLIT));
        // A long span (INK_ATT_LONG=0: never): a few blocks a SM in all, each over many positions (decode_long_kernel)
        static const int sms = [] {
            int dev = 0, n = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
            return std::max(n, 1);
        }();
        static const bool long_ok = [] {
            const char* e = std::getenv("INK_ATT_LONG");
            return !(e && std::atoi(e) == 0);
        }();
        const int most = std::max(1, kDecLongBlocksPerSm * sms / (kv_heads * rows));
        const bool longer = long_ok && extent <= DEC_NEAR && dsplits > most;
        int split_len = DEC_SPLIT;
        if (longer) {
            dsplits = most;
            split_len = (int) (grid(grid(span, dsplits), 16) * 16);
            dsplits = std::max(1, grid(span, split_len));
        }
        const dim3 gd(kv_heads, rows, dsplits);
        const auto dec = [&](auto gq) {
            constexpr int G = decltype(gq)::value;
            if (longer) {
                if (q8)
                    decode_long_kernel<G, true><<<gd, DEC_WARPS * WARP, 0, st>>>(q, q_stride, kcache, vcache, kscales,
                        vscales, rel, rel_stride, table, n_rel, extent, sink, part, heads, kv_heads, scale, pos, rows,
                        window, ring, at, split_len);
                else
                    decode_long_kernel<G, false><<<gd, DEC_WARPS * WARP, 0, st>>>(q, q_stride, kcache, vcache, nullptr,
                        nullptr, rel, rel_stride, table, n_rel, extent, sink, part, heads, kv_heads, scale, pos, rows,
                        window, ring, at, split_len);
                return;
            }
            if (q8)
                decode_kernel<G, true><<<gd, DEC_WARPS * WARP, 0, st>>>(q, q_stride, kcache, vcache, kscales, vscales,
                    rel, rel_stride, table, n_rel, extent, sink, part, heads, kv_heads, scale, pos, rows, window, ring, at);
            else
                decode_kernel<G, false><<<gd, DEC_WARPS * WARP, 0, st>>>(q, q_stride, kcache, vcache, nullptr, nullptr,
                    rel, rel_stride, table, n_rel, extent, sink, part, heads, kv_heads, scale, pos, rows, window, ring, at);
        };
        switch (g) {
        case 1: dec(std::integral_constant<int, 1>{}); break;
        case 2: dec(std::integral_constant<int, 2>{}); break;
        case 3: dec(std::integral_constant<int, 3>{}); break;
        case 4: dec(std::integral_constant<int, 4>{}); break;
        default: throw std::invalid_argument("ink_attend: 1 to 4 query heads a key head");
        }
        ck("ink_attend decode");
        combine_kernel<<<dim3(heads, rows), ATT_DIM, 0, st>>>(part, out, heads, rows, dsplits);
        ck("ink_attend combine");
        // INK_ATT_CHECK=1 (with --no-graphs): the long kernel's output against decode_kernel's, on the same inputs
        static const bool check = std::getenv("INK_ATT_CHECK") != nullptr;
        if (check && longer) {
            static float *cpart = nullptr, *cout = nullptr;
            static size_t cpart_n = 0;
            static int checked = 0;
            const int osplits = std::max(1, grid(span, DEC_SPLIT));
            const size_t need = (size_t) osplits * rows * heads * ATT_PART;
            if (need > cpart_n) {
                if (cpart) cudaFree(cpart);
                cudaMalloc(&cpart, need * sizeof(float));
                cpart_n = need;
            }
            if (!cout) cudaMalloc(&cout, (size_t) ATT_FEW * 64 * ATT_DIM * sizeof(float));
            const dim3 go(kv_heads, rows, osplits);
            const auto old_dec = [&](auto gq) {
                constexpr int G = decltype(gq)::value;
                if (q8)
                    decode_kernel<G, true><<<go, DEC_WARPS * WARP, 0, st>>>(q, q_stride, kcache, vcache, kscales, vscales,
                        rel, rel_stride, table, n_rel, extent, sink, cpart, heads, kv_heads, scale, pos, rows, window, ring, at);
                else
                    decode_kernel<G, false><<<go, DEC_WARPS * WARP, 0, st>>>(q, q_stride, kcache, vcache, nullptr, nullptr,
                        rel, rel_stride, table, n_rel, extent, sink, cpart, heads, kv_heads, scale, pos, rows, window, ring, at);
            };
            if (g == 1) old_dec(std::integral_constant<int, 1>{});
            else if (g == 2) old_dec(std::integral_constant<int, 2>{});
            else if (g == 3) old_dec(std::integral_constant<int, 3>{});
            else old_dec(std::integral_constant<int, 4>{});
            combine_kernel<<<dim3(heads, rows), ATT_DIM, 0, st>>>(cpart, cout, heads, rows, osplits);
            const size_t n = (size_t) heads * rows * ATT_DIM;
            std::vector<float> a(n), b(n);
            cudaMemcpyAsync(a.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaMemcpyAsync(b.data(), cout, n * sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            double d2 = 0, r2 = 0, mx = 0;
            for (size_t i = 0; i < n; ++i) {
                const double d = (double) a[i] - b[i];
                d2 += d * d;
                r2 += (double) b[i] * b[i];
                mx = std::max(mx, std::fabs(d));
            }
            if (checked++ % 50 == 0 || std::sqrt(d2 / std::max(r2, 1e-30)) > 1e-4)
                std::fprintf(stderr, "INK_ATT_CHECK pos %lld span %lld: long %d splits of %d vs %d of %d: rel err %.3g, max %.3g\n",
                             (long long) pos, (long long) span, dsplits, split_len, osplits, DEC_SPLIT,
                             std::sqrt(d2 / std::max(r2, 1e-30)), mx);
        }
        return;
    }
    if (!at) {   // a prompt batch on the tensor cores
        const size_t bytes = sizeof(__half) * (size_t) (PF_TOKENS * g + 2 * PF_KEYS) * PF_LD;
        const dim3 gp(kv_heads, grid(rows, PF_TOKENS));
        const auto pf = [&](auto gq) {
            constexpr int G = decltype(gq)::value;
            static bool raised = false;
            if (!raised) {
                cudaFuncSetAttribute(prefill_kernel<G, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
                cudaFuncSetAttribute(prefill_kernel<G, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
                raised = true;
            }
            if (q8)
                prefill_kernel<G, true><<<gp, 2 * G * WARP, bytes, st>>>(q, q_stride, kcache, vcache, kscales, vscales,
                    rel, rel_stride, table, n_rel, extent, sink, out, heads, kv_heads, scale, pos, rows, window, ring);
            else
                prefill_kernel<G, false><<<gp, 2 * G * WARP, bytes, st>>>(q, q_stride, kcache, vcache, nullptr, nullptr,
                    rel, rel_stride, table, n_rel, extent, sink, out, heads, kv_heads, scale, pos, rows, window, ring);
        };
        switch (g) {
        case 1: pf(std::integral_constant<int, 1>{}); break;
        case 2: pf(std::integral_constant<int, 2>{}); break;
        case 3: pf(std::integral_constant<int, 3>{}); break;
        case 4: pf(std::integral_constant<int, 4>{}); break;
        default: throw std::invalid_argument("ink_attend: 1 to 4 query heads a key head");
        }
        ck("ink_attend prefill");
        return;
    }
    if (q8) throw std::invalid_argument("ink_attend: a recorded call of more than 8 tokens over a q8_0 cache");
    const auto run = [&](auto tq, auto gq) {
        constexpr int TQ = decltype(tq)::value, G = decltype(gq)::value;
        launch_attend<TQ, G>(dim3(kv_heads, grid(rows, TQ), splits), q, q_stride, (const uint16_t*) kcache,
                             (const uint16_t*) vcache, rel, rel_stride, table, n_rel, extent, sink,
                             part, heads, kv_heads, scale, pos, rows, window, ring, at, split_len, st);
    };
    using I1 = std::integral_constant<int, 1>;
    using IT = std::integral_constant<int, ATT_TILE>;
    const bool few = rows <= ATT_FEW;
    switch (g) {
    case 1: few ? run(I1{}, I1{}) : run(IT{}, I1{}); break;
    case 2: few ? run(I1{}, std::integral_constant<int, 2>{}) : run(IT{}, std::integral_constant<int, 2>{}); break;
    case 3: few ? run(I1{}, std::integral_constant<int, 3>{}) : run(IT{}, std::integral_constant<int, 3>{}); break;
    case 4: few ? run(I1{}, std::integral_constant<int, 4>{}) : run(IT{}, std::integral_constant<int, 4>{}); break;
    default: throw std::invalid_argument("ink_attend: 1 to 4 query heads a key head");
    }
    ck("ink_attend");
    combine_kernel<<<dim3(heads, rows), ATT_DIM, 0, st>>>(part, out, heads, rows, splits);
    ck("ink_attend combine");
}

bool ink_gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows, void* stream) {
    if (rows < 1 || rows > GEMV_ROWS || n_in % 4 != 0) return false;
    const cudaStream_t st = cs(stream);
    const auto go = [&](auto rr) {
        constexpr int R = decltype(rr)::value;
        if (n_in >= 512) gemv_wide_kernel<R><<<n_out, 256, 0, st>>>(w, x, y, n_in, n_out);
        else gemv_short_kernel<R><<<grid(n_out, 8), 8 * WARP, 0, st>>>(w, x, y, n_in, n_out);
    };
    switch (rows) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    default: go(std::integral_constant<int, 8>{}); break;
    }
    ck("ink_gemv_f32");
    return true;
}

void ink_rows(const float* src, int src_stride, float* dst, int dst_stride, int n, int rows, bool add, void* stream) {
    if (rows <= 0) return;
    rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(src, src_stride, dst, dst_stride, n, add);
}

size_t ink_shortlist_floats(int rows) { return (size_t) rows * HEAD_CHUNKS; }

void ink_shortlist(const float* approx, int n, int stride, float margin, int* ids, int* count, int cap, int rows,
                   void* stream, float* scratch) {
    if (rows <= 0) return;
    const dim3 g(HEAD_CHUNKS, rows);
    head_max_kernel<<<g, 256, 0, cs(stream)>>>(approx, n, stride, scratch, count);
    shortlist_kernel<<<g, 256, 0, cs(stream)>>>(approx, n, stride, margin, scratch, ids, count, cap);
}

void ink_rescore_q8(const void* w, const float* h, int n_embd, const int* ids, const int* count, int cap,
                    float* logits, int stride, int rows, void* stream) {
    if (rows <= 0) return;
    rescore_kernel<<<dim3(grid(cap, 8), rows), 8 * WARP, 0, cs(stream)>>>((const BlockQ8*) w, h, n_embd, ids, count, cap,
                                                                          logits, stride);
}

void ink_fill_tail(float* x, int from, int n, int rows, void* stream) {
    if (rows <= 0 || from >= n) return;
    fill_tail_kernel<<<dim3(grid(n - from, 256), rows), 256, 0, cs(stream)>>>(x, from, n);
}

void ink_kv_sim(const float* x, float* y, int rows, int kvw, int block, int rot, void* stream) {
    if (rows <= 0) return;
    if (kvw % block != 0) throw std::invalid_argument("ink_kv_sim: the row is not whole blocks");
    const dim3 g(rows, kvw / block);
    if (block == 32) kv_sim_kernel<32><<<g, 32, 0, cs(stream)>>>(x, y, kvw, rot);
    else if (block == 64) kv_sim_kernel<64><<<g, 64, 0, cs(stream)>>>(x, y, kvw, rot);
    else if (block == 128) kv_sim_kernel<128><<<g, 128, 0, cs(stream)>>>(x, y, kvw, rot);
    else throw std::invalid_argument("ink_kv_sim: blocks of 32, 64 or 128");
    ck("ink_kv_sim");
}

}  // namespace strata::poly
