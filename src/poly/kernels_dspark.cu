// src/poly/kernels_dspark.cu - the DSpark drafter's own kernels for Inkling 2 (kernels.hpp, "DSpark").  The block's
// products, norms, rotary heads, attention and the head's products are the engine's shared kernels; these are the
// pieces that are the drafter's alone: the target's features into bf16 taps, their mixes, the Markov head's
// recurrence, the draft logits and the fold of the head made at load.
//
// Adapted from Basalt's src/models/inkling2/kernels/dspark.cu (https://github.com/jesdga95/basalt, MIT, Copyright the
// Basalt authors), whose forward follows vLLM's Inkling 2 DSpark draft.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int kThreads = 256;
constexpr int kMaxPer = 16;   // values a thread holds of a row: rows of up to 4096

cudaStream_t cs(void* s) { return (cudaStream_t) s; }
int grid(int64_t n, int b) { return (int) ((n + b - 1) / b); }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dspark ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(uint16_t u) { return __uint_as_float((uint32_t) u << 16); }

// to bf16, rounded to the nearest even (a NaN stays one)
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// the sum of one value a thread over a block of kThreads, in a fixed order; every thread gets it
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < kThreads / WARP; ++j) s += red[j];
    __syncthreads();
    return s;
}

__global__ void tap_kernel(const float* __restrict__ src, int src_stride, uint16_t* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(size_t) blockIdx.y * n + i] = f2bf(src[(size_t) blockIdx.y * src_stride + i]);
}

// Row r (blockIdx.x), mix j (blockIdx.y): m = sum_k alpha[j][k] f_k, the three taps then the final state; RMS-normed
// and times mix j's weights, into out's slab j (j < 3) or the Markov head's ring at the row's position (j = 3).
__global__ void __launch_bounds__(kThreads) mix_kernel(const uint16_t* __restrict__ tap, int64_t slot,
                                                       const float* __restrict__ fin, DsMix a, int n, long long pos0,
                                                       float* __restrict__ out, int rows, float* __restrict__ fr,
                                                       int fr_cap) {
    __shared__ float red[kThreads / WARP];
    const int r = blockIdx.x, j = blockIdx.y;
    float al[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) al[k] = a.alpha[j * 4 + k];
    float v[kMaxPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        float m = 0.f;
        if (c < n) {
            const size_t at = (size_t) r * n + c;
            m = al[0] * bf16f(tap[at]) + al[1] * bf16f(tap[slot + at]) + al[2] * bf16f(tap[2 * slot + at]) +
                al[3] * fin[at];
        }
        v[i] = m;
        ss += m * m;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) n + a.eps);
    float* dst = j < 3 ? out + ((size_t) j * rows + r) * n : fr + (size_t) ((pos0 + r) % fr_cap) * n;
    const float* w = a.norm[j];
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        if (c < n) dst[c] = v[i] * inv * w[c];
    }
}

// The Markov head's recurrence, kDsSteps blocks of 32 warps, a warp an index j of the rank: its gate, candidate and
// latent rows of w_sp (rank rows apart, 2 rank values: the state's then the previous token's w1 row) times
// [state; w1[*prev]], plus ctx (the step's rnn bias, hidden and context terms).
constexpr int kStepWarps = 32;
__global__ void __launch_bounds__(kStepWarps * WARP) markov_kernel(const float* __restrict__ ctx,
                                                                  const uint16_t* __restrict__ w_sp,
                                                                  const uint16_t* __restrict__ w1,
                                                                  const int32_t* __restrict__ prev, int rank,
                                                                  const float* __restrict__ state_in,
                                                                  float* __restrict__ state_out,
                                                                  float* __restrict__ latent) {
    extern __shared__ float xs[];   // 2 * rank
    const int64_t tok = *prev;
    for (int i = threadIdx.x; i < 2 * rank; i += blockDim.x)
        xs[i] = i < rank ? state_in[i] : bf16f(w1[tok * rank + (i - rank)]);
    __syncthreads();
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int j = blockIdx.x * kStepWarps + w;
    if (j >= rank) return;
    float raw[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int o = g * rank + j;
        const uint32_t* row = reinterpret_cast<const uint32_t*>(w_sp + (size_t) o * 2 * rank);   // pairs of BF16
        float acc = 0.f;
        for (int k = lane; k < rank; k += WARP) {
            const uint32_t u = row[k];
            acc += __uint_as_float(u << 16) * xs[2 * k] + __uint_as_float(u & 0xffff0000u) * xs[2 * k + 1];
        }
        raw[g] = warp_sum(acc) + ctx[o];
    }
    if (lane == 0) {
        const float gate = 1.f / (1.f + expf(-raw[0]));
        state_out[j] = gate * xs[j] + (1.f - gate) * tanhf(raw[1]);
        latent[j] = tanhf(raw[2]);
    }
}

__global__ void head_kernel(float* __restrict__ x, const float* __restrict__ fl, const float* __restrict__ fb, float cap,
                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] + fl[i] + fb[i];
    x[i] = cap > 0.f ? cap * tanhf(v / cap) : v;
}

// A warp a 32-value block of a fold row: y holds k values of `chunk` rows column by column ([k][chunk]); the row's
// block b becomes ggml's block_q8_0 (an fp16 scale, then 32 codes) at fold row v.
__global__ void fold_q8_kernel(const float* __restrict__ y, int chunk, int k, uint8_t* __restrict__ fold) {
    const int64_t wi = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP, nb = k / 32;
    if (wi >= (int64_t) chunk * nb) return;
    const int v = (int) (wi / nb), b = (int) (wi % nb);
    const float x = y[(size_t) (b * 32 + lane) * chunk + v];
    const float d = warp_max(fabsf(x)) / 127.f;   // ggml's quantize_row_q8_0_ref
    const float id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* blk = fold + ((size_t) v * nb + b) * 34;
    blk[2 + lane] = (uint8_t) (int8_t) roundf(x * id);
    if (lane == 0) {
        const __half h = __float2half(d);
        blk[0] = (uint8_t) (__half_as_ushort(h) & 0xff);
        blk[1] = (uint8_t) (__half_as_ushort(h) >> 8);
    }
}

// ggml's block_q4_0 of a warp's 32 values (quantize_row_q4_0_ref): the value of largest magnitude, with its sign,
// over -8 is the scale; code = min(15, (int) (x / d + 8.5)), value j in the low nibble of byte j, j + 16 in its high one
__device__ __forceinline__ void q4_0_block(float x, uint8_t* blk, int lane) {
    float m = x, a = fabsf(x);
#pragma unroll
    for (int o = WARP / 2; o > 0; o /= 2) {
        const float om = __shfl_xor_sync(0xffffffffu, m, o), oa = __shfl_xor_sync(0xffffffffu, a, o);
        if (oa > a || (oa == a && om < m)) {
            a = oa;
            m = om;
        }
    }
    const float d = m / -8.f, id = d != 0.f ? 1.f / d : 0.f;
    const int q = min(15, (int) (x * id + 8.5f));
    const int hi = __shfl_down_sync(0xffffffffu, q, 16);
    if (lane < 16) blk[2 + lane] = (uint8_t) (q | (hi << 4));
    if (lane == 0) {
        const __half h = __float2half(d);
        blk[0] = (uint8_t) (__half_as_ushort(h) & 0xff);
        blk[1] = (uint8_t) (__half_as_ushort(h) >> 8);
    }
}

__global__ void fold_q4_kernel(const float* __restrict__ y, int chunk, int k, uint8_t* __restrict__ fold) {
    const int64_t wi = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP, nb = k / 32;
    if (wi >= (int64_t) chunk * nb) return;
    const int v = (int) (wi / nb), b = (int) (wi % nb);
    q4_0_block(y[(size_t) (b * 32 + lane) * chunk + v], fold + ((size_t) v * nb + b) * 18, lane);
}

// The draft logits' pick in two passes: each block of the first takes a share of the n logits x = cap tanh((base +
// fl + fb) / cap) and keeps its first largest and the sum of exp(x - that); the second combines the blocks.
constexpr int kPickBlocks = 120;
struct PickPart {
    float m, s;
    int i;
};

__device__ __forceinline__ void pick_merge(float& m, float& s, int& i, float om, float os, int oi) {
    if (om > m || (om == m && oi < i)) {
        s = s * __expf(m - om) + os;
        m = om;
        i = oi;
    } else if (om > -INFINITY) {
        s += os * __expf(om - m);
    }
}

// The fold's term of row v is computed here from its Q4_0 row (R values: R / 32 blocks of 18 bytes) and the latent.
__global__ void __launch_bounds__(kThreads) pick_part_kernel(const float* __restrict__ base,
                                                             const uint8_t* __restrict__ fold,
                                                             const float* __restrict__ latent, int rank,
                                                             const float* __restrict__ fb, float cap, int n,
                                                             PickPart* __restrict__ part) {
    extern __shared__ float lat[];
    for (int i = threadIdx.x; i < rank; i += kThreads) lat[i] = latent[i];
    __syncthreads();
    const int nb = rank / 32;
    float m = -INFINITY, s = 0.f;
    int idx = 0x7fffffff;
    for (int v = blockIdx.x * kThreads + threadIdx.x; v < n; v += gridDim.x * kThreads) {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(fold + (size_t) v * nb * 18);   // 2-byte aligned
        float f = 0.f;
        for (int b = 0; b < nb; ++b) {
            const uint16_t* blk = row + b * 9;
            const float d = __half2float(__ushort_as_half(blk[0]));
            const float* l = lat + b * 32;
            float acc = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t w = blk[1 + j];   // bytes 2j and 2j + 1 of the codes
                acc += (float) ((int) (w & 15u) - 8) * l[2 * j] + (float) ((int) ((w >> 4) & 15u) - 8) * l[2 * j + 16] +
                       (float) ((int) ((w >> 8) & 15u) - 8) * l[2 * j + 1] +
                       (float) ((int) ((w >> 12) & 15u) - 8) * l[2 * j + 17];
            }
            f += d * acc;
        }
        float x = base[v] + f + fb[v];
        if (cap > 0.f) x = cap * tanhf(x / cap);
        if (x > m) {
            s = s * __expf(m - x) + 1.f;
            m = x;
            idx = v;
        } else {
            s += __expf(x - m);
        }
    }
#pragma unroll
    for (int o = WARP / 2; o > 0; o /= 2)
        pick_merge(m, s, idx, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o),
                   __shfl_xor_sync(0xffffffffu, idx, o));
    __shared__ PickPart w[kThreads / WARP];
    if (threadIdx.x % WARP == 0) w[threadIdx.x / WARP] = {m, s, idx};
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 1; j < kThreads / WARP; ++j) pick_merge(m, s, idx, w[j].m, w[j].s, w[j].i);
        part[blockIdx.x] = {m, s, idx};
    }
}

// ---- the block's attention: every row of the block reads the same keys, the context's window before it and the
// block's own rows.  A launch block takes a kv head and a split of the keys for all the rows and the heads of its group
// (QPW queries a warp, a key a lane); its partial sums are combined by the second kernel.
constexpr int kAttDim = 128, kAttTile = 32, kAttWarps = 8, kAttSplitKeys = 128, kAttMaxSplits = 64;

template <int QPW>
__global__ void __launch_bounds__(kAttWarps * WARP) block_attn_kernel(const float* __restrict__ q,
                                                                    const __half2* __restrict__ kc,
                                                                    const __half2* __restrict__ vc, int heads,
                                                                    int kv_heads, long long lo, long long hi, int cells,
                                                                    float scale, int split_keys,
                                                                    float* __restrict__ part) {
    constexpr int NQ = QPW * kAttWarps, H2 = kAttDim / 2;
    __shared__ float qs[NQ][kAttDim];
    __shared__ __half2 ks[kAttTile][H2 + 1];
    __shared__ __align__(8) __half2 vs[kAttTile][H2];
    const int split = blockIdx.x, kvh = blockIdx.y, G = heads / kv_heads;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    for (int i = threadIdx.x; i < NQ * kAttDim; i += blockDim.x) {
        const int qi = i / kAttDim, d = i % kAttDim, r = qi / G, g = qi % G;
        qs[qi][d] = q[((size_t) r * heads + (size_t) kvh * G + g) * kAttDim + d] * scale;
    }
    const long long k0 = lo + (long long) split * split_keys, k1 = min(hi, k0 + split_keys);
    float m[QPW], l[QPW], acc[QPW][4];
#pragma unroll
    for (int qq = 0; qq < QPW; ++qq) {
        m[qq] = -INFINITY;
        l[qq] = 0.f;
        acc[qq][0] = acc[qq][1] = acc[qq][2] = acc[qq][3] = 0.f;
    }
    for (long long t0 = k0; t0 < k1; t0 += kAttTile) {
        __syncthreads();
        for (int i = threadIdx.x; i < kAttTile * H2; i += blockDim.x) {
            const int j = i / H2, c = i % H2;
            const long long pos = t0 + j;
            __half2 kv = __float2half2_rn(0.f), vv = kv;
            if (pos < k1) {
                const size_t at = ((size_t) (pos % cells) * kv_heads + kvh) * H2 + c;
                kv = kc[at];
                vv = vc[at];
            }
            ks[j][c] = kv;
            vs[j][c] = vv;
        }
        __syncthreads();
        const bool valid = t0 + lane < k1;
        float p[QPW];
#pragma unroll
        for (int qq = 0; qq < QPW; ++qq) {
            const float* qv = qs[warp + kAttWarps * qq];
            float sc = 0.f;
#pragma unroll 8
            for (int c = 0; c < H2; ++c) {
                const float2 kf = __half22float2(ks[lane][c]);
                sc += qv[2 * c] * kf.x + qv[2 * c + 1] * kf.y;
            }
            sc = valid ? sc : -INFINITY;
            const float mx = fmaxf(m[qq], warp_max(sc));
            const float corr = m[qq] == -INFINITY ? 0.f : __expf(m[qq] - mx);
            p[qq] = valid ? __expf(sc - mx) : 0.f;
            l[qq] = l[qq] * corr + warp_sum(p[qq]);
            m[qq] = mx;
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[qq][e] *= corr;
        }
#pragma unroll 4
        for (int j = 0; j < kAttTile; ++j) {
            const uint2 raw = *reinterpret_cast<const uint2*>(&vs[j][2 * lane]);
            const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
            const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
#pragma unroll
            for (int qq = 0; qq < QPW; ++qq) {
                const float pj = __shfl_sync(0xffffffffu, p[qq], j);
                acc[qq][0] += pj * v01.x;
                acc[qq][1] += pj * v01.y;
                acc[qq][2] += pj * v23.x;
                acc[qq][3] += pj * v23.y;
            }
        }
    }
#pragma unroll
    for (int qq = 0; qq < QPW; ++qq) {
        const int qi = warp + kAttWarps * qq;
        float* o = part + (((size_t) split * kv_heads + kvh) * NQ + qi) * (kAttDim + 2);
#pragma unroll
        for (int e = 0; e < 4; ++e) o[4 * lane + e] = acc[qq][e];
        if (lane == 0) {
            o[kAttDim] = m[qq];
            o[kAttDim + 1] = l[qq];
        }
    }
}

// block (head h, row r), a thread a value of the head: the splits' sums combined
__global__ void block_attn_join_kernel(const float* __restrict__ part, int splits, int heads, int kv_heads, int nq,
                                       float* __restrict__ out) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x, G = heads / kv_heads;
    const int kvh = h / G, qi = r * G + h % G;
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) mx = fmaxf(mx, part[(((size_t) s * kv_heads + kvh) * nq + qi) * (kAttDim + 2) + kAttDim]);
    float lsum = 0.f, o = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* p = part + (((size_t) s * kv_heads + kvh) * nq + qi) * (kAttDim + 2);
        const float w = p[kAttDim] == -INFINITY ? 0.f : __expf(p[kAttDim] - mx);
        lsum += p[kAttDim + 1] * w;
        o += p[d] * w;
    }
    out[((size_t) r * heads + h) * kAttDim + d] = lsum > 0.f ? o / lsum : 0.f;
}

__global__ void pick_join_kernel(const PickPart* __restrict__ part, int parts, Likeliest* __restrict__ out) {
    if (threadIdx.x != 0) return;
    float m = part[0].m, s = part[0].s;
    int idx = part[0].i;
    for (int j = 1; j < parts; ++j) pick_merge(m, s, idx, part[j].m, part[j].s, part[j].i);
    out->token = idx;
    out->prob = s > 0.f ? 1.f / s : 0.f;
}

}  // namespace

void ds_tap(const float* src, int src_stride, uint16_t* dst, int n, int rows, void* stream) {
    if (rows <= 0) return;
    tap_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(src, src_stride, dst, n);
    ck("tap");
}

void ds_mix(const uint16_t* tap, int64_t slot, const float* fin, const DsMix& a, int n, int64_t pos0, float* out,
            int rows, float* fr, int fr_cap, void* stream) {
    if (rows <= 0) return;
    if (n > kThreads * kMaxPer) throw std::invalid_argument("ds_mix: rows of up to 4096 values");
    mix_kernel<<<dim3(rows, 4), kThreads, 0, cs(stream)>>>(tap, slot, fin, a, n, (long long) pos0, out, rows, fr, fr_cap);
    ck("mix");
}

void ds_markov(const float* ctx, const uint16_t* w_sp, const uint16_t* w1, const int32_t* prev, int rank,
               const float* state_in, float* state_out, float* latent, void* stream) {
    markov_kernel<<<grid(rank, kStepWarps), kStepWarps * WARP, 2 * rank * sizeof(float), cs(stream)>>>(
        ctx, w_sp, w1, prev, rank, state_in, state_out, latent);
    ck("Markov step");
}

void ds_head(float* x, const float* fl, const float* fb, float cap, int n, void* stream) {
    head_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, fl, fb, cap, n);
    ck("head");
}

void ds_fold_q4(const float* y, int chunk, int k, void* fold, void* stream) {
    if (k % 32) throw std::invalid_argument("ds_fold_q4: whole blocks of 32");
    const int64_t warps = (int64_t) chunk * (k / 32);
    fold_q4_kernel<<<grid(warps * WARP, 256), 256, 0, cs(stream)>>>(y, chunk, k, (uint8_t*) fold);
    ck("fold");
}

size_t ds_pick_bytes() { return kPickBlocks * sizeof(PickPart); }

void ds_pick(const float* base, const void* fold, const float* latent, int rank, const float* fb, float cap, int n,
             void* scratch, Likeliest* out, void* stream) {
    if (rank % 32) throw std::invalid_argument("ds_pick: a rank of whole blocks of 32");
    pick_part_kernel<<<kPickBlocks, kThreads, rank * sizeof(float), cs(stream)>>>(
        base, (const uint8_t*) fold, latent, rank, fb, cap, n, (PickPart*) scratch);
    pick_join_kernel<<<1, WARP, 0, cs(stream)>>>((const PickPart*) scratch, kPickBlocks, out);
    ck("pick");
}

size_t ds_attend_floats(int rows, int heads, int kv_heads, int dim, int64_t keys) {
    (void) dim;
    const int64_t splits = std::min<int64_t>(kAttMaxSplits, (keys + kAttSplitKeys - 1) / kAttSplitKeys);
    return (size_t) splits * kv_heads * (heads / kv_heads) * rows * (kAttDim + 2);
}

void ds_attend(const float* q, const uint16_t* kc, const uint16_t* vc, int rows, int heads, int kv_heads, int dim,
               int64_t lo, int64_t hi, int cells, float scale, float* part, float* out, void* stream) {
    const int nq = (heads / kv_heads) * rows;
    if (dim != kAttDim || heads % kv_heads || nq % kAttWarps || nq / kAttWarps > 4 || hi <= lo)
        throw std::invalid_argument("ds_attend: heads of 128 and up to 32 queries a key head, a multiple of 8");
    const int64_t keys = hi - lo;
    const int64_t splits = std::min<int64_t>(kAttMaxSplits, (keys + kAttSplitKeys - 1) / kAttSplitKeys);
    const int split_keys = (int) ((keys + splits - 1) / splits);
    const dim3 g((unsigned) splits, (unsigned) kv_heads);
    const auto k2 = (const __half2*) kc;
    const auto v2 = (const __half2*) vc;
    switch (nq / kAttWarps) {
    case 1: block_attn_kernel<1><<<g, kAttWarps * WARP, 0, cs(stream)>>>(q, k2, v2, heads, kv_heads, lo, hi, cells, scale, split_keys, part); break;
    case 2: block_attn_kernel<2><<<g, kAttWarps * WARP, 0, cs(stream)>>>(q, k2, v2, heads, kv_heads, lo, hi, cells, scale, split_keys, part); break;
    case 3: block_attn_kernel<3><<<g, kAttWarps * WARP, 0, cs(stream)>>>(q, k2, v2, heads, kv_heads, lo, hi, cells, scale, split_keys, part); break;
    default: block_attn_kernel<4><<<g, kAttWarps * WARP, 0, cs(stream)>>>(q, k2, v2, heads, kv_heads, lo, hi, cells, scale, split_keys, part); break;
    }
    block_attn_join_kernel<<<dim3((unsigned) heads, (unsigned) rows), kAttDim, 0, cs(stream)>>>(part, (int) splits, heads,
                                                                                               kv_heads, nq, out);
    ck("attend");
}

void ds_fold_q8(const float* y, int chunk, int k, void* fold, void* stream) {
    if (k % 32) throw std::invalid_argument("ds_fold_q8: whole blocks of 32");
    const int64_t warps = (int64_t) chunk * (k / 32);
    fold_q8_kernel<<<grid(warps * WARP, 256), 256, 0, cs(stream)>>>(y, chunk, k, (uint8_t*) fold);
    ck("fold");
}

}  // namespace strata::poly
