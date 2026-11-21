// src/poly/mmq_nvfp4_x2.cu - NVFP4's prompt products on RTX 50's FP4 tensor cores with each activation row as two FP4
// terms (STRATA_PREFILL_NVFP4=w4a4x2): about 8-bit activations at FP4 MMA rates.
//
// The method is Basalt's (https://github.com/jesdga95/basalt, MIT; src/prefill/moe_fp4.cu,
// docs/precision/inkling2.md): a row keeps one FP32 scale g = amax / (6 * 448); per 16 values its first term is their
// E2M1 rounding under a UE4M3 scale of their largest |value| / (6 g), and its second term the E2M1 rounding of what the
// first leaves, under a UE4M3 scale of that rest's largest |value| / (6 g).  Both terms go through each weight fragment
// of the block-scaled MMA on one FP32 chain; the row's g is applied when the tile is written, and each matrix's FP32
// scale after that, by the caller (as for W4A4).  One term (W4A4) is about 4 bits.
//
// The tiles are llama.cpp's NVFP4 FP4 tiles (ggml-cuda mmq.cuh, MIT): its weight loader, write-back and stream-k
// fixup are used as they are, and its kernel and launch are copied here with the tile loop changed - the weight tile
// is loaded once per K step and multiplied by both terms of the rows, which sit side by side in shared memory.  The
// rows are laid out as the W4A4 rows (block_fp4_mmq), the terms of a 256-column block one behind the other:
// block (2 kb + t) * rows + i is term t of row i's columns [256 kb, 256 kb + 256).
//
// The block-scaled FP4 MMA exists only in the arch-specific target, so this unit is built with 12x -> 12xa
// (CMakeLists.txt, with the W4A4 instance); Mmq takes it only on a 12.0 card.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2026 Jesús Gulfo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "common.cuh"

#define mul_mat_q_switch_J strata_poly_x2_unused_mul_mat_q_switch_J
#define mul_mat_q_case strata_poly_x2_unused_mul_mat_q_case
#include "mmq.cuh"

#include <cuda_fp8.h>

#include <climits>
#include <cstdint>

namespace {

constexpr ggml_type kT = GGML_TYPE_NVFP4;
constexpr int kQuantWarps = 8;

// ------------------------------------------------------------------------------------------------------- quantizer

#if defined(BLACKWELL_MMA_AVAILABLE)
// An E2M1 code's magnitude (the low three bits) - Basalt's e2m1_mag
__device__ __forceinline__ float e2m1_mag(unsigned c) {
    const unsigned m = c & 7u;
    if (m < 2u) return 0.5f * (float) m;
    return ((m & 1u) ? 1.5f : 1.0f) * (float) (1u << ((m >> 1) - 1u));
}

// 16 values times inv as E2M1 codes in the NVFP4 sub-block layout: byte j holds value j in its low nibble and value
// j + 8 in its high one (ggml's block_nvfp4, and quantize.cu's FP4 rows)
__device__ __forceinline__ uint2 e2m1_block(const float* v, float inv) {
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned short c;
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
            : "=h"(c) : "f"(v[j] * inv), "f"(v[j + 8] * inv));
        w[j >> 2] |= (uint32_t) (c & 0xff) << (8 * (j & 3));
    }
    return make_uint2(w[0], w[1]);
}

// v minus the 16 values the codes w stand for under the scale sg; returns the largest |value| left
__device__ __forceinline__ float e2m1_rest(float* v, uint2 w, float sg) {
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint32_t word = (j & 7) < 4 ? w.x : w.y;
        const unsigned c = (word >> (8 * (j & 3) + (j >= 8 ? 4 : 0))) & 15u;
        const float d = e2m1_mag(c) * sg;
        v[j] -= (c & 8u) ? -d : d;
        m = fmaxf(m, fabsf(v[j]));
    }
    return m;
}

// a UE4M3 scale byte for s >= 0 (E4M3, round to nearest, saturated) and its value
__device__ __forceinline__ uint8_t ue4m3(float s, float& value) {
    const __nv_fp8_storage_t b = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
    value = __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
    return (uint8_t) b;
}

#endif  // defined(BLACKWELL_MMA_AVAILABLE)

// One warp a row: row r of y is row ids[r] of x (or r), ne00 values padded with zeros to nsub sub-blocks of 16.
template <bool vec>
__global__ void __launch_bounds__(kQuantWarps * 32) quantize_x2(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                                                block_fp4_mmq* __restrict__ y, float* __restrict__ scale,
                                                                const int ne00, const int64_t ld, const int nsub,
                                                                const int64_t rows) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    // Stream-k's partial tile writes use a whole tile of row scales.
    if (blockIdx.x == 0 && threadIdx.x < 128) scale[rows + threadIdx.x] = 0.0f;
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * kQuantWarps + (threadIdx.x >> 5);
    if (r >= rows) return;
    const float* xr = x + (int64_t) (ids ? ids[r] : r) * ld;
    float amax = 0.0f;
    if (vec) {
        for (int i = 4 * lane; i < ne00; i += 4 * 32) {
            const float4 v = *(const float4*) (xr + i);
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    } else {
        for (int i = lane; i < ne00; i += 32) amax = fmaxf(amax, fabsf(xr[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float g = amax * (1.0f / (6.0f * 448.0f));
    if (lane == 0) scale[r] = g;
    const float inv_g = g > 0.0f ? 1.0f / g : 0.0f;
    for (int sb = lane; sb < nsub; sb += 32) {
        float v[16];
        const int i0 = 16 * sb;
        if (i0 < ne00) {   // ne00 is a multiple of 64: a sub-block is all in or all out
            if (vec) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float4 t = *(const float4*) (xr + i0 + 4 * q);
                    v[4 * q] = t.x; v[4 * q + 1] = t.y; v[4 * q + 2] = t.z; v[4 * q + 3] = t.w;
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) v[j] = xr[i0 + j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) v[j] = 0.0f;
        }
        float bm = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) bm = fmaxf(bm, fabsf(v[j]));
        float sd1, sd2;
        const uint8_t s1 = ue4m3(bm * inv_g * (1.0f / 6.0f), sd1);
        const uint2 q1 = e2m1_block(v, sd1 > 0.0f ? 1.0f / (sd1 * g) : 0.0f);
        const float m2 = e2m1_rest(v, q1, sd1 * g);
        const uint8_t s2 = ue4m3(m2 * inv_g * (1.0f / 6.0f), sd2);
        const uint2 q2 = e2m1_block(v, sd2 > 0.0f ? 1.0f / (sd2 * g) : 0.0f);
        const int kb = sb / 16, sub = sb % 16;
        block_fp4_mmq* b1 = y + (int64_t) (2 * kb) * rows + r;
        block_fp4_mmq* b2 = b1 + rows;
        ((uint2*) b1->qs)[sub] = q1;
        ((uint8_t*) b1->d4)[sub] = s1;
        ((uint2*) b2->qs)[sub] = q2;
        ((uint8_t*) b2->d4)[sub] = s2;
    }
#else
    GGML_UNUSED_VARS(x, ids, y, scale, ne00, ld, nsub, rows);
    NO_DEVICE_CODE;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)
}

// ---------------------------------------------------------------------------------------------------------- tiles

// mmq.cuh's ggml_cuda_mmq_vec_dot_fp4_fp4_mma with the rows' two terms: the weight fragments are loaded once and each
// goes through the MMA with term 1 and then term 2 on one FP32 chain
template <int J, bool fallback>
static __device__ __forceinline__ void vec_dot_x2(const int* __restrict__ x, const int* __restrict__ y1,
                                                  const int* __restrict__ y2, float* __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile<8, 8, int> tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(kT, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(kT, J, fallback);
    constexpr int ntx = rows_per_warp / tile_C::I;
    constexpr int nfrags = MMQ_TILE_NE_K / tile_A::J;

    const int yo = (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);
    y1 += yo;
    y2 += yo;

    const int* x_qs = (const int*) x;
    const uint32_t* x_sc = (const uint32_t*) (x_qs + 2 * MMQ_TILE_NE_K);

    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    tile_A A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * sram_stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        tile_B B1[nfrags], B2[nfrags];
        uint32_t scaleB1[nfrags], scaleB2[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B1[frag], y1 + 4 + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            load_generic(B2[frag], y2 + 4 + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB1[frag] = ((const uint32_t*) y1)[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
            scaleB2[frag] = ((const uint32_t*) y2)[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                mma_block_scaled_fp4<kT>(C, A[n][frag], B1[frag], scaleA[n][frag], scaleB1[frag]);
                mma_block_scaled_fp4<kT>(C, A[n][frag], B2[frag], scaleA[n][frag], scaleB2[frag]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}

// shared memory: the tile's ids, the two terms' row tiles, the weight tile
template <int J, bool fallback> static constexpr __device__ int y_tile_ints() {
    return GGML_PAD(J * MMQ_TILE_Y_K, ggml_cuda_mmq_get_nthreads(kT, J, fallback));
}

static size_t x2_nbytes_shared(const ggml_cuda_mmq_config& config, const int cc) {
    const size_t nbs_ids = config.J * sizeof(int);
    const size_t nbs_x = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    const size_t nbs_y = config.J * (sizeof(block_q8_1_mmq));
    return nbs_ids + nbs_x + 2 * GGML_PAD(nbs_y, config.nthreads * sizeof(int));
}

// mmq.cuh's mul_mat_q_process_tile for the two-term rows
template <int J, bool fallback, bool fixup>
static __device__ __forceinline__ void x2_process_tile(
        const char* __restrict__ x, const int offset_x, const int* __restrict__ y, const int* __restrict__ ids_dst,
        float* __restrict__ dst, float* __restrict__ tmp_fixup, const float* __restrict__ y_scale, const int stride_row_x,
        const int ncols_y, const int stride_col_dst, const int tile_x_max_i, const int tile_y_max_j, const int kb0_start,
        const int kb0_stop) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(kT, J, fallback) / warp_size;
    constexpr int qk = ggml_cuda_type_traits<kT>::qk;
    constexpr int I = ggml_cuda_mmq_get_I(kT, J, fallback);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<kT, J, fallback>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<kT, J, fallback>();
    constexpr int TY = y_tile_ints<J, fallback>();

    extern __shared__ int data_mul_mat_q[];
    int* tile_y = data_mul_mat_q + J;
    int* tile_x = tile_y + 2 * TY;

    constexpr int ITER_K = ggml_cuda_mmq_get_K_vram(kT, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;
    static_assert(ITER_K == 2 * QK_FP4_MMQ, "a K step is two blocks of rows");

    float sum[J * I / (nwarps * warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        const int kb = kb0 * qk / QK_FP4_MMQ;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int* by1 = y + ncols_y * (2 * (kb + h)) * sz;
            const int* by2 = by1 + ncols_y * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by1[l];
                tile_y[TY + l] = by2[l];
            }
            __syncthreads();
            vec_dot_x2<J, fallback>(tile_x, tile_y, tile_y + TY, sum, h * MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x * (J * I), y_scale, I, I - 1, J - 1);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// mmq.cuh's stream-k scheduling, with the two-term tile loop
template <int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(kT, J, fallback), ggml_cuda_mmq_get_occupancy(kT, J, fallback))
static __global__ void mul_mat_q_x2(
        const char* __restrict__ x, const int* __restrict__ y, const int32_t* __restrict__ ids_dst,
        const int32_t* __restrict__ expert_bounds, float* __restrict__ dst, float* __restrict__ tmp_fixup,
        const float* __restrict__ y_scale, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst, const uint3 channel_ratio,
        const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const uint3 ntx) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    static_assert(ggml_cuda_mmq_get_stream_k(kT, J, fallback));
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(kT, J, fallback) / warp_size;
    constexpr int qk = ggml_cuda_type_traits<kT>::qk;
    constexpr int I = ggml_cuda_mmq_get_I(kT, J, fallback);

    const uint32_t nty = (nrows_x + I - 1) / I;

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > J && j >= J) break;
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    constexpr int ITER_K = ggml_cuda_mmq_get_K_vram(kT, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    int kbc = int64_t(blockIdx.x) * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1) * (nsamples_y.z * nchannels_y.z * ntx.z * nty * blocks_per_ne00.z) / gridDim.x;

    kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        const int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;

        int col_low = 0;
        int col_high = ncols_dst;
        int col_diff = ncols_dst;
        int offset_y = wt * stride_sample_y + zt * stride_channel_y;
        int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * J * stride_col_dst;
        int offset_y_scale = wt * nchannels_y.z * ncols_y + zt * ncols_y;

        if (ids_dst) {
            col_low = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;
            offset_y = 0;
            offset_dst = 0;
            offset_y_scale = 0;

            if (jt * J >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);
                kb0_start = 0;
                kb0_stop = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
                const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
                if (j0 + nwarps * warp_size > J && j >= J) break;
                ids_dst_shared[j] = ids_dst[col_low + jt * J + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it * I;
        offset_y_scale += col_low + jt * J;
        const float* y_scale_tile = y_scale + offset_y_scale;

        const int tile_x_max_i = nrows_x - it * I - 1;
        const int tile_y_max_j = col_diff - jt * J - 1;
        const int offset_x = fastdiv(wt, sample_ratio) * stride_sample_x + fastdiv(zt, channel_ratio) * stride_channel_x +
                             it * I * stride_row_x;

        x2_process_tile<J, fallback, false>(x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup,
                                            y_scale_tile, stride_row_x, ncols_y, stride_col_dst, tile_x_max_i,
                                            tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);
        kb0_start = 0;
        kb0_stop = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) return;

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    int col_low = 0;
    int col_high = ncols_dst;
    int col_diff = ncols_dst;
    int offset_y = wt * stride_sample_y + zt * stride_channel_y;
    int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * J * stride_col_dst;
    int offset_y_scale = wt * nchannels_y.z * ncols_y + zt * ncols_y;

    if (ids_dst) {
        col_low = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;
        offset_y = 0;
        offset_dst = 0;
        offset_y_scale = 0;
        if (jt * J >= col_diff) return;

        // the fixup buffer is contiguous: the ids go back to the tile's own
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
            const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
            if (j0 + nwarps * warp_size > J && j >= J) break;
            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it * I;
    offset_y_scale += col_low + jt * J;
    const float* y_scale_tile = y_scale + offset_y_scale;

    const int tile_x_max_i = nrows_x - it * I - 1;
    const int tile_y_max_j = col_diff - jt * J - 1;
    const int offset_x = fastdiv(wt, sample_ratio) * stride_sample_x + fastdiv(zt, channel_ratio) * stride_channel_x +
                         it * I * stride_row_x;

    x2_process_tile<J, fallback, true>(x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup,
                                       y_scale_tile, stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j,
                                       kb0_start, kb0_stop);
#else
    GGML_UNUSED_VARS(x, y, ids_dst, expert_bounds, dst, tmp_fixup, y_scale, blocks_per_ne00, nrows_x, ncols_dst,
                     stride_row_x, ncols_y, stride_col_dst, channel_ratio, nchannels_y, stride_channel_x,
                     stride_channel_y, stride_channel_dst, sample_ratio, nsamples_y, stride_sample_x, stride_sample_y,
                     stride_sample_dst, ntx);
    NO_DEVICE_CODE;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)
}

// mmq.cuh's launch_mul_mat_q with the x2 kernel and its shared memory; the fixup is mmq.cuh's
template <int J, bool fallback>
static void launch_x2(ggml_backend_cuda_context& ctx, const mmq_args& args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(kT, J, fallback, cc);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = (int) x2_nbytes_shared(config, cc);

    const dim3 block_dims(warp_size, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_x2<J, fallback>), nbytes_shared);

    const int nty = (args.nrows_x + config.I - 1) / config.I;
    const int ntx = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y % args.nsamples_x == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio = args.nsamples_y / args.nsamples_x;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<kT>::qk);
    const uint3 ntx_fd = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd = init_fastdiv_values(sample_ratio);

    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm * tiles_nwaves);
    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);

    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30));

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;

    ggml_cuda_pool& pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) tmp_fixup.alloc(block_nums_stream_k.x * config.J * config.I);

    const dim3 block_nums_fixup(block_nums_stream_k.x, config.I / warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y / 2, block_dims.z);

    mul_mat_q_x2<J, fallback><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>(
        args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale, blocks_per_ne00_fd,
        args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst, channel_ratio_fd, nchannels_y_fd,
        args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst, sample_ratio_fd, nsamples_y_fd,
        args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst, ntx_fd);

    if (!fixup_needed) return;

    CUDA_CHECK(cudaGetLastError());
    mul_mat_q_stream_k_fixup<kT, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>(
        args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
        args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst, ntx_fd);
}

// mmq.cuh's mul_mat_q_switch_J: the tile width with the fewest tiles that fits the card's shared memory
template <bool fallback>
static void switch_J_x2(ggml_backend_cuda_context& ctx, const mmq_args& args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(kT, J, fallback, cc);
        if (config.sram_layout != GGML_CUDA_MMQ_SRAM_LAYOUT_FP4) continue;
        if (x2_nbytes_shared(config, cc) > smpbo) continue;
        const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
#define X2_J(J) case J: \
        if constexpr (ggml_cuda_mmq_get_config_blackwell(kT, J, fallback).sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_FP4) \
            launch_x2<J, fallback>(ctx, args, stream); \
        break;
        X2_J(8) X2_J(16) X2_J(24) X2_J(32) X2_J(40) X2_J(48) X2_J(56) X2_J(64)
        X2_J(72) X2_J(80) X2_J(88) X2_J(96) X2_J(104) X2_J(112) X2_J(120) X2_J(128)
#undef X2_J
        default:
            fprintf(stderr, "mmq x2: J_best=%d\n", J_best);
            GGML_ABORT("fatal error");
    }
}

}  // namespace

namespace strata::poly {

// Mmq::fits() for these tiles: a tile width whose configuration the card has and whose shared memory it holds
bool nvfp4_x2_fits(int64_t w_rows, int cc, size_t smpbo) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(kT, J, w_rows % 128 != 0, cc);
        if (c.sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_FP4 && x2_nbytes_shared(c, cc) <= smpbo) return true;
    }
    return false;
}

// Rows of cols values (a multiple of 64; ld floats apart, row i of x is row ids[i], or i) as two FP4 terms at xq,
// padded to pad_cols columns, and their scales at scale.
void nvfp4_x2_quantize(const float* x, const int32_t* ids, void* xq, float* scale, int64_t cols, int64_t ld,
                       int64_t pad_cols, int64_t rows, cudaStream_t s) {
    const bool vec = (uintptr_t) x % 16 == 0 && ld % 4 == 0;
    const unsigned grid = (unsigned) ((rows + kQuantWarps - 1) / kQuantWarps);
    const int nsub = (int) (pad_cols / 16);
    if (vec)
        quantize_x2<true><<<grid, kQuantWarps * 32, 0, s>>>(x, ids, (block_fp4_mmq*) xq, scale, (int) cols, ld, nsub,
                                                             rows);
    else
        quantize_x2<false><<<grid, kQuantWarps * 32, 0, s>>>(x, ids, (block_fp4_mmq*) xq, scale, (int) cols, ld, nsub,
                                                              rows);
}

void nvfp4_x2(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    if (a.nrows_x % 128 == 0)
        switch_J_x2<false>(ctx, a, s);
    else
        switch_J_x2<true>(ctx, a, s);
}

}  // namespace strata::poly
