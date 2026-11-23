// src/poly/cpu_dots.inc - the CPU experts' own dot products (cpu_experts.cpp), written once and compiled for each
// instruction set that has a byte dot product.  cpu_experts.cpp includes this file inside a namespace per set, with
//   KT              the functions' target attribute
//   DPBUSD(a, u, s) a plus the sums of each four products of u's unsigned bytes and s's signed ones (vpdpbusd)
// The sets compute the same integers and the same float operations in the same order, so the same bits: the AVX-512
// VNNI build (Zen 4, which has no AVX-VNNI) gives what the AVX-VNNI one gives.
//
// The types it reads (Ready, ReadyK4, Ready32, the block layouts and the tables) are cpu_experts.cpp's.

// ---- NVFP4 x Q8_0 -------------------------------------------------------------------------------------------------

// One block of one row into acc: the scale lanes of A are (s0 x4, s2 x4), of B (s1 x4, s3 x4).
KT inline void fp4_block(const block_nvfp4& x, const Ready& r, __m256& acc, const __m256i lut, const __m256i m4) {
    // lanes: lane 0 = qs 0..15 (sub 0, 1), lane 1 = qs 16..31 (sub 2, 3)
    const __m256i qs = _mm256_loadu2_m128i((const __m128i*) (x.qs + 16), (const __m128i*) x.qs);
    const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(qs, m4));
    const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(qs, 4), m4));
    const __m256i wa = _mm256_unpacklo_epi64(lo, hi);   // lane 0: sub 0, lane 1: sub 2
    const __m256i wb = _mm256_unpackhi_epi64(lo, hi);   // lane 0: sub 1, lane 1: sub 3
    __m256i sa = DPBUSD(_mm256_setzero_si256(), wa, _mm256_loadu_si256((const __m256i*) r.a));
    __m256i sb = DPBUSD(_mm256_setzero_si256(), wb, _mm256_loadu_si256((const __m256i*) r.b));
    sa = _mm256_sub_epi32(sa, _mm256_loadu_si256((const __m256i*) r.ca));
    sb = _mm256_sub_epi32(sb, _mm256_loadu_si256((const __m256i*) r.cb));
    int32_t d4;
    std::memcpy(&d4, x.d, 4);
    const __m128i di = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(d4));
    // ggml_ue4m3_to_fp32 without a gather (slow on the E-cores): 2^(e-7) (1 + m/8) / 2 = float bits (e + 119, m),
    // m 2^-10 where e is 0, and 0 for 0x7f
    const __m128i e = _mm_and_si128(_mm_srli_epi32(di, 3), _mm_set1_epi32(15));
    const __m128i m = _mm_and_si128(di, _mm_set1_epi32(7));
    const __m128 normal = _mm_castsi128_ps(
        _mm_or_si128(_mm_slli_epi32(_mm_add_epi32(e, _mm_set1_epi32(119)), 23), _mm_slli_epi32(m, 20)));
    const __m128 sub = _mm_mul_ps(_mm_cvtepi32_ps(m), _mm_set1_ps(0.0009765625f));
    __m128 ue = _mm_blendv_ps(normal, sub, _mm_castsi128_ps(_mm_cmpeq_epi32(e, _mm_setzero_si128())));
    ue = _mm_andnot_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(di, _mm_set1_epi32(0x7f)), _mm_set1_epi32(0x7f))), ue);
    const __m128 sc = _mm_mul_ps(ue, _mm_loadu_ps(r.dy));   // s0..s3
    const __m256 sc8 = _mm256_castps128_ps256(sc);
    const __m256 scale_a = _mm256_permutevar8x32_ps(sc8, _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2));
    const __m256 scale_b = _mm256_permutevar8x32_ps(sc8, _mm256_setr_epi32(1, 1, 1, 1, 3, 3, 3, 3));
    acc = _mm256_fmadd_ps(scale_a, _mm256_cvtepi32_ps(sa), acc);
    acc = _mm256_fmadd_ps(scale_b, _mm256_cvtepi32_ps(sb), acc);
}

KT inline float hsum8(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

// rows w0 and w1 (nb blocks each) against one ready activation
KT void fp4_dot2(const block_nvfp4* w0, const block_nvfp4* w1, const Ready* r, int nb, float* s0, float* s1) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kFp4Plus12));
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        fp4_block(w0[ib], r[ib], a0, lut, m4);
        fp4_block(w1[ib], r[ib], a1, lut, m4);
    }
    *s0 = hsum8(a0);
    *s1 = hsum8(a1);
}

KT float fp4_dot1(const block_nvfp4* w, const Ready* r, int nb) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kFp4Plus12));
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        fp4_block(w[ib], r[ib], a0, lut, m4);
        fp4_block(w[ib + 1], r[ib + 1], a1, lut, m4);
    }
    for (; ib < nb; ++ib) fp4_block(w[ib], r[ib], a0, lut, m4);
    return hsum8(_mm256_add_ps(a0, a1));
}

// ---- IQ4_XS x Q8_K ------------------------------------------------------------------------------------------------
// ggml's ggml_vec_dot_iq4_xs_q8_K looks the codes up as signed bytes and multiplies them with sign + maddubs + madd
// for every row.  Here a code looks up its value + 128 (unsigned, 1..241) so one vpdpbusd multiplies and sums four,
// and the 128 is taken off with the activation's sums (ReadyK4::corr, made once a call).  A super-block's sum is
// the integer ggml's is, (ls - 32) times each sub-block's; only the float adds of the lanes come in another order.

// The 8 sub-blocks' scales of a super-block, ls - 32, as int32 lanes
KT inline __m256i iq4xs_scales(const block_iq4_xs& x) {
    uint32_t sl;
    std::memcpy(&sl, x.scales_l, 4);
    const __m256i lo = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) sl),
                                                          _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)),
                                        _mm256_set1_epi32(0xf));
    const __m256i hi = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) x.scales_h),
                                                          _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
                                        _mm256_set1_epi32(3));
    return _mm256_sub_epi32(_mm256_or_si256(lo, _mm256_slli_epi32(hi, 4)), _mm256_set1_epi32(32));
}

// pair p (sub-blocks 2p and 2p + 1) of one row: lanes 0-3 sub-block 2p, 4-7 sub-block 2p + 1, times their scales
KT inline __m256i iq4xs_pair(const uint8_t* qs, const __m256i al, const __m256i ah, const __m256i corr,
                             const __m256i sc, const __m256i idx, const __m256i lut, const __m256i m4) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);   // lane 0: sub-block 2p's 16 bytes, lane 1: 2p + 1's
    const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
    const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
    const __m256i d = _mm256_sub_epi32(DPBUSD(DPBUSD(_mm256_setzero_si256(), lo, al), hi, ah), corr);
    return _mm256_mullo_epi32(d, _mm256_permutevar8x32_epi32(sc, idx));
}

// rows w0 and w1 (nb super-blocks each) against one ready activation
KT void iq4xs_dot2(const block_iq4_xs* w0, const block_iq4_xs* w1, const ReadyK4* r, int nb, float* s0, float* s1) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kIq4Plus128));
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i idx[4] = {_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3),
                            _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5), _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7)};
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const ReadyK4& R = r[ib];
        const block_iq4_xs& x0 = w0[ib];
        const block_iq4_xs& x1 = w1[ib];
        const __m256i sc0 = iq4xs_scales(x0), sc1 = iq4xs_scales(x1);
        __m256i i0 = _mm256_setzero_si256(), i1 = _mm256_setzero_si256();
        for (int p = 0; p < 4; ++p) {
            const __m256i al = _mm256_load_si256((const __m256i*) R.lo[p]);
            const __m256i ah = _mm256_load_si256((const __m256i*) R.hi[p]);
            const __m256i c = _mm256_load_si256((const __m256i*) R.corr[p]);
            i0 = _mm256_add_epi32(i0, iq4xs_pair(x0.qs + 32 * p, al, ah, c, sc0, idx[p], lut, m4));
            i1 = _mm256_add_epi32(i1, iq4xs_pair(x1.qs + 32 * p, al, ah, c, sc1, idx[p], lut, m4));
        }
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(x0.d) * R.d), _mm256_cvtepi32_ps(i0), a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(x1.d) * R.d), _mm256_cvtepi32_ps(i1), a1);
    }
    *s0 = hsum8(a0);
    *s1 = hsum8(a1);
}

// ---- IQ4_NL and Q5_0 x Q8_0 ---------------------------------------------------------------------------------------
// Two 32-value blocks a step (Ready32): a block's 16 bytes of codes in each 128-bit lane, the low nibbles against
// the activation's values 0..15 and the high ones against 16..31.  IQ4_NL looks its codes up as value + 128; Q5_0's
// codes with their fifth bit are value + 16 as they are.  The offset comes off with the activation's sums.

// [d(b) x4, d(b + 1) x4] of two fp16 block scales
KT inline __m256 scale_pair(uint16_t d0, uint16_t d1) {
    const __m128i h = _mm_shuffle_epi8(_mm_cvtsi32_si128((int) (d0 | (uint32_t) d1 << 16)),
                                       _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3));
    return _mm256_cvtph_ps(h);
}

KT float iq4nl_dot1(const block_iq4_nl* w, const Ready32* r, int nb) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kIq4Plus128));
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int i = 0; i + 1 < nb; i += 2) {
        const Ready32& R = r[i / 2];
        const __m256i q = _mm256_loadu2_m128i((const __m128i*) w[i + 1].qs, (const __m128i*) w[i].qs);
        const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
        const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
        __m256i d = DPBUSD(_mm256_setzero_si256(), lo, _mm256_load_si256((const __m256i*) R.lo));
        d = DPBUSD(d, hi, _mm256_load_si256((const __m256i*) R.hi));
        d = _mm256_sub_epi32(d, _mm256_slli_epi32(_mm256_load_si256((const __m256i*) R.sum), 7));
        const __m256 s = _mm256_mul_ps(scale_pair(w[i].d, w[i + 1].d), _mm256_load_ps(R.dy));
        if (i & 2) a1 = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(d), a1);
        else a0 = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(d), a0);
    }
    return hsum8(_mm256_add_ps(a0, a1));
}

// Q5_0's fifth bits of two blocks' values: of values 0..15 (lo) or 16..31 (hi), as 0 or 16 in the codes' bytes
KT inline __m256i q50_bits(const __m256i hv, const __m256i spread, const __m256i sel) {
    const __m256i b = _mm256_and_si256(_mm256_shuffle_epi8(hv, spread), sel);
    return _mm256_and_si256(_mm256_cmpeq_epi8(b, sel), _mm256_set1_epi8(0x10));
}

KT float q50_dot1(const block_q5_0* w, const Ready32* r, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    // the bytes of qh each value's bit is in: values 0..7 byte 0, 8..15 byte 1 (lo), 16..23 byte 2, 24..31 byte 3
    const __m256i spread_lo = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                               0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m256i spread_hi = _mm256_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                                               2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
                                         1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int i = 0; i + 1 < nb; i += 2) {
        const Ready32& R = r[i / 2];
        uint32_t h0, h1;
        std::memcpy(&h0, w[i].qh, 4);
        std::memcpy(&h1, w[i + 1].qh, 4);
        const __m256i hv = _mm256_set_m128i(_mm_cvtsi32_si128((int) h1), _mm_cvtsi32_si128((int) h0));
        const __m256i q = _mm256_loadu2_m128i((const __m128i*) w[i + 1].qs, (const __m128i*) w[i].qs);
        const __m256i lo = _mm256_or_si256(_mm256_and_si256(q, m4), q50_bits(hv, spread_lo, sel));
        const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), q50_bits(hv, spread_hi, sel));
        __m256i d = DPBUSD(_mm256_setzero_si256(), lo, _mm256_load_si256((const __m256i*) R.lo));
        d = DPBUSD(d, hi, _mm256_load_si256((const __m256i*) R.hi));
        d = _mm256_sub_epi32(d, _mm256_slli_epi32(_mm256_load_si256((const __m256i*) R.sum), 4));
        const __m256 s = _mm256_mul_ps(scale_pair(w[i].d, w[i + 1].d), _mm256_load_ps(R.dy));
        if (i & 2) a1 = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(d), a1);
        else a0 = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(d), a0);
    }
    return hsum8(_mm256_add_ps(a0, a1));
}

const Dots kDots = {STRATA_DOTS_NAME, fp4_dot2, fp4_dot1, iq4xs_dot2, iq4nl_dot1, q50_dot1};
