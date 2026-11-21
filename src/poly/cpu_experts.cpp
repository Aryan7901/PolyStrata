// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <numeric>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
constexpr int kRowsPerDownTask = 16;
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define STRATA_NVFP4_VNNI 1
// compiled for AVX2 + AVX-VNNI whatever the unit's flags (the engine is built for any x86-64), used where the CPU
// has both (fp4_ok())
#define FP4_TARGET __attribute__((target("avx2,fma,f16c,avxvnni")))
bool fp4_ok() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                           __builtin_cpu_supports("avxvnni");
    return ok;
}
// ggml's block_nvfp4 and block_q8_0 (ggml-common.h, which this unit does not include): the same bytes
struct block_nvfp4 {
    uint8_t d[4];    // UE4M3 scales, one a 16-value sub-block
    uint8_t qs[32];  // E2M1 codes: sub-block s in qs[8s .. 8s+8), its values j (low nibble) and j + 8 (high)
};
struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(block_nvfp4) == 36 && sizeof(block_q8_0) == 34, "ggml's block layouts");
// NVFP4 x Q8_0 on AVX-VNNI.  ggml's product (ggml_vec_dot_nvfp4_q8_0) reorders the activation's bytes, converts its
// scales and multiplies with maddubs + madd for every row; here the activation is made ready once a call (Ready),
// a weight's 4-bit codes look up E2M1 + 12 (unsigned, 0..24) so one vpdpbusd multiplies and sums them, and the 12 is
// taken off with the activation's sums of four.  The sums are the same integers; only the order of the float adds
// differs from ggml's.
struct Ready {                // one NVFP4 block (64 values) of an activation
    int8_t a[32], b[32];      // the values that meet sub-blocks (0, 2) and (1, 3): the weights' unpacklo / unpackhi order
    int32_t ca[8], cb[8];     // 12 * the sum of each four of a and b
    float dy[4];              // the Q8_0 scales of the sub-blocks: dy0, dy0, dy1, dy1
};
const int8_t kFp4Plus12[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

void make_ready(const block_q8_0* y, int nb, Ready* r) {
    for (int ib = 0; ib < nb; ++ib) {
        const block_q8_0& y0 = y[2 * ib];
        const block_q8_0& y1 = y[2 * ib + 1];
        // sub-block s takes q8 block s / 2, values (s % 2) * 16 ..: A = (sub 0, sub 2), B = (sub 1, sub 3)
        std::memcpy(r[ib].a, y0.qs, 16);
        std::memcpy(r[ib].a + 16, y1.qs, 16);
        std::memcpy(r[ib].b, y0.qs + 16, 16);
        std::memcpy(r[ib].b + 16, y1.qs + 16, 16);
        for (int k = 0; k < 8; ++k) {
            r[ib].ca[k] = 12 * (r[ib].a[4 * k] + r[ib].a[4 * k + 1] + r[ib].a[4 * k + 2] + r[ib].a[4 * k + 3]);
            r[ib].cb[k] = 12 * (r[ib].b[4 * k] + r[ib].b[4 * k + 1] + r[ib].b[4 * k + 2] + r[ib].b[4 * k + 3]);
        }
        const float d0 = ggml_fp16_to_fp32(y0.d), d1 = ggml_fp16_to_fp32(y1.d);
        r[ib].dy[0] = d0;
        r[ib].dy[1] = d0;
        r[ib].dy[2] = d1;
        r[ib].dy[3] = d1;
    }
}


// One block of one row into acc: the scale lanes of A are (s0 x4, s2 x4), of B (s1 x4, s3 x4).
FP4_TARGET inline void fp4_block(const block_nvfp4& x, const Ready& r, __m256& acc, const __m256i lut, const __m256i m4) {
    // lanes: lane 0 = qs 0..15 (sub 0, 1), lane 1 = qs 16..31 (sub 2, 3)
    const __m256i qs = _mm256_loadu2_m128i((const __m128i*) (x.qs + 16), (const __m128i*) x.qs);
    const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(qs, m4));
    const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(qs, 4), m4));
    const __m256i wa = _mm256_unpacklo_epi64(lo, hi);   // lane 0: sub 0, lane 1: sub 2
    const __m256i wb = _mm256_unpackhi_epi64(lo, hi);   // lane 0: sub 1, lane 1: sub 3
    __m256i sa = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wa, _mm256_loadu_si256((const __m256i*) r.a));
    __m256i sb = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wb, _mm256_loadu_si256((const __m256i*) r.b));
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

FP4_TARGET inline float hsum8(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

// rows w0 and w1 (nb blocks each) against one ready activation
FP4_TARGET void fp4_dot2(const block_nvfp4* w0, const block_nvfp4* w1, const Ready* r, int nb, float* s0, float* s1) {
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

FP4_TARGET float fp4_dot1(const block_nvfp4* w, const Ready* r, int nb) {
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
#endif

}  // namespace

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int u_type, int d_type,
                 int n_embd, int n_ff, const float* x, float limit, const Act& fn, float* out,
                 const std::function<void()>* meanwhile, const uint8_t* late, const std::function<void(int)>* wait) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* tu = traits(u_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t u_row = ggml_row_size((ggml_type) u_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t au_bytes = ggml_row_size(tu->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    const bool own_act = tu->vec_dot_type != tg->vec_dot_type;   // up's dot product takes another quantization

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, act_u, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    if (own_act) act_u.resize((size_t) rows * au_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // ggml's quantizers for the activations are scalar: microseconds a row, which on one thread kept the others
    // waiting, here and before the down products
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });
    uint8_t* actup = actp;
    if (own_act) {
        actup = act_u.data();
        const auto au_from = traits(tu->vec_dot_type)->from_float;
        workers.run(rows, [&](int t) { au_from(x + (size_t) t * n_embd, actup + (size_t) t * au_bytes, n_embd); });
    }
    const size_t au_step = own_act ? au_bytes : a_bytes;
#if defined(STRATA_NVFP4_VNNI)
    // NVFP4 gate and up: each token's activation made ready once (Ready), the rows multiplied two at a time
    const bool fp4_gu = fp4_ok() && gu_type == GGML_TYPE_NVFP4 && u_type == GGML_TYPE_NVFP4 && n_embd % 64 == 0 &&
                        tg->vec_dot_type == GGML_TYPE_Q8_0;
    const bool fp4_d = fp4_ok() && d_type == GGML_TYPE_NVFP4 && n_ff % 64 == 0 && td->vec_dot_type == GGML_TYPE_Q8_0;
    static thread_local std::vector<Ready> ready, ready_h;
    const int nb_gu = n_embd / 64, nb_d = n_ff / 64;
    if (fp4_gu) {
        ready.resize((size_t) rows * nb_gu);
        for (int t = 0; t < rows; ++t)
            make_ready((const block_q8_0*) (actp + (size_t) t * a_bytes), nb_gu, ready.data() + (size_t) t * nb_gu);
    }
    const Ready* readyp = ready.data();
#endif

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);
    // the runs in the order their products are taken: the late experts' last, so their reads go on meanwhile
    static thread_local std::vector<int> order;
    order.resize((size_t) n_run);
    std::iota(order.begin(), order.end(), 0);
    if (late)
        std::stable_partition(order.begin(), order.end(), [&](int u) { return !late[by[(size_t) run[(size_t) u]]]; });

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int* orderp = order.data();
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int u = orderp[t / gu_tasks], i0 = runp[u], i1 = runp[u + 1];
        if (late && late[byp[i0]]) (*wait)(byp[i0]);
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * u_row;
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                const uint8_t* au = actup + (size_t) experts[j].row * au_step;
                float g = 0.f, u = 0.f;
#if defined(STRATA_NVFP4_VNNI)
                if (fp4_gu) {
                    fp4_dot2((const block_nvfp4*) gate, (const block_nvfp4*) up,
                             readyp + (size_t) experts[j].row * nb_gu, nb_gu, &g, &u);
                } else
#endif
                {
                    tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                    tu->vec_dot(n_embd, &u, 0, up, 0, au, 0, 1);
                }
                g *= e.gate_scale;
                u *= e.up_scale;
                if (fn.kind == Act::kGelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                if (fn.kind == Act::kSitu) {
                    if (fn.lb > 0.f) u = fn.lb * std::tanh(u / fn.lb);
                    hp[(size_t) j * n_ff + r] = fn.beta * std::tanh(g / fn.beta) / (1.f + std::exp(-g)) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    });
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });
#if defined(STRATA_NVFP4_VNNI)
    if (fp4_d) {
        ready_h.resize((size_t) n * nb_d);
        for (int j = 0; j < n; ++j)
            make_ready((const block_q8_0*) (hqp + (size_t) j * h_bytes), nb_d, ready_h.data() + (size_t) j * nb_d);
    }
    const Ready* readyhp = ready_h.data();
#endif

    const int d_tasks = (n_embd + kRowsPerDownTask - 1) / kRowsPerDownTask;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> one;   // an output value of every use
        one.resize((size_t) n);
        float* s = one.data();
        const int r0 = t * kRowsPerDownTask, r1 = std::min(n_embd, r0 + kRowsPerDownTask);
        for (int r = r0; r < r1; ++r) {
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                for (int i = runp[u]; i < runp[u + 1]; ++i) {
#if defined(STRATA_NVFP4_VNNI)
                    if (fp4_d) {
                        s[byp[i]] = fp4_dot1((const block_nvfp4*) down, readyhp + (size_t) byp[i] * nb_d, nb_d);
                        continue;
                    }
#endif
                    td->vec_dot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
                }
            }
            // a token's sum in the order its experts were given: the order it has in a call of its own
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                out[(size_t) row * n_embd + r] = acc;
            }
        }
    });
}

}  // namespace strata::poly
