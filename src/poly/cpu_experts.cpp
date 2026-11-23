// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#include <chrono>
#include <atomic>

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cpuid.h>
#include <cstdlib>
#include <immintrin.h>
#include <mutex>
#include <numeric>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;   // 16 and 24 were within noise of it for Inkling (2026-11-21)
constexpr int kRowsPerDownTask = 16;
constexpr int kSpinsBeforeSleep = 20000;   // STRATA_WORKER_IDLE=pause: the old count of pause instructions

// An idle worker stays ready this long before it sleeps in posted_.wait().  A decode's calls come a few hundred
// microseconds apart: with the old ~1 ms of pause instructions (less on Arrow Lake) the workers had gone to sleep
// by the next one, and on an RTX 5060 laptop they started a one-expert call's first products ~46 us after it was
// posted (a futex wake and a core coming out of a deep C-state), about 55 us of its 130.  STRATA_WORKER_SPIN_US
// changes it.
const int64_t kReadyUs = [] {
    const char* s = std::getenv("STRATA_WORKER_SPIN_US");
    return s ? std::atoll(s) : 4000;
}();
const int64_t kPauseUs = [] {
    const char* s = std::getenv("STRATA_WORKER_PAUSE_US");
    return s ? std::atoll(s) : 50;
}();
const bool kOldIdle = [] {
    const char* s = std::getenv("STRATA_WORKER_IDLE");
    return s && std::strcmp(s, "pause") == 0;
}();

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// cpu_experts_use_ggml(): every product through ggml-cpu's dot products
std::atomic<bool> g_use_ggml{[] {
    const char* s = std::getenv("STRATA_CPU_GGML");
    return s && std::atoi(s) == 1;
}()};

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

// Waiting for posted_ to move.  Where the CPU has WAITPKG, the worker arms a monitor on posted_'s cache line and
// waits in C0.1 (umwait): it uses little power and wakes within a microsecond of run()'s write.  Elsewhere: pause.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
bool waitpkg_ok() {
    static const bool ok = [] {
        unsigned a = 0, b = 0, c = 0, d = 0;
        const char* s = std::getenv("STRATA_UMWAIT");
        return !(s && std::atoi(s) == 0) && __get_cpuid_count(7, 0, &a, &b, &c, &d) && (c & (1u << 5));
    }();
    return ok;
}
__attribute__((target("waitpkg"))) void wait_line(const std::atomic<uint32_t>& posted, uint32_t seen) {
    _umonitor((void*) &posted);
    if (posted.load(std::memory_order_acquire) != seen) return;   // it moved before the monitor was armed
    _umwait(1, __rdtsc() + 40000);   // C0.1, the quicker of the two to wake; the OS may end it sooner
}
void idle_wait(const std::atomic<uint32_t>& posted, uint32_t seen) {
    if (waitpkg_ok()) wait_line(posted, seen);
    else _mm_pause();
}
#else
void idle_wait(const std::atomic<uint32_t>&, uint32_t) {}
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define STRATA_CPU_DOTS 1
// ggml's block layouts (ggml-common.h, which this unit does not include): the same bytes
struct block_nvfp4 {
    uint8_t d[4];    // UE4M3 scales, one a 16-value sub-block
    uint8_t qs[32];  // E2M1 codes: sub-block s in qs[8s .. 8s+8), its values j (low nibble) and j + 8 (high)
};
struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};
struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
struct block_iq4_xs {
    uint16_t d;
    uint16_t scales_h;    // the scales' high 2 bits, sub-block ib at bits 2ib
    uint8_t scales_l[4];  // their low 4 bits, sub-block ib in byte ib / 2, nibble ib % 2
    uint8_t qs[128];      // sub-block ib in qs[16ib .. 16ib + 16): values j (low nibble) and j + 16 (high)
};
struct block_iq4_nl {
    uint16_t d;
    uint8_t qs[16];       // values j (low nibble) and j + 16 (high)
};
struct block_q5_0 {
    uint16_t d;
    uint8_t qh[4];        // the fifth bit of value j at bit j
    uint8_t qs[16];       // the low four: values j (low nibble) and j + 16 (high)
};
static_assert(sizeof(block_nvfp4) == 36 && sizeof(block_q8_0) == 34 && sizeof(block_q8_K) == 292 &&
                  sizeof(block_iq4_xs) == 136 && sizeof(block_iq4_nl) == 18 && sizeof(block_q5_0) == 22,
              "ggml's block layouts");

// NVFP4 x Q8_0 on a byte dot product (VNNI).  ggml's product (ggml_vec_dot_nvfp4_q8_0) reorders the activation's
// bytes, converts its scales and multiplies with maddubs + madd for every row; here the activation is made ready
// once a call (Ready), a weight's 4-bit codes look up E2M1 + 12 (unsigned, 0..24) so one vpdpbusd multiplies and
// sums them, and the 12 is taken off with the activation's sums of four.  The sums are the same integers; only the
// order of the float adds differs from ggml's.
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

// IQ4_XS x Q8_K: a super-block (256 values) of a Q8_K activation in the order an IQ4_XS row's codes come in, its
// sub-blocks two at a time (cpu_dots.inc)
const uint8_t kIq4Plus128[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};   // kvalues_iq4nl + 128
struct alignas(32) ReadyK4 {
    int8_t lo[4][32];     // pair p: values 0..15 of sub-blocks 2p and 2p + 1 (the codes' low nibbles)
    int8_t hi[4][32];     // and their values 16..31 (the high nibbles)
    int32_t corr[4][8];   // 128 * the sum of each lane's four lo and four hi values
    float d;
};

void make_ready_k4(const block_q8_K* y, int nb, ReadyK4* r) {
    for (int ib = 0; ib < nb; ++ib) {
        for (int p = 0; p < 4; ++p) {
            const int8_t* s0 = y[ib].qs + 64 * p;
            const int8_t* s1 = s0 + 32;
            std::memcpy(r[ib].lo[p], s0, 16);
            std::memcpy(r[ib].lo[p] + 16, s1, 16);
            std::memcpy(r[ib].hi[p], s0 + 16, 16);
            std::memcpy(r[ib].hi[p] + 16, s1 + 16, 16);
            for (int k = 0; k < 8; ++k) {
                int sum = 0;
                for (int t = 0; t < 4; ++t) sum += r[ib].lo[p][4 * k + t] + r[ib].hi[p][4 * k + t];
                r[ib].corr[p][k] = 128 * sum;
            }
        }
        r[ib].d = y[ib].d;
    }
}

// IQ4_NL and Q5_0 x Q8_0: two blocks (64 values) of a Q8_0 activation, block b's 16 values in one 128-bit lane and
// block b + 1's in the other
struct alignas(32) Ready32 {
    int8_t lo[32];      // values 0..15 of each block
    int8_t hi[32];      // values 16..31
    int32_t sum[8];     // the sum of each lane's four lo and four hi values
    float dy[8];        // the blocks' scales: d(b) x4, d(b + 1) x4
};

void make_ready32(const block_q8_0* y, int nb, Ready32* r) {
    for (int i = 0; i + 1 < nb; i += 2) {
        Ready32& R = r[i / 2];
        std::memcpy(R.lo, y[i].qs, 16);
        std::memcpy(R.lo + 16, y[i + 1].qs, 16);
        std::memcpy(R.hi, y[i].qs + 16, 16);
        std::memcpy(R.hi + 16, y[i + 1].qs + 16, 16);
        for (int k = 0; k < 8; ++k) {
            int sum = 0;
            for (int t = 0; t < 4; ++t) sum += R.lo[4 * k + t] + R.hi[4 * k + t];
            R.sum[k] = sum;
        }
        const float d0 = ggml_fp16_to_fp32(y[i].d), d1 = ggml_fp16_to_fp32(y[i + 1].d);
        for (int k = 0; k < 4; ++k) R.dy[k] = d0, R.dy[4 + k] = d1;
    }
}

// The products of one instruction set (cpu_dots.inc)
struct Dots {
    const char* name;
    void (*fp4_dot2)(const block_nvfp4*, const block_nvfp4*, const Ready*, int, float*, float*);
    float (*fp4_dot1)(const block_nvfp4*, const Ready*, int);
    void (*iq4xs_dot2)(const block_iq4_xs*, const block_iq4_xs*, const ReadyK4*, int, float*, float*);
    float (*iq4nl_dot1)(const block_iq4_nl*, const Ready32*, int);
    float (*q50_dot1)(const block_q5_0*, const Ready32*, int);
};

// compiled for these sets whatever the unit's flags (the engine is built for any x86-64), used where the CPU has them
namespace avx_vnni {   // Intel from Alder Lake, AMD from Zen 5
#define KT __attribute__((target("avx2,fma,f16c,avxvnni")))
#define DPBUSD(a, u, s) _mm256_dpbusd_avx_epi32(a, u, s)
#define STRATA_DOTS_NAME "AVX-VNNI"
#include "cpu_dots.inc"
#undef KT
#undef DPBUSD
#undef STRATA_DOTS_NAME
}  // namespace avx_vnni
namespace avx512_vnni {   // the same products on 256-bit AVX-512 VNNI: Zen 4, Intel server parts without AVX-VNNI
#define KT __attribute__((target("avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512vnni")))
#define DPBUSD(a, u, s) _mm256_dpbusd_epi32(a, u, s)
#define STRATA_DOTS_NAME "AVX-512 VNNI"
#include "cpu_dots.inc"
#undef KT
#undef DPBUSD
#undef STRATA_DOTS_NAME
}  // namespace avx512_vnni

// The set this CPU runs, or none (ggml's products for every type).  STRATA_CPU_ISA=avxvnni | avx512vnni | none picks
// one, where the CPU has it (avx512vnni on a CPU with both: the AVX-512 build, for testing it).
const Dots* dots() {
    static const Dots* const d = [] {
        const bool base = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                          __builtin_cpu_supports("f16c");
        const bool v256 = base && __builtin_cpu_supports("avxvnni");
        const bool v512 = base && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
                          __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vnni");
        const char* want = std::getenv("STRATA_CPU_ISA");
        const Dots* pick = v256 ? &avx_vnni::kDots : v512 ? &avx512_vnni::kDots : nullptr;
        if (want && std::strcmp(want, "none") == 0) pick = nullptr;
        else if (want && std::strcmp(want, "avx512vnni") == 0) pick = v512 ? &avx512_vnni::kDots : nullptr;
        else if (want && std::strcmp(want, "avxvnni") == 0) pick = v256 ? &avx_vnni::kDots : nullptr;
        std::fprintf(stderr, "strata-poly: CPU experts: %s byte dot products for NVFP4, IQ4_XS, IQ4_NL and Q5_0%s\n",
                     pick ? pick->name : "no", pick ? "" : " (ggml's products)");
        return pick;
    }();
    return d;
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
    std::chrono::steady_clock::time_point idle_since{};
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
        if (kOldIdle) {
            if (++idle < kSpinsBeforeSleep) {
                _mm_pause();
                continue;
            }
        } else {
            const auto t = std::chrono::steady_clock::now();
            if (idle++ == 0) idle_since = t;
            // the first moments with pause: a call's next batch (its down products) comes within microseconds, and a
            // worker woken from umwait came to it ~10 us late
            if (t - idle_since < std::chrono::microseconds(kPauseUs)) {
                _mm_pause();
                continue;
            }
            if (t - idle_since < std::chrono::microseconds(kReadyUs)) {
                idle_wait(posted_, seen);
                continue;
            }
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

void cpu_experts_use_ggml(bool on) { g_use_ggml.store(on); }

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

// INK_CPU_PROF=1: one-token calls' time by phase and by expert count, printed every 20000 calls (a diagnosis)
namespace {
struct Prof {
    double calls[9] = {}, ph[9][5] = {};
    int64_t all = 0;
    // the gate|up phase of 1- and 2-expert calls: the GPU launches (meanwhile), the workers' start, the tail
    double mw[3] = {}, start_avg[3] = {}, start_max[3] = {}, tail[3] = {}, busy_avg[3] = {};
    // by the gate|up type (one-token calls): calls, experts, microseconds
    double by_type[64][3] = {};
};
constexpr int kMaxThreads = 64;
std::atomic<int> next_widx{0};
thread_local int widx = -1;
using PClock = std::chrono::steady_clock;
PClock::time_point t_first[kMaxThreads], t_last[kMaxThreads];
bool t_used[kMaxThreads];
double busy[kMaxThreads];
Prof prof;
const bool prof_on = std::getenv("INK_CPU_PROF") != nullptr;
double us_between(PClock::time_point a, PClock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
}  // namespace

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int u_type, int d_type,
                 int n_embd, int n_ff, const float* x, float limit, const Act& fn, float* out,
                 const std::function<void()>* meanwhile, const uint8_t* late, const std::function<void(int)>* wait) {
    const bool timed = prof_on && rows == 1 && n <= 8 && !late;
    PClock::time_point pt[5];
    if (timed) pt[0] = PClock::now();
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
#if defined(STRATA_CPU_DOTS)
    // This file's products (cpu_dots.inc), where the CPU has a byte dot product: each token's activation made ready
    // once (Ready, ReadyK4), gate's and up's rows multiplied two at a time; the down rows against each expert's
    // ready h (Ready, Ready32).  Others: ggml's.
    const Dots* const dk = g_use_ggml.load(std::memory_order_relaxed) ? nullptr : dots();
    enum { kGgml, kFp4, kIq4xs, kIq4nl, kQ50 };
    const bool same_gu = gu_type == u_type;
    const int gu_k = !dk || !same_gu ? kGgml
                     : gu_type == GGML_TYPE_NVFP4 && n_embd % 64 == 0 && tg->vec_dot_type == GGML_TYPE_Q8_0 ? kFp4
                     : gu_type == GGML_TYPE_IQ4_XS && n_embd % 256 == 0 && tg->vec_dot_type == GGML_TYPE_Q8_K ? kIq4xs
                     : kGgml;
    const int d_k = !dk || td->vec_dot_type != GGML_TYPE_Q8_0 ? kGgml
                    : d_type == GGML_TYPE_NVFP4 && n_ff % 64 == 0 ? kFp4
                    : d_type == GGML_TYPE_IQ4_NL && n_ff % 64 == 0 ? kIq4nl
                    : d_type == GGML_TYPE_Q5_0 && n_ff % 64 == 0 ? kQ50
                    : kGgml;
    static thread_local std::vector<Ready> ready, ready_h;
    static thread_local std::vector<ReadyK4> ready_k4;
    static thread_local std::vector<Ready32> ready_32;
    const int nb_gu = gu_k == kIq4xs ? n_embd / 256 : n_embd / 64, nb_d = n_ff / 64;
    if (gu_k == kFp4) {
        ready.resize((size_t) rows * nb_gu);
        for (int t = 0; t < rows; ++t)
            make_ready((const block_q8_0*) (actp + (size_t) t * a_bytes), nb_gu, ready.data() + (size_t) t * nb_gu);
    } else if (gu_k == kIq4xs) {
        ready_k4.resize((size_t) rows * nb_gu);
        for (int t = 0; t < rows; ++t)
            make_ready_k4((const block_q8_K*) (actp + (size_t) t * a_bytes), nb_gu, ready_k4.data() + (size_t) t * nb_gu);
    }
    const Ready* readyp = ready.data();
    const ReadyK4* readyk4p = ready_k4.data();
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

    if (timed) pt[1] = PClock::now();
    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int* orderp = order.data();
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    const bool fine = timed && n <= 2;
    double mw_us = 0;
    if (fine) std::fill(t_used, t_used + kMaxThreads, false), std::fill(busy, busy + kMaxThreads, 0.0);
    const PClock::time_point t_post = PClock::now();
    workers.run(first + n_run * gu_tasks, [&](int t) {
        struct Stamp {
            bool on;
            PClock::time_point a;
            int w;
            ~Stamp() {
                if (!on) return;
                const PClock::time_point b = PClock::now();
                if (!t_used[w]) t_used[w] = true, t_first[w] = a;
                t_last[w] = b;
                busy[w] += us_between(a, b);
            }
        } st{fine, fine ? PClock::now() : PClock::time_point{}, 0};
        if (fine) {
            if (widx < 0) widx = next_widx.fetch_add(1) % kMaxThreads;
            st.w = widx;
        }
        if (t < first) {
            const PClock::time_point a = PClock::now();
            (*meanwhile)();
            if (fine) mw_us = us_between(a, PClock::now());
            return;
        }
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
#if defined(STRATA_CPU_DOTS)
                if (gu_k == kFp4) {
                    dk->fp4_dot2((const block_nvfp4*) gate, (const block_nvfp4*) up,
                                 readyp + (size_t) experts[j].row * nb_gu, nb_gu, &g, &u);
                } else if (gu_k == kIq4xs) {
                    dk->iq4xs_dot2((const block_iq4_xs*) gate, (const block_iq4_xs*) up,
                                   readyk4p + (size_t) experts[j].row * nb_gu, nb_gu, &g, &u);
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
    if (timed) pt[2] = PClock::now();
    if (fine) {
        std::vector<double> starts, ends, busys;
        for (int w = 0; w < kMaxThreads; ++w)
            if (t_used[w]) {
                starts.push_back(us_between(t_post, t_first[w]));
                ends.push_back(us_between(t_post, t_last[w]));
                busys.push_back(busy[w]);
            }
        std::sort(starts.begin(), starts.end());
        std::sort(ends.begin(), ends.end());
        double sa = 0, ba = 0;
        for (double v : starts) sa += v;
        for (double v : busys) ba += v;
        prof.mw[n] += mw_us;
        prof.start_avg[n] += sa / starts.size();
        prof.start_max[n] += starts.back();
        prof.tail[n] += ends.back() - ends[ends.size() / 2];
        prof.busy_avg[n] += ba / busys.size();
    }
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });
#if defined(STRATA_CPU_DOTS)
    if (d_k == kFp4) {
        ready_h.resize((size_t) n * nb_d);
        for (int j = 0; j < n; ++j)
            make_ready((const block_q8_0*) (hqp + (size_t) j * h_bytes), nb_d, ready_h.data() + (size_t) j * nb_d);
    } else if (d_k == kIq4nl || d_k == kQ50) {
        ready_32.resize((size_t) n * nb_d);
        for (int j = 0; j < n; ++j)
            make_ready32((const block_q8_0*) (hqp + (size_t) j * h_bytes), 2 * nb_d, ready_32.data() + (size_t) j * nb_d);
    }
    const Ready* readyhp = ready_h.data();
    const Ready32* ready32p = ready_32.data();
#endif

    if (timed) pt[3] = PClock::now();
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
#if defined(STRATA_CPU_DOTS)
                    if (d_k == kFp4) {
                        s[byp[i]] = dk->fp4_dot1((const block_nvfp4*) down, readyhp + (size_t) byp[i] * nb_d, nb_d);
                        continue;
                    }
                    if (d_k == kIq4nl) {
                        s[byp[i]] = dk->iq4nl_dot1((const block_iq4_nl*) down, ready32p + (size_t) byp[i] * nb_d, 2 * nb_d);
                        continue;
                    }
                    if (d_k == kQ50) {
                        s[byp[i]] = dk->q50_dot1((const block_q5_0*) down, ready32p + (size_t) byp[i] * nb_d, 2 * nb_d);
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
    if (timed) {
        pt[4] = PClock::now();
        prof.calls[n] += 1;
        for (int i = 0; i < 4; ++i) prof.ph[n][i] += us_between(pt[i], pt[i + 1]);
        if (gu_type >= 0 && gu_type < 64) {
            double* bt = prof.by_type[gu_type];
            bt[0] += 1;
            bt[1] += n;
            bt[2] += us_between(pt[0], pt[4]);
        }
        if (++prof.all % 20000 == 0)
            for (int k = 1; k <= 8; ++k)
                if (prof.calls[k] > 0)
                    std::fprintf(stderr, "INK_CPU_PROF %d experts: %6.0f calls, prep %5.1f  gate_up %6.1f  hq %5.1f  down %6.1f us\n",
                                 k, prof.calls[k], prof.ph[k][0] / prof.calls[k], prof.ph[k][1] / prof.calls[k],
                                 prof.ph[k][2] / prof.calls[k], prof.ph[k][3] / prof.calls[k]);
        if (prof.all % 20000 == 0)
            for (int k = 0; k < 64; ++k)
                if (prof.by_type[k][0] > 0)
                    std::fprintf(stderr, "INK_CPU_PROF type %2d (%s): %6.0f calls, %6.0f experts, %6.1f us a call, %6.1f us an expert\n",
                                 k, ggml_type_name((ggml_type) k), prof.by_type[k][0], prof.by_type[k][1],
                                 prof.by_type[k][2] / prof.by_type[k][0], prof.by_type[k][2] / prof.by_type[k][1]);
        if (prof.all % 20000 == 0)
            for (int k = 1; k <= 2; ++k)
                if (prof.calls[k] > 0)
                    std::fprintf(stderr, "INK_CPU_PROF %d experts gate_up: gpu launches %5.1f, threads start avg %5.1f max %5.1f, "
                                 "tail (last - median end) %5.1f, busy avg %5.1f us\n", k,
                                 prof.mw[k] / prof.calls[k], prof.start_avg[k] / prof.calls[k], prof.start_max[k] / prof.calls[k],
                                 prof.tail[k] / prof.calls[k], prof.busy_avg[k] / prof.calls[k]);
    }
}

}  // namespace strata::poly
