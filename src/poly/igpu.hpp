// src/poly/igpu.hpp - routed experts of the CPU's share computed on an Intel integrated GPU beside the CPU (opt-in).
//
// Measured on an Arrow Lake laptop (Core Ultra 7 255HX, Xe-LPG iGPU): the iGPU streams system RAM at ~55 GB/s while
// the CPU's 13 threads keep their own ~40 GB/s - the two add up.  A decode call's CPU experts are split between them:
// the iGPU takes some whole experts, the CPU the rest, both at once.  The iGPU reads the experts where the engine keeps
// them, so they must live in its shared host memory (alloc_host()): Engine-side code copies them there at load.
//
// Only NVFP4 experts with Inkling 2's SITU activation, one token's row of input an expert.  Intel's OpenCL runtime
// (NEO); direct submission is turned on (NEOReadDebugKeys / EnableDirectSubmission) before it starts, which takes a
// submit-and-wait from ~55 us to ~6 us.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::poly {

/// One expert for the iGPU: its gate, up and down (NVFP4, in alloc_host() memory) and the scales and weight CpuExpert has.
struct IgpuExpert {
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    float gate_scale, up_scale, weight;
};

class Igpu {
public:
    /// The iGPU, or null when there is none (or no OpenCL runtime for it); the reason goes to stderr.
    static std::unique_ptr<Igpu> open(int n_embd, int n_ff, float beta, float lb);
    ~Igpu();
    /// Host memory the iGPU reads in place (Intel USM host memory); null on failure.  Freed with the Igpu.
    void* alloc_host(size_t bytes);
    /// Starts n experts (n <= kMax) of one token whose input is x (n_embd floats); their weighted outputs land in
    /// part() (n rows of n_embd) once wait() returns.
    void start(const IgpuExpert* e, int n, const float* x);
    /// False when the batch did not finish (the iGPU hung: i915 resets the context, and the runtime is not told under
    /// direct submission): part() is then not the experts' and the iGPU is used no more (take() is 0).
    bool wait();
    /// How many of a call's n CPU experts to hand over: the split that ends soonest by what calls have taken so far
    /// (the CPU's time for j experts, the iGPU's for k, both learned as the calls run), 0 when none gains.
    int take(int n);
    /// The CPU's time for a call of j experts (us), to learn from.
    void cpu_took(int j, double us);
    const float* part() const { return part_; }
    bool busy() const { return busy_; }
    static constexpr int kMax = 8;

private:
    Igpu() = default;
    struct Cl;
    std::unique_ptr<Cl> cl_;
    int n_embd_ = 0, n_ff_ = 0;
    float beta_ = 0, lb_ = 0;
    float* x_ = nullptr;      ///< the input row
    float* h_ = nullptr;      ///< kMax rows of n_ff
    float* part_ = nullptr;   ///< kMax rows of output
    uint64_t* gu_ = nullptr;  ///< per expert: gate, up, down addresses
    float* sc_ = nullptr;     ///< per expert: gate scale, up scale, weight
    bool busy_ = false, dead_ = false;
    int k_ = 0;                 ///< experts of the batch in flight
    double cpu_us_[kMax + 1] = {};   ///< the CPU's time for a call of j experts
    double ig_fixed_ = 20.0, ig_per_ = 200.0;   ///< the iGPU's: fixed + per expert (us)
    int64_t calls_ = 0;
};

/// The accelerator cpu_experts() hands experts to, when one is set (Inkling with INK_IGPU=1), and whether a range of
/// memory is its: drop_pages() must leave such ranges alone (MADV_DONTNEED would zero them).
void set_igpu(Igpu* g);
Igpu* igpu();
bool igpu_owns(const void* p);
void igpu_register(const void* p, size_t n);

}  // namespace strata::poly
