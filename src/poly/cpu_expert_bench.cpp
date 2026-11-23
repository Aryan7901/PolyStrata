// src/poly/cpu_expert_bench.cpp - times cpu_experts() the way a decode uses it: one token, a few routed experts a
// call, drawn from a pool larger than the CPU's caches, in a layer's formats (gate|up and down may differ, as in
// Unsloth's Inkling 2 files).  Prints the weights' bytes a second and, with --check, how far this file's own
// kernels are from ggml-cpu's dot products on the same calls (cpu_experts_use_ggml()).
//
// usage: poly_cpu_expert_bench GU_TYPE [D_TYPE] [experts_per_call] [threads] [calls] [--check]
//   types by ggml name: nvfp4 q8_0 q4_k q5_k iq4_xs q5_0 iq4_nl ...; D_TYPE defaults to GU_TYPE
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <sys/mman.h>

using namespace strata::poly;

namespace {

int type_of(const std::string& name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n && name == n) return t;
    }
    std::fprintf(stderr, "unknown type %s\n", name.c_str());
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
    bool check = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--check") check = true;
        else args.push_back(argv[i]);
    }
    const int gt = type_of(args.size() > 0 ? args[0] : "nvfp4");
    const int dt = args.size() > 1 ? type_of(args[1]) : gt;
    const int per_call = args.size() > 2 ? std::atoi(args[2].c_str()) : 2;
    const int threads = args.size() > 3 ? std::atoi(args[3].c_str()) : 13;
    const int calls = args.size() > 4 ? std::atoi(args[4].c_str()) : 400;
    // Inkling 2's shape; POLY_BENCH_SHAPE=n_embd,n_ff another (2048,512: Tiel-Coder's, whose rows K-quants fit)
    int n_embd = 2304, n_ff = 1152;
    if (const char* sh = std::getenv("POLY_BENCH_SHAPE")) std::sscanf(sh, "%d,%d", &n_embd, &n_ff);
    const int pool = 160;
    ggml_cpu_init();
    const size_t gu_row = ggml_row_size((ggml_type) gt, n_embd), d_row = ggml_row_size((ggml_type) dt, n_ff);
    const size_t gu = gu_row * n_ff, dn = d_row * n_embd, one = (2 * gu + dn + 63) / 64 * 64;
    const size_t pool_bytes = (one * pool + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    uint8_t* wp = (uint8_t*) mmap(nullptr, pool_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    // weights made by quantizing random floats, so the scales are real ones
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.02f);
    {
        std::vector<float> f((size_t) n_embd * 64), g((size_t) n_ff * 128);
        // a few distinct experts, copied through the pool: the i-quants' quantizers take seconds an expert
        constexpr int kUniq = 6;
        for (int e = 0; e < pool; ++e) {
            uint8_t* p = wp + one * e;
            if (e >= kUniq) {
                std::memcpy(p, wp + one * (e % kUniq), one);
                continue;
            }
            for (size_t at = 0; at < 2 * gu; at += gu_row * 64) {
                for (float& v : f) v = nd(rng);
                ggml_quantize_chunk((ggml_type) gt, f.data(), p + at, 0, 64, n_embd, nullptr);
            }
            for (size_t at = 0; at < dn; at += d_row * 128) {
                for (float& v : g) v = nd(rng);
                ggml_quantize_chunk((ggml_type) dt, g.data(), p + 2 * gu + at, 0, 128, n_ff, nullptr);
            }
        }
    }
    std::vector<float> x(n_embd), out(n_embd), ref(n_embd);
    for (float& v : x) v = nd(rng) * 50.f;
    Workers workers(threads);
    Act act;
    act.kind = Act::kSitu;
    act.beta = 4.f;
    act.lb = 25.f;
    std::uniform_int_distribution<int> pick(0, pool - 1);
    double sum = 0.0, err2 = 0.0, ref2 = 0.0, maxrel = 0.0;
    std::vector<CpuExpert> ex((size_t) per_call);
    auto run = [&](int i) {
        for (int k = 0; k < per_call; ++k) {
            const uint8_t* p = wp + one * (size_t) pick(rng);
            ex[(size_t) k] = CpuExpert{p, p + gu, p + 2 * gu, 0.125f, 0, 1.f, 1.f};
        }
        cpu_experts(workers, ex.data(), per_call, 1, gt, gt, dt, n_embd, n_ff, x.data(), INFINITY, act, out.data());
        if (i >= 0)
            for (float v : out) sum += v;
        if (check) {
            cpu_experts_use_ggml(true);
            cpu_experts(workers, ex.data(), per_call, 1, gt, gt, dt, n_embd, n_ff, x.data(), INFINITY, act, ref.data());
            cpu_experts_use_ggml(false);
            double e2 = 0, r2 = 0;
            for (int j = 0; j < n_embd; ++j) {
                e2 += (double) (out[j] - ref[j]) * (out[j] - ref[j]);
                r2 += (double) ref[j] * ref[j];
            }
            err2 += e2;
            ref2 += r2;
            maxrel = std::max(maxrel, std::sqrt(e2 / std::max(r2, 1e-30)));
        }
    };
    for (int i = 0; i < 20; ++i) run(-1);
    rng.seed(11);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < calls; ++i) run(i);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double bytes = (double) (2 * gu + dn) * per_call * calls;
    std::printf("%s/%s: %d experts a call, %d threads: %.1f us a call, %.1f GB/s of weights, checksum %.6g",
                ggml_type_name((ggml_type) gt), ggml_type_name((ggml_type) dt), per_call, threads, s / calls * 1e6,
                bytes / s / 1e9, sum);
    if (check) std::printf("; vs ggml: rel err %.3g (max a call %.3g)", std::sqrt(err2 / std::max(ref2, 1e-30)), maxrel);
    std::printf("\n");
    return 0;
}
