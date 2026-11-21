// src/poly/mmq_tiles_test.cu - the size-class launches of STRATA_POLY_STAGED_TILES (engine.cpp, moe_rows) give the same
// results as one launch: to the bit, except on a card whose MMQ uses stream-k (Mmq::split_exact()), where they are
// the same products added in another order and must agree within rounding.  llama.cpp's MMQ takes a launch's tile
// width from the largest expert in it, so the engine launches the experts of up to N rows apart from the larger
// ones, each class with bounds in which the other classes' experts have no rows, over rows laid out class after
// class.  That layout is built here the way the engine builds it, for row counts with many small experts, a few
// large, empty ones and equal ones.
//
// usage: poly_mmq_tiles_test   (exit 0: every case agrees; 77: no card)
#include "mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using strata::poly::Mmq;

namespace {

struct Quant {
    const char* name;
    int type;     // ggml_type
    size_t block; // bytes per block
    int block_cols = 256, matrix_rows = 256, matrix_cols = 512;
};
constexpr Quant kQuants[] = {{"Q4_K", 12, 144}, {"Q5_K", 13, 176}, {"Q6_K", 14, 210}};
constexpr Quant kNvfp4 = {"NVFP4", 40, 36, 64, 130, 576};  // output tile and K-step tails

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

int tile_of(const std::vector<int>& t, int rows) {
    int k = 0;
    while (k < (int) t.size() && rows > t[(size_t) k]) ++k;
    return k;
}

// one case: the rows of each expert, the classes' limits; returns the number of rows that differ from one launch,
// to the bit where `exact`, else by more than 8 float epsilons of the case's largest value
int run_case(const Quant& q, const std::vector<int>& rows, const std::vector<int>& limits, bool exact,
             std::mt19937& rng) {
    const int kRows = q.matrix_rows, kCols = q.matrix_cols;
    const int n = (int) rows.size();
    const size_t per = (size_t) kRows * (kCols / q.block_cols) * q.block;
    std::vector<uint8_t> w(per * n + 4096, 0);   // zero bytes behind the matrices (kTail)
    for (size_t i = 0; i < per * n; ++i) w[i] = (uint8_t) rng();
    for (size_t b = 0; b < per * n / q.block; ++b) {
        uint8_t* p = &w[b * q.block];
        if (q.type == kNvfp4.type) std::fill_n(p, 4, 0x38);  // UE4M3 scales of 1
        else if (q.type == 14) { p[q.block - 2] = 0; p[q.block - 1] = 0x30; }
        else { p[0] = 0; p[1] = 0x30; p[2] = 0; p[3] = 0x30; }  // half scales of 2^-3
    }
    const int total = std::accumulate(rows.begin(), rows.end(), 0);
    std::vector<float> x((size_t) total * kCols);
    for (auto& v : x) v = (float) ((rng() & 0xffff) / 32768.0 - 1.0);
    std::vector<int32_t> start(n + 1, 0);
    for (int e = 0; e < n; ++e) start[(size_t) e + 1] = start[(size_t) e] + rows[(size_t) e];

    // the reference: rows by expert, one launch
    std::vector<int32_t> iota(total + 128), gather_ref(total);  // MMQ reads full tiles of output row IDs
    std::iota(iota.begin(), iota.begin() + total, 0);
    std::iota(gather_ref.begin(), gather_ref.end(), 0);
    // the classes: rows class after class (an expert's rows in order), a bounds array for each class
    const int nclass = (int) limits.size() + 1;
    std::vector<int32_t> gather_cls, bounds((size_t) nclass * (n + 1)), most(nclass, 0);
    for (int k = 0, at = 0; k < nclass; ++k) {
        for (int e = 0; e < n; ++e) {
            bounds[(size_t) k * (n + 1) + e] = at;
            if (rows[(size_t) e] > 0 && tile_of(limits, rows[(size_t) e]) == k) {
                for (int r = 0; r < rows[(size_t) e]; ++r) gather_cls.push_back(start[(size_t) e] + r);
                at += rows[(size_t) e];
                most[(size_t) k] = std::max(most[(size_t) k], rows[(size_t) e]);
            }
        }
        bounds[(size_t) k * (n + 1) + n] = at;
    }
    std::vector<int32_t> ref_bounds(start.begin(), start.end());

    Mmq mmq;
    uint8_t* dw = (uint8_t*) upload(w);
    float* dx = upload(x);
    int32_t *d_iota = upload(iota), *d_gref = upload(gather_ref), *d_gcls = upload(gather_cls), *d_rb = upload(ref_bounds),
            *d_cb = upload(bounds);
    void* xq = nullptr;
    float *dref = nullptr, *dcls = nullptr;
    bool good = dw && dx && d_iota && d_gref && d_gcls && d_rb && d_cb;
    good = good && ok(cudaMalloc(&xq, Mmq::q8_bytes(total, kCols)), "xq") &&
           ok(cudaMalloc((void**) &dref, (size_t) total * kRows * 4), "ref") &&
           ok(cudaMalloc((void**) &dcls, (size_t) total * kRows * 4), "cls");
    int bad = -1;
    if (good) {
        cudaMemset(dref, 0, (size_t) total * kRows * 4);
        cudaMemset(dcls, 0, (size_t) total * kRows * 4);
        const int maxrows = *std::max_element(rows.begin(), rows.end());
        mmq.quantize(dx, d_gref, xq, q.type, kCols, kCols, total, nullptr);
        mmq.mul(dw, q.type, kRows, kCols, per, n, xq, d_rb, d_iota, total, maxrows, dref, kRows, nullptr);
        mmq.quantize(dx, d_gcls, xq, q.type, kCols, kCols, total, nullptr);
        for (int k = 0; k < nclass; ++k)
            if (most[(size_t) k] > 0)
                mmq.mul(dw, q.type, kRows, kCols, per, n, xq, d_cb + (size_t) k * (n + 1), d_iota, total, most[(size_t) k],
                        dcls, kRows, nullptr);
        if (ok(cudaDeviceSynchronize(), "run")) {
            std::vector<float> a((size_t) total * kRows), b((size_t) total * kRows);
            cudaMemcpy(a.data(), dref, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), dcls, b.size() * 4, cudaMemcpyDeviceToHost);
            float big = 0;
            for (float v : a) big = std::max(big, std::fabs(v));
            const float tol = 8 * FLT_EPSILON * big;
            const auto near = [tol](float u, float v) { return std::fabs(u - v) <= tol; };
            bad = q.type == kNvfp4.type && !(big > 0) ? total : 0;
            // Row i of the class layout is row gather_cls[i] of the reference.
            for (int i = 0; i < total; ++i) {
                const float* c = &b[(size_t) i * kRows];
                const float* r = &a[(size_t) gather_cls[(size_t) i] * kRows];
                if (exact ? std::memcmp(c, r, kRows * 4) != 0 : !std::equal(c, c + kRows, r, near)) ++bad;
            }
        }
    }
    for (void* p : {(void*) dw, (void*) dx, (void*) d_iota, (void*) d_gref, (void*) d_gcls, (void*) d_rb, (void*) d_cb, xq,
                    (void*) dref, (void*) dcls})
        if (p) cudaFree(p);
    return bad;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no card: skipped\n");
        return 77;
    }
    std::mt19937 rng(7);
    const char* nvfp4_mode = std::getenv("STRATA_PREFILL_NVFP4");
    if (nvfp4_mode && std::strcmp(nvfp4_mode, "w4a4x2") == 0) {
        int device = 0;
        cudaDeviceProp prop{};
        if (!ok(cudaGetDevice(&device), "device") || !ok(cudaGetDeviceProperties(&prop, device), "device properties"))
            return 1;
        if (prop.major != 12 || prop.minor != 0) {
            std::printf("no RTX 50: skipped\n");
            return 77;
        }
        const auto& q = kNvfp4;
        if (!Mmq::fits(q.type, q.matrix_rows, q.matrix_cols)) {
            std::printf("FAIL NVFP4: no MMQ tile\n");
            return 1;
        }
        const int bad = run_case(q, {8, 0, 9, 17, 65, 1, 6}, {8, 16, 32}, false, rng);
        if (bad != 0) {
            std::printf("FAIL NVFP4: %d rows differ\n", bad);
            return 1;
        }
        std::printf("PASS NVFP4 size classes and tile tails\n");
        return 0;
    }

    struct Dist { const char* name; std::vector<int> rows; };
    std::vector<Dist> dists;
    {   // many small experts and a few large, as a prompt batch's staged experts are
        std::vector<int> r;
        for (int e = 0; e < 96; ++e) r.push_back(1 + (int) (rng() % 16) * (rng() % 4 == 0 ? 6 : 1));
        for (int e : {5, 40, 77}) r[(size_t) e] = 200 + (int) (rng() % 300);
        dists.push_back({"skewed", r});
    }
    {   // empty experts between the others, one with a single row
        std::vector<int> r(40, 0);
        for (int e = 0; e < 40; e += 3) r[(size_t) e] = 1 + (int) (rng() % 90);
        r[1] = 1;
        dists.push_back({"with empty", r});
    }
    dists.push_back({"all small", std::vector<int>(33, 7)});
    dists.push_back({"one class edge", std::vector<int>{8, 9, 16, 17, 32, 33, 64, 65, 8, 16, 32, 64}});
    const std::vector<std::vector<int>> limits = {{8, 16, 32}, {16}, {4, 8, 12, 16, 24, 32, 48}, {64}};
    int failures = 0, cases = 0;
    for (const Quant& q : kQuants) {
        const bool exact = Mmq::split_exact(q.type, q.matrix_rows);
        if (!exact) std::printf("%s: this card's MMQ uses stream-k, compared within rounding\n", q.name);
        for (const Dist& d : dists)
            for (const auto& l : limits) {
                const int bad = run_case(q, d.rows, l, exact, rng);
                ++cases;
                if (bad != 0) {
                    ++failures;
                    std::printf("FAIL %s %s, %zu classes: %d rows differ%s\n", q.name, d.name, l.size() + 1, bad,
                                exact ? "" : " by more than rounding");
                }
            }
    }
    if (failures) {
        std::printf("FAIL: %d of %d cases\n", failures, cases);
        return 1;
    }
    std::printf("PASS cases=%d\n", cases);
    return 0;
}
