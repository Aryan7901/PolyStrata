// src/poly/norm_q8_test.cu - kernels::native_add_rms_norm_q8_1() against the launches it stands for: axpy(),
// rms_norm() and native_quantize_q8_1(), with and without the sum; and native_rms_norm_q8_1_planes() against a
// rms_norm() and native_quantize_q8_1() a plane.  The residual, the normed rows and the Q8_1 blocks must be the same
// bytes.
//
// usage: poly_norm_q8_test   (exit 0: every byte the same)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<unsigned char> bytes(const void* d, size_t n) {
    std::vector<unsigned char> v(n);
    cudaMemcpy(v.data(), d, n, cudaMemcpyDeviceToHost);
    return v;
}

int check(int n, int rows, bool with_add, float spread, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, spread);
    std::vector<float> r((size_t) n * rows), add(r.size()), w(n);
    for (float& x : r) x = nd(rng);
    for (float& x : add) x = nd(rng);
    for (float& x : w) x = 1.f + 0.1f * nd(rng);
    const size_t qb = kernels::native_q8_1_bytes(n) * rows, fb = r.size() * sizeof(float);
    float *r1 = upload(r), *r2 = upload(r), *a = upload(add), *dw = upload(w), *y1 = nullptr, *y2 = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    cudaMalloc(&y1, fb), cudaMalloc(&y2, fb), cudaMalloc(&q1, qb), cudaMalloc(&q2, qb);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const float eps = 1e-5f;
    if (with_add) poly::axpy(a, 1.f, r1, n * rows, true, s);
    poly::rms_norm(r1, dw, y1, n, rows, eps, s);
    kernels::native_quantize_q8_1(y1, q1, n * rows, 1, s);
    kernels::native_add_rms_norm_q8_1(r2, with_add ? a : nullptr, dw, y2, q2, n, rows, eps, poly::sum_threads(n), s);
    cudaStreamSynchronize(s);
    const bool same_r = bytes(r1, fb) == bytes(r2, fb), same_y = bytes(y1, fb) == bytes(y2, fb),
               same_q = bytes(q1, qb) == bytes(q2, qb);
    cudaFree(r1), cudaFree(r2), cudaFree(a), cudaFree(dw), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(q2);
    cudaStreamDestroy(s);
    const bool ok = same_r && same_y && same_q;
    std::printf("%s n %5d rows %4d %s spread %g: residual %s, normed %s, q8_1 %s\n", ok ? "ok  " : "FAIL", n, rows,
                with_add ? "add " : "norm", spread, same_r ? "same" : "DIFF", same_y ? "same" : "DIFF",
                same_q ? "same" : "DIFF");
    return ok ? 0 : 1;
}

// planes groups of rows, each from its own input (two of them may be one) with its own weights
int check_planes(int n, int rows, int planes, bool shared_input, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) 3 * n * rows), w((size_t) 3 * n);
    for (float& v : x) v = nd(rng);
    for (float& v : w) v = 1.f + 0.1f * nd(rng);
    const size_t plane = (size_t) n * rows, qp = kernels::native_q8_1_bytes(n) * rows;
    float *dx = upload(x), *dw = upload(w), *y1 = nullptr, *y2 = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    cudaMalloc(&y1, 3 * plane * sizeof(float)), cudaMalloc(&y2, 3 * plane * sizeof(float));
    cudaMalloc(&q1, 3 * qp), cudaMalloc(&q2, 3 * qp);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const float eps = 1e-6f;
    const float* xs[3];
    const float* ws[3];
    for (int g = 0; g < planes; ++g) {
        xs[g] = dx + (shared_input ? 0 : g) * plane;
        ws[g] = dw + (size_t) g * n;
        poly::rms_norm(xs[g], ws[g], y1 + g * plane, n, rows, eps, s);
        kernels::native_quantize_q8_1(y1 + g * plane, (char*) q1 + g * qp, n * rows, 1, s);
    }
    kernels::native_rms_norm_q8_1_planes(xs, ws, planes, y2, q2, n, rows, eps, poly::sum_threads(n), s);
    cudaStreamSynchronize(s);
    const size_t fb = planes * plane * sizeof(float);
    const bool same_y = bytes(y1, fb) == bytes(y2, fb), same_q = bytes(q1, planes * qp) == bytes(q2, planes * qp);
    cudaFree(dx), cudaFree(dw), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(q2);
    cudaStreamDestroy(s);
    const bool ok = same_y && same_q;
    std::printf("%s n %5d rows %4d planes %d%s: normed %s, q8_1 %s\n", ok ? "ok  " : "FAIL", n, rows, planes,
                shared_input ? " (one input)" : "", same_y ? "same" : "DIFF", same_q ? "same" : "DIFF");
    return ok ? 0 : 1;
}

// ink_mul_sigmoid_q8 against mul_sigmoid() and native_quantize_q8_1()
int check_sigmoid_q8(int n, int rows, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const int gs = 2 * n + 64;
    std::vector<float> a((size_t) n * rows), g((size_t) gs * rows);
    for (float& v : a) v = nd(rng);
    for (float& v : g) v = 3.f * nd(rng);
    const size_t qb = kernels::native_q8_1_bytes(n) * rows, fb = a.size() * sizeof(float);
    float *a1 = upload(a), *a2 = upload(a), *dg = upload(g);
    void *q1 = nullptr, *q2 = nullptr;
    cudaMalloc(&q1, qb), cudaMalloc(&q2, qb);
    cudaStream_t s;
    cudaStreamCreate(&s);
    poly::mul_sigmoid(a1, dg, n, rows, gs, false, s);
    kernels::native_quantize_q8_1(a1, q1, n * rows, 1, s);
    poly::ink_mul_sigmoid_q8(a2, dg, n, rows, gs, q2, s);
    cudaStreamSynchronize(s);
    const bool same_a = bytes(a1, fb) == bytes(a2, fb), same_q = bytes(q1, qb) == bytes(q2, qb);
    cudaFree(a1), cudaFree(a2), cudaFree(dg), cudaFree(q1), cudaFree(q2);
    cudaStreamDestroy(s);
    const bool ok = same_a && same_q;
    std::printf("%s n %5d rows %4d mul_sigmoid_q8: gated %s, q8_1 %s\n", ok ? "ok  " : "FAIL", n, rows,
                same_a ? "same" : "DIFF", same_q ? "same" : "DIFF");
    return ok ? 0 : 1;
}

// ink_gate_gemv against ink_gemv_f32() and ink_gate(), jobs of their own inputs, one of them in place
int check_gate_gemv(int n_in, int n, int rows, int jobs, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    auto vec = [&](size_t m, float sc) {
        std::vector<float> v(m);
        for (float& x : v) x = sc * nd(rng);
        return v;
    };
    int bad = 0;
    std::vector<float*> w(jobs), h(jobs), cur1(jobs), cur2(jobs), msg(jobs), gf(jobs);
    for (int j = 0; j < jobs; ++j) {
        w[j] = upload(vec((size_t) 2 * n * n_in, 0.1f));
        h[j] = upload(vec((size_t) rows * n_in, 1.f));
        auto c = vec((size_t) rows * n, 1.f);
        cur1[j] = upload(c);
        cur2[j] = upload(c);
        msg[j] = upload(vec((size_t) rows * n, 1.f));
        cudaMalloc(&gf[j], (size_t) rows * 2 * n * sizeof(float));
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int j = 0; j < jobs; ++j) {
        poly::ink_gemv_f32(w[j], h[j], gf[j], n_in, 2 * n, rows, s);
        poly::ink_gate(cur1[j], n, msg[j], n, gf[j], gf[j] + n, 2 * n, cur1[j], n, rows, s);
    }
    std::vector<const float*> cw(w.begin(), w.end()), ch(h.begin(), h.end()), cc(cur2.begin(), cur2.end()),
        cm(msg.begin(), msg.end());
    const bool ran = poly::ink_gate_gemv(cw.data(), ch.data(), cc.data(), n, cm.data(), n, cur2.data(), jobs, n_in, n,
                                         rows, s);
    cudaStreamSynchronize(s);
    for (int j = 0; j < jobs; ++j) bad += bytes(cur1[j], (size_t) rows * n * 4) != bytes(cur2[j], (size_t) rows * n * 4);
    for (int j = 0; j < jobs; ++j) cudaFree(w[j]), cudaFree(h[j]), cudaFree(cur1[j]), cudaFree(cur2[j]), cudaFree(msg[j]), cudaFree(gf[j]);
    cudaStreamDestroy(s);
    const bool ok = ran && bad == 0;
    std::printf("%s n_in %d n %d rows %d jobs %d gate_gemv: %s\n", ok ? "ok  " : "FAIL", n_in, n, rows, jobs,
                !ran ? "not run" : bad ? "DIFF" : "same");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    int bad = 0;
    for (int n : {512, 1024, 2048, 2816, 4096, 5120})
        for (int rows : {1, 3, 8, 64})
            for (bool add : {false, true}) bad += check(n, rows, add, 1.f, rng);
    bad += check(2048, 1024, true, 1.f, rng);   // a prompt batch
    bad += check(2048, 8, true, 300.f, rng);    // large activations
    for (int n : {2304, 4096})
        for (int rows : {1, 2, 7})
            for (int planes : {1, 3}) bad += check_planes(n, rows, planes, false, rng);
    bad += check_planes(2304, 1, 3, true, rng);
    for (int rows : {1, 3, 8}) bad += check_sigmoid_q8(3072, rows, rng);
    for (int rows : {1, 2, 8})
        for (int jobs : {1, 3}) bad += check_gate_gemv(64, 2304, rows, jobs, rng);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
