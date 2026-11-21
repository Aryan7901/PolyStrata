// src/poly/igpu.cpp - see the header.  The kernels were tuned on an Arrow Lake iGPU: a
// sub-group of 16 lanes a row, the lanes on one 64-value block at a time (contiguous reads of the input in local
// memory), four blocks' codes and scales loaded ahead of their use; gate and up rows together with SITU, then down.
#include "igpu.hpp"

// OpenCL's loader is opened at run time (dlopen): nothing links against it, and a machine without it runs as before
#if !defined(STRATA_IGPU) && defined(__linux__) && __has_include(<CL/cl.h>)
#define STRATA_IGPU 1
#endif
#if STRATA_IGPU
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace strata::poly {
namespace {
Igpu* g_igpu = nullptr;
std::vector<std::pair<const uint8_t*, const uint8_t*>> g_ranges;   // set at load, read after
}  // namespace

void set_igpu(Igpu* g) { g_igpu = g; }
Igpu* igpu() { return g_igpu; }

void igpu_register(const void* p, size_t n) {
    g_ranges.push_back({(const uint8_t*) p, (const uint8_t*) p + n});
}

bool igpu_owns(const void* p) {
    for (const auto& [a, b] : g_ranges)
        if ((const uint8_t*) p >= a && (const uint8_t*) p < b) return true;
    return false;
}

// INK_IGPU_SHARE, when set, fixes the iGPU's share of a call's experts instead (an experiment).
int Igpu::take(int n) {
    if (n < 2 || dead_) return 0;
    static const double share = [] {
        const char* s = std::getenv("INK_IGPU_SHARE");
        return s ? std::atof(s) : -1.0;
    }();
    if (share >= 0) return std::max(0, std::min({(int) (n * share + 0.5), n - 1, kMax}));
    ++calls_;
    int best = 0;
    double best_us = cpu_us_[std::min(n, kMax)];
    for (int k = 1; k < n && k <= kMax; ++k) {
        const double us = std::max(cpu_us_[n - k], ig_fixed_ + ig_per_ * k);
        if (us < best_us * 0.97) best = k, best_us = us;   // a gain worth the iGPU's power
    }
    // one call in 64 that would not use it gives it one expert, so a clock that rose is noticed
    if (best == 0 && n >= 3 && calls_ % 64 == 0) best = 1;
    return best;
}

void Igpu::cpu_took(int j, double us) {
    if (j < 1 || j > kMax) return;
    double& c = cpu_us_[j];
    c = c == 0 ? us : 0.95 * c + 0.05 * us;
}
}  // namespace strata::poly

#if !STRATA_IGPU
namespace strata::poly {
struct Igpu::Cl {};
std::unique_ptr<Igpu> Igpu::open(int, int, float, float) {
    std::fprintf(stderr, "strata-poly: iGPU experts: built without OpenCL\n");
    return nullptr;
}
Igpu::~Igpu() = default;
void* Igpu::alloc_host(size_t) { return nullptr; }
void Igpu::start(const IgpuExpert*, int, const float*) {}
bool Igpu::wait() { return false; }
int Igpu::take(int) { return 0; }
void Igpu::cpu_took(int, double) {}
}  // namespace strata::poly
#else

#ifndef CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL
#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL 0x4200
#endif

namespace strata::poly {
namespace {

const char* kSrc = R"CLC(
// NE, NF: n_embd and n_ff (build options)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// the activation as ggml's CPU dot products see it: Q8_0 blocks of 32, the scale amax / 127 kept as a half, values
// rounded to the nearest (even) step - else the iGPU's experts differ from the CPU's by ~1% and change the answers
inline void q8_round(__local float* xs, int n) {
    for (int b = get_local_id(0); b < n / 32; b += get_local_size(0)) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmax(amax, fabs(xs[b * 32 + j]));
        const float id = amax != 0.f ? 127.f / amax : 0.f;
        const float d = (float) convert_half_rte(amax / 127.f);
        for (int j = 0; j < 32; ++j) xs[b * 32 + j] = rint(xs[b * 32 + j] * id) * d;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
float ue4m3(uchar x) {
    if (x == 0 || x == 0x7f) return 0.f;
    int e = (x >> 3) & 15, m = x & 7;
    float raw = e == 0 ? ldexp((float) m, -9) : ldexp(1.f + m / 8.f, e - 7);
    return raw * 0.5f;
}
// one 64-value block's dot with x (lanes on 4 values each); c: this lane's code word already shifted
inline float blk_dot(uint c, float4 xv, const float* kvp) {
    float t = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint n = (c >> (8 * j)) & 15;
        const float mag = kvp[n & 7];
        t += ((n & 8) ? -mag : mag) * (j == 0 ? xv.x : j == 1 ? xv.y : j == 2 ? xv.z : xv.w);
    }
    return t;
}
#define U 4
// expert e = group (., e): gate row r and up row r together, SITU, into h[e][r]
__kernel __attribute__((intel_reqd_sub_group_size(16)))
void gate_up_situ(__global const ulong* ptr, __global const float* sc, __global const float* x, __global float* h,
                  float beta, float lb) {
    __local float xs[NE];
    __local float uet[256];
    for (int i = get_local_id(0); i < NE; i += get_local_size(0)) xs[i] = x[i];
    for (int i = get_local_id(0); i < 256; i += get_local_size(0)) uet[i] = ue4m3((uchar) i);
    barrier(CLK_LOCAL_MEM_FENCE);
    q8_round(xs, NE);
    const int e = get_group_id(1);
    const int sg = get_sub_group_id(), nsg = get_num_sub_groups(), lane = get_sub_group_local_id();
    const int s = lane >> 2, q = lane & 3;
    const int byte_off = 4 + s * 8 + (q & 1) * 4, shift = (q >> 1) * 4;
    const int nb = NE / 64;
    const size_t row_bytes = (size_t) nb * 36;
    const float kvp[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
    __global const uchar* gw = (__global const uchar*) ptr[3 * e];
    __global const uchar* uw = (__global const uchar*) ptr[3 * e + 1];
    for (int r = get_group_id(0) * nsg + sg; r < NF; r += get_num_groups(0) * nsg) {
        __global const uchar* wg = gw + (size_t) r * row_bytes;
        __global const uchar* wu = uw + (size_t) r * row_bytes;
        float ag = 0.f, au = 0.f;
        for (int b0 = 0; b0 < nb; b0 += U) {
            uint cg[U], cu[U], dg[U], du[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = min(b0 + u, nb - 1);
                cg[u] = *(__global const uint*) (wg + b * 36 + byte_off);
                cu[u] = *(__global const uint*) (wu + b * 36 + byte_off);
                dg[u] = wg[b * 36 + s];
                du[u] = wu[b * 36 + s];
            }
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = b0 + u;
                if (b >= nb) break;
                const float4 xv = vload4(lane, xs + b * 64);
                ag += uet[dg[u]] * blk_dot(cg[u] >> shift, xv, kvp);
                au += uet[du[u]] * blk_dot(cu[u] >> shift, xv, kvp);
            }
        }
        ag = sub_group_reduce_add(ag);
        au = sub_group_reduce_add(au);
        if (lane == 0) {
            const float g = ag * sc[3 * e];
            float uu = au * sc[3 * e + 1];
            if (lb > 0.f) uu = lb * tanh(uu / lb);
            h[(size_t) e * NF + r] = beta * tanh(g / beta) / (1.f + exp(-g)) * uu;
        }
    }
}
// down: expert e's row r (NF values) times h[e], weighted, into part[e][r]
__kernel __attribute__((intel_reqd_sub_group_size(16)))
void down(__global const ulong* ptr, __global const float* sc, __global const float* h, __global float* part) {
    __local float xs[NF];
    __local float uet[256];
    const int e = get_group_id(1);
    for (int i = get_local_id(0); i < NF; i += get_local_size(0)) xs[i] = h[(size_t) e * NF + i];
    for (int i = get_local_id(0); i < 256; i += get_local_size(0)) uet[i] = ue4m3((uchar) i);
    barrier(CLK_LOCAL_MEM_FENCE);
    q8_round(xs, NF);
    const int sg = get_sub_group_id(), nsg = get_num_sub_groups(), lane = get_sub_group_local_id();
    const int s = lane >> 2, q = lane & 3;
    const int byte_off = 4 + s * 8 + (q & 1) * 4, shift = (q >> 1) * 4;
    const int nb = NF / 64;
    const size_t row_bytes = (size_t) nb * 36;
    const float kvp[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
    __global const uchar* w = (__global const uchar*) ptr[3 * e + 2];
    for (int r = get_group_id(0) * nsg + sg; r < NE; r += get_num_groups(0) * nsg) {
        __global const uchar* wr = w + (size_t) r * row_bytes;
        float acc = 0.f;
        for (int b0 = 0; b0 < nb; b0 += U) {
            uint cw[U], dw[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = min(b0 + u, nb - 1);
                cw[u] = *(__global const uint*) (wr + b * 36 + byte_off);
                dw[u] = wr[b * 36 + s];
            }
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = b0 + u;
                if (b >= nb) break;
                acc += uet[dw[u]] * blk_dot(cw[u] >> shift, vload4(lane, xs + b * 64), kvp);
            }
        }
        acc = sub_group_reduce_add(acc);
        if (lane == 0) part[(size_t) e * NE + r] = acc * sc[3 * e + 2];
    }
}
)CLC";

typedef void* (*HostAllocFn)(cl_context, const cl_mem_properties_intel*, size_t, cl_uint, cl_int*);
typedef cl_int (*MemFreeFn)(cl_context, void*);
typedef cl_int (*SetArgMemFn)(cl_kernel, cl_uint, const void*);

// the calls this file makes, from the loader
struct Api {
    decltype(&::clGetPlatformIDs) GetPlatformIDs = nullptr;
    decltype(&::clGetPlatformInfo) GetPlatformInfo = nullptr;
    decltype(&::clGetDeviceIDs) GetDeviceIDs = nullptr;
    decltype(&::clCreateContext) CreateContext = nullptr;
    decltype(&::clCreateCommandQueueWithProperties) CreateCommandQueueWithProperties = nullptr;
    decltype(&::clGetExtensionFunctionAddressForPlatform) GetExtensionFunctionAddressForPlatform = nullptr;
    decltype(&::clCreateProgramWithSource) CreateProgramWithSource = nullptr;
    decltype(&::clBuildProgram) BuildProgram = nullptr;
    decltype(&::clGetProgramBuildInfo) GetProgramBuildInfo = nullptr;
    decltype(&::clCreateKernel) CreateKernel = nullptr;
    decltype(&::clSetKernelExecInfo) SetKernelExecInfo = nullptr;
    decltype(&::clSetKernelArg) SetKernelArg = nullptr;
    decltype(&::clGetDeviceInfo) GetDeviceInfo = nullptr;
    decltype(&::clFinish) Finish = nullptr;
    decltype(&::clReleaseKernel) ReleaseKernel = nullptr;
    decltype(&::clReleaseProgram) ReleaseProgram = nullptr;
    decltype(&::clReleaseCommandQueue) ReleaseCommandQueue = nullptr;
    decltype(&::clReleaseContext) ReleaseContext = nullptr;
    decltype(&::clEnqueueNDRangeKernel) EnqueueNDRangeKernel = nullptr;
    decltype(&::clFlush) Flush = nullptr;
    decltype(&::clGetEventInfo) GetEventInfo = nullptr;
    decltype(&::clReleaseEvent) ReleaseEvent = nullptr;
    decltype(&::clGetEventProfilingInfo) GetEventProfilingInfo = nullptr;
};

const Api* api() {
    static const Api* a = [] () -> const Api* {
        void* lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) return nullptr;
        static Api t;
        bool ok = true;
        ok = ok && (t.GetPlatformIDs = (decltype(&::clGetPlatformIDs)) dlsym(lib, "clGetPlatformIDs"));
        ok = ok && (t.GetPlatformInfo = (decltype(&::clGetPlatformInfo)) dlsym(lib, "clGetPlatformInfo"));
        ok = ok && (t.GetDeviceIDs = (decltype(&::clGetDeviceIDs)) dlsym(lib, "clGetDeviceIDs"));
        ok = ok && (t.CreateContext = (decltype(&::clCreateContext)) dlsym(lib, "clCreateContext"));
        ok = ok && (t.CreateCommandQueueWithProperties = (decltype(&::clCreateCommandQueueWithProperties)) dlsym(lib, "clCreateCommandQueueWithProperties"));
        ok = ok && (t.GetExtensionFunctionAddressForPlatform = (decltype(&::clGetExtensionFunctionAddressForPlatform)) dlsym(lib, "clGetExtensionFunctionAddressForPlatform"));
        ok = ok && (t.CreateProgramWithSource = (decltype(&::clCreateProgramWithSource)) dlsym(lib, "clCreateProgramWithSource"));
        ok = ok && (t.BuildProgram = (decltype(&::clBuildProgram)) dlsym(lib, "clBuildProgram"));
        ok = ok && (t.GetProgramBuildInfo = (decltype(&::clGetProgramBuildInfo)) dlsym(lib, "clGetProgramBuildInfo"));
        ok = ok && (t.CreateKernel = (decltype(&::clCreateKernel)) dlsym(lib, "clCreateKernel"));
        ok = ok && (t.SetKernelExecInfo = (decltype(&::clSetKernelExecInfo)) dlsym(lib, "clSetKernelExecInfo"));
        ok = ok && (t.SetKernelArg = (decltype(&::clSetKernelArg)) dlsym(lib, "clSetKernelArg"));
        ok = ok && (t.GetDeviceInfo = (decltype(&::clGetDeviceInfo)) dlsym(lib, "clGetDeviceInfo"));
        ok = ok && (t.Finish = (decltype(&::clFinish)) dlsym(lib, "clFinish"));
        ok = ok && (t.ReleaseKernel = (decltype(&::clReleaseKernel)) dlsym(lib, "clReleaseKernel"));
        ok = ok && (t.ReleaseProgram = (decltype(&::clReleaseProgram)) dlsym(lib, "clReleaseProgram"));
        ok = ok && (t.ReleaseCommandQueue = (decltype(&::clReleaseCommandQueue)) dlsym(lib, "clReleaseCommandQueue"));
        ok = ok && (t.ReleaseContext = (decltype(&::clReleaseContext)) dlsym(lib, "clReleaseContext"));
        ok = ok && (t.EnqueueNDRangeKernel = (decltype(&::clEnqueueNDRangeKernel)) dlsym(lib, "clEnqueueNDRangeKernel"));
        ok = ok && (t.Flush = (decltype(&::clFlush)) dlsym(lib, "clFlush"));
        ok = ok && (t.GetEventInfo = (decltype(&::clGetEventInfo)) dlsym(lib, "clGetEventInfo"));
        ok = ok && (t.ReleaseEvent = (decltype(&::clReleaseEvent)) dlsym(lib, "clReleaseEvent"));
        ok = ok && (t.GetEventProfilingInfo = (decltype(&::clGetEventProfilingInfo)) dlsym(lib, "clGetEventProfilingInfo"));
        return ok ? &t : nullptr;
    }();
    return a;
}

}  // namespace

struct Igpu::Cl {
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel gu = nullptr, down = nullptr;
    cl_event done = nullptr, began = nullptr;
    HostAllocFn host_alloc = nullptr;
    MemFreeFn mem_free = nullptr;
    SetArgMemFn set_arg_mem = nullptr;
    std::vector<void*> allocs;
};

std::unique_ptr<Igpu> Igpu::open(int n_embd, int n_ff, float beta, float lb) {
    if (n_embd > 8192 || n_ff > 8192 || n_embd % 64 || n_ff % 64) {
        std::fprintf(stderr, "strata-poly: iGPU experts: unsupported widths\n");
        return nullptr;
    }
    // before the runtime reads its settings: submissions straight to the ring (55 -> 6 us a round trip)
    setenv("NEOReadDebugKeys", "1", 0);
    setenv("EnableDirectSubmission", "1", 0);
    if (!api()) {
        std::fprintf(stderr, "strata-poly: iGPU experts: no OpenCL loader (libOpenCL.so.1)\n");
        return nullptr;
    }
    const Api& A = *api();
    std::unique_ptr<Igpu> g(new Igpu());
    g->cl_ = std::make_unique<Cl>();
    Cl& c = *g->cl_;
    cl_uint np = 0;
    if (A.GetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        std::fprintf(stderr, "strata-poly: iGPU experts: no OpenCL platform\n");
        return nullptr;
    }
    std::vector<cl_platform_id> ps(np);
    A.GetPlatformIDs(np, ps.data(), nullptr);
    for (cl_platform_id p : ps) {
        char name[256] = {};
        A.GetPlatformInfo(p, CL_PLATFORM_NAME, sizeof name, name, nullptr);
        if (std::strstr(name, "Intel") && A.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &c.dev, nullptr) == CL_SUCCESS) {
            c.plat = p;
            break;
        }
    }
    if (!c.plat) {
        std::fprintf(stderr, "strata-poly: iGPU experts: no Intel GPU in OpenCL (intel-compute-runtime)\n");
        return nullptr;
    }
    cl_int err = CL_SUCCESS;
    c.ctx = A.CreateContext(nullptr, 1, &c.dev, nullptr, nullptr, &err);
    const cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_PRIORITY_KHR,
                                      CL_QUEUE_PRIORITY_HIGH_KHR, 0};
    c.q = err == CL_SUCCESS ? A.CreateCommandQueueWithProperties(c.ctx, c.dev, qp, &err) : nullptr;
    c.host_alloc = (HostAllocFn) A.GetExtensionFunctionAddressForPlatform(c.plat, "clHostMemAllocINTEL");
    c.mem_free = (MemFreeFn) A.GetExtensionFunctionAddressForPlatform(c.plat, "clMemBlockingFreeINTEL");
    c.set_arg_mem = (SetArgMemFn) A.GetExtensionFunctionAddressForPlatform(c.plat, "clSetKernelArgMemPointerINTEL");
    if (err != CL_SUCCESS || !c.host_alloc || !c.set_arg_mem) {
        std::fprintf(stderr, "strata-poly: iGPU experts: no USM on the Intel GPU\n");
        return nullptr;
    }
    c.prog = A.CreateProgramWithSource(c.ctx, 1, &kSrc, nullptr, &err);
    const std::string opts = "-cl-std=CL3.0 -cl-mad-enable -DNE=" + std::to_string(n_embd) + " -DNF=" + std::to_string(n_ff);
    if (err != CL_SUCCESS || A.BuildProgram(c.prog, 1, &c.dev, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        char log[8192] = {};
        A.GetProgramBuildInfo(c.prog, c.dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, nullptr);
        std::fprintf(stderr, "strata-poly: iGPU experts: the kernels did not build:\n%s\n", log);
        return nullptr;
    }
    c.gu = A.CreateKernel(c.prog, "gate_up_situ", &err);
    c.down = A.CreateKernel(c.prog, "down", &err);
    const cl_bool yes = CL_TRUE;
    A.SetKernelExecInfo(c.gu, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof yes, &yes);
    A.SetKernelExecInfo(c.down, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof yes, &yes);
    g->n_embd_ = n_embd;
    g->n_ff_ = n_ff;
    g->beta_ = beta;
    g->lb_ = lb;
    g->x_ = (float*) g->alloc_host((size_t) n_embd * sizeof(float));
    g->h_ = (float*) g->alloc_host((size_t) kMax * n_ff * sizeof(float));
    g->part_ = (float*) g->alloc_host((size_t) kMax * n_embd * sizeof(float));
    g->gu_ = (uint64_t*) g->alloc_host((size_t) kMax * 3 * sizeof(uint64_t));
    g->sc_ = (float*) g->alloc_host((size_t) kMax * 3 * sizeof(float));
    if (!g->x_ || !g->h_ || !g->part_ || !g->gu_ || !g->sc_) {
        std::fprintf(stderr, "strata-poly: iGPU experts: no shared memory for the buffers\n");
        return nullptr;
    }
    c.set_arg_mem(c.gu, 0, g->gu_);
    c.set_arg_mem(c.gu, 1, g->sc_);
    c.set_arg_mem(c.gu, 2, g->x_);
    c.set_arg_mem(c.gu, 3, g->h_);
    A.SetKernelArg(c.gu, 4, sizeof(float), &g->beta_);
    A.SetKernelArg(c.gu, 5, sizeof(float), &g->lb_);
    c.set_arg_mem(c.down, 0, g->gu_);
    c.set_arg_mem(c.down, 1, g->sc_);
    c.set_arg_mem(c.down, 2, g->h_);
    c.set_arg_mem(c.down, 3, g->part_);
    char name[256] = {};
    A.GetDeviceInfo(c.dev, CL_DEVICE_NAME, sizeof name, name, nullptr);
    std::fprintf(stderr, "strata-poly: iGPU experts on %s\n", name);
    return g;
}

Igpu::~Igpu() {
    if (!cl_) return;
    const Api& A = *api();
    if (busy_) wait();
    if (dead_) return;   // a lost batch: clFinish and the frees would wait on it; the process ends soon anyway
    if (cl_->q) A.Finish(cl_->q);
    for (void* p : cl_->allocs)
        if (cl_->mem_free) cl_->mem_free(cl_->ctx, p);
    if (cl_->gu) A.ReleaseKernel(cl_->gu);
    if (cl_->down) A.ReleaseKernel(cl_->down);
    if (cl_->prog) A.ReleaseProgram(cl_->prog);
    if (cl_->q) A.ReleaseCommandQueue(cl_->q);
    if (cl_->ctx) A.ReleaseContext(cl_->ctx);
}

void* Igpu::alloc_host(size_t bytes) {
    cl_int err = CL_SUCCESS;
    void* p = cl_->host_alloc(cl_->ctx, nullptr, bytes, 4096, &err);
    if (err != CL_SUCCESS || !p) return nullptr;
    cl_->allocs.push_back(p);
    igpu_register(p, bytes);
    return p;
}

void Igpu::start(const IgpuExpert* e, int n, const float* x) {
    const Api& A = *api();
    if (busy_) wait();
    std::memcpy(x_, x, (size_t) n_embd_ * sizeof(float));
    for (int i = 0; i < n; ++i) {
        gu_[3 * i] = (uint64_t) (uintptr_t) e[i].gate;
        gu_[3 * i + 1] = (uint64_t) (uintptr_t) e[i].up;
        gu_[3 * i + 2] = (uint64_t) (uintptr_t) e[i].down;
        sc_[3 * i] = e[i].gate_scale;
        sc_[3 * i + 1] = e[i].up_scale;
        sc_[3 * i + 2] = e[i].weight;
    }
    // a work-group of 16 sub-groups, a row each at a time: n_ff / 16 groups for gate|up, n_embd / 16 for down (tuned)
    const size_t g1[2] = {(size_t) (n_ff_ / 16) * 256, (size_t) n}, l1[2] = {256, 1};
    const size_t g2[2] = {(size_t) (n_embd_ / 16) * 256, (size_t) n}, l2[2] = {256, 1};
    A.EnqueueNDRangeKernel(cl_->q, cl_->gu, 2, nullptr, g1, l1, 0, nullptr, &cl_->began);
    A.EnqueueNDRangeKernel(cl_->q, cl_->down, 2, nullptr, g2, l2, 0, nullptr, &cl_->done);
    A.Flush(cl_->q);
    busy_ = true;
    k_ = n;
}

bool Igpu::wait() {
    if (!busy_) return !dead_;
    const Api& A = *api();
    // a batch takes a millisecond or two; 200 ms means it will never end
    const auto began = std::chrono::steady_clock::now();
    cl_int st = CL_QUEUED;
    while (st > CL_COMPLETE &&
           A.GetEventInfo(cl_->done, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, nullptr) == CL_SUCCESS) {
        if (std::chrono::steady_clock::now() - began > std::chrono::milliseconds(200)) {
            std::fprintf(stderr, "strata-poly: iGPU experts: a batch did not finish (a GPU hang?); the CPU computes "
                                 "them from now on\n");
            dead_ = true;
            busy_ = false;
            return false;   // the events are left: releasing them could wait on the lost batch
        }
    }
    if (st < 0) {
        std::fprintf(stderr, "strata-poly: iGPU experts: a batch failed (%d); the CPU computes them from now on\n", st);
        dead_ = true;
    }
    // the iGPU's time for the batch, from its first kernel's start to its second's end: what take() plans with
    cl_ulong t0 = 0, t1 = 0;
    if (A.GetEventProfilingInfo(cl_->began, CL_PROFILING_COMMAND_START, sizeof t0, &t0, nullptr) == CL_SUCCESS &&
        A.GetEventProfilingInfo(cl_->done, CL_PROFILING_COMMAND_END, sizeof t1, &t1, nullptr) == CL_SUCCESS && t1 > t0 &&
        k_ > 0) {
        const double per = ((double) (t1 - t0) / 1e3) / k_;
        ig_per_ = 0.9 * ig_per_ + 0.1 * per;
    }
    A.ReleaseEvent(cl_->began);
    A.ReleaseEvent(cl_->done);
    cl_->began = cl_->done = nullptr;
    busy_ = false;
    return !dead_;
}

}  // namespace strata::poly
#endif
