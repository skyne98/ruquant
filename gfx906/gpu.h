// One CUDA-spelled host API for both toolchains: nvcc builds a harness as CUDA, hipcc as HIP.
// Harnesses get validated on the 3090 and then run unchanged on the MI50.
#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                         \
    do {                                                                                 \
        cudaError_t e_ = (x);                                                            \
        if (e_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

// Dynamic shared memory above the default needs an opt-in on CUDA; gfx906 gives every
// workgroup up to 64 KB without one.
template <class F> inline void rq_allow_smem(F kernel, size_t bytes) {
#if defined(__HIPCC__)
    (void)kernel;
    (void)bytes;
#else
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
#endif
}

// Blocks of `kernel` that fit on one CU / SM at once.
template <class F> inline int rq_blocks_per_cu(F kernel, int block, size_t smem = 0) {
    int n = 0;
#if defined(__HIPCC__)
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, reinterpret_cast<const void*>(kernel), block, smem));
#else
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, block, smem));
#endif
    return n;
}

// A constant-rate tick, to turn clock64() cycle counts into the real shader clock.
__device__ inline unsigned long long rq_walltick() {
#if defined(__HIPCC__)
    return wall_clock64(); // s_memrealtime
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

inline double rq_walltick_hz(int dev) {
#if defined(__HIPCC__)
    int khz = 0;
    CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev));
    return khz * 1e3;
#else
    (void)dev;
    return 1e9; // globaltimer counts nanoseconds
#endif
}

// Hides a register's value from the optimizer (no instruction emitted), so benchmark chains
// of identical ops can't be folded.
__device__ inline void rq_opaque(int& x) {
#if defined(__HIPCC__)
    asm volatile("" : "+v"(x));
#else
    asm volatile("" : "+r"(x));
#endif
}

// Peak engine clock (CUDA 13 dropped cudaDeviceProp::clockRate).
inline double rq_max_clock_mhz(int dev) {
    int khz = 0;
#if defined(__HIPCC__)
    CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, dev));
#else
    CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
#endif
    return khz / 1e3;
}

__device__ inline int rq_dot4(int a, int b, int c) {
#if defined(__HIPCC__)
    return __builtin_amdgcn_sdot4(a, b, c, false); // v_dot4_i32_i8
#else
    return __dp4a(a, b, c);
#endif
}

inline const char* rq_toolchain() {
#if defined(__HIPCC__)
    return "hip";
#else
    return "cuda";
#endif
}

// Times `body` (which launches work) with events: warm-up runs, then the median of `reps`.
// With batch > 1 a sample is `batch` calls back to back between one event pair, divided:
// the steady-state cost in a stream, without the ~10 us an event pair adds on ROCm around a
// single small launch.
template <class F> inline double rq_time_ms(F body, int reps = 20, int warmup = 3, int batch = 1) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) body();
    CHECK(cudaDeviceSynchronize());
    double times[256];
    reps = reps > 256 ? 256 : reps;
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaEventRecord(a));
        for (int j = 0; j < batch; ++j) body();
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        times[i] = ms / batch;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    for (int i = 1; i < reps; ++i) // insertion sort, reps is small
        for (int j = i; j > 0 && times[j] < times[j - 1]; --j) {
            const double t = times[j];
            times[j] = times[j - 1];
            times[j - 1] = t;
        }
    return times[reps / 2];
}
