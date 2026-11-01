#include <cuda_runtime.h>
#include <cstdint>

__global__ void rq_scale_kernel(const float * __restrict__ src, float * __restrict__ dst, int64_t n, float mul) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] * mul;
    }
}

extern "C" void rq_scale_launch(const float * src, float * dst, int64_t n, float mul, cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int) ((n + threads - 1) / threads);
    rq_scale_kernel<<<blocks, threads, 0, stream>>>(src, dst, n, mul);
}
