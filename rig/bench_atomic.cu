// Split-K reduction cost: the skinny kernel's epilogue, alone. Grid (N/wave, G) one-wave
// blocks, each lane adds T int32 partials for its channel, G-way contention per address.
// G = 1 is a plain store, the floor. The cost model charges 2*G*T*N*4 B at L2 bandwidth;
// this measures the real rate (--atomic-gbs in calibrate.py).
//   ./bench_atomic [device]     -> JSON on stdout
#include "../gfx906/gpu.h"
#include "../gfx906/w4a4_skinny.h"

#include <cstdio>
#include <cstdlib>

__global__ void partials(int32_t* __restrict__ acc, int T, int N, int atomic) {
    const int n = blockIdx.x * RQ_WAVE + threadIdx.x;
    const int v = n ^ blockIdx.y;
    for (int t = 0; t < T; ++t) {
        int32_t* out = &acc[(size_t)t * N + n];
        if (atomic) atomicAdd(out, v + t);
        else *out = v + t;
    }
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    int32_t* acc;
    const size_t max_elems = (size_t)64 * 17408;
    CHECK(cudaMalloc(&acc, max_elems * sizeof(int32_t)));

    std::printf("{\n  \"bench\": \"atomic\", \"toolchain\": \"%s\", \"device\": %d, \"name\": \"%s\",\n"
                "  \"points\": [\n", rq_toolchain(), dev, prop.name);
    bool first = true;
    for (int N : {4096, 17408})
        for (int T : {8, 32, 64})
            for (int G : {1, 2, 4, 8, 16}) {
                const dim3 grid(N / RQ_WAVE, G);
                const int atomic = G > 1;
                // batched, and without zeroing acc in between: a real pipeline zeroes it in
                // the epilogue that reads it, not with a launch of its own
                const double ms = rq_time_ms([&] { partials<<<grid, RQ_WAVE>>>(acc, T, N, atomic); }, 10, 3, 10);
                const double ops = (double)G * T * N;
                std::printf("%s    {\"N\": %d, \"T\": %d, \"G\": %d, \"us\": %.2f, \"gops\": %.2f, "
                            "\"model_bytes_gbs\": %.1f}",
                            first ? "" : ",\n", N, T, G, ms * 1e3, ops / (ms * 1e-3) / 1e9,
                            2.0 * G * T * N * 4 / (ms * 1e-3) / 1e9);
                first = false;
            }
    std::printf("\n  ]\n}\n");
    CHECK(cudaFree(acc));
    return 0;
}
