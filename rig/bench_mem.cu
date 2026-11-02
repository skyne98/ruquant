// Memory system numbers the cost model takes as flags:
//   - streaming read bandwidth over a sweep of bytes in flight (waves x loads per lane):
//     the plateau gives --bw-eff, the rising part gives loaded latency by Little's law
//     (latency = bytes in flight / bandwidth), which is --latency;
//   - copy bandwidth;
//   - idle latency by pointer chasing, per footprint (L1 / L2 / HBM / TLB reach).
//   ./bench_mem [device] [MiB]     -> JSON on stdout
#include "../gfx906/gpu.h"
#include "../gfx906/w4a4_skinny.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

struct alignas(16) v4 {
    int x, y, z, w;
};

// Every lane keeps U independent 16-byte loads in flight per pass over a grid-stride loop.
template <int U> __global__ void __launch_bounds__(256) stream_read(const v4* __restrict__ p, size_t n, int* out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    for (; i + (U - 1) * stride < n; i += U * stride) {
        v4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x7fffffff) out[0] = acc; // practically never: keeps the loads alive
}

__global__ void __launch_bounds__(256) stream_copy(const v4* __restrict__ src, v4* __restrict__ dst, size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

// One lane follows a random cycle; each hop is a dependent load. Slots are 256 B apart.
// The index starts from threadIdx.x (0 here) so the compiler treats it as per-lane and emits
// vector loads, the path weights take; a uniform index would go through the scalar cache.
__global__ void chase(const int* __restrict__ next, int hops, int* out, unsigned long long* t) {
    int j = threadIdx.x;
    rq_opaque(j);
    const unsigned long long c0 = clock64(), w0 = rq_walltick();
    for (int h = 0; h < hops; ++h) j = next[j];
    const unsigned long long c1 = clock64(), w1 = rq_walltick();
    out[0] = j;
    t[0] = c1 - c0;
    t[1] = w1 - w0;
}

// One point of the sweep: `blocks` blocks of `threads` lanes, U loads in flight per lane.
template <int U>
static void read_point(const v4* buf, size_t n, int* out, int cus, int blocks, int threads, bool& first) {
    const double ms = rq_time_ms([&] { stream_read<U><<<blocks, threads>>>(buf, n, out); }, 5, 1);
    const double gbs = n * sizeof(v4) / (ms * 1e-3) / 1e9;
    const int resident = std::min(blocks, cus * rq_blocks_per_cu(stream_read<U>, threads));
    const double inflight = (double)resident * threads * U * sizeof(v4);
    std::printf("%s    {\"blocks\": %d, \"threads\": %d, \"resident_blocks\": %d, \"loads_per_lane\": %d, "
                "\"bytes_in_flight\": %.0f, \"gbs\": %.1f}",
                first ? "" : ",\n", blocks, threads, resident, U, inflight, gbs);
    first = false;
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    const size_t mib = argc > 2 ? std::atoll(argv[2]) : 1024;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    const int cus = prop.multiProcessorCount;
    const size_t bytes = mib << 20, n = bytes / sizeof(v4);

    v4 *a, *b;
    int* out;
    CHECK(cudaMalloc(&a, bytes));
    CHECK(cudaMalloc(&b, bytes));
    CHECK(cudaMalloc(&out, 64));
    CHECK(cudaMemset(a, 1, bytes));

    std::printf("{\n  \"bench\": \"mem\", \"toolchain\": \"%s\", \"device\": %d, \"name\": \"%s\", \"cus\": %d, "
                "\"buffer_mib\": %zu,\n  \"read_sweep\": [\n",
                rq_toolchain(), dev, prop.name, cus, mib);
    // from a few single-wave blocks (latency-bound, the Little's-law ramp) to every CU full
    bool first = true;
    for (int threads : {RQ_WAVE, 256})
        for (int blocks : {cus / 8, cus / 4, cus / 2, cus, 2 * cus, 4 * cus, 8 * cus}) {
            if (threads == RQ_WAVE && blocks > 2 * cus) continue;
            read_point<1>(a, n, out, cus, blocks, threads, first);
            read_point<2>(a, n, out, cus, blocks, threads, first);
            read_point<4>(a, n, out, cus, blocks, threads, first);
            read_point<8>(a, n, out, cus, blocks, threads, first);
        }
    const double copy_ms = rq_time_ms([&] { stream_copy<<<cus * 8, 256>>>(a, b, n); }, 7, 2);
    std::printf("\n  ],\n  \"copy_gbs\": %.1f,\n", 2.0 * bytes / (copy_ms * 1e-3) / 1e9);

    // pointer chase
    std::printf("  \"chase\": [\n");
    const size_t slot = 256 / sizeof(int);
    std::mt19937 rng(906);
    int *d_next, *d_out;
    unsigned long long* d_t;
    CHECK(cudaMalloc(&d_out, 64));
    CHECK(cudaMalloc(&d_t, 16));
    const size_t footprints_kib[] = {8, 16, 32, 64, 256, 1024, 2048, 4096, 8192, 32768, 262144};
    const int nfp = sizeof(footprints_kib) / sizeof(footprints_kib[0]);
    for (int f = 0; f < nfp; ++f) {
        const size_t fp = footprints_kib[f] << 10;
        if (fp > bytes) break;
        const size_t slots = fp / 256;
        std::vector<int> order(slots);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin() + 1, order.end(), rng);
        std::vector<int> next(slots * slot, 0);
        for (size_t s = 0; s < slots; ++s) next[order[s] * slot] = (int)(order[(s + 1) % slots] * slot);
        CHECK(cudaMalloc(&d_next, next.size() * sizeof(int)));
        CHECK(cudaMemcpy(d_next, next.data(), next.size() * sizeof(int), cudaMemcpyHostToDevice));
        const int hops = fp <= (4u << 20) ? 200000 : 100000;
        chase<<<1, 1>>>(d_next, (int)std::min<size_t>(slots, 4096), d_out, d_t); // warm the TLB / caches
        chase<<<1, 1>>>(d_next, hops, d_out, d_t);
        CHECK(cudaDeviceSynchronize());
        unsigned long long t[2];
        CHECK(cudaMemcpy(t, d_t, sizeof(t), cudaMemcpyDeviceToHost));
        const double ns = t[1] / rq_walltick_hz(dev) * 1e9 / hops;
        std::printf("    {\"footprint_kib\": %zu, \"ns\": %.1f, \"cycles\": %.1f}%s\n", footprints_kib[f], ns,
                    (double)t[0] / hops, (f + 1 < nfp && (footprints_kib[f + 1] << 10) <= bytes) ? "," : "");
        CHECK(cudaFree(d_next));
    }
    std::printf("  ]\n}\n");
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    return 0;
}
