// MI50 read-bandwidth microbenchmark shaped like the skinny W4A4 kernel's weight stream: each wave reads its own
// contiguous run of 1 KB loads (64 lanes x 16 B), D loads in flight (software-pipelined, the next D issued before the
// current D are used), optionally A v_dot8 per lane per load to stand in for the kernel's math.
//   stream_bench WGS THREADS D NT A [KB_PER_WAVE [ROT]]   NT: 1 = nontemporal loads; prints GB/s (median of 10 samples)
// KB_PER_WAVE: each wave's run (default: 256 MB over all waves). ROT: wave w starts its run at (w * ROT) KB, wrapping, so
// that concurrent streams sit at different offsets (the order of a sum doesn't matter to the skinny kernel).
// Build: hipcc -O3 --offload-arch=gfx906 stream_bench.hip -o stream_bench (rig/membench/build.sh)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// a clang vector type: __builtin_nontemporal_load takes these, not HIP's int4
typedef int v4i __attribute__((ext_vector_type(4)));

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

template <int D, bool NT, int A>
__global__ void __launch_bounds__(1024) stream_kernel(const v4i * __restrict__ p, int steps, int rot, int * out) {
    const int lane = threadIdx.x % 64;
    const int wave = __builtin_amdgcn_readfirstlane((blockIdx.x * blockDim.x + threadIdx.x) / 64);
    const v4i * q = p + (size_t) wave * steps * 64 + lane;
    const int phase = (int) (((long long) wave * rot) % steps) / D * D; // whole D-step blocks
    auto at = [&](int s) { const int t = s + phase; return q + (size_t) (t < steps ? t : t - steps) * 64; };
    int acc = 0;
    v4i cur[D], nxt[D];
#pragma unroll
    for (int d = 0; d < D; ++d) cur[d] = NT ? __builtin_nontemporal_load(at(d)) : *at(d);
    for (int s = 0; s < steps; s += D) {
        const int sn = s + D < steps ? s + D : s;
#pragma unroll
        for (int d = 0; d < D; ++d) nxt[d] = NT ? __builtin_nontemporal_load(at(sn + d)) : *at(sn + d);
#pragma unroll
        for (int d = 0; d < D; ++d) {
            acc ^= cur[d].x ^ cur[d].y ^ cur[d].z ^ cur[d].w;
#pragma unroll
            for (int a = 0; a < A; ++a) acc = __builtin_amdgcn_sdot8(cur[d].x, acc | a, acc, false);
        }
#pragma unroll
        for (int d = 0; d < D; ++d) cur[d] = nxt[d];
    }
    if (acc == 0x7f3a21c5) out[0] = acc;
}

template <int D, bool NT, int A>
static void launch(int wgs, int threads, const v4i * p, int steps, int rot, int * out) {
    stream_kernel<D, NT, A><<<wgs, threads>>>(p, steps, rot, out);
}

using launch_fn = void (*)(int, int, const v4i *, int, int, int *);

template <int D, bool NT>
static launch_fn pick_a(int a) {
    switch (a) {
        case 0:  return launch<D, NT, 0>;
        case 8:  return launch<D, NT, 8>;
        case 16: return launch<D, NT, 16>;
        case 32: return launch<D, NT, 32>;
        default: return nullptr;
    }
}

template <bool NT>
static launch_fn pick_d(int d, int a) {
    switch (d) {
        case 1: return pick_a<1, NT>(a);
        case 2: return pick_a<2, NT>(a);
        case 4: return pick_a<4, NT>(a);
        case 8: return pick_a<8, NT>(a);
        default: return nullptr;
    }
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s WGS THREADS D NT A [KB_PER_WAVE]\n", argv[0]);
        return 1;
    }
    const int wgs = atoi(argv[1]), threads = atoi(argv[2]), D = atoi(argv[3]), nt = atoi(argv[4]), a = atoi(argv[5]);
    const int kb_per_wave = argc > 6 ? atoi(argv[6]) : 0;
    const int rot = argc > 7 ? atoi(argv[7]) : 0;
    const int waves = wgs * threads / 64;
    // the same total bytes (~256 MB) whatever the grid, unless the run length per wave is given
    int steps = kb_per_wave > 0 ? kb_per_wave : (int) ((256ull << 20) / 1024 / waves);
    steps = std::max(D, steps / D * D);
    const size_t bytes = (size_t) waves * steps * 1024;
    launch_fn f = nt ? pick_d<true>(D, a) : pick_d<false>(D, a);
    if (f == nullptr) {
        fprintf(stderr, "unsupported D %d / A %d\n", D, a);
        return 1;
    }
    v4i * p;
    int * out;
    CHECK(hipMalloc(&p, bytes + D * 1024));
    CHECK(hipMalloc(&out, 4));
    CHECK(hipMemset(p, 1, bytes + D * 1024));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    // each sample times BATCH launches back to back (env, default 50), as in a forward: a single launch between syncs
    // also measures the GPU leaving its idle clocks
    const int batch = getenv("BATCH") ? std::max(1, atoi(getenv("BATCH"))) : 50;
    std::vector<float> ms;
    for (int r = 0; r < 13; ++r) {
        CHECK(hipEventRecord(e0));
        for (int b = 0; b < batch; ++b) f(wgs, threads, p, steps, rot, out);
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float t;
        CHECK(hipEventElapsedTime(&t, e0, e1));
        if (r >= 3) ms.push_back(t / batch);
    }
    std::sort(ms.begin(), ms.end());
    const float t = ms[ms.size() / 2];
    printf("wgs %d threads %d waves %d D %d NT %d A %d rot %d: %zu MB, %d KB per wave, %.1f us, %.1f GB/s\n", wgs, threads, waves,
           D, nt, a, rot, bytes >> 20, steps, t * 1000, bytes / (t * 1e-3) / 1e9);
    return 0;
}
