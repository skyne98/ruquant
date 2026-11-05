// rig/hipgraph_bench.cpp: hipcc -O2 --offload-arch=gfx906 rig/hipgraph_bench.cpp -o hipgraph_bench && ./hipgraph_bench 2000 0 10
// Direct launches vs a captured HIP graph for a chain of N small kernels on one stream.
//   graphbench [N=2000] [WORK=0] [REPS=20]   WORK = inner loop count per kernel (0: empty kernel)
// Prints, per mode: host time to enqueue one chain, wall time per chain (enqueue + finish), and GPU time per chain
// measured with events around the chain.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <thread>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

struct Args { float * a; float * b; int n; int work; int pad[12]; }; // ~80 bytes of kernel arguments, like ggml's

__global__ void busy(float * p, int n) { float v = p[threadIdx.x]; for (int i = 0; i < n; ++i) v = v * 1.0001f + 0.5f; p[threadIdx.x + 64] = v; }

__global__ void mark(int * flag, int v) { __threadfence_system(); *(volatile int *) flag = v; __threadfence_system(); }

__global__ void k(Args p) {
    float v = p.a[threadIdx.x];
    for (int i = 0; i < p.work; ++i) v = v * 1.0001f + 0.5f;
    if (blockIdx.x == 0) p.b[threadIdx.x] = v;
}

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2000, work = argc > 2 ? atoi(argv[2]) : 0, reps = argc > 3 ? atoi(argv[3]) : 20;
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    float *a, *b; CK(hipMalloc(&a, 4096)); CK(hipMalloc(&b, 4096)); CK(hipMemset(a, 0, 4096));
    Args p{a, b, 64, work, {}};
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto chain = [&] { for (int i = 0; i < N; ++i) k<<<4, 64, 0, s>>>(p); };

    for (int r = 0; r < 3; ++r) chain();
    CK(hipStreamSynchronize(s));
    double enq = 0, wall = 0; float gpu = 0;
    for (int r = 0; r < reps; ++r) {
        CK(hipEventRecord(e0, s));
        const double t0 = now_us();
        chain();
        const double t1 = now_us();
        CK(hipEventRecord(e1, s));
        CK(hipEventSynchronize(e1));
        const double t2 = now_us();
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        enq += t1 - t0; wall += t2 - t0; gpu += ms;
    }
    printf("direct: N %d work %d | enqueue %8.1f us (%.2f us/kernel) | wall %8.1f us | gpu %8.1f us (%.2f us/kernel)\n",
           N, work, enq / reps, enq / reps / N, wall / reps, gpu * 1000 / reps, gpu * 1000 / reps / N);

    hipGraph_t g; hipGraphExec_t ge;
    double tc = now_us();
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    chain();
    CK(hipStreamEndCapture(s, &g));
    const double tcap = now_us() - tc;
    tc = now_us();
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    const double tinst = now_us() - tc;
    for (int r = 0; r < 3; ++r) CK(hipGraphLaunch(ge, s));
    CK(hipStreamSynchronize(s));
    enq = 0; wall = 0; gpu = 0;
    for (int r = 0; r < reps; ++r) {
        CK(hipEventRecord(e0, s));
        const double t0 = now_us();
        CK(hipGraphLaunch(ge, s));
        const double t1 = now_us();
        CK(hipEventRecord(e1, s));
        CK(hipEventSynchronize(e1));
        const double t2 = now_us();
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        enq += t1 - t0; wall += t2 - t0; gpu += ms;
    }
    // pre-queued: a long kernel first, so the whole chain is in the queue before the GPU reaches it
    auto prequeued = [&](bool graph) {
        float tot = 0;
        for (int r = 0; r < reps; ++r) {
            busy<<<1, 64, 0, s>>>(a, 20000000);
            CK(hipEventRecord(e0, s));
            if (graph) { CK(hipGraphLaunch(ge, s)); } else { chain(); }
            CK(hipEventRecord(e1, s));
            CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); tot += ms;
        }
        return tot * 1000 / reps;
    };
    const float pq_direct = prequeued(false), pq_graph = prequeued(true);
    printf("pre-queued chain on the GPU: direct %8.1f us (%.2f us/kernel), graph %8.1f us (%.2f us/kernel)\n",
           pq_direct, pq_direct / N, pq_graph, pq_graph / N);
    // start-up latency: the first kernel writes a flag to pinned host memory; a spinning host thread notes when it
    // flips, against the start of hipGraphLaunch (or of the direct chain)
    {
        volatile int * flag; CK(hipHostMalloc((void **) &flag, 64, hipHostMallocMapped | hipHostMallocCoherent));
        hipGraph_t g2; hipGraphExec_t ge2;
        int v = 0;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        mark<<<1, 1, 0, s>>>((int *) flag, 0);
        chain();
        CK(hipStreamEndCapture(s, &g2));
        CK(hipGraphInstantiate(&ge2, g2, nullptr, nullptr, 0));
        // the captured mark writes 0; rewrite its argument per replay is not possible, so clear the flag to -1 first
        auto measure = [&](bool graph) {
            double tot = 0;
            for (int r = 0; r < reps; ++r) {
                *flag = -1;
                std::atomic<double> seen{0};
                std::thread spin([&] { while (*flag != (graph ? 0 : r + 1)) {} seen = now_us(); });
                const double t0 = now_us();
                if (graph) { CK(hipGraphLaunch(ge2, s)); } else { mark<<<1, 1, 0, s>>>((int *) flag, r + 1); chain(); }
                spin.join();
                CK(hipStreamSynchronize(s));
                tot += seen - t0;
            }
            return tot / reps;
        };
        for (int r = 0; r < 2; ++r) CK(hipGraphLaunch(ge2, s));
        CK(hipStreamSynchronize(s));
        const double lg = measure(true), ld = measure(false);
        printf("first kernel runs after: graph %.1f us, direct %.1f us (from the start of the launch call)\n", lg, ld);
        (void) v;
    }
    printf("graph:  N %d work %d | launch  %8.1f us (%.2f us/kernel) | wall %8.1f us | gpu %8.1f us (%.2f us/kernel) | capture %.0f us, instantiate %.0f us\n",
           N, work, enq / reps, enq / reps / N, wall / reps, gpu * 1000 / reps, gpu * 1000 / reps / N, tcap, tinst);
    return 0;
}
