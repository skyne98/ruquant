// rig/hipgraph_args_bench.cpp: hipcc -O2 --offload-arch=gfx906 rig/hipgraph_args_bench.cpp -o hipgraph_args_bench && HIP_VISIBLE_DEVICES=0,1,2,3 ./hipgraph_args_bench 2000 32 18
// hipGraphLaunch scaling with ggml-like graphs: 32 distinct kernels with 18 arguments, cycled through an N-node chain.
//   graphpar2 [N=2000] [KINDS=32] [ARGS=1|18]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <utility>

template <int I> __global__ void kk(float * a, float * b, const float * c, const int * d, float * e, float * f,
                                    int n0, int n1, int n2, int n3, int s0, int s1, int s2, int s3, float x, float y, int m, int z) {
    if (blockIdx.x == 0 && n0 < 0) a[threadIdx.x] = b[0] + c[0] + d[0] + e[0] + f[0] + n1 + n2 + n3 + s0 + s1 + s2 + s3 + x + y + m + z + I;
}
template <int I> __global__ void k1(float * a) { if (blockIdx.x == 0 && a == nullptr) a[threadIdx.x] = I; }

using launch_fn = void (*)(hipStream_t, float *, int);
template <int I> void L(hipStream_t s, float * p, int args) {
    if (args > 1) kk<I><<<4, 64, 0, s>>>(p, p, p, (const int *) p, p, p, 1, 2, 3, 4, 5, 6, 7, 8, 1.0f, 2.0f, 9, 10);
    else          k1<I><<<4, 64, 0, s>>>(p);
}
template <int... I> static std::vector<launch_fn> table(std::integer_sequence<int, I...>) { return { &L<I>... }; }

int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2000, kinds = argc > 2 ? atoi(argv[2]) : 32, args = argc > 3 ? atoi(argv[3]) : 18;
    const auto fns = table(std::make_integer_sequence<int, 32>{});
    int nd = 0; hipGetDeviceCount(&nd);
    std::vector<hipGraphExec_t> ge(nd); std::vector<hipStream_t> st(nd);
    for (int d = 0; d < nd; ++d) {
        hipSetDevice(d); hipStreamCreateWithFlags(&st[d], hipStreamNonBlocking);
        float * p; hipMalloc(&p, 4096);
        hipGraph_t g;
        hipStreamBeginCapture(st[d], hipStreamCaptureModeThreadLocal);
        for (int i = 0; i < N; ++i) fns[i % kinds](st[d], p, args);
        hipStreamEndCapture(st[d], &g);
        hipGraphInstantiate(&ge[d], g, nullptr, nullptr, 0);
        hipGraphLaunch(ge[d], st[d]); hipStreamSynchronize(st[d]);
    }
    for (int nt = 1; nt <= nd; nt *= 2) {
        std::vector<double> us(nt); std::vector<std::thread> th; std::atomic<int> go{0};
        for (int t = 0; t < nt; ++t) th.emplace_back([&, t] {
            hipSetDevice(t);
            while (!go) {}
            double tot = 0;
            for (int r = 0; r < 20; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                hipGraphLaunch(ge[t], st[t]);
                tot += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                hipStreamSynchronize(st[t]);
            }
            us[t] = tot / 20;
        });
        go = 1;
        for (auto & x : th) x.join();
        double m = 0; for (double u : us) m = std::max(m, u);
        printf("kinds %2d args %2d, threads %d: %5.0f us per launch (%.2f us/node)\n", kinds, args, nt, m, m / N);
    }
    for (int nt = 1; nt <= nd; nt *= 2) {
        std::vector<double> us(nt); std::vector<std::thread> th; std::atomic<int> go{0};
        for (int t = 0; t < nt; ++t) th.emplace_back([&, t] {
            hipSetDevice(t);
            float * p; hipMalloc(&p, 4096);
            while (!go) {}
            double tot = 0;
            for (int r = 0; r < 20; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) { fns[i % kinds](st[t], p, args); if (i % 512 == 511) hipStreamSynchronize(st[t]); }
                tot += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                hipStreamSynchronize(st[t]);
            }
            us[t] = tot / 20;
        });
        go = 1;
        for (auto & x : th) x.join();
        double m = 0; for (double u : us) m = std::max(m, u);
        printf("direct: kinds %2d args %2d, threads %d: %5.0f us per chain (%.2f us/kernel)\n", kinds, args, nt, m, m / N);
    }
}
