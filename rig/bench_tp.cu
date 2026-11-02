// Tensor-parallel numbers for the cost model, on the first N GPUs (default: all visible):
//   - peer copy bandwidth for every ordered pair, a ring with every link busy at once, and
//     every GPU sending to GPU 0 at once (its ingress limit);
//   - all-reduce of activation-sized f16 buffers (T tokens x 5120 x 2 B) over the first 2 and
//     all N GPUs, four ways:
//       rccl   RCCL's ncclAllReduce (HIP build, when rccl.h is found)
//       copy   every GPU pulls every peer's partial with copy engines, then adds
//       read   one kernel per GPU reads every peer's partial over P2P and adds
//       write  every GPU stores its partial into every peer, then each adds locally
//     Ops run back to back and each waits for the previous op on every GPU, as layer l+1
//     waits for layer l's sum. One op from a known fill is checked first ("ok"). "issue_us"
//     is host time spent enqueueing: when it is close to "us", the host is the limit.
//   ./bench_tp [N]     -> JSON on stdout
#include "../gfx906/gpu.h"
#ifdef RQ_RCCL
#include <rccl/rccl.h>
#define NCHECK(x)                                                                          \
    do {                                                                                   \
        ncclResult_t r_ = (x);                                                             \
        if (r_ != ncclSuccess) {                                                           \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r_)); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int MAXG = 8;
constexpr size_t MAX_BYTES = 4 << 20; // largest all-reduce
constexpr size_t BIG = 32 << 20;      // bandwidth tests

struct Ptrs {
    const __half* p[MAXG];
};
struct Slots {
    __half* p[MAXG];
};

__global__ void fill(__half* p, size_t n, float v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2half(v);
}

// dst = sum of the N buffers in src, 8 halves per lane. src.p[g] may live on another GPU; all
// N loads are issued before the first add, so a PCIe round trip is paid once.
template <int N> __global__ void sum_into(Ptrs src, __half* __restrict__ dst, size_t n8) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 v[N];
#pragma unroll
    for (int g = 0; g < N; ++g) v[g] = reinterpret_cast<const uint4*>(src.p[g])[i];
    float acc[8] = {};
#pragma unroll
    for (int g = 0; g < N; ++g) {
        const __half* h = reinterpret_cast<const __half*>(&v[g]);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += __half2float(h[e]);
    }
    uint4 out;
    __half* o = reinterpret_cast<__half*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = __float2half(acc[e]);
    reinterpret_cast<uint4*>(dst)[i] = out;
}

// this GPU's partial into its slot on every peer (null for itself): posted stores over P2P
__global__ void push(const __half* __restrict__ src, Slots dst, int n, size_t n8) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 v = reinterpret_cast<const uint4*>(src)[i];
    for (int g = 0; g < n; ++g)
        if (dst.p[g]) reinterpret_cast<uint4*>(dst.p[g])[i] = v;
}

static void launch_sum(int n, const Ptrs& src, __half* dst, size_t n8, cudaStream_t s) {
    const unsigned blocks = (unsigned)((n8 + 255) / 256);
    switch (n) {
    case 2: sum_into<2><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    case 3: sum_into<3><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    case 4: sum_into<4><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    case 5: sum_into<5><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    case 6: sum_into<6><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    case 7: sum_into<7><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    default: sum_into<8><<<blocks, 256, 0, s>>>(src, dst, n8); break;
    }
}

struct Gpu {
    cudaStream_t s;
    cudaEvent_t done[2], pushed; // end of this GPU's part of op k (by parity); end of its push
    __half* buf[2];              // partial in, sum out; the two swap every op
    __half* slot;                // MAXG landing slots of MAX_BYTES for peers' partials
    __half *big, *land;          // bandwidth tests: BIG source, MAXG x BIG landing
};

enum Method { RCCL, COPY, READ, WRITE };
static const char* NAMES[] = {"rccl", "copy", "read", "write"};

struct Group {
    std::vector<Gpu>& g;
    int n;
    size_t bytes = 0;
    long k = 0; // ops issued, for buffer and event parity
#ifdef RQ_RCCL
    ncclComm_t comm[MAXG];
#endif
};

static void sync_all(Group& G) {
    for (int i = 0; i < G.n; ++i) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(G.g[i].s));
    }
}

// before op k on GPU i: every peer has finished op k-1, so the buffers op k reads are written
// and the ones it overwrites are no longer being read
static void wait_peers(Group& G, int i, int parity) {
    for (int j = 0; j < G.n; ++j)
        if (j != i) CHECK(cudaStreamWaitEvent(G.g[i].s, G.g[j].done[parity], 0));
}

static void op(Group& G, Method m) {
    auto& g = G.g;
    const int cur = G.k & 1, prev = cur ^ 1;
    const size_t n8 = G.bytes / 16, elems = MAX_BYTES / sizeof(__half);
    if (m == RCCL) {
#ifdef RQ_RCCL
        NCHECK(ncclGroupStart());
        for (int i = 0; i < G.n; ++i)
            NCHECK(ncclAllReduce(g[i].buf[0], g[i].buf[0], G.bytes / 2, ncclFloat16, ncclSum, G.comm[i], g[i].s));
        NCHECK(ncclGroupEnd());
#endif
    } else if (m == WRITE) {
        for (int i = 0; i < G.n; ++i) {
            CHECK(cudaSetDevice(i));
            wait_peers(G, i, prev); // peers are done reading their slots from op k-1
            Slots dst{};
            for (int j = 0; j < G.n; ++j)
                if (j != i) dst.p[j] = g[j].slot + i * elems;
            push<<<(unsigned)((n8 + 255) / 256), 256, 0, g[i].s>>>(g[i].buf[cur], dst, G.n, n8);
            CHECK(cudaEventRecord(g[i].pushed, g[i].s));
        }
        for (int i = 0; i < G.n; ++i) {
            CHECK(cudaSetDevice(i));
            Ptrs src{};
            src.p[0] = g[i].buf[cur];
            for (int j = 0, c = 1; j < G.n; ++j)
                if (j != i) {
                    CHECK(cudaStreamWaitEvent(g[i].s, g[j].pushed, 0));
                    src.p[c++] = g[i].slot + j * elems;
                }
            launch_sum(G.n, src, g[i].buf[prev], n8, g[i].s);
            CHECK(cudaEventRecord(g[i].done[cur], g[i].s));
        }
    } else {
        for (int i = 0; i < G.n; ++i) {
            CHECK(cudaSetDevice(i));
            wait_peers(G, i, prev);
            Ptrs src{};
            src.p[0] = g[i].buf[cur];
            for (int j = 0, c = 1; j < G.n; ++j) {
                if (j == i) continue;
                if (m == COPY) {
                    CHECK(cudaMemcpyPeerAsync(g[i].slot + j * elems, i, g[j].buf[cur], j, G.bytes, g[i].s));
                    src.p[c++] = g[i].slot + j * elems;
                } else {
                    src.p[c++] = g[j].buf[cur];
                }
            }
            launch_sum(G.n, src, g[i].buf[prev], n8, g[i].s);
            CHECK(cudaEventRecord(g[i].done[cur], g[i].s));
        }
    }
    ++G.k;
}

// one op from GPU i holding i+1 everywhere: every GPU must end with n(n+1)/2
static bool check(Group& G, Method m) {
    const size_t elems = G.bytes / sizeof(__half);
    for (int i = 0; i < G.n; ++i) {
        CHECK(cudaSetDevice(i));
        fill<<<(unsigned)((elems + 255) / 256), 256, 0, G.g[i].s>>>(G.g[i].buf[0], elems, (float)(i + 1));
    }
    sync_all(G);
    G.k = 0;
    op(G, m);
    sync_all(G);
    const float want = G.n * (G.n + 1) / 2.0f;
    bool ok = true;
    std::vector<__half> h(elems);
    for (int i = 0; i < G.n; ++i) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpy(h.data(), G.g[i].buf[m == RCCL ? 0 : 1], G.bytes, cudaMemcpyDeviceToHost));
        for (size_t e = 0; e < elems; ++e) ok &= __half2float(h[e]) == want;
    }
    return ok;
}

static void time_op(Group& G, Method m, int ops, double& us, double& issue_us) {
    for (int w = 0; w < 5; ++w) op(G, m);
    sync_all(G);
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    for (int r = 0; r < ops; ++r) op(G, m);
    const auto t1 = clk::now();
    sync_all(G);
    const auto t2 = clk::now();
    us = std::chrono::duration<double, std::micro>(t2 - t0).count() / ops;
    issue_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ops;
}

// wall time of `reps` rounds of the given (src, dst) copies issued together, each on its
// source GPU's stream
static double copies_gbs(std::vector<Gpu>& g, int n, const std::vector<std::pair<int, int>>& pairs, int reps) {
    auto sync = [&] {
        for (int i = 0; i < n; ++i) {
            CHECK(cudaSetDevice(i));
            CHECK(cudaStreamSynchronize(g[i].s));
        }
    };
    auto issue = [&] {
        for (auto [a, b] : pairs) {
            CHECK(cudaSetDevice(a));
            CHECK(cudaMemcpyPeerAsync(g[b].land + a * (BIG / sizeof(__half)), b, g[a].big, a, BIG, g[a].s));
        }
    };
    issue();
    sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) issue();
    sync();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)reps * BIG / s / 1e9; // per copy
}

int main(int argc, char** argv) {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    const int N = std::min({argc > 1 ? std::atoi(argv[1]) : count, count, MAXG});
#ifdef RQ_RCCL
    const bool have_rccl = true;
#else
    const bool have_rccl = false;
#endif
    std::printf("{\n  \"bench\": \"tp\", \"toolchain\": \"%s\", \"devices\": %d", rq_toolchain(), N);
    if (N < 2) {
        std::printf(", \"skipped\": \"needs two GPUs\"\n}\n");
        return 0;
    }
    bool p2p = true;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            if (a != b) {
                int ok = 0;
                CHECK(cudaDeviceCanAccessPeer(&ok, a, b));
                p2p &= ok != 0;
            }
    std::vector<Gpu> g(N);
    for (int i = 0; i < N; ++i) {
        CHECK(cudaSetDevice(i));
        if (p2p)
            for (int j = 0; j < N; ++j)
                if (j != i) CHECK(cudaDeviceEnablePeerAccess(j, 0));
        CHECK(cudaStreamCreate(&g[i].s));
        for (auto& e : g[i].done) CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&g[i].pushed, cudaEventDisableTiming));
        for (auto& b : g[i].buf) CHECK(cudaMalloc(&b, MAX_BYTES));
        CHECK(cudaMalloc(&g[i].slot, MAXG * MAX_BYTES));
        CHECK(cudaMalloc(&g[i].big, BIG));
        CHECK(cudaMalloc(&g[i].land, MAXG * BIG));
        CHECK(cudaMemset(g[i].big, 1, BIG));
    }
    std::printf(", \"p2p\": %s, \"rccl\": %s,\n", p2p ? "true" : "false", have_rccl ? "true" : "false");

    // bandwidth: every ordered pair alone, then a ring and a fan-in into GPU 0 with every
    // participating link busy at once
    std::printf("  \"pair_gbs\": [");
    for (int a = 0; a < N; ++a) {
        std::printf("%s[", a ? ", " : "");
        for (int b = 0; b < N; ++b)
            std::printf("%s%.2f", b ? ", " : "", a == b ? 0.0 : copies_gbs(g, N, {{a, b}}, 4));
        std::printf("]");
    }
    std::vector<std::pair<int, int>> ring, fanin;
    for (int i = 0; i < N; ++i) ring.push_back({i, (i + 1) % N});
    for (int i = 1; i < N; ++i) fanin.push_back({i, 0});
    std::printf("],\n  \"ring_gbs_per_link\": %.2f, \"fanin_gbs_per_link\": %.2f,\n", copies_gbs(g, N, ring, 4),
                copies_gbs(g, N, fanin, 4));

    // all-reduce sweep
    std::printf("  \"allreduce\": [\n");
    const size_t tokens[] = {1, 2, 4, 8, 16, 32, 64, 409}; // x 5120 x 2 B; 409 ~ 4 MiB
    std::vector<int> sizes = {2};
    if (N > 2) sizes.push_back(N);
    bool first = true;
    for (int n : sizes) {
        Group G{g, n};
#ifdef RQ_RCCL
        int devs[MAXG];
        for (int i = 0; i < n; ++i) devs[i] = i;
        NCHECK(ncclCommInitAll(G.comm, n, devs));
#endif
        for (int m = RCCL; m <= WRITE; ++m) {
            if (m == RCCL && !have_rccl) continue;
            if ((m == READ || m == WRITE) && !p2p) continue;
            for (size_t t : tokens) {
                G.bytes = t * 5120 * 2;
                const bool ok = check(G, (Method)m);
                double us = 0, issue_us = 0;
                time_op(G, (Method)m, G.bytes <= (1u << 20) ? 200 : 50, us, issue_us);
                std::printf("%s    {\"gpus\": %d, \"method\": \"%s\", \"tokens\": %zu, \"bytes\": %zu, \"ok\": %s, "
                            "\"us\": %.2f, \"issue_us\": %.2f}",
                            first ? "" : ",\n", n, NAMES[m], t, G.bytes, ok ? "true" : "false", us, issue_us);
                first = false;
            }
        }
#ifdef RQ_RCCL
        for (int i = 0; i < n; ++i) ncclCommDestroy(G.comm[i]);
#endif
    }
    std::printf("\n  ]\n}\n");
    return 0;
}
