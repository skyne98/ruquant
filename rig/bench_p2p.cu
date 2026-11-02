// Two-GPU numbers for the multi-GPU terms of the cost model:
//   - peer copy bandwidth both ways (or staged through pinned host memory if P2P is off);
//   - one-way latency of an activation-sized hand-off (layer split: --handoff-us);
//   - a 2-GPU all-reduce built from two peer copies and an add (tensor split: --allreduce-us).
//     RCCL may do better or worse; run rccl-tests' all_reduce_perf too if it is installed.
//   ./bench_p2p     -> JSON on stdout
#include "../gfx906/gpu.h"

#include <cstdio>
#include <cstdlib>

__global__ void add_into(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

static double time_ms(int reps, cudaStream_t s0, cudaStream_t s1, void (*fn)(void*), void* ctx) {
    cudaEvent_t a, b;
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    fn(ctx);
    CHECK(cudaStreamSynchronize(s0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaStreamSynchronize(s1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(a, s0));
    for (int i = 0; i < reps; ++i) fn(ctx);
    CHECK(cudaSetDevice(1));
    CHECK(cudaStreamSynchronize(s1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(b, s0));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

struct Ctx {
    float *buf0, *buf1, *tmp0, *tmp1, *host;
    size_t bytes;
    bool p2p;
    cudaStream_t s0, s1;
};

static void copy_0_to_1(Ctx* c) {
    if (c->p2p) {
        CHECK(cudaMemcpyPeerAsync(c->buf1, 1, c->buf0, 0, c->bytes, c->s0));
    } else {
        CHECK(cudaMemcpyAsync(c->host, c->buf0, c->bytes, cudaMemcpyDeviceToHost, c->s0));
        CHECK(cudaMemcpyAsync(c->buf1, c->host, c->bytes, cudaMemcpyHostToDevice, c->s0));
    }
}

static void one_way(void* p) { copy_0_to_1((Ctx*)p); }

static void ping_pong(void* p) {
    Ctx* c = (Ctx*)p;
    copy_0_to_1(c);
    if (c->p2p) {
        CHECK(cudaMemcpyPeerAsync(c->buf0, 0, c->buf1, 1, c->bytes, c->s0));
    } else {
        CHECK(cudaMemcpyAsync(c->host, c->buf1, c->bytes, cudaMemcpyDeviceToHost, c->s0));
        CHECK(cudaMemcpyAsync(c->buf0, c->host, c->bytes, cudaMemcpyHostToDevice, c->s0));
    }
    CHECK(cudaStreamSynchronize(c->s0));
}

static void allreduce(void* p) {
    Ctx* c = (Ctx*)p;
    const size_t n = c->bytes / sizeof(float);
    CHECK(cudaSetDevice(0));
    CHECK(cudaMemcpyPeerAsync(c->tmp1, 1, c->buf0, 0, c->bytes, c->s0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaMemcpyPeerAsync(c->tmp0, 0, c->buf1, 1, c->bytes, c->s1));
    CHECK(cudaStreamSynchronize(c->s1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamSynchronize(c->s0));
    add_into<<<(unsigned)((n + 255) / 256), 256, 0, c->s0>>>(c->buf0, c->tmp0, n);
    CHECK(cudaSetDevice(1));
    add_into<<<(unsigned)((n + 255) / 256), 256, 0, c->s1>>>(c->buf1, c->tmp1, n);
    CHECK(cudaStreamSynchronize(c->s1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamSynchronize(c->s0));
}

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    std::printf("{\n  \"bench\": \"p2p\", \"toolchain\": \"%s\", \"devices\": %d", rq_toolchain(), count);
    if (count < 2) {
        std::printf(", \"skipped\": \"needs two GPUs\"\n}\n");
        return 0;
    }
    int a01 = 0, a10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&a01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&a10, 1, 0));
    Ctx c{};
    c.p2p = a01 && a10;
    const size_t max_bytes = 256 << 20;
    CHECK(cudaSetDevice(0));
    if (c.p2p) CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaMalloc(&c.buf0, max_bytes));
    CHECK(cudaMalloc(&c.tmp0, max_bytes));
    CHECK(cudaStreamCreate(&c.s0));
    CHECK(cudaSetDevice(1));
    if (c.p2p) CHECK(cudaDeviceEnablePeerAccess(0, 0));
    CHECK(cudaMalloc(&c.buf1, max_bytes));
    CHECK(cudaMalloc(&c.tmp1, max_bytes));
    CHECK(cudaStreamCreate(&c.s1));
    CHECK(cudaMallocHost(&c.host, max_bytes));
    CHECK(cudaSetDevice(0));

    std::printf(", \"p2p\": %s,\n  \"bandwidth\": [\n", c.p2p ? "true" : "false");
    const size_t bw_sizes[] = {1 << 20, 16 << 20, 256 << 20};
    for (int i = 0; i < 3; ++i) {
        c.bytes = bw_sizes[i];
        const double ms = time_ms(10, c.s0, c.s1, one_way, &c);
        std::printf("    {\"bytes\": %zu, \"gbs\": %.2f}%s\n", c.bytes, c.bytes / (ms * 1e-3) / 1e9, i < 2 ? "," : "");
    }
    // activation hand-offs: T tokens x hidden x 2 B, hidden 2560..5120, T 1..32
    std::printf("  ],\n  \"latency\": [\n");
    const size_t small[] = {5120, 10240, 81920, 327680};
    for (int i = 0; i < 4; ++i) {
        c.bytes = small[i];
        const double pp = time_ms(200, c.s0, c.s1, ping_pong, &c);
        const double ar = time_ms(200, c.s0, c.s1, allreduce, &c);
        std::printf("    {\"bytes\": %zu, \"one_way_us\": %.2f, \"allreduce_us\": %.2f}%s\n", c.bytes,
                    pp * 1e3 / 2, ar * 1e3, i < 3 ? "," : "");
    }
    std::printf("  ]\n}\n");
    return 0;
}
