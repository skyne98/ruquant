// What limits w4a4_skinny at T = 32 (R = 1, KA = 1, TB = 32)? The same loop, prefetch on, in
// four variants that each remove one input path:
//   0  as the kernel: weights streamed, activations scalar-loaded block by block
//   1  activations loaded once before the loop: no scalar loads in the loop
//   2  activations scalar-loaded every block, but always the same 512 B: scalar-cache hits
//   3  weights loaded once before the loop: no vector loads in the loop
//   4  weights loaded every step, but always the same 1 KB per wave: L2 hits, no DRAM
//   5  as 0, but weights two steps ahead instead of one
// If 1 is much faster than 0, the scalar path is the limit; 2 then says whether it is the
// cache's hit bandwidth (2 ~ 0) or refills from L2 (2 ~ 1). 3 does the same for the weights.
// Results are garbage on purpose; only the time matters. Best W x G split per variant.
//   ./probe_t32 [device]
#include "../gfx906/gpu.h"
#include "../gfx906/w4a4_skinny.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int T = 32, TB = 32, NB = RQ_KW; // KA = 1: one block per k-word, 4 per step

template <int MODE>
__global__ void __launch_bounds__(RQ_WAVE * 8)
probe(const int32_t* __restrict__ Ac, const int32_t* __restrict__ Wt, int32_t* __restrict__ acc, int N, int Kw,
      int kw_split, int zero) {
    const int lane = threadIdx.x % RQ_WAVE;
    const int wave = rq_wave_id(), W = blockDim.x / RQ_WAVE;
    const int g0 = blockIdx.x;
    const int wg_begin = blockIdx.y * kw_split;
    const int wg_steps = (min(wg_begin + kw_split, Kw) - wg_begin) / RQ_KW;
    const int per_wave = (wg_steps + W - 1) / W;
    const int s_lo = min(wave * per_wave, wg_steps);
    const int steps = min(s_lo + per_wave, wg_steps) - s_lo;
    const int w_begin = wg_begin + s_lo * RQ_KW;
    const rq_words<RQ_KW>* wp =
        reinterpret_cast<const rq_words<RQ_KW>*>(Wt) + ((size_t)g0 * (Kw / RQ_KW) + w_begin / RQ_KW) * RQ_WAVE + lane;
    const rq_words<TB>* ap = reinterpret_cast<const rq_words<TB>*>(Ac) + (size_t)w_begin;

    int sum[T];
#pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0;
    const int nblocks = steps * NB;
    rq_words<RQ_KW> cur, nxt, nxt2;
    rq_words<TB> a_cur;
    if (steps > 0) {
        cur = wp[0];
        a_cur = ap[0];
        if (MODE == 5) nxt2 = wp[(size_t)(steps > 1 ? 1 : 0) * RQ_WAVE];
    }
    for (int s = 0; s < steps; ++s) {
        const int s_next = s + 1 < steps ? s + 1 : s;
        if (MODE == 3) nxt = cur;
        else if (MODE == 4) nxt = wp[(size_t)(s & zero) * RQ_WAVE];
        else if (MODE == 5) {
            nxt = nxt2;
            nxt2 = wp[(size_t)(s + 2 < steps ? s + 2 : steps - 1) * RQ_WAVE];
        } else nxt = wp[(size_t)s_next * RQ_WAVE];
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int qi = s * NB + q;
            const rq_words<TB> a = a_cur;
            rq_words<TB> a_nxt;
            sum[0] = rq_dot8(a.v[0], cur.v[q], sum[0]);
            rq_block_fence();
            if (MODE == 1) a_nxt = a_cur;
            else if (MODE == 2) a_nxt = ap[q + (s & zero)]; // zero = 0 at run time: not hoistable
            else a_nxt = ap[qi + 1 < nblocks ? qi + 1 : qi];
#pragma unroll
            for (int t = 1; t < T; ++t) sum[t] = rq_dot8(a.v[t], cur.v[q], sum[t]);
            rq_block_fence();
            a_cur = a_nxt;
        }
        cur = nxt;
    }
#pragma unroll
    for (int t = 0; t < T; ++t) atomicAdd(&acc[(size_t)t * N + g0 * RQ_WAVE + lane], sum[t]);
}

struct Shape {
    const char* name;
    int K, N;
};

template <int MODE> static double best_us(const Shape& s, int& bw, int& bg) {
    const int Kw = s.K / 8, N = s.N;
    const size_t w_bytes = (size_t)N * Kw * 4, a_bytes = (size_t)T * Kw * 4;
    const int copies = (int)std::max<size_t>(1, (64u << 20) / w_bytes + 1);
    std::vector<int32_t*> wt(copies);
    for (auto& p : wt) {
        CHECK(cudaMalloc(&p, w_bytes));
        CHECK(cudaMemset(p, 0x11, w_bytes));
    }
    int32_t *ac, *acc;
    CHECK(cudaMalloc(&ac, a_bytes));
    CHECK(cudaMemset(ac, 0x22, a_bytes));
    CHECK(cudaMalloc(&acc, (size_t)T * N * 4));
    double best = 1e30;
    for (int W : {1, 2, 4, 8})
        for (int G : {1, 2, 4, 8, 16, 32}) {
            int kw_split = ((Kw + G - 1) / G + 3) / 4 * 4;
            if (kw_split < 4 * W) continue;
            const dim3 grid(N / RQ_WAVE, (Kw + kw_split - 1) / kw_split);
            int rep = 0;
            const double us = 1e3 * rq_time_ms([&] {
                probe<MODE><<<grid, RQ_WAVE * W>>>(ac, wt[rep++ % copies], acc, N, Kw, kw_split, 0);
            }, 10, 3, 10);
            if (us < best) best = us, bw = W, bg = G;
        }
    for (auto p : wt) CHECK(cudaFree(p));
    CHECK(cudaFree(ac));
    CHECK(cudaFree(acc));
    return best;
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    CHECK(cudaSetDevice(dev));
    const Shape shapes[] = {{"27b mlp.down", 17408, 5120}, {"27b mlp.gate/up", 5120, 17408},
                            {"27b attn.q+gate", 5120, 12288}, {"27b attn.o", 6144, 5120}};
    std::printf("%-18s %13s %13s %13s %13s %13s %13s   us (TMAC/s)\n", "T=32 shape", "0 kernel", "1 no act ld",
                "2 act hits", "3 no wt ld", "4 wt L2 hits", "5 wt 2 ahead");
    for (const Shape& s : shapes) {
        std::printf("%-18s", s.name);
        double us[6];
        int W[6], G[6];
        us[0] = best_us<0>(s, W[0], G[0]);
        us[1] = best_us<1>(s, W[1], G[1]);
        us[2] = best_us<2>(s, W[2], G[2]);
        us[3] = best_us<3>(s, W[3], G[3]);
        us[4] = best_us<4>(s, W[4], G[4]);
        us[5] = best_us<5>(s, W[5], G[5]);
        for (int m = 0; m < 6; ++m)
            std::printf("  %6.1f (%4.1f)", us[m], (double)s.K * s.N * T / (us[m] * 1e-6) / 1e12);
        std::printf("\n");
    }
    return 0;
}
