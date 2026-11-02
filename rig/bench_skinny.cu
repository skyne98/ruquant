// Times w4a4_skinny on the Qwen3.8 / DFlash2 shapes the cost model uses, at every split-K
// choice (W waves per workgroup x G workgroups along K), so calibrate.py can put measured
// next to predicted. At T >= 32 it also times R = 2 (two channel groups per wave), which
// halves scalar-cache traffic per dot8: if R = 2 is much faster there, the scalar cache is
// the limit the model warned about.
//
// Weights rotate through enough copies to overflow L2, as in a real forward pass where every
// layer's weights come from HBM. A G > 1 point includes the memset of the int32 accumulators.
//   ./bench_skinny [device] [--quick]     -> JSON on stdout
#include "../gfx906/gpu.h"
#include "../gfx906/w4a4_skinny.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Shape {
    const char* name;
    int K, N;
};

// unique (K, N) of gfx906/model.py's Qwen3.8-27B, Flash-Next and DFlash2 drafter layers
static const Shape SHAPES[] = {
    {"27b attn.q+gate", 5120, 12288}, {"27b attn.k/v", 5120, 1024}, {"27b attn.o / dn.out", 6144, 5120},
    {"27b dn.qkvz", 5120, 16384},     {"27b mlp.gate/up", 5120, 17408}, {"27b mlp.down", 17408, 5120},
    {"27b drafter q", 5120, 4096},    {"27b drafter o", 4096, 5120},    {"27b drafter fc", 25600, 5120},
    {"fn qsa.q+gate", 2560, 12288},   {"fn dn.qkvz", 2560, 16384},      {"fn qsa.o / dn.out", 6144, 2560},
    {"fn expert gate/up", 2560, 640}, {"fn expert down", 640, 2560},
};

static bool first_point = true;

template <int T, int R, int KA, int TB> static void sweep(const Shape& s, int lds_max) {
    const int Kw = s.K / 8;
    const int N = (s.N + RQ_WAVE * R - 1) / (RQ_WAVE * R) * (RQ_WAVE * R); // pad to whole groups
    const size_t w_bytes = (size_t)N * Kw * 4, a_bytes = (size_t)T * Kw * 4, acc_bytes = (size_t)T * N * 4;
    const int copies = (int)std::max<size_t>(1, (64u << 20) / w_bytes + 1);

    std::vector<int32_t> host(std::max(w_bytes, a_bytes) / 4);
    std::mt19937 rng(906);
    for (auto& v : host) v = (int32_t)rng();
    std::vector<int32_t*> wt(copies);
    for (auto& p : wt) {
        CHECK(cudaMalloc(&p, w_bytes));
        CHECK(cudaMemcpy(p, host.data(), w_bytes, cudaMemcpyHostToDevice));
    }
    int32_t *ac, *acc;
    CHECK(cudaMalloc(&ac, a_bytes));
    CHECK(cudaMemcpy(ac, host.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&acc, acc_bytes));

    auto kernel = w4a4_skinny<T, R, KA, TB>;
    int last_splits = 0;
    for (int W : {1, 2, 4, 8}) {
        const size_t lds = rq_lds_bytes<T, R>(W);
        if (lds > (size_t)lds_max) continue;
        rq_allow_smem(kernel, lds);
        last_splits = 0;
        for (int G : {1, 2, 4, 8, 16, 32}) {
            int kw_split = (Kw + G - 1) / G;
            kw_split = (kw_split + 3) / 4 * 4;
            const int splits = (Kw + kw_split - 1) / kw_split;
            if (splits == last_splits || kw_split < 4 * W) continue; // same launch, or idle waves
            last_splits = splits;
            const dim3 grid(N / (RQ_WAVE * R), splits);
            int rep = 0;
            // batched (back-to-back launches, as in a forward pass), and without zeroing acc
            // between split-K launches: the epilogue that reads acc would zero it
            const double ms = rq_time_ms([&] {
                kernel<<<grid, RQ_WAVE * W, lds>>>(ac, wt[rep++ % copies], acc, N, Kw, kw_split);
            }, 10, 3, 10);
            std::printf("%s    {\"shape\": \"%s\", \"K\": %d, \"N\": %d, \"T\": %d, \"R\": %d, \"W\": %d, "
                        "\"G\": %d, \"us\": %.2f, \"weight_gbs\": %.1f}",
                        first_point ? "" : ",\n", s.name, s.K, s.N, T, R, W, splits, ms * 1e3,
                        (double)s.N * s.K / 2 / (ms * 1e-3) / 1e9);
            first_point = false;
        }
    }
    for (auto p : wt) CHECK(cudaFree(p));
    CHECK(cudaFree(ac));
    CHECK(cudaFree(acc));
}

template <int T> static void preset(const Shape& s, int lds_max) {
    sweep<T, rq_cfg<T>::R, rq_cfg<T>::KA, rq_cfg<T>::TB>(s, lds_max);
}

int main(int argc, char** argv) {
    const int dev = argc > 1 && argv[1][0] != '-' ? std::atoi(argv[1]) : 0;
    bool quick = false;
    for (int i = 1; i < argc; ++i) quick |= std::strcmp(argv[i], "--quick") == 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    const int lds_max = RQ_WAVE == 64 ? 64 << 10 : 99 << 10;

    std::printf("{\n  \"bench\": \"skinny\", \"toolchain\": \"%s\", \"device\": %d, \"name\": \"%s\", "
                "\"wave\": %d, \"quick\": %s,\n  \"points\": [\n",
                rq_toolchain(), dev, prop.name, RQ_WAVE, quick ? "true" : "false");
    const int nshapes = quick ? 2 : (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));
    for (int i = 0; i < nshapes; ++i) {
        const Shape& s = quick ? SHAPES[i * 5] : SHAPES[i];
        preset<1>(s, lds_max);
        preset<4>(s, lds_max);
        preset<8>(s, lds_max);
        preset<16>(s, lds_max);
        preset<32>(s, lds_max);
        sweep<32, 2, 1, 32>(s, lds_max);
        if (!quick) {
            preset<64>(s, lds_max);
            sweep<64, 2, 1, 32>(s, lds_max);
        }
    }
    std::printf("\n  ]\n}\n");
    return 0;
}
