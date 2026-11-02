// Correctness of the skinny W4A4 kernels against a CPU reference: integer results must match
// exactly, and the epilogue must match the spec's fp32 -> f16 rounding bit for bit.
// On the 3090 (nvcc) waves are 32 lanes and dot8 is emulated, so this checks the indexing;
// on the MI50 (hipcc) it runs the real v_dot8_i32_i4 with 64-lane waves.
//   nvcc -O2 -arch=sm_86 -std=c++17 gfx906/test_skinny.cu -o gfx906/build/test_skinny
//   hipcc -x hip --offload-arch=gfx906 -O3 -std=c++17 gfx906/test_skinny.cu -o test_skinny
//   ./test_skinny [device]
#include "gpu.h"
#include "w4a4_skinny.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(906);

// spec packing: element 8w+j in bits [4j+3:4j], two's complement
static std::vector<int32_t> pack(const std::vector<int8_t>& codes, int rows, int K) {
    std::vector<int32_t> out((size_t)rows * K / 8, 0);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k)
            out[((size_t)r * K + k) / 8] |= (uint32_t)(codes[(size_t)r * K + k] & 0xF) << (4 * (k % 8));
    return out;
}

static uint16_t f32_to_f16_rne(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, ax = x & 0x7fffffff;
    if (ax >= 0x7f800000) return sign | (ax > 0x7f800000 ? 0x7e00 : 0x7c00);
    if (ax >= 0x477ff000) return sign | 0x7c00; // >= 65520 rounds to inf
    if (ax < 0x38800000) {                       // f16 subnormal: round(|f| * 2^24)
        float af;
        std::memcpy(&af, &ax, 4);
        return sign | (uint16_t)std::nearbyint((double)af * 16777216.0);
    }
    uint32_t h = ((((ax >> 23) - 127 + 15) << 10) | ((ax & 0x7fffff) >> 13));
    const uint32_t rem = ax & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return sign | h;
}

template <typename T> static T* upload(const std::vector<T>& v) {
    T* d;
    CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <int T> static bool run(int N, int K, int kw_split, int W) {
    constexpr int R = rq_cfg<T>::R, KA = rq_cfg<T>::KA;
    const int Kw = K / 8;
    std::uniform_int_distribution<int> code(-8, 7);
    std::vector<int8_t> a((size_t)T * K), w((size_t)N * K);
    for (auto& v : a) v = (int8_t)code(rng);
    for (auto& v : w) v = (int8_t)code(rng);

    const auto a_rows = pack(a, T, K), w_rows = pack(w, N, K);
    std::vector<int32_t> ac(a_rows.size()), wt(w_rows.size());
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < Kw; ++k) ac[rq_act_index(t, k, T, KA)] = a_rows[(size_t)t * Kw + k];
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < Kw; ++k) wt[rq_wt_index(n, k, Kw)] = w_rows[(size_t)n * Kw + k];

    std::vector<int32_t> ref((size_t)T * N, 0), colsum(N, 0);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) colsum[n] += w[(size_t)n * K + k];
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n) {
            int s = 0;
            for (int k = 0; k < K; ++k) s += a[(size_t)t * K + k] * w[(size_t)n * K + k];
            ref[(size_t)t * N + n] = s;
        }

    int32_t *d_ac = upload(ac), *d_wt = upload(wt), *d_acc;
    CHECK(cudaMalloc(&d_acc, ref.size() * 4));
    const dim3 grid(N / (RQ_WAVE * R), (Kw + kw_split - 1) / kw_split);
    // one split along y writes instead of adding: fill with garbage to prove it overwrites
    CHECK(cudaMemset(d_acc, grid.y > 1 ? 0 : 0xAB, ref.size() * 4));
    const size_t lds = rq_lds_bytes<T>(W);
    rq_allow_smem(w4a4_skinny<T>, lds);
    w4a4_skinny<T><<<grid, RQ_WAVE * W, lds>>>(d_ac, d_wt, d_acc, N, Kw, kw_split);
    CHECK(cudaGetLastError());
    std::vector<int32_t> got(ref.size());
    CHECK(cudaMemcpy(got.data(), d_acc, got.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != ref[i];

    // epilogue, spec rounding
    std::uniform_real_distribution<float> sa(0.01f, 0.1f), sw(0.001f, 0.01f);
    std::uniform_int_distribution<int> zd(0, 15);
    std::vector<float> s_a(T), s_w(N);
    std::vector<int32_t> z(T);
    for (int t = 0; t < T; ++t) s_a[t] = sa(rng), z[t] = zd(rng);
    for (int n = 0; n < N; ++n) s_w[n] = sw(rng);
    float *d_sa = upload(s_a), *d_sw = upload(s_w);
    int32_t *d_z = upload(z), *d_cs = upload(colsum);
    rq_half* d_y;
    CHECK(cudaMalloc(&d_y, ref.size() * 2));
    w4a4_epilogue<<<dim3((N + 255) / 256, T), 256>>>(d_acc, d_sa, d_z, d_sw, d_cs, d_y, T, N);
    CHECK(cudaGetLastError());
    std::vector<uint16_t> y(ref.size());
    CHECK(cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost));
    int bad_y = 0;
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n) {
            const int corr = ref[(size_t)t * N + n] + (8 - z[t]) * colsum[n];
            volatile float p = (float)corr * s_a[t]; // two separate fp32 roundings
            const float yf = p * s_w[n];
            bad_y += y[(size_t)t * N + n] != f32_to_f16_rne(yf);
        }

    for (void* p : {(void*)d_ac, (void*)d_wt, (void*)d_acc, (void*)d_sa, (void*)d_sw, (void*)d_z,
                    (void*)d_cs, (void*)d_y})
        CHECK(cudaFree(p));
    std::printf("T=%-2d R=%d KA=%d  N=%-3d K=%-4d kw_split=%-3d W=%d  acc %s  epilogue %s\n", T, R, KA, N,
                K, kw_split, W, bad ? "MISMATCH" : "exact", bad_y ? "MISMATCH" : "bit-exact");
    return bad == 0 && bad_y == 0;
}

template <int T> static bool all_splits() {
    const size_t smem = RQ_WAVE == 64 ? 64 << 10 : 99 << 10; // gfx906 LDS / sm_86 opt-in max
    const int w_max = rq_lds_bytes<T>(2) <= smem ? 8 : 1; // the LDS buffer does not grow with W
    return run<T>(512, 1024, 48, 1)        // global splits only, last one shorter
         & run<T>(256, 4096, 512, 4)       // one global split (plain store), 4 waves reduce in LDS
         & run<T>(512, 1024, 64, 3)        // both, uneven wave slices (6, 6, 4 steps)
         & run<T>(256, 256, 16, w_max);    // more waves than steps: some waves get none
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %d: %s (%s, %d-lane waves)\n", dev, prop.name, rq_toolchain(), RQ_WAVE);
    bool ok = all_splits<1>() & all_splits<2>() & all_splits<4>() & all_splits<8>() & all_splits<16>() &
              all_splits<24>() & all_splits<32>() & all_splits<48>() & all_splits<64>();
    std::printf(ok ? "ALL OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
