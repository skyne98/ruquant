// Skinny W4A4 GEMM for speculative-decode verify widths (T = 1..64 tokens), shaped for gfx906.
//
// gfx906 has no matrix unit, so every instruction that is not v_dot8_i32_i4 takes a VALU slot
// that could have done 8 multiply-adds. The inner loop is therefore weight loads + dot8 only:
//   - each lane owns one output channel per channel group (R groups per wave), so there is
//     no cross-lane reduction and no LDS in the loop;
//   - activations are wave-uniform, so they come in through scalar loads and feed dot8 from
//     SGPRs (gfx9 VOP3 may read one SGPR, which is exactly one dot8 operand). They are stored
//     step-major so one step's T*KA words are contiguous: a few wide s_loads, one pointer;
//   - weights are repacked offline so a wave's load is one contiguous 16 B/lane run. The
//     layout is the same for every T, so one copy of the weights serves every verify width;
//   - split-K in two levels: the waves of a workgroup reduce through LDS after the loop, then
//     workgroups add int32 partials atomically. Integer adds are exact and order-independent,
//     so either way this stays bit-exact with the spec's single dot8 chain.
//
// The same source builds for CUDA (wave = 32 lanes, dot8 emulated) so the 3090 can check the
// indexing; only the gfx906 build is meant to be fast.
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(__HIPCC__)
#define RQ_WAVE 64
__device__ __forceinline__ int rq_dot8(int a, int b, int c) {
    return __builtin_amdgcn_sdot8(a, b, c, false); // v_dot8_i32_i4
}
// Without this the scheduler hoists every token block's s_loads to the top of the step,
// which needs more SGPRs than a wave has and spills them through v_writelane/v_readlane.
__device__ __forceinline__ void rq_block_fence() { __builtin_amdgcn_sched_barrier(0); }
// wave index in an SGPR, so per-wave K offsets keep the activation loads scalar
__device__ __forceinline__ int rq_wave_id() { return __builtin_amdgcn_readfirstlane(threadIdx.x / RQ_WAVE); }
#else
#define RQ_WAVE 32
__device__ __forceinline__ int rq_dot8(int a, int b, int c) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        c += ((a << (28 - 4 * j)) >> 28) * ((b << (28 - 4 * j)) >> 28);
    }
    return c;
}
__device__ __forceinline__ void rq_block_fence() {}
__device__ __forceinline__ int rq_wave_id() { return threadIdx.x / RQ_WAVE; }
#endif

template <int N> struct alignas(16) rq_words {
    int32_t v[N];
};

// Weights, 4 k-words (32 int4) per lane per step, for every T:
//   Wt[group][step][lane][4]   group = channel / RQ_WAVE, step = k_word / 4
constexpr int RQ_KW = 4;
inline size_t rq_wt_index(int channel, int k_word, int Kw) {
    const int group = channel / RQ_WAVE, lane = channel % RQ_WAVE;
    return (((size_t)group * (Kw / RQ_KW) + k_word / RQ_KW) * RQ_WAVE + lane) * RQ_KW + k_word % RQ_KW;
}

// Verify-width presets: R channel groups per wave (R*T accumulator VGPRs), KA activation
// words per token per chunk, TB tokens per scalar load (TB*KA SGPRs). Picked from the gfx906
// asm by build.sh / isa.py: fewest instructions per dot8 without SGPR spills (v_writelane).
template <int T> struct rq_cfg;
template <> struct rq_cfg<1>  { static constexpr int R = 4, KA = 4, TB = 1; };
template <> struct rq_cfg<2>  { static constexpr int R = 4, KA = 4, TB = 2; };
template <> struct rq_cfg<4>  { static constexpr int R = 4, KA = 4, TB = 4; };
template <> struct rq_cfg<8>  { static constexpr int R = 2, KA = 4, TB = 8; };
template <> struct rq_cfg<16> { static constexpr int R = 2, KA = 4, TB = 16; };
template <> struct rq_cfg<24> { static constexpr int R = 1, KA = 1, TB = 24; };
template <> struct rq_cfg<32> { static constexpr int R = 1, KA = 1, TB = 32; };
template <> struct rq_cfg<48> { static constexpr int R = 1, KA = 1, TB = 24; };
template <> struct rq_cfg<64> { static constexpr int R = 1, KA = 1, TB = 32; };

// Activations, step-major: Ac[chunk][t][KA], chunk = k_word / KA. The quantizer writes this.
inline size_t rq_act_index(int t, int k_word, int T, int KA) {
    return ((size_t)(k_word / KA) * T + t) * KA + k_word % KA;
}

// Bytes of dynamic LDS for a launch with W waves per workgroup.
template <int T> constexpr size_t rq_lds_bytes(int W) {
    return (size_t)(W - 1) * rq_cfg<T>::R * T * RQ_WAVE * sizeof(int32_t);
}

// Ac:  activation words (spec values: element 8w+j in bits [4j+3:4j], stored as q-8), rq_act_index
// Wt:  weight words, rq_wt_index
// acc: [T][N] int32. Zeroed before launch when gridDim.y > 1 (atomic adds), else just written.
// grid (N / (RQ_WAVE*R), ceil(Kw / kw_split)), block RQ_WAVE*W (W <= 8), rq_lds_bytes<T>(W)
// dynamic LDS; Kw and kw_split multiples of 4.
// Split-K happens in two places: the W waves of a workgroup take slices of its K range and
// reduce through LDS (cheap), and workgroups along y take kw_split chunks and add atomically
// in L2 (expensive at large T: 4 bytes per output per split).
template <int T, int R = rq_cfg<T>::R, int KA = rq_cfg<T>::KA, int TB = rq_cfg<T>::TB>
__global__ void __launch_bounds__(RQ_WAVE * 8)
w4a4_skinny(const int32_t* __restrict__ Ac, const int32_t* __restrict__ Wt,
            int32_t* __restrict__ acc, int N, int Kw, int kw_split) {
    static_assert(RQ_KW % KA == 0 && T % TB == 0, "bad preset");
    constexpr int CHUNKS = RQ_KW / KA;
    const int lane = threadIdx.x % RQ_WAVE;
    const int wave = rq_wave_id(), W = blockDim.x / RQ_WAVE;
    const int g0 = blockIdx.x * R;
    const int wg_begin = blockIdx.y * kw_split;
    const int wg_steps = (min(wg_begin + kw_split, Kw) - wg_begin) / RQ_KW;
    const int per_wave = (wg_steps + W - 1) / W;
    const int s_lo = min(wave * per_wave, wg_steps);
    const int steps = min(s_lo + per_wave, wg_steps) - s_lo;
    const int w_begin = wg_begin + s_lo * RQ_KW;

    const rq_words<RQ_KW>* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wp[r] = reinterpret_cast<const rq_words<RQ_KW>*>(Wt) +
                ((size_t)(g0 + r) * (Kw / RQ_KW) + w_begin / RQ_KW) * RQ_WAVE + lane;
    }
    const rq_words<TB * KA>* ap =
        reinterpret_cast<const rq_words<TB * KA>*>(Ac) + (size_t)(w_begin / KA) * (T / TB);

    int sum[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) sum[r][t] = 0;

    // software pipeline: the next step's weights are in flight while this step's dot8s run
    rq_words<RQ_KW> cur[R], nxt[R];
    if (steps > 0)
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = wp[r][0];

    for (int s = 0; s < steps; ++s) {
        const int s_next = s + 1 < steps ? s + 1 : s; // last step re-reads, no branch
#pragma unroll
        for (int r = 0; r < R; ++r) nxt[r] = wp[r][(size_t)s_next * RQ_WAVE];

#pragma unroll
        for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
            for (int b = 0; b < T / TB; ++b) {
                const rq_words<TB * KA> a = ap[((size_t)s * CHUNKS + c) * (T / TB) + b]; // uniform -> SGPRs
#pragma unroll
                for (int tt = 0; tt < TB; ++tt)
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int j = 0; j < KA; ++j) {
                            const int t = b * TB + tt;
                            sum[r][t] = rq_dot8(a.v[tt * KA + j], cur[r].v[c * KA + j], sum[r][t]);
                        }
                rq_block_fence();
            }
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = nxt[r];
    }

    // reduce the workgroup's K slices: waves 1..W-1 park their sums in LDS, wave 0 adds them
    extern __shared__ int32_t rq_lds[];
    if (W > 1) {
        if (wave > 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int t = 0; t < T; ++t) rq_lds[(((wave - 1) * R + r) * T + t) * RQ_WAVE + lane] = sum[r][t];
        __syncthreads();
        if (wave == 0)
            for (int v = 0; v < W - 1; ++v)
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int t = 0; t < T; ++t) sum[r][t] += rq_lds[((v * R + r) * T + t) * RQ_WAVE + lane];
    }
    if (wave != 0) return;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            int32_t* out = &acc[(size_t)t * N + (g0 + r) * RQ_WAVE + lane];
            if (gridDim.y == 1) *out = sum[r][t];
            else atomicAdd(out, sum[r][t]);
        }
}

// Spec epilogue (mi50emu spec_np.py): corr = acc + (8 - z)*colsum; y = f16_rn((f32(corr)*s_a)*s_w)
#if defined(__HIPCC__)
typedef _Float16 rq_half;
__device__ __forceinline__ rq_half rq_to_half(float y) {
    __asm__ volatile("" : "+v"(y)); // stops LLVM folding fmul+fptrunc into v_fma_mixlo_f16
    return (rq_half)y;
}
#else
#include <cuda_fp16.h>
typedef __half rq_half;
__device__ __forceinline__ rq_half rq_to_half(float y) { return __float2half_rn(y); }
#endif

__global__ void __launch_bounds__(256)
w4a4_epilogue(const int32_t* __restrict__ acc, const float* __restrict__ s_a,
              const int32_t* __restrict__ z, const float* __restrict__ s_w,
              const int32_t* __restrict__ colsum, rq_half* __restrict__ Y, int T, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (n >= N || t >= T) return;
    const int corr = acc[(size_t)t * N + n] + (8 - z[t]) * colsum[n];
    const float y = ((float)corr * s_a[t]) * s_w[n];
    Y[(size_t)t * N + n] = rq_to_half(y);
}
