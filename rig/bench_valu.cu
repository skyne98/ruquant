// VALU issue rate of the instructions the cost model leans on, and the shader clock under load.
//
// Each lane runs ACC independent chains of one op, enough waves to fill every SIMD. Throughput
// is reported as lane-instructions per clock per CU: a full-rate 32-bit VALU op on GCN is 64
// (one wave64 instruction per 4 clocks on each of 4 SIMD16s). The clock comes from clock64()
// over a constant-rate wall tick, measured inside the kernel, so it is the clock under load.
//
// The model assumes dot8 at 64. A latency mode (1 chain, 1 wave per SIMD) gives dependent-issue
// cycles per instruction.
//   ./bench_valu [device] [iters]     -> JSON on stdout
#include "../gfx906/gpu.h"
#include "../gfx906/w4a4_skinny.h"

#include <cstdio>
#include <cstdlib>

enum Op { DOT8, DOT4, FMA32, PKFMA16 };
static const char* op_name[] = {"dot8_i32_i4", "dot4_i32_i8", "fma_f32", "pk_fma_f16"};
static const int op_macs[] = {8, 4, 1, 2};

template <int OP, int ACC>
__global__ void __launch_bounds__(256) valu(int iters, int seed, int* out, unsigned long long* clocks) {
    int a = seed ^ (int)threadIdx.x, b = seed * 7 + (int)threadIdx.x;
    int iacc[ACC];
    float facc[ACC];
    __half2 hacc[ACC];
    const float fa = 1.0f + 1e-7f * a, fb = 0.999999f;
    const __half2 ha = __floats2half2_rn(1.0f, 0.5f), hb = __floats2half2_rn(0.25f, 1.0f);
#pragma unroll
    for (int i = 0; i < ACC; ++i) {
        iacc[i] = i + (int)threadIdx.x;
        facc[i] = (float)iacc[i];
        hacc[i] = __floats2half2_rn((float)i, 1.0f);
    }
    const unsigned long long c0 = clock64(), w0 = rq_walltick();
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            rq_opaque(a); // a chain of identical integer ops would otherwise fold into one
            rq_opaque(b);
#pragma unroll
            for (int i = 0; i < ACC; ++i) {
                if (OP == DOT8) iacc[i] = rq_dot8(a, b, iacc[i]);
                if (OP == DOT4) iacc[i] = rq_dot4(a, b, iacc[i]);
                if (OP == FMA32) facc[i] = fmaf(fa, facc[i], fb);
                if (OP == PKFMA16) hacc[i] = __hfma2(ha, hacc[i], hb);
            }
        }
    }
    const unsigned long long c1 = clock64(), w1 = rq_walltick();
    int s = 0;
#pragma unroll
    for (int i = 0; i < ACC; ++i) s += iacc[i] + (int)facc[i] + (int)__low2float(hacc[i]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        clocks[0] = c1 - c0;
        clocks[1] = w1 - w0;
    }
}

__global__ void empty_kernel() {}

// Per-kernel launch cost, back to back in one stream (no graph): the floor every small
// matmul, quantize and epilogue kernel of a decode step pays.
static double launch_us() {
    constexpr int n = 1000;
    const double ms = rq_time_ms([] {
        for (int i = 0; i < n; ++i) empty_kernel<<<1, RQ_WAVE>>>();
    }, 5, 1);
    return ms * 1e3 / n;
}

struct Result {
    double ms, clock_mhz, lane_instr_per_clk_cu, tmacs, cycles_per_instr;
};

template <int OP, int ACC>
static Result run(const cudaDeviceProp& prop, int dev, int iters, int blocks, int threads) {
    int* out;
    unsigned long long* clocks;
    CHECK(cudaMalloc(&out, (size_t)blocks * threads * sizeof(int)));
    CHECK(cudaMalloc(&clocks, 2 * sizeof(unsigned long long)));
    const double ms = rq_time_ms([&] { valu<OP, ACC><<<blocks, threads>>>(iters, 12345, out, clocks); }, 5, 1);
    unsigned long long h[2];
    CHECK(cudaMemcpy(h, clocks, sizeof(h), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(out));
    CHECK(cudaFree(clocks));
    const double secs_inside = h[1] / rq_walltick_hz(dev);
    const double clock_hz = h[0] / secs_inside;
    const double instr = (double)blocks * threads * iters * 8 * ACC;
    Result r;
    r.ms = ms;
    r.clock_mhz = clock_hz / 1e6;
    r.lane_instr_per_clk_cu = instr / (ms * 1e-3) / clock_hz / prop.multiProcessorCount;
    r.tmacs = instr * op_macs[OP] / (ms * 1e-3) / 1e12;
    r.cycles_per_instr = (double)h[0] / ((double)iters * 8 * ACC);
    return r;
}

template <int OP> static void op(const cudaDeviceProp& prop, int dev, int iters, bool last) {
    const int cus = prop.multiProcessorCount;
    // throughput: 8 independent chains per lane, 8 blocks of 256 lanes per CU
    const Result t = run<OP, 8>(prop, dev, iters, cus * 8, 256);
    // latency: one chain, one wave per SIMD (4 waves per CU on GCN)
    const Result l = run<OP, 1>(prop, dev, iters, cus, 4 * RQ_WAVE);
    std::printf("    {\"op\": \"%s\", \"macs_per_lane_instr\": %d, \"ms\": %.3f, \"clock_mhz\": %.1f, "
                "\"lane_instr_per_clk_per_cu\": %.2f, \"full_rate_fraction\": %.3f, \"tmacs\": %.2f, "
                "\"dependent_cycles_per_instr\": %.2f}%s\n",
                op_name[OP], op_macs[OP], t.ms, t.clock_mhz, t.lane_instr_per_clk_cu,
                t.lane_instr_per_clk_cu / 64.0, t.tmacs, l.cycles_per_instr, last ? "" : ",");
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 1 << 15;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("{\n  \"bench\": \"valu\", \"toolchain\": \"%s\", \"device\": %d, \"name\": \"%s\", \"cus\": %d, "
                "\"max_clock_mhz\": %.1f, \"iters\": %d,\n  \"note\": \"full rate = 64 lane-instructions/clk/CU "
                "(GCN wave64)\",\n  \"ops\": [\n",
                rq_toolchain(), dev, prop.name, prop.multiProcessorCount, rq_max_clock_mhz(dev), iters);
    op<DOT8>(prop, dev, iters, false);
    op<DOT4>(prop, dev, iters, false);
    op<FMA32>(prop, dev, iters, false);
    op<PKFMA16>(prop, dev, iters, true);
    std::printf("  ],\n  \"launch_us\": %.2f\n}\n", launch_us());
    return 0;
}
