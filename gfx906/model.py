#!/usr/bin/env python3
"""MI50 verify-step time vs verify width T, from gfx906 asm (gfx906/build.sh) plus a roofline.

Two paths for the linear layers of a LLaMA-3-8B-sized model (32 layers):
  today  q4_0 weights x q8_1 activations, llama.cpp's MMVQ (T <= 8) / MMQ (T > 8), dot4
  w4a4   per-channel int4 x per-token int4, w4a4_skinny.h, dot8

Per kernel, the steady-state loop's instruction mix (counted from the asm) gives CU-cycles per
multiply-add on each pipe: VALU (1 wave-instruction/clk/CU), SALU (1/clk/CU), LDS (128 B/clk),
vector L1 (64 B/clk), scalar cache. Compute time is the busiest pipe; memory time is HBM bytes
at an assumed achievable bandwidth; a kernel takes the larger of the two.

For w4a4 the model also picks, per matmul, how to split K: W waves per workgroup reducing in
LDS, times G workgroups adding atomically in L2. Splitting buys memory parallelism (Little's
law: bandwidth <= bytes in flight / latency, one prefetched 16 B/lane step per wave in flight,
residency limited by VGPRs and LDS) and costs reduction traffic. Today's kernels get no
latency penalty: ggml is assumed to saturate memory on its own.

What it leaves out, so read the absolute numbers as a floor and the shape as the result:
bank conflicts, per-wave prologue/epilogue instructions, launch overhead, and attention /
norms / RoPE / lm_head / KV cache. Those are the same for both paths or favour today's path.

    python3 gfx906/model.py [--clock 1.725] [--bw 1024] [--bw-eff 0.8]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANES = 64

# LLaMA-3-8B linear layers: (name, K in, N out); q/k/v and gate/up share an input
LAYER = [("q", 4096, 4096), ("k", 4096, 1024), ("v", 4096, 1024), ("o", 4096, 4096),
         ("gate", 4096, 14336), ("up", 4096, 14336), ("down", 14336, 4096)]
QUANT_INPUTS = [4096, 4096, 4096, 14336]  # qkv, o, gate/up, down: one activation quant each
LAYERS = 32
WIDTHS = [1, 2, 4, 8, 16, 24, 32, 48, 64]

# today's MMQ on GCN for q4_0 (mmq-config-gcn.cuh): 256 threads, 64 x J tile, 256 K per iteration
MMQ_THREADS, MMQ_Y, MMQ_ITER_K = 256, 64, 256
MMQ_K01_TRIPS = 32 // (2 * 4)  # MMQ_TILE_NE_K / (QR4_0 * VDR_Q4_0_Q8_1_MMQ), loop left rolled

# online Hadamard before down_proj, 14336 = 28 * 512: 9 radix-2 stages + a 28-point dense
# Hadamard, fp32 (fixed order, for parity), per element per token
HADAMARD_FLOPS_PER_ELEM = 9 + 27


def isa(path, pattern):
    out = subprocess.run([sys.executable, str(HERE / "isa.py"), str(path), "--kernel", pattern, "--json"],
                         capture_output=True, text=True, check=True).stdout
    res = json.loads(out)
    if len(res) != 1:
        sys.exit(f"{path}: {len(res)} kernels match {pattern}")
    return res[0]


def add(a, b, k=1):
    out = dict(a)
    for key, v in b.items():
        out[key] = out.get(key, 0) + k * v
    return out


def loop_mix(lp):
    c = dict(lp["counts"])
    c["lds_B"] = lp["lds_bytes_per_wave"]
    c["l1_B"] = lp["vmem_bytes_per_wave"]
    c["smem_B"] = lp["smem_bytes_per_wave"]
    return c


def dots(c):
    return c.get("dot8", 0) + c.get("dot4", 0)


def lane_macs(c):
    return 8 * c.get("dot8", 0) + 4 * c.get("dot4", 0)


def single_loop(k):
    """Kernels whose K loop is one loop with the dots fully unrolled inside (MMVQ, skinny)."""
    dot_loops = [lp for lp in k["loops"] if dots(lp["counts"])]
    inner = min(dot_loops, key=lambda lp: lp["range"][1] - lp["range"][0])
    if dots(inner["counts"]) != max(dots(lp["counts"]) for lp in dot_loops):
        sys.exit(f"{k['kernel']}: dots are not all in one innermost loop")
    return loop_mix(inner)


def mmq_loop(k, J):
    """MMQ: the K loop holds two rolled k01 loops (one per vec_dot call) that run 4 times each."""
    def contains(a, b):
        return a["range"][0] <= b["range"][0] and b["range"][1] <= a["range"][1] and a is not b
    dot_loops = [lp for lp in k["loops"] if dots(lp["counts"])]
    leaves = [lp for lp in dot_loops if not any(contains(lp, o) for o in dot_loops)]
    outer = min((lp for lp in dot_loops if sum(contains(lp, l) for l in leaves) >= 2),
                key=lambda lp: lp["range"][1] - lp["range"][0])
    mix = loop_mix(outer)
    for leaf in leaves:
        if contains(outer, leaf):
            mix = add(mix, loop_mix(leaf), MMQ_K01_TRIPS - 1)
    want = MMQ_Y * J * MMQ_ITER_K // (MMQ_THREADS // LANES) // LANES // 4
    if mix.get("dot4", 0) != want:
        sys.exit(f"MMQ J={J}: {mix.get('dot4')} dot4 per wave-iteration, the tile needs {want}")
    return mix


class GPU:
    def __init__(self, a):
        self.cus, self.simds, self.hz = 60, 240, a.clock * 1e9
        self.bw = a.bw * 1e9 * a.bw_eff
        self.latency = a.latency * 1e-9
        self.lds, self.l1 = 128.0, 64.0
        self.lds_per_cu = 64 * 1024
        self.smem = a.smem  # scalar cache B/clk per CU (shared cache): unverified assumption
        self.l2 = 16 * 64 * self.hz  # 16 channels x 64 B/clk, for split-K partials

    def cycles_per_mac(self, c):
        """CU-cycles per multiply-add on each pipe, from one loop iteration's mix."""
        macs = LANES * lane_macs(c)
        return {
            "valu": (c.get("valu", 0) + dots(c)) / macs,
            "salu": c.get("salu", 0) / macs,
            "lds": c["lds_B"] / self.lds / macs,
            "l1": c["l1_B"] / self.l1 / macs,
            "smem": c["smem_B"] / self.smem / macs,
        }

    def compute_s(self, macs, c):
        pipes = self.cycles_per_mac(c)
        busiest = max(pipes, key=pipes.get)
        return macs * pipes[busiest] / (self.cus * self.hz), busiest


def kernels(build):
    today, w4a4 = {}, {}
    for T in WIDTHS:
        if T <= 8:
            today[T] = single_loop(isa(build / "mmvq.s", rf"mul_mat_vec_qIL9ggml_type2ELi{T}ELb0ELb0ELb0E"))
        else:
            today[T] = mmq_loop(isa(build / "mmq_q4_0.s", rf"mul_mat_qIL9ggml_type2ELi{T}ELb0E"), T)
        k = isa(build / "w4a4_skinny.s", rf"w4a4_skinnyILi{T}ELi")
        r = int(re.search(rf"skinnyILi{T}ELi(\d+)E", k["kernel"]).group(1))
        w4a4[T] = (single_loop(k), r, k["meta"])
    return today, w4a4


def w4a4_matmul(gpu, T, K, N, mix, R, occupancy):
    """Best (seconds, W, G, limit) over the split-K choices for one skinny matmul."""
    macs = T * N * K
    hbm_bytes = N * K / 2 + N * 8 + T * K / 2 + T * 8 + T * N * 2  # weights+scale/colsum, acts, f16 out
    comp, pipe = gpu.compute_s(macs, mix)
    groups, steps = N // (LANES * R), K // 8 // 4
    best = None
    for W in (1, 2, 4, 8):
        lds = (W - 1) * R * T * LANES * 4
        if lds > gpu.lds_per_cu or steps < W:
            continue
        wg_per_cu = gpu.lds_per_cu // lds if lds else 99
        per_simd = min(occupancy, wg_per_cu * W / 4)
        G = 1
        while True:
            waves = groups * W * G
            resident = min(waves, gpu.simds * per_simd)
            bw = min(gpu.bw, resident * R * LANES * 16 / gpu.latency)
            mem = hbm_bytes / bw
            reduce_lds = 2 * (W - 1) * T * N * 4 / (gpu.lds * gpu.cus * gpu.hz)
            partials = (2 * G * T * N * 4 + T * N * 4) / gpu.l2 if G > 1 else 0.0  # atomics + memset
            t = max(mem, comp) + reduce_lds + partials
            limit = pipe if comp >= mem else ("hbm" if bw >= gpu.bw else "latency")
            if best is None or t < best[0]:
                best = (t, W, G, limit)
            if steps // (W * G * 2) < 1:
                break
            G *= 2
    return best


def step(gpu, T, today_mix, w4a4_mix, R, occupancy, detail=False):
    """One verify step (all linear layers, LAYERS layers) at width T, seconds, for both paths."""
    t_today = t_w4a4 = 0.0
    limits = {"today": set(), "w4a4": set()}
    for name, K, N in LAYER:
        macs = T * N * K
        # today: q4_0 weights (18 B / 32), q8_1 activations (36 B / 32), fp32 out
        mem = (N * K * 18 / 32 + T * K * 36 / 32 + T * N * 4) / gpu.bw
        comp, pipe = gpu.compute_s(macs, today_mix)
        t_today += max(mem, comp)
        limits["today"].add("hbm" if mem >= comp else pipe)
        t, W, G, limit = w4a4_matmul(gpu, T, K, N, w4a4_mix, R, occupancy)
        t_w4a4 += t
        limits["w4a4"].add(limit)
        if detail:
            print(f"    {name:<5} K={K:<6} N={N:<6} today {max(mem, comp)*1e6:7.1f} us   "
                  f"w4a4 {t*1e6:7.1f} us  (W={W} G={G}, {limit})")
    for K in QUANT_INPUTS:  # activation quantization: memory-bound pass over fp32 input
        t_today += (T * K * 4 + T * K * 36 / 32) / gpu.bw
        t_w4a4 += (T * K * 4 + T * K / 2) / gpu.bw
    # Hadamard on the down_proj input: plain fp32 VALU (non-packed, fixed order for parity)
    t_w4a4 += T * 14336 * HADAMARD_FLOPS_PER_ELEM / LANES / (gpu.cus * gpu.hz)
    return t_today * LAYERS, t_w4a4 * LAYERS, limits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clock", type=float, default=1.725, help="GHz (MI50 peak engine clock)")
    p.add_argument("--bw", type=float, default=1024, help="GB/s HBM2 peak")
    p.add_argument("--bw-eff", type=float, default=0.80, help="achievable fraction of --bw (measure on the card)")
    p.add_argument("--smem", type=float, default=4.0, help="scalar cache B/clk per CU (assumed 16 B/clk per 4 CUs)")
    p.add_argument("--latency", type=float, default=1000, help="ns, HBM latency under load (guess; measure)")
    p.add_argument("--detail", type=int, action="append", default=[], help="per-matmul breakdown at this T")
    a = p.parse_args()
    gpu = GPU(a)
    today, w4a4 = kernels(HERE / "build")

    print("Steady-state K loop, from gfx906 asm (CU-cycles per 512 multiply-adds = one dot8 wave-instruction)")
    print(f"{'T':>3}  {'today kernel':<14}{'VALU':>6}{'LDS':>6}{'L1':>6}  {'w4a4':<10}{'VALU':>6}{'SALU':>6}{'SMEM':>6}{'L1':>6}  vgpr occ  VALU ratio")
    for T in WIDTHS:
        ct = gpu.cycles_per_mac(today[T])
        mix, R, meta = w4a4[T]
        cw = gpu.cycles_per_mac(mix)
        name = "MMVQ" if T <= 8 else f"MMQ J={T}"
        print(f"{T:>3}  {name:<14}{ct['valu']*512:6.2f}{ct['lds']*512:6.2f}{ct['l1']*512:6.2f}  "
              f"R={R:<8}{cw['valu']*512:6.2f}{cw['salu']*512:6.2f}{cw['smem']*512:6.2f}{cw['l1']*512:6.2f}  "
              f"{meta.get('NumVgprs'):>4} {meta.get('Occupancy'):>3}  {ct['valu']/cw['valu']:6.1f}x")

    print(f"\nVerify step, linear layers of {LAYERS} LLaMA-3-8B layers, MI50 at {a.clock} GHz, "
          f"{a.bw:.0f} GB/s x {a.bw_eff}, {a.latency:.0f} ns latency")
    print(f"{'T':>3}  {'today ms':>9} {'x T=1':>6}  {'w4a4 ms':>8} {'x T=1':>6}  {'speedup':>7}  limits (today | w4a4)")
    base_t = base_w = None
    for T in WIDTHS:
        mix, R, meta = w4a4[T]
        tt, tw, lim = step(gpu, T, today[T], mix, R, meta.get("Occupancy", 1))
        base_t, base_w = base_t or tt, base_w or tw
        print(f"{T:>3}  {tt*1e3:9.2f} {tt/base_t:6.2f}  {tw*1e3:8.2f} {tw/base_w:6.2f}  {tt/tw:6.2f}x  "
              f"{','.join(sorted(lim['today'])):<10} | {','.join(sorted(lim['w4a4']))}")
    for T in a.detail:
        mix, R, meta = w4a4[T]
        print(f"\n  T={T}, one layer:")
        step(gpu, T, today[T], mix, R, meta.get("Occupancy", 1), detail=True)


if __name__ == "__main__":
    main()
