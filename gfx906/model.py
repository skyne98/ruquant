#!/usr/bin/env python3
"""MI50 speculative-decoding step time, from gfx906 asm (gfx906/build.sh) plus a roofline.

Targets and drafters, shapes from their Hugging Face config.json:
  Qwen/Qwen3.8-27B          64 layers: 48 Gated DeltaNet + 16 gated full attention, dense MLP
  z-lab/Qwen3.8-27B-DFlash2 5-layer block-diffusion drafter, block 8
  Qwen/Qwen3.8-Flash-Next   48 layers: 36 Gated DeltaNet + 12 Qwen Sparse Attention,
                            512-expert MoE (top-10 + shared expert)
  incoai/Qwen3.8-Flash-Next-DFlash2   5-layer drafter, block 4 on the model card

DFlash2 drafts a whole block in one pass and the target verifies one path through it, so a
step is: drafter forward over the block, lm_head over the block (drafter logits), target
forward over the block, lm_head again (target logits).

Two paths for every quantized linear layer:
  today  q4_0 weights x q8_1 activations, llama.cpp's MMVQ (T <= 8) / MMQ (T > 8), dot4
  w4a4   per-channel int4 x per-token int4, w4a4_skinny.h, dot8. Pays an online Hadamard on
         every input whose rotation can't be folded offline: down_proj, and on Qwen3.8 also
         o_proj and the DeltaNet out_proj, because the output gates sit in front of them.

Per kernel, the steady-state loop's instruction mix (counted from the asm) gives CU-cycles per
multiply-add on each pipe: VALU (1 wave-instruction/clk/CU), SALU (1/clk/CU), LDS (128 B/clk),
vector L1 (64 B/clk), scalar cache. Compute time is the busiest pipe; memory time is HBM bytes
at an achievable bandwidth; a kernel takes the larger of the two. For w4a4 the model picks a
split-K per matmul against Little's law (bytes in flight / latency) and reduction traffic.

Shared by both paths, so they dilute the w4a4 gain: lm_head (q6_K, ggml kernel), attention and
the KV cache (fp16, packed-fp16 VALU at --attn-eff), DeltaNet recurrent state and its update.

Left out, so read absolute numbers as a floor and the comparison as the result: bank conflicts,
launch overhead, norms / RoPE / convs / router softmax / hyper-connections / n-gram embedding
lookups, the drafter's selector, expert-routing overlap between tokens (experts are assumed
independent, which is the bandwidth worst case), and linear-attention rollback beyond one
state read + write per step. Multi-GPU is ideal tensor parallelism plus a fixed all-reduce cost.

    python3 gfx906/model.py [--ctx 8192] [--detail 27b:8]
"""

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
LANES = 64
WIDTHS = [1, 2, 4, 8, 16, 24, 32, 48, 64]

# today's MMQ on GCN for q4_0 (mmq-config-gcn.cuh): 256 threads, 64 x J tile, 256 K per iteration
MMQ_THREADS, MMQ_Y, MMQ_ITER_K = 256, 64, 256
MMQ_K01_TRIPS = 32 // (2 * 4)  # MMQ_TILE_NE_K / (QR4_0 * VDR_Q4_0_Q8_1_MMQ), loop left rolled

Q4_0, Q8_1 = 18 / 32, 36 / 32  # bytes per element


# --------------------------------------------------------------------------------------------
# architectures


@dataclass
class Attn:
    layers: int
    q_heads: int
    head_dim: int
    kv_heads: int
    window: Optional[int] = None  # tokens a query attends to: sliding window or sparse budget
    index: Optional[tuple] = None  # (heads, head_dim, compress): indexer scan of the whole context


@dataclass
class Net:
    layers: int
    dense: list  # (name, K, N, count): quantized linears, count = instances per forward
    inputs: list  # (K, count): activation vectors quantized per token (one per shared input)
    hadamard: list  # (K, count, adds per element): online Hadamard in front of a w4a4 linear
    attn: list = field(default_factory=list)
    deltanet: Optional[tuple] = None  # (layers, v_heads, k_dim, v_dim): fp32 recurrent state
    moe: Optional[tuple] = None  # (layers, experts, top_k, d_model, d_ff)


@dataclass
class Arch:
    name: str
    target: Net
    drafter: Net
    lm_head: tuple  # (K, N)
    block: int  # DFlash2 block on the model card = tokens the target verifies per step
    acceptance: tuple  # (low, high) accepted tokens per step at `block`, model card
    gpus: int


# Hadamard of size a*b as H_a (dense, a adds per element) x H_b (FWHT, log2 b adds per element)
def had(a, b):
    return a + int(math.log2(b))


QWEN38_27B = Arch(
    name="Qwen3.8-27B + DFlash2",
    target=Net(
        layers=64,
        dense=[("attn.q+gate", 5120, 12288, 16), ("attn.k", 5120, 1024, 16), ("attn.v", 5120, 1024, 16),
               ("attn.o", 6144, 5120, 16),
               ("dn.qkvz", 5120, 16384, 48), ("dn.ba", 5120, 96, 48), ("dn.out", 6144, 5120, 48),
               ("mlp.gate", 5120, 17408, 64), ("mlp.up", 5120, 17408, 64), ("mlp.down", 17408, 5120, 64)],
        inputs=[(5120, 16 + 48 + 64), (6144, 16 + 48), (17408, 64)],
        hadamard=[(17408, 64, had(68, 256)), (6144, 16 + 48, had(12, 512))],
        attn=[Attn(16, 24, 256, 4)],
        deltanet=(48, 48, 128, 128),
    ),
    drafter=Net(
        layers=5,
        dense=[("q", 5120, 4096, 5), ("k", 5120, 1024, 5), ("v", 5120, 1024, 5), ("o", 4096, 5120, 5),
               ("gate", 5120, 17408, 5), ("up", 5120, 17408, 5), ("down", 17408, 5120, 5),
               ("fc", 5 * 5120, 5120, 1)],
        inputs=[(5120, 10), (4096, 5), (17408, 5), (5 * 5120, 1)],
        hadamard=[(17408, 5, had(68, 256))],
        attn=[Attn(5, 32, 128, 8, window=2048)],
    ),
    lm_head=(5120, 248320),
    block=8,
    acceptance=(4.10, 5.46),  # MT-Bench .. GSM8K, H200, temperature 1.0
    gpus=1,
)

QWEN38_FLASH_NEXT = Arch(
    name="Qwen3.8-Flash-Next + DFlash2",
    target=Net(
        layers=48,
        dense=[("qsa.q+gate", 2560, 12288, 12), ("qsa.k", 2560, 512, 12), ("qsa.v", 2560, 512, 12),
               ("qsa.o", 6144, 2560, 12), ("qsa.idx_q", 2560, 512, 12), ("qsa.idx_k", 2560, 128, 12),
               ("dn.qkvz", 2560, 16384, 36), ("dn.ba", 2560, 96, 36), ("dn.out", 6144, 2560, 36),
               ("moe.router", 2560, 512, 48), ("shared.gate", 2560, 640, 48), ("shared.up", 2560, 640, 48),
               ("shared.down", 640, 2560, 48)],
        inputs=[(2560, 12 + 36 + 48), (6144, 12 + 36), (640, 48 + 48 * 10)],
        hadamard=[(6144, 12 + 36, had(12, 512)), (640, 48 + 48 * 10, had(20, 32))],
        attn=[Attn(12, 24, 256, 2, window=2048, index=(4, 128, 4))],
        deltanet=(36, 48, 128, 128),
        moe=(48, 512, 10, 2560, 640),
    ),
    drafter=Net(
        layers=5,
        dense=[("q", 2560, 4096, 5), ("k", 2560, 1024, 5), ("v", 2560, 1024, 5), ("o", 4096, 2560, 5),
               ("gate", 2560, 7680, 5), ("up", 2560, 7680, 5), ("down", 7680, 2560, 5),
               ("fc", 5 * 2560, 2560, 1)],
        inputs=[(2560, 10), (4096, 5), (7680, 5), (5 * 2560, 1)],
        hadamard=[(7680, 5, had(60, 128))],
        attn=[Attn(5, 32, 128, 8, window=2048)],
    ),
    lm_head=(2560, 248320),
    block=4,  # limited by today's QSA implementation, per the model card
    acceptance=(2.83, 3.26),
    gpus=4,  # ~62-70 GB of 4-bit weights; n-gram embedding table assumed in host memory
)

ARCHS = {"27b": QWEN38_27B, "flash-next": QWEN38_FLASH_NEXT}
STEP_WIDTHS = [1, 4, 8, 16, 32]


# --------------------------------------------------------------------------------------------
# instruction mixes from the asm


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


# --------------------------------------------------------------------------------------------
# the machine


class GPU:
    def __init__(self, a):
        self.cus, self.simds, self.hz = 60, 240, a.clock * 1e9
        self.bw = a.bw * 1e9 * a.bw_eff
        self.latency = a.latency * 1e-9
        self.lds, self.l1 = 128.0, 64.0
        self.lds_per_cu = 64 * 1024
        self.smem = a.smem  # scalar cache B/clk per CU (shared cache): unverified assumption
        # split-K partials: assumed at L2 bandwidth (16 channels x 64 B/clk) unless measured
        self.l2 = a.atomic_gbs * 1e9 if a.atomic_gbs else 16 * 64 * self.hz
        self.dot8_rate = a.dot8_rate  # fraction of a full-rate VALU op (1.0 assumed)
        self.attn_eff = a.attn_eff
        self.wave_rate = self.cus * self.hz  # VALU wave-instructions per second, whole GPU

    def cycles_per_mac(self, c):
        """CU-cycles per multiply-add on each pipe, from one loop iteration's mix."""
        macs = LANES * lane_macs(c)
        return {
            "valu": (c.get("valu", 0) + c.get("dot4", 0) + c.get("dot8", 0) / self.dot8_rate) / macs,
            "salu": c.get("salu", 0) / macs,
            "lds": c["lds_B"] / self.lds / macs,
            "l1": c["l1_B"] / self.l1 / macs,
            "smem": c["smem_B"] / self.smem / macs,
        }

    def compute_s(self, macs, c):
        pipes = self.cycles_per_mac(c)
        busiest = max(pipes, key=pipes.get)
        return macs * pipes[busiest] / self.wave_rate, busiest


def mix_for(T):
    """Nearest compiled width at or above T (kernels exist for WIDTHS only)."""
    return next(w for w in WIDTHS if w >= T)


def w4a4_matmul(gpu, T, K, N, mix, R, occupancy):
    """Best (seconds, W, G, limit) over the split-K choices for one skinny matmul."""
    macs = T * N * K
    hbm_bytes = N * K / 2 + N * 8 + T * K / 2 + T * 8 + T * N * 2  # weights+scale/colsum, acts, f16 out
    comp, pipe = gpu.compute_s(macs, mix)
    groups, steps = max(1, math.ceil(N / (LANES * R))), max(1, K // 8 // 4)
    best = None
    for W in (1, 2, 4, 8):
        lds = R * T * LANES * 4 if W > 1 else 0  # one buffer, waves 1..W-1 add into it
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
            reduce_lds = 2 * W * T * N * 4 / (gpu.lds * gpu.wave_rate) if W > 1 else 0.0  # store, adds, load
            partials = (2 * G * T * N * 4 + T * N * 4) / gpu.l2 if G > 1 else 0.0  # atomics + memset
            t = max(mem, comp) + reduce_lds + partials
            limit = pipe if comp >= mem else ("hbm" if bw >= gpu.bw else "latency")
            if best is None or t < best[0]:
                best = (t, W, G, limit)
            if steps // (W * G * 2) < 1:
                break
            G *= 2
    return best


class Measured:
    """Matmul times the rig measured on the card (rig/results/<dir>), used instead of the roofline
    for every shape it has: skinny-gpu0.json for w4a4 (best split and R per point), ggml-gpu0.json
    for today (q4_0 linears, q6_K lm_head). T between measured widths is interpolated linearly,
    past the widest extrapolated from the last two. A tensor split still takes 1/gpus of the
    full-size time, which flatters it: the real per-GPU slices are smaller and less efficient."""

    def __init__(self, d):
        self.d, self.pts = Path(d), {}  # (kind, K, N) -> {T: seconds}
        for name in ("skinny-gpu0.json", "ggml-gpu0.json"):
            f = self.d / name
            if not f.exists():
                continue
            for p in json.loads(f.read_text())["points"]:
                kind = "w4a4" if name.startswith("skinny") else ("lm" if p["type"] == "q6_K" else "today")
                ts = self.pts.setdefault((kind, p["K"], p["N"]), {})
                ts[p["T"]] = min(ts.get(p["T"], math.inf), p["us"] * 1e-6)

    def kinds(self):
        return sorted({k[0] for k in self.pts})

    def get(self, kind, T, K, N):
        ts = self.pts.get((kind, K, N))
        if not ts:
            return None
        if T in ts:
            return ts[T]
        xs = sorted(ts)
        if len(xs) == 1:
            return ts[xs[0]]
        lo, hi = [x for x in xs if x < T], [x for x in xs if x > T]
        a, b = (lo[-1], hi[0]) if lo and hi else ((xs[-2], xs[-1]) if lo else (xs[0], xs[1]))
        return ts[a] + (ts[b] - ts[a]) * (T - a) / (b - a)


class Model:
    def __init__(self, gpu, today, w4a4, a, measured=None):
        self.gpu, self.today, self.w4a4, self.a, self.measured = gpu, today, w4a4, a, measured

    def linear(self, path, T, K, N):
        if self.measured and (t := self.measured.get(path, T, K, N)) is not None:
            return t
        g = self.gpu
        if path == "today":
            mem = (N * K * Q4_0 + T * K * Q8_1 + T * N * 4) / g.bw
            comp, _ = g.compute_s(T * N * K, self.today[mix_for(T)])
            return max(mem, comp)
        mix, R, meta = self.w4a4[mix_for(T)]
        return w4a4_matmul(g, T, K, N, mix, R, meta.get("Occupancy", 1))[0]

    def attention(self, attn, T, ctx):
        """fp16 KV cache read once per step, QK^T and PV on packed-fp16 VALU; same for both paths."""
        g = self.gpu
        t = 0.0
        for at in attn:
            seen = min(ctx, at.window) if at.window else ctx
            kv_bytes = at.layers * seen * at.kv_heads * at.head_dim * 2 * 2
            macs = at.layers * T * seen * at.q_heads * at.head_dim * 2
            if at.index:
                heads, dim, compress = at.index
                kv_bytes += at.layers * (ctx // compress) * dim * 2
                macs += at.layers * T * (ctx // compress) * heads * dim
            comp = macs / (2 * LANES) / g.wave_rate / g.attn_eff
            t += max(kv_bytes / g.bw, comp)
        return t

    def forward(self, net, path, T, ctx):
        """Seconds per component for one forward pass of `net` over T tokens on one GPU."""
        g = self.gpu
        parts = {"linear": 0.0, "moe": 0.0, "attn": 0.0, "deltanet": 0.0, "quant": 0.0, "hadamard": 0.0}
        for _, K, N, count in net.dense:
            parts["linear"] += count * self.linear(path, T, K, N)
        if net.moe:
            layers, experts, k, d, f = net.moe
            # experts touched by T tokens; reuse 0 = independent routing, the bandwidth worst case
            t_eff = 1 + (T - 1) * (1 - self.a.expert_reuse)
            unique = experts * (1 - (1 - k / experts) ** t_eff)
            bpw = Q4_0 if path == "today" else 0.5
            mem = layers * unique * 3 * d * f * bpw / g.bw
            mix = self.today[1] if path == "today" else self.w4a4[1][0]  # ~1 token per expert
            comp, _ = g.compute_s(layers * T * k * 3 * d * f, mix)
            parts["moe"] = max(mem, comp)
        parts["attn"] = self.attention(net.attn, T, ctx)
        if net.deltanet:
            layers, heads, dk, dv = net.deltanet
            state = layers * heads * dk * dv * 4
            update = T * layers * heads * dk * dv * 2 / LANES / g.wave_rate  # fp32 FMAs
            parts["deltanet"] = max(2 * state / g.bw, update)
        act = Q8_1 if path == "today" else 0.5
        parts["quant"] = sum(T * K * c * (4 + act) for K, c in net.inputs) / g.bw
        if path == "w4a4":
            parts["hadamard"] = sum(T * K * c * adds for K, c, adds in net.hadamard) / LANES / g.wave_rate
        return parts

    def lm_head(self, arch, T):
        """q6_K output projection with ggml's kernel in both paths (A4 on lm_head costs quality)."""
        K, N = arch.lm_head
        if self.measured and (t := self.measured.get("lm", T, K, N)) is not None:
            return t
        mem = (N * K * self.a.lm_bpw / 8) / self.gpu.bw
        comp, _ = self.gpu.compute_s(T * N * K, self.today[mix_for(T)])
        return max(mem, comp)

    def allreduce_s(self, nbytes):
        """One all-reduce across the GPUs of a tensor split: a fixed cost plus bytes at the
        all-reduce's effective rate (both fitted by rig/calibrate.py from bench_tp)."""
        rate = self.a.allreduce_gbs * 1e9 if self.a.allreduce_gbs > 0 else float("inf")
        return self.a.allreduce_us * 1e-6 + nbytes / rate

    def step(self, arch, path, T, ctx, split="layer"):
        """One decoding step. T=1 is plain autoregressive decoding; T>1 is DFlash2 with block T.

        split "layer": each GPU holds a run of layers and they take turns (llama.cpp's default):
        capacity scales, bandwidth doesn't; one activation hand-off per GPU boundary.
        split "tensor": every GPU holds a slice of every layer: ideal 1/gpus of the work plus
        two all-reduces per layer of T x hidden f16 activations."""
        gpus = arch.gpus
        target = self.forward(arch.target, path, T, ctx)
        share = 1 / gpus if split == "tensor" else 1.0
        out = {"target": sum(target.values()) * share, "lm_head": self.lm_head(arch, T) * share}
        if gpus == 1:
            out["comm"] = 0.0
        elif split == "tensor":
            out["comm"] = 2 * arch.target.layers * self.allreduce_s(T * arch.lm_head[0] * 2)
        else:
            out["comm"] = (gpus - 1) * self.a.handoff_us * 1e-6
        if T > 1:
            out["drafter"] = sum(self.forward(arch.drafter, path, T, ctx).values())  # on one GPU
            out["lm_head"] *= 2  # drafter logits for the block, then target logits
        else:
            out["drafter"] = 0.0
        out["total"] = sum(out.values())
        out["parts"] = target
        return out


def weights_gb(arch, path):
    bpw = Q4_0 if path == "today" else 0.5
    dense = sum(K * N * c for _, K, N, c in arch.target.dense)
    if arch.target.moe:
        layers, experts, _, d, f = arch.target.moe
        dense += layers * experts * 3 * d * f
    drafter = sum(K * N * c for _, K, N, c in arch.drafter.dense)
    K, N = arch.lm_head
    return (dense + drafter) * bpw / 1e9, K * N * 6.5625 / 8 / 1e9


# --------------------------------------------------------------------------------------------


def print_kernels(gpu, today, w4a4):
    print("Steady-state K loop, from gfx906 asm (CU-cycles per 512 multiply-adds = one dot8 wave-instruction)")
    print(f"{'T':>3}  {'today kernel':<14}{'VALU':>6}{'LDS':>6}{'L1':>6}  {'w4a4':<10}{'VALU':>6}{'SALU':>6}"
          f"{'SMEM':>6}{'L1':>6}  vgpr occ  VALU ratio")
    for T in WIDTHS:
        ct = gpu.cycles_per_mac(today[T])
        mix, R, meta = w4a4[T]
        cw = gpu.cycles_per_mac(mix)
        name = "MMVQ" if T <= 8 else f"MMQ J={T}"
        print(f"{T:>3}  {name:<14}{ct['valu']*512:6.2f}{ct['lds']*512:6.2f}{ct['l1']*512:6.2f}  "
              f"R={R:<8}{cw['valu']*512:6.2f}{cw['salu']*512:6.2f}{cw['smem']*512:6.2f}{cw['l1']*512:6.2f}  "
              f"{meta.get('NumVgprs'):>4} {meta.get('Occupancy'):>3}  {ct['valu']/cw['valu']:6.1f}x")


def print_arch(model, arch, ctx):
    wt, lm = weights_gb(arch, "today")
    ww, _ = weights_gb(arch, "w4a4")
    print(f"\n{arch.name}: {arch.gpus} x MI50, ctx {ctx}, weights incl. drafter {wt:.1f} GB (q4_0) / "
          f"{ww:.1f} GB (w4a4) + lm_head {lm:.2f} GB (q6_K), per GPU {(wt + lm) / arch.gpus:.1f} / "
          f"{(ww + lm) / arch.gpus:.1f} GB")
    for split in (("layer", "tensor") if arch.gpus > 1 else ("layer",)):
        print_steps(model, arch, ctx, split)


def print_steps(model, arch, ctx, split):
    if arch.gpus > 1:
        how = ("GPUs take turns over layer runs" if split == "layer"
               else f"tensor parallel, all-reduce {model.a.allreduce_us:.0f} us"
               + (f" + bytes at {model.a.allreduce_gbs:.1f} GB/s" if model.a.allreduce_gbs > 0 else ""))
        print(f"  -- {split} split: {how}")
    print(f"  step ms       {'today':>34}   {'w4a4':>34}")
    print(f"  {'T':>4}  {'target':>8}{'drafter':>8}{'lm_head':>8}{'step':>9}    "
          f"{'target':>8}{'drafter':>8}{'lm_head':>8}{'step':>9}   speedup")
    steps = {}
    for T in STEP_WIDTHS:
        st, sw = model.step(arch, "today", T, ctx, split), model.step(arch, "w4a4", T, ctx, split)
        steps[T] = (st["total"], sw["total"])
        mark = " AR" if T == 1 else (" <- model card" if T == arch.block else "")
        print(f"  {T:>4}  {st['target']*1e3:8.2f}{st['drafter']*1e3:8.2f}{st['lm_head']*1e3:8.2f}"
              f"{st['total']*1e3:9.2f}    {sw['target']*1e3:8.2f}{sw['drafter']*1e3:8.2f}"
              f"{sw['lm_head']*1e3:8.2f}{sw['total']*1e3:9.2f}   {st['total']/sw['total']:5.2f}x{mark}")

    lo, hi = arch.acceptance
    B = arch.block
    print(f"  tokens/s, block {B} with the model card's acceptance {lo}-{hi} per step:")
    for i, path in enumerate(("today", "w4a4")):
        ar, sp = steps[1][i], steps[B][i]
        print(f"    {path:<6} AR {1/ar:6.1f}   DFlash2 {lo/sp:6.1f} - {hi/sp:6.1f}   ({lo*ar/sp:.2f}x - {hi*ar/sp:.2f}x AR)")
    print(f"  wider blocks: accepted tokens per step must grow by this factor over block {B} to pay off")
    for T in (t for t in STEP_WIDTHS if t > B):
        print(f"    block {T:>2}: today x{steps[T][0]/steps[B][0]:.2f}   w4a4 x{steps[T][1]/steps[B][1]:.2f}")


def print_detail(model, arch, T, ctx):
    print(f"\n  {arch.name}, T={T}, one GPU's share of the target forward (ms):")
    for path in ("today", "w4a4"):
        p = model.forward(arch.target, path, T, ctx)
        line = "  ".join(f"{k} {v*1e3/arch.gpus:.2f}" for k, v in p.items())
        print(f"    {path:<6} {line}")


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clock", type=float, default=1.725, help="GHz (MI50 peak engine clock)")
    p.add_argument("--bw", type=float, default=1024, help="GB/s HBM2 peak")
    p.add_argument("--bw-eff", type=float, default=0.80, help="achievable fraction of --bw (measure on the card)")
    p.add_argument("--smem", type=float, default=4.0, help="scalar cache B/clk per CU (assumed 16 B/clk per 4 CUs)")
    p.add_argument("--dot8-rate", type=float, default=1.0, help="v_dot8 issue rate as a fraction of full VALU rate")
    p.add_argument("--atomic-gbs", type=float, default=0.0, help="split-K partials rate, GB/s (default: L2 bandwidth)")
    p.add_argument("--latency", type=float, default=1000, help="ns, HBM latency under load (guess; measure)")
    p.add_argument("--ctx", type=int, default=8192, help="tokens in the KV cache")
    p.add_argument("--attn-eff", type=float, default=0.5, help="fraction of packed-fp16 peak attention reaches")
    p.add_argument("--lm-bpw", type=float, default=6.5625, help="lm_head bits per weight (q6_K)")
    p.add_argument("--allreduce-us", type=float, default=15, help="fixed cost per all-reduce, tensor split (guess)")
    p.add_argument("--allreduce-gbs", type=float, default=0.0,
                   help="all-reduce bytes per second on top of the fixed cost, GB/s (default: free)")
    p.add_argument("--handoff-us", type=float, default=10, help="per GPU boundary, layer split (guess)")
    p.add_argument("--expert-reuse", type=float, default=0.0,
                   help="0 = draft tokens route independently (worst case); 1 = all share experts")
    p.add_argument("--gpus-27b", type=int, default=QWEN38_27B.gpus)
    p.add_argument("--gpus-flash-next", type=int, default=QWEN38_FLASH_NEXT.gpus)
    p.add_argument("--measured", metavar="DIR",
                   help="rig results dir: use its measured matmul times wherever it has the shape")
    p.add_argument("--detail", action="append", default=[], help="ARCH:T component breakdown, e.g. 27b:8")
    p.add_argument("--kernels", action="store_true", help="also print the per-kernel instruction table")
    return p


def main():
    a = make_parser().parse_args()
    QWEN38_27B.gpus, QWEN38_FLASH_NEXT.gpus = a.gpus_27b, a.gpus_flash_next

    gpu = GPU(a)
    today, w4a4 = kernels(HERE / "build")
    if a.kernels:
        print_kernels(gpu, today, w4a4)
    measured = Measured(a.measured) if a.measured else None
    model = Model(gpu, today, w4a4, a, measured)
    print(f"MI50 at {a.clock} GHz, {a.bw:.0f} GB/s x {a.bw_eff}, {a.latency:.0f} ns latency, "
          f"attention at {a.attn_eff} of packed-fp16 peak")
    if measured:
        print(f"matmuls measured on the card ({', '.join(measured.kinds()) or 'none found'}) from {measured.d}; "
              "the rest from the roofline")
    for arch in ARCHS.values():
        print_arch(model, arch, a.ctx)
    for d in a.detail:
        name, T = d.split(":")
        print_detail(model, ARCHS[name], int(T), a.ctx)


if __name__ == "__main__":
    main()
