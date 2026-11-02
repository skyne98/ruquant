#!/usr/bin/env python3
"""Turn a rig results directory into gfx906/model.py flags, then put the model's per-matmul
predictions next to what the rig measured.

    python3 rig/calibrate.py results/<host>-<date>      # run here, where gfx906/build/*.s exist

Reads (any subset): valu-gpu0.json, mem-gpu0.json, atomic-gpu0.json, p2p.json, tp.json,
skinny-gpu0.json (w4a4 measured), ggml-gpu0.json (today's path measured).
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gfx906"))
import model  # noqa: E402


def load(d, name):
    p = d / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"  {name}: unreadable ({e})")
        return None


def flags_from(d):
    flags, notes = {}, []
    valu = load(d, "valu-gpu0.json")
    if valu:
        ops = {o["op"]: o for o in valu["ops"]}
        dot8 = ops["dot8_i32_i4"]
        flags["--clock"] = round(dot8["clock_mhz"] / 1e3, 3)
        flags["--dot8-rate"] = round(min(1.0, dot8["full_rate_fraction"]), 3)
        notes.append(f"dot8 at {dot8['full_rate_fraction']:.2f} of full rate, "
                     f"{dot8['tmacs']:.1f} TMAC/s, clock under load {dot8['clock_mhz']:.0f} MHz, "
                     f"dependent issue {dot8['dependent_cycles_per_instr']:.1f} cycles")
        for name in ("dot4_i32_i8", "fma_f32", "pk_fma_f16"):
            o = ops[name]
            notes.append(f"{name} at {o['full_rate_fraction']:.2f} of full rate")
        if "launch_us" in valu:
            notes.append(f"kernel launch {valu['launch_us']:.1f} us back to back (not in the model yet: "
                         "every small matmul below pays it)")
    mem = load(d, "mem-gpu0.json")
    if mem:
        pts = mem["read_sweep"]
        peak = max(p["gbs"] for p in pts)
        flags["--bw-eff"] = round(peak / 1024, 3)
        ramp = sorted(p["bytes_in_flight"] / (p["gbs"] * 1e9) * 1e9 for p in pts if p["gbs"] < 0.7 * peak)
        if ramp:
            flags["--latency"] = round(statistics.median(ramp))
        chase = {c["footprint_kib"]: c["ns"] for c in mem.get("chase", [])}
        notes.append(f"peak read {peak:.0f} GB/s, copy {mem['copy_gbs']:.0f} GB/s, loaded latency "
                     f"{flags.get('--latency', '?')} ns, idle chase: "
                     + ", ".join(f"{k} KiB {v:.0f} ns" for k, v in chase.items()))
    atomic = load(d, "atomic-gpu0.json")
    if atomic:
        rates = [p["model_bytes_gbs"] for p in atomic["points"] if p["G"] >= 4]
        if rates:
            flags["--atomic-gbs"] = round(statistics.median(rates))
            notes.append(f"split-K partials at {flags['--atomic-gbs']} GB/s (model's L2 guess: "
                         f"{16 * 64 * flags.get('--clock', 1.725):.0f})")
    p2p = load(d, "p2p.json")
    if p2p and "latency" in p2p:
        lat = {x["bytes"]: x for x in p2p["latency"]}
        pick = lat.get(10240) or next(iter(lat.values()))
        flags["--handoff-us"] = round(pick["one_way_us"], 1)
        flags["--allreduce-us"] = round(pick["allreduce_us"], 1)
        notes.append(f"P2P {'on' if p2p['p2p'] else 'OFF (staged through host)'}, "
                     + ", ".join(f"{b['bytes'] >> 20} MiB {b['gbs']:.1f} GB/s" for b in p2p["bandwidth"]))
    tp = load(d, "tp.json")
    if tp and tp.get("allreduce"):
        notes.append("peer copy GB/s [src][dst]: " + "  ".join(
            "[" + " ".join(f"{x:.1f}" for x in row) + "]" for row in tp["pair_gbs"])
            + f"; all links busy: ring {tp['ring_gbs_per_link']:.1f}, into GPU 0 {tp['fanin_gbs_per_link']:.1f} per link")
        fits = {}
        for n in sorted({p["gpus"] for p in tp["allreduce"]}):
            pts = [p for p in tp["allreduce"] if p["gpus"] == n and p["ok"]]
            best = {}
            for p in pts:
                if p["bytes"] not in best or p["us"] < best[p["bytes"]]["us"]:
                    best[p["bytes"]] = p
            # us = a + bytes * c over verify-sized all-reduces (up to 64 tokens of 5120)
            xs = [b for b in sorted(best) if best[b]["tokens"] <= 64]
            ys = [best[b]["us"] for b in xs]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            a = my - c * mx
            if a < 0:  # e.g. staged through the host, where small sizes don't follow the line
                a, c = 0.0, sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
            fits[n] = (round(a, 1), round(1 / (c * 1e3), 2))
            notes.append(f"all-reduce over {n} GPUs (best method per size): " + ", ".join(
                f"{best[b]['tokens']} tok {best[b]['us']:.0f} us {best[b]['method']}" for b in sorted(best))
                + f"; fit {fits[n][0]} us + bytes at {fits[n][1]} GB/s")
        n = max(fits)
        flags["--allreduce-us"], flags["--allreduce-gbs"] = fits[n]
        notes.append(f"(all-reduce flags are for {n} GPUs; other counts above)")
    return flags, notes


class _LmHead:
    def __init__(self, K, N):
        self.lm_head = (K, N)


def compare(d, flags):
    skinny, ggml = load(d, "skinny-gpu0.json"), load(d, "ggml-gpu0.json")
    if not (skinny or ggml):
        return
    argv = [str(x) for kv in flags.items() for x in kv]
    a = model.make_parser().parse_args(argv)
    gpu = model.GPU(a)
    today, w4a4 = model.kernels(ROOT / "gfx906" / "build")
    m = model.Model(gpu, today, w4a4, a)

    print("\nmeasured vs model, per matmul (us; ratio > 1 means slower than predicted)")
    print(f"  {'shape':<22}{'T':>3}   {'today meas':>10}{'pred':>8}{'ratio':>7}   "
          f"{'w4a4 meas':>10}{'pred':>8}{'ratio':>7}  best split")
    best = {}
    for p in (skinny or {}).get("points", []):
        k = (p["shape"], p["T"])
        if k not in best or p["us"] < best[k]["us"]:
            best[k] = p
    rows = {}
    for p in (ggml or {}).get("points", []):
        rows.setdefault((p["shape"], p["T"]), {})["ggml"] = p
    for k, p in best.items():
        rows.setdefault(k, {})["skinny"] = p
    for (shape, T), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        g, s = r.get("ggml"), r.get("skinny")
        K, N = (g or s)["K"], (g or s)["N"]
        line = f"  {shape:<22}{T:>3}   "
        if g:
            pred = (m.lm_head(_LmHead(K, N), T) if g["type"] == "q6_K" else m.linear("today", T, K, N)) * 1e6
            line += f"{g['us']:10.1f}{pred:8.1f}{g['us'] / pred:7.2f}   "
        else:
            line += " " * 28
        if s and T in model.WIDTHS:
            pred = m.linear("w4a4", T, K, N) * 1e6
            line += f"{s['us']:10.1f}{pred:8.1f}{s['us'] / pred:7.2f}  R={s['R']} W={s['W']} G={s['G']}"
        print(line)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = Path(sys.argv[1])
    flags, notes = flags_from(d)
    print(f"results: {d}")
    for n in notes:
        print(f"  {n}")
    print("\nmodel flags from these measurements:")
    print("  python3 gfx906/model.py " + " ".join(f"{k} {v}" for k, v in flags.items()))
    compare(d, flags)


if __name__ == "__main__":
    main()
