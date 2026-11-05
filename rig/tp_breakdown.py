# Where a prefill forward's time goes, per GPU, from rocprof kernel traces of llama-bench runs (one timed forward,
# -r 1, so the trace holds the warm-up and the timed forward; each ends in the lm_head mat-vec).
#   python3 rig/tp_breakdown.py ONE_GPU.csv TP.csv [TP.csv ...]
# Prints, in ms per forward, each kernel category on one GPU and per lane (mean over lanes) under TP, with the excess
# over (one GPU)/N, plus idle time inside the forward. Make a trace on the box with
#   HIP_VISIBLE_DEVICES=0,1 rocprof --stats --basenames on -o tr.csv ./llama-bench -m M -ngl 99 -fa 1 -sm tensor \
#       -n 0 -p 512 -b 512 -ub 512 -r 1
import collections, csv, sys

CATS = (("rq_gemm", "w4a4 gemm"), ("rq_skinny", "w4a4 skinny"), ("rq_quantize_act", "act quantize"),
        ("rq_fwht", "hadamard"), ("rq_paley", "hadamard"), ("nccl", "all-reduce (RCCL)"),
        ("k_broadcast_reduce", "all-reduce (custom)"), ("k_twoshot", "all-reduce (custom)"),
        ("convert_unary", "all-reduce f16 converts"), ("k_ar_", "all-reduce quantize"),
        ("dequantize_block", "all-reduce dequantize"), ("gated_delta_net", "deltanet"), ("ssm_conv", "ssm_conv"),
        ("mmq_gemm_repacked", "q8_0 ssm_out"), ("Cijk", "rocblas f32 (alpha/beta)"), ("flash_attn", "flash attn"),
        ("rms_norm", "rms_norm"), ("unary_gated", "swiglu/gate"), ("cpy", "copies"), ("concat", "copies"),
        ("copyBuffer", "copies"), ("k_bin_bcast", "add/mul"), ("rope", "rope"), ("quantize_mmq", "q8 act quant"))


def cat(name):
    return next((c for key, c in CATS if key in name), "other")


def lanes(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[r["gpu-id"]].append((int(r["BeginNs"]), int(r["EndNs"]), r["KernelName"]))
    out = []
    for v in by.values():
        v.sort()
        heads = [i for i, k in enumerate(v) if "mul_mat_vec_q8_0_repacked" in k[2]]
        f = v[heads[-2] + 1:heads[-1] + 1]
        while not f[0][2].startswith("k_get_rows"):
            f = f[1:]
        t = collections.Counter()
        last = f[0][0]
        for b, e, n in f:
            t[cat(n)] += (e - b) / 1e6
            t["idle inside the forward"] += max(0, b - last) / 1e6
            last = max(last, e)
        t["forward"] = (f[-1][1] - f[0][0]) / 1e6
        out.append(t)
    return len(out), {k: sum(o[k] for o in out) / len(out) for k in set().union(*out)}


one = lanes(sys.argv[1])[1]
tps = [lanes(p) for p in sys.argv[2:]]
keys = sorted(set(one).union(*(t for _, t in tps)) - {"forward"}, key=lambda c: -max([one.get(c, 0)] + [t.get(c, 0) for _, t in tps]))
print("%-26s %8s" % ("ms per forward", "1 GPU") + "".join(" | %8s %8s" % ("TP%d" % n, "excess") for n, _ in tps))
for k in keys + ["forward"]:
    a = one.get(k, 0)
    print("%-26s %8.1f" % (k, a) + "".join(" | %8.1f %+8.1f" % (t.get(k, 0), t.get(k, 0) - a / n) for n, t in tps))
