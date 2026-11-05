#!/usr/bin/env python3
# Splits a one-GPU rocprof dispatch trace (--timestamp on) of a DFlash2 server run into its graphs and reports
# where the drafter's GPU time goes:
#   python3 rig/draft_breakdown.py results.csv [gap_us]
# A graph is a run of dispatches without an idle gap over gap_us (default 50). Graphs with DeltaNet kernels are the
# target's, graphs with the top-k radix kernels are the drafter's draft, the rest (between a target graph and a
# draft) the drafter's context injection.
import collections, csv, sys

path = sys.argv[1]
gap_ns = float(sys.argv[2] if len(sys.argv) > 2 else 50) * 1e3
rows = sorted(csv.DictReader(open(path)), key=lambda r: int(r["BeginNs"]))

segs, cur = [], []
for r in rows:
    b, e = int(r["BeginNs"]), int(r["EndNs"])
    if cur and b - cur[-1][2] > gap_ns:
        segs.append(cur)
        cur = []
    cur.append((r["KernelName"].split("(")[0], b, e, r["grd"]))
if cur:
    segs.append(cur)

def kind(seg):
    names = {k[0] for k in seg}
    if any(n.startswith(("gated_delta_net", "ssm_conv")) for n in names):
        return "target"
    if any(n.startswith("top_k_radix") for n in names):
        return "draft"
    return "other"

def short(n):
    for p in ("rq_skinny_kernel", "rq_quantize_act", "rq_codes", "rq_hadamard_stats", "rq_fwht", "rq_paley",
              "flash_attn", "mul_mat_vec", "mul_mat_q", "top_k_radix", "k_get_rows", "rms_norm", "k_bin_bcast",
              "concat", "cpy", "soft_max", "rope", "unary", "scale", "argmax"):
        if n.startswith(p) or p in n:
            return p
    return n[:40]

by_kind = collections.defaultdict(list)
for s in segs:
    by_kind[kind(s)].append(s)
for k, ss in by_kind.items():
    walls = sorted((s[-1][2] - s[0][1]) / 1e3 for s in ss)
    busy = sorted(sum(e - b for _, b, e, _ in s) / 1e3 for s in ss)
    print("%-6s %5d graphs, median wall %8.1f us, busy %8.1f us, %6.0f dispatches each" %
          (k, len(ss), walls[len(walls) // 2], busy[len(busy) // 2], sum(len(s) for s in ss) / len(ss)))

for k in ("draft", "other"):
    ss = by_kind.get(k, [])
    if not ss:
        continue
    acc = collections.Counter()
    cnt = collections.Counter()
    for s in ss:
        for n, b, e, g in s:
            key = (short(n), g if short(n) in ("rq_skinny_kernel", "mul_mat_vec", "mul_mat_q") else "")
            acc[key] += (e - b) / 1e3
            cnt[key] += 1
    tot = sum(acc.values()) / len(ss)
    print("== %s graphs: %.1f us of kernels each" % (k, tot))
    for key, v in acc.most_common(18):
        print("  %-24s %-9s %4.0f calls %8.1f us  %4.1f%%" % (key[0], key[1], cnt[key] / len(ss), v / len(ss), 100 * v / len(ss) / tot))
