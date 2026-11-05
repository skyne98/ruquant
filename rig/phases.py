# Splits a rig/prof_spec.sh trace (results.csv) into speculative-decoding phases at GPU idle gaps > 40 us and
# prints each phase's kernel count, span, busy time and top kernels, for two verify steps.
#   python3 rig/phases.py OUT/results.csv
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1])))
# a step starts at the target lm_head: the biggest per-step kernel right before the argmax ops
argmax = [j for j, r in enumerate(rows) if r["KernelName"].startswith("argmax") and r["grd"] == "1024"]
starts = []
for j in argmax:
    k = j - 1
    while rows[k]["KernelName"].startswith(("pad_f32", "argmax", "k_get_rows")):
        k -= 1
    if not starts or k > starts[-1] + 50:
        starts.append(k)
for s in (len(starts) // 3, len(starts) // 2):
    a, b = starts[s], starts[s + 1]
    seg = rows[a:b]
    t0 = int(seg[0]["BeginNs"])
    print("step period %.2f ms" % ((int(rows[b]["BeginNs"]) - t0) / 1e6))
    phases, cur = [], [seg[0]]
    for p, r in zip(seg, seg[1:]):
        gap = int(r["BeginNs"]) - int(p["EndNs"])
        if gap > 40000:
            phases.append((cur, gap)); cur = []
        cur.append(r)
    phases.append((cur, 0))
    for ph, gap in phases:
        busy = sum(int(r["DurationNs"]) for r in ph) / 1e3
        span = (int(ph[-1]["EndNs"]) - int(ph[0]["BeginNs"])) / 1e3
        top = Counter()
        for r in ph:
            top[r["KernelName"][:22]] += int(r["DurationNs"]) / 1e3
        print("  %5d kernels span %7.1f us busy %7.1f us, then gap %6.1f us | %s" % (len(ph), span, busy, gap / 1e3,
              ", ".join("%s %.0f" % kv for kv in top.most_common(5))))
