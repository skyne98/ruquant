#!/usr/bin/env python3
# Where a DFlash2 step's time goes at depth: from a rocprofv3 kernel trace of one GPU running rig/long_ctx_bench.py
# (a long document with questions, then short prompts), splits the decode into verify steps and compares deep steps
# with short ones, per kernel (GPU time per step) plus the GPU's idle time per step.
#   python3 rig/step_depth.py PROFILE_DIR_OR_CSV [SPLIT_US]
# A step starts at every 16th q4 attention call (rq_fa_q4_kernel): 16 calls per target forward. Steps whose attention calls average more than SPLIT_US (default 300) are deep, under SPLIT_US / 4 short.
import collections, csv, re, sys
from pathlib import Path

path = Path(sys.argv[1])
split = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
csvs = [path] if path.suffix == ".csv" else sorted(path.rglob("*kernel_trace.csv"))
rows = []
for f in csvs:
    for r in csv.DictReader(open(f)):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), re.sub(r"\(.*", "", r["Kernel_Name"].replace("void ", ""))[:48]))
rows.sort()

# target forwards: every 16 attention calls (prefill ubatches use llama.cpp's kernels, the drafter an f16 cache); a
# step spans from one forward's first attention call to the next's
fa = [i for i, r in enumerate(rows) if r[2].startswith("rq_fa_q4_kernel")]
starts = fa[::16]
steps = []
for a, b in zip(starts, starts[1:]):
    seg = rows[a:b]
    fa_us = [(e - s) / 1e3 for s, e, k in seg if k.startswith("rq_fa_q4_kernel")]
    wall = (rows[b][0] - rows[a][0]) / 1e6
    if len(fa_us) != 16 or wall > 200:  # a prompt in between
        continue
    busy = sum(e - s for s, e, k in seg) / 1e6
    steps.append((sum(fa_us) / 16, wall, busy, seg))

groups = {"deep": [s for s in steps if s[0] > split], "short": [s for s in steps if s[0] < split / 4]}
print(f"{len(steps)} steps: {len(groups['deep'])} deep, {len(groups['short'])} short")
per = {}
for g, ss in groups.items():
    if not ss:
        continue
    k = collections.Counter()
    for _, _, _, seg in ss:
        for s, e, name in seg:
            k[name] += (e - s) / 1e3
    n = len(ss)
    per[g] = {name: v / n for name, v in k.items()}
    wall = sorted(s[1] for s in ss)[n // 2]
    busy = sorted(s[2] for s in ss)[n // 2]
    print(f"{g:5s}: step {wall:7.2f} ms wall, {busy:7.2f} ms GPU busy, {wall - busy:6.2f} ms idle (medians of {n})")
if len(per) == 2:
    print("\nper step, us: deep - short, the kernels that grow most")
    names = set(per["deep"]) | set(per["short"])
    diff = sorted(((per["deep"].get(n, 0) - per["short"].get(n, 0), n) for n in names), reverse=True)
    for d, n in diff[:15]:
        print(f"  {n:48s} {per['short'].get(n, 0):9.1f} -> {per['deep'].get(n, 0):9.1f}  ({d:+8.1f})")
