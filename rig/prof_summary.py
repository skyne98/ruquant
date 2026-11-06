#!/usr/bin/env python3
# Summarizes a recorded profile (rig/prof.sh): a profile directory, its out/ directory or a pulled .tar.gz.
#   python3 rig/prof_summary.py PROFILE [KERNEL_REGEX]
# Kernel traces: per kernel and grid (workgroups), the count and the median / min duration in us.
# Counters: per kernel and grid, the median of each counter over its dispatches.
import collections, csv, io, re, sys, tarfile
from pathlib import Path


def csv_files(path):
    p = Path(path)
    if p.suffix == ".gz":
        with tarfile.open(p) as t:
            for m in t.getmembers():
                if m.name.endswith(".csv"):
                    yield m.name, io.StringIO(t.extractfile(m).read().decode())
        return
    for f in sorted(p.rglob("*.csv")):
        yield str(f), open(f)


def short(name):
    return re.sub(r"\(.*", "", name.replace("void ", ""))[:60]


def main():
    pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    for name, fh in csv_files(sys.argv[1]):
        rows = list(csv.DictReader(fh))
        if not rows or "Kernel_Name" not in rows[0]:
            continue
        rows = [r for r in rows if pat is None or pat.search(r["Kernel_Name"])]
        if name.endswith("kernel_trace.csv"):
            d = collections.defaultdict(list)
            for r in rows:
                wgs = int(r["Grid_Size_X"]) // int(r["Workgroup_Size_X"]) * int(r.get("Grid_Size_Y") or 1)
                d[(short(r["Kernel_Name"]), wgs)].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000)
            print(f"{name}: kernel trace")
            for (k, wgs), v in sorted(d.items(), key=lambda kv: -sum(kv[1])):
                v.sort()
                print(f"  {k:60s} wgs {wgs:5d}  n {len(v):6d}  med {v[len(v) // 2]:9.2f} us  min {v[0]:9.2f} us")
        elif name.endswith("counter_collection.csv"):
            d = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in rows:
                d[(short(r["Kernel_Name"]), int(r["Grid_Size"]) // int(r["Workgroup_Size"]))][r["Counter_Name"]].append(
                    float(r["Counter_Value"]))
            print(f"{name}: counters (median per dispatch)")
            for (k, wgs), cs in sorted(d.items()):
                n = max(len(v) for v in cs.values())
                vals = "  ".join(f"{c}={sorted(v)[len(v) // 2]:.4g}" for c, v in cs.items())
                print(f"  {k:60s} wgs {wgs:5d}  n {n:6d}  {vals}")


main()
