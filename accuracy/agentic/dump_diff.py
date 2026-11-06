#!/usr/bin/env python3
"""Where a client's next prompt stops matching the previous prompt + the model's raw answer (LLAMA_DUMP_DIR dumps).

    python3 dump_diff.py DUMP_DIR
"""
import os
import sys

d = sys.argv[1]
n = len([f for f in os.listdir(d) if f.endswith("-prompt.txt")])
read = lambda i, k: open(os.path.join(d, f"{i:04d}-{k}.txt"), encoding="utf-8", errors="replace").read()
for i in range(1, n):
    try:
        p, a, q = read(i, "prompt"), read(i, "answer"), read(i + 1, "prompt")
    except FileNotFoundError:
        continue
    want = p + a
    k = next((j for j, (x, y) in enumerate(zip(want, q)) if x != y), min(len(want), len(q)))
    if k < len(p) * 0.9:
        continue  # another conversation (title, summary)
    if k >= len(want):
        print(f"{i:4d} -> {i + 1}: matches prompt + all {len(a)} answer chars")
        continue
    off = k - len(p)
    print(f"{i:4d} -> {i + 1}: differs {off} chars into a {len(a)}-char answer ({off / max(1, len(a)):.0%})")
    print(f"       answer:  {want[max(0, k - 60):k + 60]!r}")
    print(f"       resent:  {q[max(0, k - 60):k + 60]!r}")
