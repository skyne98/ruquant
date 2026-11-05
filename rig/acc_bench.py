#!/usr/bin/env python3
# Draft acceptance of a target + drafter pair on rig/acc_prompts.json (math, code, chat), through llama-server.
#   python3 rig/acc_bench.py MODEL DRAFT|'' N_MAX [SPEC]     env: BIN, ROCM_LIB (default /opt/rocm/lib), GPUS,
#                                                            N (tokens, default 256), PORT, EXTRA
# Greedy chat completions. Prints per category the mean accepted length (completion tokens / verify steps, the
# DFlash2 model card's acceptance length; here steps = drafted tokens / N_MAX) and the acceptance at each draft
# position (the share of verify steps that kept at least that many drafts, from the server's trace log).
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

model, draft, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = sys.argv[4] if len(sys.argv) > 4 else "draft-dflash"
here = os.path.dirname(os.path.abspath(__file__))
prompts = json.load(open(os.path.join(here, "acc_prompts.json")))
bin_dir = os.environ.get("BIN", os.path.expanduser("~/mx-llama.cpp/build/bin"))
port = os.environ.get("PORT", "8096")
n_tok = int(os.environ.get("N", "256"))

args = [os.path.join(bin_dir, "llama-server"), "-m", model, "-ngl", "99", "-fa", "1", "-lm", "dio", "-c", "8192",
        "-np", "1", "--port", port, "--host", "127.0.0.1", "--no-webui", "-lv", "4", "--spec-type", spec,
        "--spec-draft-n-max", str(nmax), "-bs", "--samplers", "temperature"]
if draft:
    args += ["-md", draft, "-ngld", "99"]
args += os.environ.get("EXTRA", "").split()
env = dict(os.environ, HIP_VISIBLE_DEVICES=os.environ.get("GPUS", "0"),
           LD_LIBRARY_PATH=bin_dir + ":" + os.environ.get("ROCM_LIB", "/opt/rocm/lib"))
log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
try:
    for _ in range(900):
        try:
            urllib.request.urlopen("http://127.0.0.1:%s/health" % port, timeout=2)
            break
        except Exception:
            if server.poll() is not None:
                sys.exit("server exited:\n" + open(log.name).read()[-3000:])
            time.sleep(1)
    rows = []
    for cat, prompt in prompts:
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": n_tok,
                           "temperature": 0, "cache_prompt": False}).encode()
        req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, body,
                                     {"Content-Type": "application/json"})
        t = json.load(urllib.request.urlopen(req, timeout=600))["timings"]
        rows.append((cat, t))
finally:
    server.terminate()
    server.wait()

# the server prints one "acc per pos" line per request, in order
log_text = open(log.name).read()
if os.environ.get("KEEP_LOG"):  # a copy of the server log, e.g. to read its token-graph lines
    open(os.environ["KEEP_LOG"], "w").write(log_text)
per_pos = [[float(x) for x in m.group(1).split(",")]
           for m in re.finditer(r"acc per pos = \(([^)]*)\)", log_text)]
os.unlink(log.name)
# with LLAMA_STEP_TRACE=1 the server logs host time per step by phase, and each context its graph reuse, input
# upload and dispatch time; show the last (cumulative) line of each
step_trace = {}
for m in re.finditer(r"(\w+: step trace[^,]*), (.*)", log_text):
    step_trace[m.group(1)] = m.group(0)
for line in step_trace.values():
    print(line)
cats = {}
for i, (cat, t) in enumerate(rows):
    steps = t["draft_n"] / nmax if t.get("draft_n") else t["predicted_n"]
    c = cats.setdefault(cat, {"tokens": 0, "steps": 0, "ms": 0.0, "pos": []})
    c["tokens"] += t["predicted_n"]
    c["steps"] += steps
    c["ms"] += t["predicted_ms"]
    if i < len(per_pos):
        c["pos"].append(per_pos[i])
tot = {"tokens": 0, "steps": 0, "ms": 0.0, "pos": []}
for cat, c in cats.items():
    for k in ("tokens", "steps", "ms"):
        tot[k] += c[k]
    tot["pos"] += c["pos"]
for cat, c in list(cats.items()) + [("all", tot)]:
    pos = [sum(p[j] for p in c["pos"]) / len(c["pos"]) for j in range(len(c["pos"][0]))] if c["pos"] else []
    print("%-5s accept len %5.2f  %6.1f t/s  %5.1f ms/step  per pos %s" % (
        cat, c["tokens"] / c["steps"], c["tokens"] / c["ms"] * 1000, c["ms"] / c["steps"],
        " ".join("%.2f" % x for x in pos)))
