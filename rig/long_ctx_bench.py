#!/usr/bin/env python3
# DFlash2 decoding after a long context, through llama-server: a DEPTH-token document (wikitext) followed by a question,
# greedy, and the same questions without the document.
#   python3 rig/long_ctx_bench.py MODEL DRAFT N_MAX DEPTH [TEXT]
#   env: BIN, ROCM_LIB (default ~/therock-7.12/lib), GPUS, EXTRA (e.g. "-sm tensor -devd ROCm0"), PORT, N (tokens, 256),
#        LOG (keep the server log there)
# Prints per request the prompt tokens, prefill time and rate, decode rate and acceptance length (completion tokens per
# verify step), then the server's KV cache, recurrent state and compute buffer sizes.
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

model, draft, nmax, depth = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text_path = sys.argv[5] if len(sys.argv) > 5 else os.path.expanduser("~/ref/wiki.test.raw")
bin_dir = os.environ.get("BIN", os.path.expanduser("~/mx-llama.cpp/build-r712/bin"))
port = os.environ.get("PORT", "8097")
n_tok = int(os.environ.get("N", "256"))
url = "http://127.0.0.1:%s" % port

args = [os.path.join(bin_dir, "llama-server"), "-m", model, "-ngl", "99", "-fa", "1", "-lm", "dio", "-c", str(depth + 4096),
        "-b", "2048", "-ub", "2048", "-np", "1", "--port", port, "--host", "127.0.0.1", "--no-webui", "-lv", "4",
        "--spec-type", "draft-dflash", "--spec-draft-n-max", str(nmax), "-bs", "--samplers", "temperature"]
if draft:
    args += ["-md", draft, "-ngld", "99"]
args += os.environ.get("EXTRA", "").split()
env = dict(os.environ, HIP_VISIBLE_DEVICES=os.environ.get("GPUS", "0"), GGML_META_TOKEN_GRAPH="1",
           LD_LIBRARY_PATH=bin_dir + ":" + os.environ.get("ROCM_LIB", os.path.expanduser("~/therock-7.12/lib")))
env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
env.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")
env.setdefault("GGML_META_SPIN", "2000000")
log = open(os.environ["LOG"], "w+") if os.environ.get("LOG") else tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)


def post(path, body, timeout=3600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


questions = [
    "Summarize the text above in five sentences.",
    "Which people, places and dates does the text above mention? Answer as a list.",
    "Write a Python function that counts how often each word occurs in a text, with a short explanation.",
]
server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
try:
    for _ in range(1800):
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            break
        except Exception:
            if server.poll() is not None:
                sys.exit("server exited:\n" + open(log.name).read()[-3000:])
            time.sleep(1)
    toks = post("/tokenize", {"content": open(text_path, errors="replace").read()})["tokens"]
    doc = post("/detokenize", {"tokens": toks[:depth]})["content"] if depth > 0 else ""
    print("model %s, drafter %s, draft %d, depth %d (document of %d tokens)" % (
        os.path.basename(model), os.path.basename(draft), nmax, depth, min(depth, len(toks))), flush=True)
    for with_doc in ([True, False] if depth > 0 else [False]):
        for q in questions:
            content = (doc + "\n\n" + q) if with_doc else q
            r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": n_tok,
                                              "temperature": 0, "cache_prompt": True})
            t = r["timings"]
            steps = t["draft_n"] / nmax if t.get("draft_n") else t["predicted_n"]
            print("%-9s prompt %6d tok %8.0f ms (%6.1f t/s)  decode %4d tok %6.1f t/s  accept len %4.2f  %5.1f ms/step" % (
                "document" if with_doc else "short", t["prompt_n"], t["prompt_ms"], t["prompt_per_second"],
                t["predicted_n"], t["predicted_per_second"], t["predicted_n"] / steps, t["predicted_ms"] / steps),
                flush=True)
finally:
    server.terminate()
    server.wait()
log.seek(0)
seen = set()
for line in log.read().splitlines():
    m = re.search(r"(KV buffer size|RS buffer size|compute buffer size|KV self size|recurrent.*size|llama_kv_cache.*size)", line)
    if m:
        s = re.sub(r"^.*?(llama_\w+|\w+_context|sched_reserve|\w+): +", r"\1: ", line).strip()
        if s not in seen:
            seen.add(s)
            print("  " + s)
