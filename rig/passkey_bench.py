#!/usr/bin/env python3
# Long-context retrieval through llama-server with DFlash2: a pass key hidden at several positions of a DEPTH-token
# wikitext document, asked for at the end, greedy, thinking off. Scores whether the answer contains the key.
#   python3 rig/passkey_bench.py MODEL DRAFT N_MAX DEPTH [TEXT]
#   env: BIN, ROCM_LIB (default ~/therock-7.12/lib), GPUS, EXTRA (e.g. "-sm tensor -devd ROCm0 -ctk q4_0 -ctv q4_0"),
#        PORT, POS (fractions of the document, default 0.05,0.25,0.5,0.75,0.95), SEED, LOG (keep the server log there)
# Requests go from the deepest key to the shallowest, so each one shares the document up to its key with the last.
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

model, draft, nmax, depth = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text_path = sys.argv[5] if len(sys.argv) > 5 else os.path.expanduser("~/ref/wiki.test.raw")
bin_dir = os.environ.get("BIN", os.path.expanduser("~/mx-llama.cpp/build-r712/bin"))
port = os.environ.get("PORT", "8098")
positions = [float(p) for p in os.environ.get("POS", "0.05,0.25,0.5,0.75,0.95").split(",")]
url = "http://127.0.0.1:%s" % port

args = [os.path.join(bin_dir, "llama-server"), "-m", model, "-ngl", "99", "-fa", "1", "-lm", "dio", "-c", str(depth + 1024),
        "-b", "2048", "-ub", "2048", "-np", "1", "--port", port, "--host", "127.0.0.1", "--no-webui",
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


rng = random.Random(int(os.environ.get("SEED", "1")))
question = "What is the pass key mentioned in the text above? Reply with the number only."
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
    toks = post("/tokenize", {"content": open(text_path, errors="replace").read()})["tokens"][:depth]
    print("model %s, depth %d, extra '%s'" % (os.path.basename(model), len(toks), os.environ.get("EXTRA", "")), flush=True)
    hits = 0
    for frac in sorted(positions, reverse=True):
        key = rng.randrange(10000, 100000)
        p = int(frac * len(toks))
        needle = "\n\nThe pass key is %d. Remember it. %d is the pass key.\n\n" % (key, key)
        content = (post("/detokenize", {"tokens": toks[:p]})["content"] + needle +
                   post("/detokenize", {"tokens": toks[p:]})["content"] + "\n\n" + question)
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 48,
                                          "temperature": 0, "cache_prompt": True,
                                          "chat_template_kwargs": {"enable_thinking": False}})
        answer = r["choices"][0]["message"]["content"] or ""
        ok = str(key) in re.findall(r"\d+", answer)
        hits += ok
        t = r["timings"]
        print("key at %4.0f%% (token %6d)  %-5s  key %d  answer %-24r  prompt %6d tok %7.0f ms" % (
            100 * frac, p, "ok" if ok else "MISS", key, answer.strip()[:24], t["prompt_n"], t["prompt_ms"]), flush=True)
    print("found %d / %d" % (hits, len(positions)), flush=True)
finally:
    server.terminate()
    server.wait()
