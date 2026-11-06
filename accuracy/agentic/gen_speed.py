#!/usr/bin/env python3
"""Decode speed of one long greedy generation through llama-server (to time generation checkpoints).

    python3 gen_speed.py [URL] [N_PREDICT] [REPEATS]
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


prompt = post("/apply-template", {"messages": [{"role": "user", "content":
    "Write a long, detailed Python tutorial on building a web scraper, with complete code for every step."}],
    "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
for i in range(REP):
    r = post("/completion", {"prompt": prompt, "n_predict": N, "temperature": 0, "ignore_eos": True, "cache_prompt": False})
    t = r["timings"]
    acc = t.get("draft_n_accepted", 0) / max(1, t.get("draft_n", 1))
    print(f"run {i + 1}: {t['predicted_n']} tokens in {t['predicted_ms'] / 1000:.2f} s = {t['predicted_per_second']:.2f} t/s "
          f"(draft acceptance {acc:.3f})", flush=True)
