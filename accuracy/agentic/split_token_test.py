#!/usr/bin/env python3
"""A prompt cached with one word split into two tokens, then sent back as text: how much does the server redo?

    python3 split_token_test.py [URL]
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


tok = lambda s: post("/tokenize", {"content": s})["tokens"]
text = " ".join(f"Section {i}: the inventory module tracks warehouses, shipments and quarterly reports." for i in range(400))
canon = tok(text)
k = len(canon) // 4  # a word a quarter of the way in
piece = post("/detokenize", {"tokens": [canon[k]]})["content"]
cut = len(piece) // 2
split = tok(piece[:cut]) + tok(piece[cut:])
assert len(split) >= 2 and post("/detokenize", {"tokens": split})["content"] == piece, (piece, split)
noncanon = canon[:k] + split + canon[k + 1:]
post("/completion", {"prompt": noncanon, "n_predict": 1, "cache_prompt": True})
r = post("/completion", {"prompt": text + " Section 400: done.", "n_predict": 1, "cache_prompt": True})
t = r["timings"]
print(f"{len(canon)} tokens, token {k} ({piece!r}) cached as {len(split)} tokens; the text sent back: "
      f"re-processed {t['prompt_n']} tokens, cache hit {t.get('cache_n')}")
