#!/usr/bin/env python3
"""Can the model see? One question per image through llama-server (--mmproj), greedy, thinking off; an answer passes
when it contains every expected keyword (case-insensitive, any one of each group's alternatives).

    python3 vision_check.py [URL] [EXTRA_IMAGE QUESTION KEYWORD1|ALT,KEYWORD2 ...]
"""
import base64
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
CASES = [
    ("chart.png", "What is the title of this chart, which day has the most orders, and how many orders were there on "
                  "Thursday?", [["orders per day"], ["tue"], ["6", "six"]]),
    ("sign.png", "Transcribe all the text on this sign.", [["bridge closed"], ["elm street"], ["friday"], ["14:30"]]),
    ("shapes.png", "List the shapes in this image from left to right, with their colors.",
     [["red circle"], ["blue square", "blue rectangle"], ["green triangle"]]),
]
extra = sys.argv[2:]
for i in range(0, len(extra) - 2, 3):
    CASES.append((extra[i], extra[i + 1], [k.split("|") for k in extra[i + 2].split(",")]))


def ask(path, question):
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    img = base64.b64encode(open(p, "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        {"type": "text", "text": question}]}],
        "temperature": 0, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"] or "", r["timings"]


passed = 0
for path, q, keys in CASES:
    answer, t = ask(path, q)
    low = answer.lower()
    ok = all(any(k in low for k in group) for group in keys)
    passed += ok
    print(f"== {os.path.basename(path)}: {'PASS' if ok else 'FAIL'} ({t['prompt_n']} prompt tokens, {t['predicted_n']} generated)")
    print("   " + answer.strip().replace("\n", "\n   ")[:900])
print(f"passed {passed} / {len(CASES)}")
