#!/usr/bin/env python3
"""Does a chat turn survive the server's parse -> client -> chat template round trip byte for byte?

    python3 template_roundtrip.py [URL]

Greedy, so /completion on the rendered prompt and /v1/chat/completions on the messages produce the same answer: the
first gives the raw text the model generated (reasoning and tool call markup included), the second the parsed message
a client stores and sends back. The next prompt, rendered by /apply-template with that message and a tool result,
should start with prompt + raw text; the first byte where it does not is where the server's cache match ends.
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
TOOLS = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


msgs = [{"role": "system", "content": "You are a coding agent. Use the write tool to create files."},
        {"role": "user", "content": "Create fizz.py with a fizzbuzz function and a main that prints 1..20. Think briefly "
                                    "first, then make one write call."}]
prompt = post("/apply-template", {"messages": msgs, "tools": TOOLS})["prompt"]
raw = post("/completion", {"prompt": prompt, "n_predict": 3000, "temperature": 0, "cache_prompt": False})["content"]
r = post("/v1/chat/completions", {"messages": msgs, "tools": TOOLS, "max_tokens": 3000, "temperature": 0})
m = r["choices"][0]["message"]
assistant = {"role": "assistant", "content": m.get("content") or "", "tool_calls": m.get("tool_calls") or []}
if m.get("reasoning_content"):
    assistant["reasoning_content"] = m["reasoning_content"]
nxt = msgs + [assistant]
if assistant["tool_calls"]:
    nxt.append({"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "ok"})
nxt.append({"role": "user", "content": "Thanks."})
again = post("/apply-template", {"messages": nxt, "tools": TOOLS})["prompt"]
want = prompt + raw
i = next((k for k, (a, b) in enumerate(zip(want, again)) if a != b), min(len(want), len(again)))
print(f"raw answer {len(raw)} chars; parsed: reasoning {len(m.get('reasoning_content') or '')} chars, content "
      f"{len(m.get('content') or '')} chars, {len(assistant['tool_calls'])} tool call(s)")
if i >= len(want):
    print("the next prompt starts with prompt + raw answer: the cache can match all of it")
else:
    off = i - len(prompt)
    print(f"first difference {off} chars into the answer ({off / max(1, len(raw)):.0%}):")
    print(f"  generated:   {want[max(len(prompt), i - 80):i + 80]!r}")
    print(f"  re-rendered: {again[max(len(prompt), i - 80):i + 80]!r}")
