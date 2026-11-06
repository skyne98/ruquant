#!/usr/bin/env python3
"""Why a llama-server conversation re-processes a long answer, and whether generation checkpoints fix it.

    python3 cache_roundtrip.py [URL]          (default http://127.0.0.1:8099)

1. Tokenization round trip: /completion with return_tokens, sampled as an agent client would (temperature 1, top-k
   20, top-p 0.95), then the returned text re-tokenized. Where the tokens differ, a client sending the text back can
   never match the cache past that point.
2. Agent turn: a chat request that writes a long file through a tool call (reasoning on), then the next request with
   that assistant message and a tool result, as an agent client sends it. Reports how much of the conversation the
   server re-processed (timings.prompt_n), first with the message as returned, then with one character changed in
   the middle of the answer, which forces a divergence there. Without generation checkpoints the server rolls back
   to the start of the answer; with them (LLAMA_GEN_CKPT_STEP) to the last checkpoint before the change.
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": 7}


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def roundtrip():
    prompt = post("/apply-template", {"messages": [{"role": "user", "content":
        "Write a detailed technical essay (about 2500 words) on how B-trees work, with Python code examples."}]})["prompt"]
    r = post("/completion", {"prompt": prompt, "n_predict": 4096, "return_tokens": True, **SAMPLING})
    gen = r["tokens"]
    text = r["content"]
    re_tok = post("/tokenize", {"content": text, "parse_special": True})["tokens"]
    first = next((i for i, (a, b) in enumerate(zip(gen, re_tok)) if a != b), None)
    print(f"round trip: {len(gen)} sampled tokens, {len(re_tok)} after re-tokenizing the text; "
          + ("identical" if first is None and len(gen) == len(re_tok) else f"first difference at token {first}"))
    if first is not None:
        ctx = lambda toks: post("/detokenize", {"tokens": toks[max(0, first - 3):first + 3]})["content"]
        print(f"  sampled {gen[first - 3:first + 3]} {ctx(gen)!r}\n  re-tokenized {re_tok[first - 3:first + 3]} {ctx(re_tok)!r}")


TOOLS = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]


def agent_turn():
    msgs = [{"role": "system", "content": "You are a coding agent. Use the write tool to create files."},
            {"role": "user", "content": "Create inventory.py: an inventory module (products, stock movements, a CSV "
                                        "export, a report function and a small CLI, with docstrings), about 300 lines. Don't "
                                        "plan at length: write it right away, in one write call."}]
    r = post("/v1/chat/completions", {"messages": msgs, "tools": TOOLS, "max_tokens": 16384, **SAMPLING})
    m = r["choices"][0]["message"]
    if not m.get("tool_calls"):  # thought past the limit: the same request with a reasoning budget
        print(f"  no tool call after {r['timings']['predicted_n']} tokens; again with thinking_budget_tokens 2048")
        r = post("/v1/chat/completions", {"messages": msgs, "tools": TOOLS, "max_tokens": 16384,
                                          "thinking_budget_tokens": 2048, **SAMPLING})
        m = r["choices"][0]["message"]
    t = r["timings"]
    print(f"answer: {t['predicted_n']} tokens generated after {t['prompt_n']} prompt tokens "
          f"(reasoning {len(m.get('reasoning_content') or '')} chars, tool calls {len(m.get('tool_calls') or [])})")
    if not m.get("tool_calls"):
        print("  no tool call; skipping the follow-up")
        return
    assistant = {"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]}
    if m.get("reasoning_content"):
        assistant["reasoning_content"] = m["reasoning_content"]
    tool = {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"], "content": "wrote inventory.py"}
    follow = {"role": "user", "content": "Thanks. In one sentence: what does the report generator output?"}
    for label, a in [("as returned", assistant), ("one character changed at 85% of the file", changed(assistant))]:
        r2 = post("/v1/chat/completions", {"messages": msgs + [a, tool, follow], "tools": TOOLS, "max_tokens": 64,
                                           **SAMPLING})
        t2 = r2["timings"]
        print(f"follow-up, {label}: re-processed {t2['prompt_n']} tokens (cache hit {t2.get('cache_n')}), "
              f"{t2['prompt_ms'] / 1000:.1f} s")
        # restore the cache to the unchanged conversation before the next variant
        post("/v1/chat/completions", {"messages": msgs + [assistant, tool, follow], "tools": TOOLS, "max_tokens": 1})


def changed(assistant):
    a = json.loads(json.dumps(assistant))
    args = json.loads(a["tool_calls"][0]["function"]["arguments"])
    c = args["content"]
    i = len(c) * 85 // 100
    args["content"] = c[:i] + ("#" if c[i] != "#" else "@") + c[i + 1:]
    a["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    return a


if __name__ == "__main__":
    roundtrip()
    agent_turn()
