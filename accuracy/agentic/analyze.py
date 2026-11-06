#!/usr/bin/env python3
"""Scores the agentic runs and looks for garbage in what the model wrote.

    python3 analyze.py LABEL [LABEL ...] [--detail]

For every run in accuracy/results/agentic/LABEL/: hidden and suite test results, steps, tool calls and tool errors,
tokens, the largest context a request reached, and DFlash2 acceptance (from the llama-server log slice). Then every
reasoning and text part of the exported session is checked for:
  loop      a line of 20+ characters 4+ times in a row, a paragraph of 200+ characters 3+ times, a span of 20-200
            characters repeated 4+ times in a row, or a part over 2000 characters that zlib compresses below 12%
  script    characters outside Latin / common punctuation / box drawing / emoji: CJK, Cyrillic, Arabic, ... (Qwen's
            failure mode is drifting into Chinese)
  junk      U+FFFD, control characters, or chat-template tokens (<|im_start|>, <tool_call>, </think>) leaking into
            text, which means the output parser saw malformed structure
  cutoff    a step that ended because it hit the output token limit
Tool errors are split into argument errors (the call itself was malformed: bad JSON, unknown tool, schema violation)
and ordinary failures (a command failed, an edit's old string wasn't found).
"""
import argparse
import gzip
import json
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "agentic"
TEMPLATE_TOKENS = re.compile(r"<\|im_(start|end)\|>|<\|endoftext\|>|</?tool_call>|</?think>|<\|[a-z_]+\|>")
REPEAT_SPAN = re.compile(r"(.{20,200}?)\1{3,}", re.S)
ARG_ERROR = re.compile(r"invalid (arguments|input|parameters)|unknown tool|not a valid|schema|JSON|"
                       r"expected (string|object|number)|required property|Unrecognized", re.I)


def foreign_chars(text: str) -> Counter:
    out = Counter()
    for ch in text:
        o = ord(ch)
        if o < 0x250 or 0x2000 <= o < 0x2c00 or 0x1f000 <= o < 0x1fb00 or 0xfe00 <= o < 0xfe10:
            continue  # Latin, punctuation, arrows, math, box drawing, emoji, variation selectors
        try:
            out[unicodedata.name(ch).split()[0]] += 1
        except ValueError:
            out["UNNAMED"] += 1
    return out


def check_text(text: str) -> list[str]:
    flags = []
    # a line of 20+ characters 4+ times in a row (blank lines skipped); the same line scattered through code or tests
    # (setup calls, separator comments) is normal and not counted
    run, prev, worst = 0, None, (0, "")
    for l in (l.strip() for l in text.splitlines()):
        if not l:
            continue
        run = run + 1 if l == prev else 1
        prev = l
        if len(l) >= 20 and run > worst[0]:
            worst = (run, l)
    if worst[0] >= 4:
        flags.append(f"loop: line x{worst[0]} in a row: {worst[1][:80]!r}")
    paras = Counter(p.strip() for p in text.split("\n\n") if len(p.strip()) >= 200)
    if paras and (top := paras.most_common(1)[0])[1] >= 3:
        flags.append(f"loop: paragraph x{top[1]}: {top[0][:80]!r}")
    if len(text) < 200_000 and (m := REPEAT_SPAN.search(text)):
        flags.append(f"loop: span x{len(m.group(0)) // len(m.group(1))}: {m.group(1)[:80]!r}")
    if len(text) > 2000 and (r := len(zlib.compress(text.encode())) / len(text.encode())) < 0.12:
        flags.append(f"loop: compresses to {r:.0%}")
    if (f := foreign_chars(text)) and sum(f.values()) >= 3:
        flags.append(f"script: {dict(f.most_common(4))}")
    if "�" in text or any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text):
        flags.append("junk: replacement or control characters")
    if m := TEMPLATE_TOKENS.search(text):
        flags.append(f"junk: template token {m.group(0)!r}")
    return flags


def server_stats(log: Path) -> dict:
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    acc = [(int(a), int(g)) for a, g in re.findall(r"draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated", text)]
    ctx = [int(x) for x in re.findall(r"stop processing: n_tokens = (\d+)", text)]
    pp = [(int(n), float(ms)) for ms, n in re.findall(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", text)]
    tg = [(int(n), float(ms)) for ms, n in re.findall(r"\n[^\n]*\seval time =\s+([\d.]+) ms /\s+(\d+) tokens", text)
          if True]
    return {
        "requests": len(ctx),
        "max_ctx": max(ctx, default=0),
        "acceptance": round(sum(a for a, _ in acc) / max(1, sum(g for _, g in acc)), 3),
        "prompt_tokens": sum(n for n, _ in pp),
        "prompt_s": round(sum(ms for _, ms in pp) / 1000, 1),
        "gen_tokens": sum(n for n, _ in tg),
        "gen_s": round(sum(ms for _, ms in tg) / 1000, 1),
    }


def analyze_run(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    r = {"run": d.name, "task": meta["task"], "hidden": meta["hidden"], "suite": meta["suite"],
         "turn_status": [t["status"] for t in meta["turns"]], "seconds": meta["seconds"],
         "stage_hidden": [t["hidden"] for t in meta["turns"] if "hidden" in t]}
    flags, tools, tool_err, arg_err, cutoffs = [], Counter(), [], [], 0
    steps = reasoning_chars = text_chars = 0
    sess = d / "session.json"
    if sess.exists() and sess.stat().st_size:
        s = json.loads(sess.read_text())
        r["tokens"] = s["info"].get("tokens")
        for m in s["messages"]:
            if m.get("type") != "assistant":
                continue
            steps += 1
            if m.get("finish") == "length" or m.get("error"):
                pass  # cutoffs are counted from the event streams below
                if m.get("error"):
                    flags.append(("error", str(m["error"])[:200]))
            for p in m.get("content", []):
                if p["type"] in ("reasoning", "text"):
                    if p["type"] == "reasoning":
                        reasoning_chars += len(p["text"])
                    else:
                        text_chars += len(p["text"])
                    flags += [(p["type"], f) for f in check_text(p["text"])]
                elif p["type"] == "tool":
                    tools[p["name"]] += 1
                    st = p.get("state", {})
                    if st.get("status") == "error":
                        msg = json.dumps(st.get("error") or st.get("content") or st)[:300]
                        (arg_err if ARG_ERROR.search(msg) else tool_err).append(f"{p['name']}: {msg}")
                    for v in (st.get("input") or {}).values():
                        if isinstance(v, str) and len(v) > 40:  # file contents, edits, commands
                            flags += [(f"tool:{p['name']}", f) for f in check_text(v) if not f.startswith("script")]
    for k in range(1, len(meta["turns"]) + 1):
        ev = d / f"events-{k}.jsonl.gz"
        if ev.exists():
            for line in gzip.open(ev, "rt"):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") == "step_finish" and e["part"].get("reason") == "length":
                    cutoffs += 1
                if e.get("type") == "error":
                    flags.append(("error", json.dumps(e.get("error") or e)[:200]))
    r.update(steps=steps, tools=dict(tools), tool_errors=tool_err, arg_errors=arg_err, cutoffs=cutoffs, flags=flags,
             reasoning_chars=reasoning_chars, text_chars=text_chars, server=server_stats(d / "server.log"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    for label in a.labels:
        runs = [analyze_run(d) for d in sorted((RESULTS / label).iterdir()) if (d / "meta.json").exists()]
        (RESULTS / label / "analysis.json").write_text(json.dumps(runs, indent=1))
        print(f"\n== {label}: {len(runs)} runs")
        print(f"{'run':24} {'hidden':>9} {'suite':>5} {'steps':>5} {'tools':>5} {'t.err':>5} {'arg':>3} "
              f"{'cut':>3} {'flags':>5} {'max ctx':>7} {'acc':>5} {'min':>5}")
        for r in runs:
            h = r["hidden"]
            print(f"{r['run']:24} {h['passed']:>4}/{h['passed'] + h['failed']:<4} {'ok' if r['suite']['ok'] else 'FAIL':>5} "
                  f"{r['steps']:>5} {sum(r['tools'].values()):>5} {len(r['tool_errors']):>5} {len(r['arg_errors']):>3} "
                  f"{r['cutoffs']:>3} {len(r['flags']):>5} {r['server'].get('max_ctx', 0):>7} "
                  f"{r['server'].get('acceptance', 0):>5} {r['seconds'] / 60:>5.1f}"
                  + ("" if all(s == "ok" for s in r["turn_status"]) else f"  turns {r['turn_status']}"))
            if a.detail:
                for kind, f in r["flags"]:
                    print(f"    [{kind}] {f}")
                for e in r["arg_errors"]:
                    print(f"    [arg error] {e}")
        n = len(runs)
        if n:
            ok = sum(r["hidden"]["ok"] for r in runs)
            frac = sum(r["hidden"]["passed"] / max(1, r["hidden"]["passed"] + r["hidden"]["failed"]) for r in runs) / n
            print(f"all hidden tests pass: {ok}/{n}; mean hidden pass rate {frac:.1%}; suite ok "
                  f"{sum(r['suite']['ok'] for r in runs)}/{n}; flagged parts {sum(len(r['flags']) for r in runs)}; "
                  f"argument errors {sum(len(r['arg_errors']) for r in runs)} of {sum(sum(r['tools'].values()) for r in runs)} "
                  f"tool calls; cutoffs {sum(r['cutoffs'] for r in runs)}")


if __name__ == "__main__":
    main()
