#!/usr/bin/env python3
"""A logging proxy for an OpenAI-compatible server: forwards every request, streams the answer through, and writes
each exchange to OUT_DIR/NNNN.json (the request body, and the streamed answer reassembled into a message).

    python3 log_proxy.py LISTEN_PORT UPSTREAM OUT_DIR       e.g. 8100 http://127.0.0.1:8099 /tmp/exchanges

Then compare_turns() pairs each answer with the assistant message the client sends back in the next request.
"""
import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

lock = threading.Lock()
counter = [0]


def reassemble(raw: bytes) -> dict:
    msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            ch = json.loads(line[6:])["choices"][0]
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        d = ch.get("delta", {})
        msg["content"] += d.get("content") or ""
        msg["reasoning_content"] += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            t = msg["tool_calls"].setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
            t["id"] = tc.get("id") or t["id"]
            f = tc.get("function") or {}
            t["name"] += f.get("name") or ""
            t["arguments"] += f.get("arguments") or ""
    msg["tool_calls"] = [msg["tool_calls"][k] for k in sorted(msg["tool_calls"])]
    return msg


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.forward(None)

    def do_POST(self):
        self.forward(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def forward(self, body):
        print(self.command, self.path, self.request_version, len(body or b""), flush=True)
        req = urllib.request.Request(UPSTREAM + self.path, body, {k: v for k, v in self.headers.items()
                                                                  if k.lower() not in ("host", "content-length")},
                                     method=self.command)
        with urllib.request.urlopen(req, timeout=7200) as r:
            self.send_response(r.status)
            for k, v in r.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = []
            while True:
                c = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not c:
                    break
                chunks.append(c)
                self.wfile.write(c)
                self.wfile.flush()
        if body and self.path.endswith("/chat/completions"):
            raw = b"".join(chunks)
            with lock:
                counter[0] += 1
                n = counter[0]
            try:
                answer = reassemble(raw) if b"data: " in raw else json.loads(raw)["choices"][0]["message"]
            except Exception as e:  # keep the raw bytes when the answer can't be parsed
                answer = {"error": str(e), "raw": raw.decode(errors="replace")[:20000]}
            with open(os.path.join(OUT, f"{n:04d}.json"), "w") as f:
                json.dump({"request": json.loads(body), "answer": answer}, f)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port, UPSTREAM, OUT = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
