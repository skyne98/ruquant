#!/usr/bin/env python3
"""One line per ggml_bench JSON on stdin: shape/T:us for every point."""
import json
import sys

d = json.load(sys.stdin)
print("  ".join("%s %s/%d:%.0f" % (p["shape"].replace("27b ", "")[:9], p["type"], p["T"], p["us"]) for p in d["points"]))
