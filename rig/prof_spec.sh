#!/usr/bin/env bash
# Kernel time per speculative-decoding step, through llama-server under rocprof, on an MI50 box.
#   rig/prof_spec.sh MODEL DRAFT [N_MAX] [OUT]    env as rig/spec_bench.sh: BIN, GPUS, SPEC, EXTRA, N, PORT;
#                                                 PROF (extra rocprof flags, e.g. --hip-trace for host API calls)
# Runs one greedy chat completion (the math prompt) and prints the kernels by total time, in ms per verify
# step (steps = drafted tokens / N_MAX). Load-time kernels are included but show up with few calls.
set -euo pipefail
model=$(realpath "$1")
draft=$(realpath "$2")
nmax=${3:-5}
out=$(realpath -m "${4:-$HOME/prof}")
bin=${BIN:-$HOME/mx-llama.cpp/build/bin}
port=${PORT:-8098}
n=${N:-256}
mkdir -p "$out"
cd "$bin"   # the HIP backend is loaded from the working directory
args=(-m "$model" -ngl 99 -fa 1 -lm dio -c 8192 -np 1 --port "$port" --host 127.0.0.1 --no-webui
      -md "$draft" -ngld 99 --spec-type "${SPEC:-draft-dflash}" --spec-draft-n-max "$nmax")
# shellcheck disable=SC2206
args+=(${EXTRA:-})
HIP_VISIBLE_DEVICES=${GPUS:-0} LD_LIBRARY_PATH=$bin:/opt/rocm/lib \
    rocprof --stats --basenames on ${PROF:-} -o "$out/results.csv" ./llama-server "${args[@]}" > "$out/server.log" 2>&1 &
prof=$!
for _ in $(seq 900); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
    kill -0 $prof 2>/dev/null || { tail -30 "$out/server.log"; exit 1; }
    sleep 1
done
prompt="Solve: a train leaves at 3:15 pm travelling 80 km/h; a second train leaves the same station at 4:00 pm at 110 km/h on the same track. When does the second catch up? Show your work."
body=$(python3 -c 'import json,sys; print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": int(sys.argv[2]), "temperature": 0, "cache_prompt": False}))' "$prompt" "$n")
curl -s "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" > "$out/response.json"
server=$(pgrep -x llama-server | head -1)
kill -INT "$server"
wait $prof || true
python3 - "$out" "$nmax" <<'EOF'
import csv, json, sys
out, nmax = sys.argv[1], float(sys.argv[2])
t = json.load(open(out + "/response.json"))["timings"]
steps = t["draft_n"] / nmax
print("decode %.1f t/s, %d tokens, draft %d/%d, %.0f steps, %.1f ms/step wall"
      % (t["predicted_per_second"], t["predicted_n"], t["draft_n_accepted"], t["draft_n"], steps, t["predicted_ms"] / steps))
rows = list(csv.DictReader(open(out + "/results.stats.csv")))
total = sum(float(r["TotalDurationNs"]) for r in rows)
print("all kernels %.1f ms/step (includes prompt and load)" % (total / 1e6 / steps))
for r in sorted(rows, key=lambda r: -float(r["TotalDurationNs"]))[:40]:
    print("%7.2f ms/step %7d calls %8.1f us  %s" % (float(r["TotalDurationNs"]) / 1e6 / steps, int(r["Calls"]),
          float(r["AverageNs"]) / 1e3, r["Name"][:110]))
EOF
