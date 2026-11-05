#!/usr/bin/env bash
# Decode speed with and without a speculative drafter, through llama-server, on an MI50 box.
#   rig/spec_bench.sh MODEL [DRAFT] [N_MAX]      DRAFT '' with SPEC=draft-mtp uses the model's MTP layer; env: GPUS (HIP_VISIBLE_DEVICES, default 0), BIN (llama.cpp
#                                                build/bin), EXTRA (more server flags), N (tokens, default 256),
#                                                TEXT (file that gets the generated texts, to compare builds),
#                                                LOG (keeps the server log, e.g. with EXTRA=-lv 4 for acceptance per position)
# Greedy chat completions (RAW=1: raw text completions) of a few fixed prompts; prints prompt and decode t/s and the draft acceptance per prompt,
# then the mean decode t/s.
set -euo pipefail
model=$1
draft=${2:-}
nmax=${3:-7}
bin=${BIN:-$HOME/mx-llama.cpp/build/bin}
port=${PORT:-8099}
n=${N:-256}
args=(-m "$model" -ngl 99 -fa 1 -lm dio -c 8192 -np 1 --port "$port" --host 127.0.0.1 --no-webui)
if [ -n "$draft" ]; then
    args+=(-md "$draft" -ngld 99 --spec-type "${SPEC:-draft-dspark}" --spec-draft-n-max "$nmax")
elif [ -n "${SPEC:-}" ]; then # a drafter inside the model (SPEC=draft-mtp)
    args+=(--spec-type "$SPEC" --spec-draft-n-max "$nmax")
fi
# shellcheck disable=SC2206
args+=(${EXTRA:-})
log=$(mktemp)
HIP_VISIBLE_DEVICES=${GPUS:-0} LD_LIBRARY_PATH=$bin:/opt/rocm/lib "$bin/llama-server" "${args[@]}" > "$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; [ -n "${LOG:-}" ] && cp "$log" "$LOG"; rm -f "$log"' EXIT
for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { tail -30 "$log"; exit 1; }
    sleep 1
done
prompts=(
    "Write a Python function that parses a CSV file and returns the average of each numeric column. Explain the code."
    "Explain how a transformer language model generates text, step by step, for a curious high school student."
    "Solve: a train leaves at 3:15 pm travelling 80 km/h; a second train leaves the same station at 4:00 pm at 110 km/h on the same track. When does the second catch up? Show your work."
    "Write a short story about a lighthouse keeper who finds a message in a bottle."
)
sum=0
for p in "${prompts[@]}"; do
    if [ -n "${RAW:-}" ]; then
        endpoint=completion
        body=$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0, "cache_prompt": False}))' "$p" "$n")
    else
        endpoint=v1/chat/completions
        body=$(python3 -c 'import json,sys; print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": int(sys.argv[2]), "temperature": 0, "cache_prompt": False}))' "$p" "$n")
    fi
    curl -s "http://127.0.0.1:$port/$endpoint" -H 'Content-Type: application/json' -d "$body" | python3 -c '
import json, sys
r = json.load(sys.stdin)
t = r["timings"]
NMAX = float(sys.argv[1])
if sys.argv[2]:
    text = r["content"] if "content" in r else r["choices"][0]["message"]  # message keeps reasoning_content too
    open(sys.argv[2], "a").write(json.dumps(text) + "\n")
acc = "  draft %d/%d" % (t.get("draft_n_accepted", 0), t["draft_n"]) if t.get("draft_n") else ""
steps = t["draft_n"] / NMAX if t.get("draft_n") else t["predicted_n"]
print("prompt %7.1f t/s  decode %6.1f t/s (%d tokens)%s  %.1f ms/step" % (t["prompt_per_second"], t["predicted_per_second"], t["predicted_n"], acc, t["predicted_ms"] / steps))
sys.stderr.write("%f\n" % t["predicted_per_second"])' "$nmax" "${TEXT:-}" 2>> "$log.tps"
done
python3 -c 'import sys; v=[float(x) for x in open(sys.argv[1])]; print("mean decode %.1f t/s" % (sum(v)/len(v)))' "$log.tps"
rm -f "$log.tps"
