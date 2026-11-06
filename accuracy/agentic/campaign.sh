#!/usr/bin/env bash
# The agentic runs on the 3090: every task REPS times with the q4_0 cache (INT4 tensor-core attention, DFlash2), then
# with the f16 cache (no drafter: an f16 cache at 64k and the drafter don't fit 24 GB together; with sampling,
# speculative decoding doesn't change the output distribution). Results in accuracy/results/agentic/{q4tc,f16}.
#   campaign.sh SCRATCH [REPS]      env AGENTIC_ENV (the sandbox toolchain), SKIP_Q4=1
set -uo pipefail
cd "$(dirname "$0")"
S=$1
reps=${2:-2}
export AGENTIC_SCRATCH=$S/runs
wait_up() {
    for _ in $(seq 1 150); do
        curl -s -m 2 127.0.0.1:8099/health | grep -q ok && return 0
        kill -0 "$1" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}
if [ "${SKIP_Q4:-0}" != 1 ]; then
    pid=$(cat "$S/server-q4tc.pid")
    python3 run.py q4tc --reps "$reps" --server-log "$S/server-q4tc.log"
    kill "$pid"; sleep 5
fi
pid=$(KV=f16 NODRAFT=1 UB=1024 ./serve.sh "$S/server-f16.log")
echo "$pid" > "$S/server-f16.pid"
wait_up "$pid" || { echo "f16 server did not start"; tail -20 "$S/server-f16.log"; exit 1; }
python3 run.py f16 --reps "$reps" --server-log "$S/server-f16.log"
kill "$pid"
python3 analyze.py q4tc f16
