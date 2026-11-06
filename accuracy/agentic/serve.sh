#!/usr/bin/env bash
# llama-server on the 3090 for the agentic runs: our W4A4 model with its DFlash2 drafter (draft 7). The W4A4 matmuls
# run on the INT4 tensor cores (rq-w4a4-tc.cuh).
#   KV=q4 (default): q4_0 K and V; decode and verify (<= 16 query tokens) take rq-fattn.cu, int4 Q x int4 K on the
#                    INT4 tensor cores with the gfx906 dot8 kernel's numerics; prefill llama.cpp's kernels, as on gfx906
#   KV=f16:          the reference cache
#   serve.sh LOG [extra llama-server args]     env KV, CTX (default 65536), PORT (8099), UB (2048), NODRAFT=1
set -euo pipefail
cd "$(dirname "$0")/../.."
log=$1
shift
kv=${KV:-q4}
args=(-m models/qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf -ngl 99 -fa on -c "${CTX:-65536}" -np 1 --jinja
      -b "${UB:-2048}" -ub "${UB:-2048}" --port "${PORT:-8099}" --host 127.0.0.1 --no-webui)
if [ "${NODRAFT:-0}" != 1 ]; then
    args+=(-md models/dspark/dflash2-ost44-h4-l4-c8-g64-rb4.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 7)
fi
if [ "$kv" = q4 ]; then
    args+=(-ctk q4_0 -ctv q4_0)
fi
nohup build-llama/build/bin/llama-server "${args[@]}" "$@" > "$log" 2>&1 < /dev/null &
echo $!
