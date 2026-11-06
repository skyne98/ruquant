#!/usr/bin/env bash
# The server's text-level cache match (LLAMA_TEXT_MATCH) on a box: split_token_test.py with it off and on, TP4 published
# command, recorded in ~/profiles/<time>-text-match.
#   bash rig/text_match_bench.sh MODEL DRAFT      env BIN (default ~/mx-llama.cpp/build-tm/bin), ROCM_LIB, TESTS, GPUS
set -uo pipefail
model=$1
draft=$2
bin=${BIN:-$HOME/mx-llama.cpp/build-tm/bin}
rl=${ROCM_LIB:-$HOME/therock-7.12/lib}
tests=${TESTS:-$HOME/rq-test}
d=$HOME/profiles/$(date -u +%Y%m%dT%H%M%SZ)-text-match
mkdir -p "$d"
cp "$0" "$d/"
echo "label: text-match; binary: $bin (libllama-server-impl.so md5 $(md5sum < "$bin/libllama-server-impl.so" | cut -c1-32)); gpus ${GPUS:-0,1,2,3}" > "$d/meta.txt"
port=8096
for tm in 0 1; do
    HIP_VISIBLE_DEVICES=${GPUS:-0,1,2,3} LLAMA_TEXT_MATCH=$tm LD_LIBRARY_PATH=$bin:$rl HSA_FORCE_FINE_GRAIN_PCIE=1 \
        GGML_META_TOKEN_GRAPH=1 GGML_META_SPIN=2000000 nohup "$bin/llama-server" -m "$model" -md "$draft" -ngl 99 -ngld 99 \
        -fa 1 -sm tensor -devd ROCm0 --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature -c 65536 \
        --host 127.0.0.1 --port $port > "$d/server-tm$tm.log" 2>&1 < /dev/null &
    pid=$!
    for _ in $(seq 1 300); do curl -s -m 2 127.0.0.1:$port/health | grep -q ok && break; sleep 2; done
    echo "== LLAMA_TEXT_MATCH=$tm" | tee -a "$d/results.txt"
    python3 "$tests/accuracy/agentic/split_token_test.py" http://127.0.0.1:$port 2>&1 | tail -1 | tee -a "$d/results.txt"
    kill $pid
    sleep 10
done
echo "exit: 0" >> "$d/meta.txt"
echo "$d"
