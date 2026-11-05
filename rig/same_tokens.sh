#!/bin/bash
# Greedy completion through llama-server; prints a hash of the generated text, to check that a change keeps the
# output byte for byte (env passes through, e.g. RQ_FUSE_HADAMARD=0; BIN and ROCM_LIB pick the build):
#   rig/same_tokens.sh MODEL GPUS "EXTRA"      e.g. rig/same_tokens.sh ~/models/m.gguf 0,1,2,3 "-sm tensor"
bin=${BIN:-$HOME/mx-llama.cpp/build/bin}
export LD_LIBRARY_PATH=$bin:${ROCM_LIB:-/opt/rocm/lib}
log=$(mktemp)
HIP_VISIBLE_DEVICES=$2 "$bin/llama-server" -m "$1" -ngl 99 -fa 1 -c 4096 -np 1 --port 8095 --host 127.0.0.1 --no-webui \
    --cache-ram 0 $3 > "$log" 2>&1 &
pid=$!
for i in $(seq 300); do curl -sf http://127.0.0.1:8095/health > /dev/null 2>&1 && break; sleep 1; done
curl -s http://127.0.0.1:8095/completion -H 'Content-Type: application/json' -d '{"prompt": "Explain how a transistor works, step by step.", "n_predict": 160, "temperature": 0, "cache_prompt": false}' |
    python3 -c 'import json, sys, hashlib; c = json.load(sys.stdin)["content"]; print(hashlib.md5(c.encode()).hexdigest(), repr(c[:60]))'
kill $pid
wait $pid 2> /dev/null
rm -f "$log"
