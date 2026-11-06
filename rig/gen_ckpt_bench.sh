#!/usr/bin/env bash
# Generation checkpoints (LLAMA_GEN_CKPT_STEP, the fork's server) on a box, recorded in ~/profiles/<time>-gen-ckpt:
#   speed     decode t/s of a greedy 8192-token generation, twice, with checkpoints off (0) and every 2048 tokens
#   reprefill the tokens the server re-processes after a long agent answer (cache_roundtrip.py), off vs on
#   vision    with MMPROJ set: the vision check (vision_check.py) at the last GPU set
# for each GPU set in GPU_SETS (default "0,1 0,1,2,3": TP2 like the community report, TP4), the published server
# command (-sm tensor -devd ROCm0, DFlash2 draft 7, -bs --samplers temperature) at -c 65536.
#   bash rig/gen_ckpt_bench.sh MODEL DRAFT      env BIN (default ~/mx-llama.cpp/build-ckpt/bin), ROCM_LIB, TESTS
#                                               (default ~/rq-test), GPU_SETS, STEPS (default "0 2048"), MMPROJ
set -uo pipefail
model=$1
draft=$2
bin=${BIN:-$HOME/mx-llama.cpp/build-ckpt/bin}
rl=${ROCM_LIB:-$HOME/therock-7.12/lib}
tests=${TESTS:-$HOME/rq-test}
d=$HOME/profiles/$(date -u +%Y%m%dT%H%M%SZ)-gen-ckpt
mkdir -p "$d"
cp "$0" "$d/"
{
    echo "label: gen-ckpt"
    echo "time: $(date -u +%FT%TZ)"
    echo "binary: $bin (libggml-hip.so md5 $(md5sum < "$bin/libggml-hip.so" | cut -c1-32), llama-server md5 $(md5sum < "$bin/llama-server" | cut -c1-32))"
    echo "model: $model"
    echo "drafter: $draft"
    echo "mmproj: ${MMPROJ:-none}"
    echo "gpu sets: ${GPU_SETS:-0,1 0,1,2,3}; steps: ${STEPS:-0 2048}"
} > "$d/meta.txt"
res=$d/results.txt
port=8097
up() {
    for _ in $(seq 1 300); do
        curl -s -m 2 127.0.0.1:$port/health | grep -q ok && return 0
        kill -0 "$1" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}
serve() { # serve LOG GPUS STEP [extra args]
    local log=$1 gpus=$2 step=$3
    shift 3
    HIP_VISIBLE_DEVICES=$gpus LLAMA_GEN_CKPT_STEP=$step LD_LIBRARY_PATH=$bin:$rl HSA_FORCE_FINE_GRAIN_PCIE=1 \
        GGML_META_TOKEN_GRAPH=1 GGML_META_SPIN=2000000 nohup "$bin/llama-server" -m "$model" -md "$draft" -ngl 99 -ngld 99 \
        -fa 1 -sm tensor -devd ROCm0 --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature -c 65536 \
        --host 127.0.0.1 --port $port "$@" > "$log" 2>&1 < /dev/null &
    echo $!
}
last=""
for gpus in ${GPU_SETS:-0,1 0,1,2,3}; do
    last=$gpus
    for step in ${STEPS:-0 2048}; do
        log=$d/server-gpus${gpus//,/}-step$step.log
        pid=$(serve "$log" "$gpus" "$step")
        if ! up "$pid"; then
            echo "== GPUs $gpus, step $step: server did not start" | tee -a "$res"
            tail -5 "$log" >> "$res"
            kill "$pid" 2>/dev/null
            continue
        fi
        echo "== GPUs $gpus, LLAMA_GEN_CKPT_STEP=$step" | tee -a "$res"
        python3 "$tests/accuracy/agentic/gen_speed.py" http://127.0.0.1:$port 8192 2 2>&1 | tee -a "$res"
        python3 "$tests/accuracy/agentic/cache_roundtrip.py" http://127.0.0.1:$port 2>&1 | tee -a "$res"
        kill "$pid"
        sleep 8
    done
done
if [ -n "${MMPROJ:-}" ]; then
    log=$d/server-vision.log
    pid=$(serve "$log" "$last" 2048 --mmproj "$MMPROJ")
    if up "$pid"; then
        echo "== vision, GPUs $last, $MMPROJ" | tee -a "$res"
        python3 "$tests/accuracy/vision/vision_check.py" http://127.0.0.1:$port 2>&1 | tee -a "$res"
    else
        echo "== vision: server did not start" | tee -a "$res"
        tail -5 "$log" >> "$res"
    fi
    kill "$pid" 2>/dev/null
fi
echo "exit: 0" >> "$d/meta.txt"
echo "$d"
