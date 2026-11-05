#!/usr/bin/env bash
# The final numbers of a W4A4 target with its DFlash2 drafter on box 1 (4x MI50 32 GB) at one GPU, TP2 and TP4, with
# the runtime settings that won (RESULTS.md):
#   rig/final_mi50.sh MODEL DRAFTER            env BIN (default ~/mx-llama.cpp/build/bin), ROCM_LIB (default
#                                              /opt/rocm/lib), GRAPHS=1 (whole-token HIP graphs and ggml-cuda graphs: the
#                                              ROCm 7.12 default; on ROCm 5.7 leave it off), NMAX (draft length,
#                                              default 7), SCORE=0 (skip the KLD runs, e.g. for a baseline model),
#                                              DEVD (drafter placement under TP, default "-devd ROCm0"; DEVD= replicates
#                                              it, which a drafter without its own lm_head needs, e.g. incoai's)
#   ROCm 7.12: BIN=~/mx-llama.cpp/build-r712/bin ROCM_LIB=~/therock-7.12/lib GRAPHS=1 rig/final_mi50.sh MODEL DRAFTER
# - llama-bench, 3 repetitions: prefill pp512 (ubatch 512) and pp2048 (ubatch 2048), plain decode tg128; the 8-token
#   forward a verify step runs (pp8) as the median of repetitions 11-30, once any graph is recorded.
# - DFlash2 decoding: draft NMAX, rig/acc_prompts.json (30 prompts x 256 tokens, greedy), the drafter on GPU 0.
# - Quality: KLD / top-1 against the 16-chunk UD-Q8_K_XL reference (~/ref/ud-q8_k_xl-16.kld), one GPU and TP4 (the
#   TP4 prefill all-reduces in f16, see rig/score_box.sh).
set -uo pipefail
model=$1
draft=$2
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-$HOME/mx-llama.cpp/build/bin}
export LD_LIBRARY_PATH=$bin:${ROCM_LIB:-/opt/rocm/lib} HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_SPIN=2000000
if [ "${GRAPHS:-0}" = 1 ]; then
    unset GGML_CUDA_DISABLE_GRAPHS
    export GGML_META_TOKEN_GRAPH=1
else
    export GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_TOKEN_GRAPH=0
fi
unset RQ_ACT_CLIP RQ_ACT_CLIP_H
show() {
    python3 -c 'import sys, json
for l in sys.stdin:
    r = json.loads(l)
    name = "pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"]
    extra = "  (%.1f ms per forward)" % (1000 * r["n_prompt"] / r["avg_ts"]) if r["n_prompt"] == 8 else ""
    print("  %-7s %8.1f t/s  +- %.1f%s" % (name, r["avg_ts"], r["stddev_ts"], extra), flush=True)'
}
echo "model $(basename "$model"), drafter $(basename "$draft"), draft ${NMAX:-7}, $(date -u +%F)"
for g in 0 0,1 0,1,2,3; do
    sm=
    devd=
    if [ $g != 0 ]; then
        sm="-sm tensor"
        devd=${DEVD--devd ROCm0}
    fi
    echo "== GPUs $g"
    HIP_VISIBLE_DEVICES=$g "$bin/llama-bench" -m "$model" -ngl 99 -fa 1 -lm dio $sm -p 512 -n 128 -r 3 -o jsonl 2>/dev/null | show
    HIP_VISIBLE_DEVICES=$g "$bin/llama-bench" -m "$model" -ngl 99 -fa 1 -lm dio $sm -p 8 -n 0 -r 30 -o jsonl 2>/dev/null |
        python3 -c 'import sys, json
for l in sys.stdin:
    r = json.loads(l); ms = sorted(8000 / x for x in r["samples_ts"][10:])
    print("  pp8     %8.1f t/s  (%.2f ms per forward, median of repetitions 11-30)" % (8000 / ms[10], ms[10]), flush=True)'
    HIP_VISIBLE_DEVICES=$g "$bin/llama-bench" -m "$model" -ngl 99 -fa 1 -lm dio $sm -p 2048 -n 0 -b 2048 -ub 2048 -r 3 \
        -o jsonl 2>/dev/null | show
    BIN=$bin ROCM_LIB=${ROCM_LIB:-/opt/rocm/lib} GPUS=$g EXTRA="$sm $devd" python3 "$here/acc_bench.py" "$model" "$draft" "${NMAX:-7}" 2>&1 | grep -E "^(math|code|chat|all) |rror"
done
[ "${SCORE:-1}" = 0 ] && exit 0
echo "== quality, 16 chunks"
BIN=$bin ROCM_LIB=${ROCM_LIB:-/opt/rocm/lib} GPUS=0 "$here/score_box.sh" "$model" "one GPU"
BIN=$bin ROCM_LIB=${ROCM_LIB:-/opt/rocm/lib} GPUS=0,1,2,3 EXTRA="-sm tensor" "$here/score_box.sh" "$model" "TP4"
