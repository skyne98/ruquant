#!/usr/bin/env bash
# accuracy/score.sh for an MI50 box: llama-perplexity against the uploaded UD-Q8_K_XL 16-chunk reference.
#   rig/score_box.sh MODEL [label]      env: GPUS (default 1), EXTRA (more llama-perplexity args, e.g. "-sm tensor
#                                       -ub 8"), RQ_ACT_CLIP etc. pass through. Under -sm tensor the prefill all-reduces
#                                       stay f16 unless GGML_NCCL_AR_WIRE says otherwise: perplexity would quantize every
#                                       token's, where serving quantizes only prompts
set -euo pipefail
model=$1
label=${2:-$(basename "$model" .gguf)}
bin=${BIN:-$HOME/mx-llama.cpp/build/bin}
log=$(mktemp)
HIP_VISIBLE_DEVICES=${GPUS:-1} GGML_CUDA_DISABLE_GRAPHS=1 GGML_NCCL_AR_WIRE=${GGML_NCCL_AR_WIRE:-f16} LD_LIBRARY_PATH=$bin:/opt/rocm/lib "$bin/llama-perplexity" -m "$model" \
    -c 2048 -b 2048 -fa on -ngl 99 -lm dio ${EXTRA:-} \
    --kl-divergence-base "$HOME/ref/ud-q8_k_xl-16.kld" --kl-divergence > "$log" 2>&1 || { tail -5 "$log"; exit 1; }
get() { sed -n '/Perplexity statistics/,$p' "$log" | grep -E "$1" | head -1 | awk -F: '{print $2}' | awk '{print $1}'; }
printf "%-40s chunks 16  ppl/base %s  KLD %s  top-1 %s %%\n" "$label" \
    "$(get 'Mean PPL\(Q\)/PPL\(base\)')" "$(get 'Mean    KLD')" "$(get 'Same top p')"
rm -f "$log"
