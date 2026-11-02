#!/usr/bin/env bash
# Scores a GGUF against the UD-Q8_K_XL reference log-probs with our llama.cpp fork and prints one
# line: perplexity ratio, mean KLD, top-1 agreement. CHUNKS 16 (default) uses the first-16-chunk
# reference (ud-q8_k_xl-16.kld, ~1.5 min), 64 the full one (llama-perplexity scores whatever the
# reference file holds).
#   accuracy/score.sh MODEL.gguf [16|64] [label]       env (e.g. RQ_ACT_CLIP) passes through
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1
chunks=${2:-16}
label=${3:-$(basename "$model" .gguf)}
log=$(mktemp)
LD_LIBRARY_PATH=/run/opengl-driver/lib:${LD_LIBRARY_PATH:-} build-llama/build/bin/llama-perplexity -m "$model" \
    -c 2048 -b 2048 -fa on \
    --kl-divergence-base "accuracy/ref/ud-q8_k_xl$([ "$chunks" = 64 ] || echo "-$chunks").kld" --kl-divergence > "$log" 2>&1 || { tail -5 "$log"; exit 1; }
get() { sed -n '/Perplexity statistics/,$p' "$log" | grep -E "$1" | head -1 | awk -F: '{print $2}' | awk '{print $1}'; }
printf "%-34s chunks %2d  ppl/base %s  KLD %s  top-1 %s %%\n" "$label" "$chunks" \
    "$(get 'Mean PPL\(Q\)/PPL\(base\)')" "$(get 'Mean    KLD')" "$(get 'Same top p')"
rm -f "$log"
