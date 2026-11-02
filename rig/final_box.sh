#!/usr/bin/env bash
# Final measurements of a W4A4 target on an MI50 box, with the clips stored in the model (no RQ_ACT_CLIP*):
# quality against the 16-chunk UD-Q8_K_XL reference on GPU 1 while GPU 0 runs llama-bench and the DFlash2
# speculative bench.
#   rig/final_box.sh MODEL DRAFTER [label]
set -uo pipefail
model=$1
draft=$2
label=${3:-$(basename "$model" .gguf)}
cd ~/models
unset RQ_ACT_CLIP RQ_ACT_CLIP_H
(GPUS=1 ~/rig/score_box.sh "$model" "$label" > ~/ref/final-$label.txt 2>&1 &)
bin=~/mx-llama.cpp/build/bin
echo "== $label forward (llama-bench, ms = 1000 * T / t/s)"
HIP_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 LD_LIBRARY_PATH=$bin:/opt/rocm/lib $bin/llama-bench -m "$model" -ngl 99 -fa 1 -lm dio \
    -p 1,6,8 -n 0 -r 10 2>&1 | grep -E "\| +pp" | awk -F"|" '{print $9, $10}'
for nm in 4 5; do
    echo "== $label + $(basename "$draft" .gguf), draft $nm"
    GGML_CUDA_DISABLE_GRAPHS=1 SPEC=draft-dflash N=256 EXTRA="-bs --samplers temperature" ~/rig/spec_bench.sh "$model" "$draft" $nm 2>&1 | tail -5
done
while pgrep -x llama-perplexit > /dev/null; do sleep 10; done  # -x: the process name, never this script
cat ~/ref/final-$label.txt
