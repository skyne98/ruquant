#!/usr/bin/env bash
# After accuracy/ostquant_qwen.py: fold its R1 + smoothing scales into unsloth's Q4_0 type map, GPTQ against
# Hessians from the folded model itself, and score.
#   accuracy/ost_pipeline.sh OSTDIR TAG      e.g. models/ost ost100
# Writes models/qwen3.8-27b-TAG-{rtn,gptq}.gguf, models/hess-TAG/, appends scores to models/scores.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
ost=$1
tag=$2
ud=$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502
in=$ud/Qwen3.8-27B-UD-Q8_K_XL.gguf
map=$(ls "$ud"/Qwen3.8-27B-Q4_0.gguf 2>/dev/null || ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-Q4_0.gguf | head -1)
rtn=models/qwen3.8-27b-$tag-rtn.gguf
gptq=models/qwen3.8-27b-$tag-gptq.gguf
hess=models/hess-$tag

./run.sh convert "$in" "$rtn" --ost "$ost" --types-from "$map" > "models/convert-$tag-rtn.log" 2>&1
tail -1 "models/convert-$tag-rtn.log"
for clip in 1.0 0.9; do
    RQ_ACT_CLIP=$clip accuracy/score.sh "$rtn" 16 "$tag rtn clip $clip" | tee -a models/scores.txt
done
RQ_ACT_CLIP=${HESS_CLIP:-1.0} ./run.sh hessian "$rtn" accuracy/ref/wikitext/wiki.train.raw "$hess" > "models/hessian-$tag.log" 2>&1
tail -1 "models/hessian-$tag.log"
./run.sh convert "$in" "$gptq" --ost "$ost" --types-from "$map" --hessians "$hess" > "models/convert-$tag-gptq.log" 2>&1
tail -1 "models/convert-$tag-gptq.log"
for clip in 1.0 0.9 0.8; do
    RQ_ACT_CLIP=$clip accuracy/score.sh "$gptq" 16 "$tag gptq clip $clip" | tee -a models/scores.txt
done
