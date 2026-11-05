#!/usr/bin/env bash
# Fast-iteration proxy for quality work: Qwen3.5-4B, the same hybrid layout as Qwen3.8-27B (3 DeltaNet : 1 attention
# layer, the same head dims, hidden 2560 = 20 x 128 and FFN 9216 = 72 x 128, Paley x Sylvester like 5120 and 17408)
# with tied embeddings, which convert unties. The 27B pipeline at 1/7 the size: unsloth's Q4_0 type map, ssm_out
# Q8_0, RTN -> Hessians from the RTN model -> GPTQ with weight groups; scored against BF16 log-probs (16 chunks).
#   accuracy/proxy.sh TAG [extra convert args, e.g. --ost DIR]
# Writes models/q35-4b-TAG-{rtn,gptq}.gguf and models/hess-q35-4b-TAG/, appends scores to models/scores-4b.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
tag=$1
shift
d=$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523
in=$d/Qwen3.5-4B-BF16.gguf
map=$d/Qwen3.5-4B-Q4_0.gguf
rtn=models/q35-4b-$tag-rtn.gguf
gptq=models/q35-4b-$tag-gptq.gguf
hess=models/hess-q35-4b-$tag
export REF=accuracy/ref/q35-4b-bf16-16.kld
./run.sh convert "$in" "$rtn" --types-from "$map" --set ssm_out=q8_0 "$@" > "models/convert-q35-4b-$tag-rtn.log" 2>&1
tail -1 "models/convert-q35-4b-$tag-rtn.log"
accuracy/score.sh "$rtn" 16 "4b $tag rtn" | tee -a models/scores-4b.txt
./run.sh hessian "$rtn" accuracy/ref/wikitext/wiki.train.raw "$hess" > "models/hessian-q35-4b-$tag.log" 2>&1
tail -1 "models/hessian-q35-4b-$tag.log"
./run.sh convert "$in" "$gptq" --types-from "$map" --set ssm_out=q8_0 --hessians "$hess" --wgroups "$@" \
    > "models/convert-q35-4b-$tag-gptq.log" 2>&1
tail -1 "models/convert-q35-4b-$tag-gptq.log"
accuracy/score.sh "$gptq" 16 "4b $tag gptq" | tee -a models/scores-4b.txt
