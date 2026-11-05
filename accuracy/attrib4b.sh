#!/usr/bin/env bash
# Error map of the Qwen3.5-4B proxy (accuracy/proxy.sh TAG first): for each class of linears, the score with only its
# activations unquantized (ruquant a16: its own int4 weights at F16) and with the class kept at Q8_0 (rotated, no
# quantization error at all). The gap to the base is that class's share of the damage; a16 / keep split it into
# activation and weight error.
#   accuracy/attrib4b.sh TAG      appends to models/scores-4b.txt
set -euo pipefail
cd "$(dirname "$0")/.."
tag=$1
d=$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523
base=models/q35-4b-$tag-gptq.gguf
v=models/q35-4b-variant.gguf
export REF=accuracy/ref/q35-4b-bf16-16.kld
score() { accuracy/score.sh "$v" 16 "4b $tag $1" | tee -a models/scores-4b.txt; }
./run.sh a16 "$base" "$v" > /dev/null && score "a16 all"
while read -r label names; do
    ./run.sh a16 "$base" "$v" --only "$names" > /dev/null && score "a16 $label"
    ./run.sh convert "$d/Qwen3.5-4B-BF16.gguf" "$v" --types-from "$d/Qwen3.5-4B-Q4_0.gguf" --set ssm_out=q8_0 \
        --hessians "models/hess-q35-4b-$tag" --wgroups --keep "$names" > /dev/null && score "keep $label"
done <<'LIST'
ffn_down       ffn_down.
ffn_gate+up    ffn_gate.,ffn_up.
dn_in          attn_qkv.,attn_gate.
attn_in        attn_q.,attn_k.,attn_v.
attn_out       attn_output.
first4         blk.0.,blk.1.,blk.2.,blk.3.
last4          blk.28.,blk.29.,blk.30.,blk.31.
LIST
rm -f "$v"
