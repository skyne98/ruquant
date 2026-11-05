#!/usr/bin/env bash
# The 4B proxy's error map (accuracy/attrib4b.sh) on Qwen3.8-27B: per class of linears, the score with only its
# activations unquantized (ruquant a16, Q8_0) and with the class taken from the all-Q8_0 donor (ruquant graft).
# Variants bigger than the 3090 run with NGL layers on the GPU.
#   accuracy/attrib27b.sh BASE.gguf DONOR.gguf     appends to models/scores.txt
set -euo pipefail
cd "$(dirname "$0")/.."
base=$1
donor=$2
v=models/q38-27b-variant.gguf
export NGL=${NGL:-44}
score() { accuracy/score.sh "$v" 16 "27b $1" | tee -a models/scores.txt; rm -f "$v"; }
while read -r label names; do
    ./run.sh graft "$base" "$donor" "$v" --only "$names" > /dev/null && score "keep $label"
    ./run.sh a16 "$base" "$v" --only "$names" --type q8_0 > /dev/null && score "a16 $label"
done <<'LIST'
ffn_gate+up    ffn_gate.,ffn_up.
ffn_down       ffn_down.
dn_in          attn_qkv.,attn_gate.
attn_in+out    attn_q.,attn_k.,attn_v.,attn_output.
first4         blk.0.,blk.1.,blk.2.,blk.3.
last4          blk.60.,blk.61.,blk.62.,blk.63.
LIST
