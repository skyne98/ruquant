#!/usr/bin/env bash
# Error map: which linear classes cost the most? Each variant leaves one class at Q8_0 (rotated).
set -euo pipefail
cd "$(dirname "$0")/.."
G=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-UD-Q8_K_XL.gguf)
export RQ_ACT_CLIP=${RQ_ACT_CLIP:-0.8}
while read -r label keep; do
    ./run.sh convert "$G" models/variant.gguf --keep "$keep" > models/variant-convert.log 2>&1
    accuracy/score.sh models/variant.gguf 16 "keep $label (clip $RQ_ACT_CLIP)"
done <<'LIST'
ffn_down         ffn_down
attn_out+ssm_out attn_output,ssm_out
ffn_gate+up      ffn_gate,ffn_up
readers          attn_q,attn_k,attn_v,attn_qkv,attn_gate,ssm_alpha,ssm_beta
first4+last4     blk.0.,blk.1.,blk.2.,blk.3.,blk.60.,blk.61.,blk.62.,blk.63.
LIST
rm -f models/variant.gguf
