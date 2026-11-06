#!/usr/bin/env bash
# The q4_0 KV cache with our dot8 attention measured end to end on a box, recorded in ~/profiles/<time>-kv-measure:
#   kld      quality at 32k context against the UD-Q8_K_XL reference run at 32k (REF, one 32768-token chunk scored over
#            positions 16k-32k), one GPU per config, four at a time. -ub 8 runs decode / verify-width kernels (W4A4
#            skinny, and for q4_0 our attention); -ub 2048 the prefill kernels (for q4_0 the stock tile kernel).
#   passkey  retrieval: a pass key at 5 positions of a 64k document (rig/passkey_bench.py, TP4 server with DFlash2).
#   speed    llama-bench at depth (rig/depth_bench.sh): TP2 at 0 / 32k / 64k, one GPU at 32k, TP4 at 64k, f16 vs q4_0.
#   context  how much fits: one GPU with -c 262144 (q4_0) and 131072 (f16), drafter included; DFlash2 after a
#            131072-token document at TP4 (rig/long_ctx_bench.py).
#   bash rig/kv_box_bench.sh MODEL DRAFT     env BIN (default ~/mx-llama.cpp/build-kv/bin), ROCM_LIB, REF, PARTS
set -uo pipefail
model=$1
draft=$2
bin=${BIN:-$HOME/mx-llama.cpp/build-kv/bin}
rl=${ROCM_LIB:-$HOME/therock-7.12/lib}
ref=${REF:-$HOME/ref/ud-q8_k_xl-32k.kld}
parts=${PARTS:-kld passkey speed context}
d=$HOME/profiles/$(date -u +%Y%m%dT%H%M%SZ)-kv-measure
mkdir -p "$d"
cp "$0" ~/rig/passkey_bench.py "$d/"
{
    echo "label: kv-measure"
    echo "time: $(date -u +%FT%TZ)"
    echo "binary: $bin (libggml-hip.so md5 $(md5sum < "$bin/libggml-hip.so" | cut -c1-32))"
    echo "model: $model"
    echo "drafter: $draft"
    echo "reference: $ref (sha256 $(sha256sum < "$ref" | cut -c1-16)...)"
    echo "parts: $parts"
} > "$d/meta.txt"
res=$d/results.txt
q4="-ctk q4_0 -ctv q4_0"

# kld NAME GPU UB [VAR=v ...] -- [llama-perplexity args]
kld() {
    local name=$1 g=$2 ub=$3 envs=()
    shift 3
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    env "${envs[@]}" HIP_VISIBLE_DEVICES=$g GGML_CUDA_DISABLE_GRAPHS=1 LD_LIBRARY_PATH=$bin:$rl "$bin/llama-perplexity" \
        -m "$model" -f "$HOME/ref/wiki.test.raw" -c 32768 -b 2048 -ub "$ub" --chunks 1 -fa on -ngl 99 -lm dio -t 3 "$@" \
        --kl-divergence-base "$ref" --kl-divergence > "$d/kld-$name.log" 2>&1
}
kld_line() {
    local log=$d/kld-$1.log
    printf "%-24s %s  %s\n" "$1" \
        "$(sed -n '/KL divergence statistics/,$p' "$log" | grep -E 'Mean    KLD|99.0%   KLD|Same top p' | tr -s ' ' | tr '\n' ' ')" \
        "$(grep -m1 'Mean PPL(Q)/PPL(base)' "$log" | tr -s ' ')" | tee -a "$res"
}

if [[ " $parts " == *" kld "* ]]; then
    echo "== kld: 32k context against UD-Q8_K_XL at 32k, positions 16k-32k, one GPU each" | tee -a "$res"
    kld f16-ub8 0 8 -- &
    kld q4_0-stock-ub8 1 8 RQ_FA_DOT8=0 -- $q4 &
    kld q4_0-dot8-ub8 2 8 -- $q4 &
    kld q4_0-dot8-int8q-ub8 3 8 RQ_FA_DOT8=2 -- $q4 &
    wait
    kld q8_0-ub8 0 8 -- -ctk q8_0 -ctv q8_0 &
    kld f16-ub2048 1 2048 -- &
    kld q4_0-ub2048 2 2048 -- $q4 &
    kld q4_0-dot8-norot-ub8 3 8 LLAMA_ATTN_ROT_DISABLE=1 -- $q4 &
    wait
    for n in f16-ub8 q8_0-ub8 q4_0-stock-ub8 q4_0-dot8-ub8 q4_0-dot8-int8q-ub8 q4_0-dot8-norot-ub8 f16-ub2048 q4_0-ub2048; do
        kld_line $n
    done
fi

if [[ " $parts " == *" passkey "* ]]; then
    echo "== passkey: 64k document, TP4, DFlash2" | tee -a "$res"
    for kv in f16 q4_0; do
        BIN=$bin ROCM_LIB=$rl GPUS=0,1,2,3 LOG=$d/passkey-$kv.log EXTRA="-sm tensor -devd ROCm0 -ctk $kv -ctv $kv" \
            python3 ~/rig/passkey_bench.py "$model" "$draft" 7 65536 2>&1 | tee -a "$res"
    done
fi

if [[ " $parts " == *" speed "* ]]; then
    echo "== speed: llama-bench at depth" | tee -a "$res"
    for spec in "0,1 0,32768,65536" "0 32768" "0,1,2,3 65536"; do
        for kv in f16 q4_0; do
            echo "-- GPUs ${spec% *}, depths ${spec#* }, kv $kv" | tee -a "$res"
            GPUS_LIST=${spec% *} DEPTHS=${spec#* } BIN=$bin ROCM_LIB=$rl EXTRA_BENCH="-ctk $kv -ctv $kv" OUT=$res \
                bash ~/rig/depth_bench.sh "$model" > /dev/null 2>> "$d/depth.err"
        done
    done
fi

if [[ " $parts " == *" context "* ]]; then
    echo "== context: what fits on one GPU (short prompts only), and DFlash2 after 131072 tokens at TP4" | tee -a "$res"
    for spec in "q4_0 262144" "f16 131072"; do
        kv=${spec% *}
        echo "-- one GPU, kv $kv, -c ${spec#* }" | tee -a "$res"
        BIN=$bin ROCM_LIB=$rl GPUS=0 LOG=$d/context-1gpu-$kv.log EXTRA="-ctk $kv -ctv $kv -c ${spec#* }" \
            python3 ~/rig/long_ctx_bench.py "$model" "$draft" 7 0 2>&1 | tee -a "$res"
    done
    for kv in f16 q4_0; do
        echo "-- TP4, kv $kv, 131072-token document" | tee -a "$res"
        BIN=$bin ROCM_LIB=$rl GPUS=0,1,2,3 LOG=$d/context-tp4-$kv.log EXTRA="-sm tensor -devd ROCm0 -ctk $kv -ctv $kv" \
            python3 ~/rig/long_ctx_bench.py "$model" "$draft" 7 131072 2>&1 | tee -a "$res"
    done
fi
echo "exit: 0" >> "$d/meta.txt"
echo "$d"
