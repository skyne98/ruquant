#!/usr/bin/env bash
# Quantized KV cache on a box (llama.cpp's Hadamard attention rotation on, unless LLAMA_ATTN_ROT_DISABLE=1), recorded
# in ~/profiles/<time>-kvquant:
#   1. quality at 2048 context: KLD against the UD-Q8_K_XL reference (rig/score_box.sh, 16 chunks, one GPU) per KV type;
#   2. quality at 32k context: one 32768-token chunk, KLD of each KV type against the same model with an f16 cache
#      (TP4, 16-bit all-reduces), so only the cache's error counts, over positions 16k-32k;
#   3. speed: pp512 / tg128 / pp8 at depth 0 and 65536 per KV type, one GPU and TP4.
#   bash rig/kvquant_bench.sh MODEL      env BIN (default ~/mx-llama.cpp/build-r712/bin), ROCM_LIB, KVS (K/V pairs)
set -uo pipefail
model=$1
bin=${BIN:-$HOME/mx-llama.cpp/build-r712/bin}
rl=${ROCM_LIB:-$HOME/therock-7.12/lib}
kvs=${KVS:-"q8_0/q8_0 q8_0/q4_0 q4_0/q4_0"}
d=$HOME/profiles/$(date -u +%Y%m%dT%H%M%SZ)-kvquant
mkdir -p "$d"
cp "$0" "$d/"
{
    echo "label: kvquant"
    echo "time: $(date -u +%FT%TZ)"
    echo "binary: $bin (libggml-hip.so md5 $(md5sum < "$bin/libggml-hip.so" | cut -c1-32))"
    echo "model: $model"
    echo "kv types: f16/f16 $kvs; q4_0/q4_0 also without rotation (LLAMA_ATTN_ROT_DISABLE=1) in part 1"
} > "$d/meta.txt"
kv_args() { echo "-ctk ${1%/*} -ctv ${1#*/}"; }

echo "== 1. KLD against UD-Q8_K_XL, 2048 context, 16 chunks, one GPU" | tee -a "$d/results.txt"
for kv in f16/f16 $kvs; do
    BIN=$bin ROCM_LIB=$rl GPUS=0 EXTRA="$(kv_args $kv)" bash ~/rig/score_box.sh "$model" "kv $kv" | tee -a "$d/results.txt"
done
LLAMA_ATTN_ROT_DISABLE=1 BIN=$bin ROCM_LIB=$rl GPUS=0 EXTRA="$(kv_args q4_0/q4_0)" bash ~/rig/score_box.sh "$model" \
    "kv q4_0/q4_0 unrotated" | tee -a "$d/results.txt"

echo "== 2. KLD against the f16 cache, one 32768-token chunk, TP4" | tee -a "$d/results.txt"
base=$HOME/tp/kv32k-f16.kld
ppl() {
    HIP_VISIBLE_DEVICES=0,1,2,3 GGML_CUDA_DISABLE_GRAPHS=1 GGML_NCCL_AR_WIRE=f16 HSA_FORCE_FINE_GRAIN_PCIE=1 \
        LD_LIBRARY_PATH=$bin:$rl "$bin/llama-perplexity" -m "$model" -f "$HOME/ref/wiki.test.raw" -c 32768 -b 2048 -ub 2048 \
        --chunks 1 -fa on -ngl 99 -lm dio -sm tensor "$@"
}
ppl --kl-divergence-base "$base" > "$d/ppl32k-base.log" 2>&1
for kv in f16/f16 $kvs; do
    log=$d/ppl32k-${kv/\//-}.log
    ppl $(kv_args $kv) --kl-divergence-base "$base" --kl-divergence > "$log" 2>&1
    printf "%-12s %s\n" "kv $kv" "$(sed -n '/KL divergence statistics/,$p' "$log" | grep -E 'Mean    KLD|Same top p|99.9%   KLD' | tr -s ' ' | tr '\n' ' ')" \
        | tee -a "$d/results.txt"
done

echo "== 3. speed at depth 0 and 65536" | tee -a "$d/results.txt"
for g in 0 0,1,2,3; do
    for kv in f16/f16 $kvs; do
        echo "-- GPUs $g, kv $kv" | tee -a "$d/results.txt"
        DEPTHS=0,65536 GPUS_LIST=$g BIN=$bin ROCM_LIB=$rl EXTRA_BENCH="$(kv_args $kv)" OUT=$d/results.txt \
            bash ~/rig/depth_bench.sh "$model" > /dev/null 2>> "$d/depth.err"
    done
done
echo "exit: 0" >> "$d/meta.txt"
echo "$d"
