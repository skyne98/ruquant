#!/usr/bin/env bash
# Speed against context depth on a box: prefill (pp512), plain decode (tg128) and the DFlash2 verify forward (pp8, 8
# tokens) after DEPTHS tokens already in the context, on one GPU, TP2 and TP4, with ROCm 7.12 and HIP graphs.
#   bash rig/depth_bench.sh MODEL      env BIN (default ~/mx-llama.cpp/build-r712/bin), ROCM_LIB, DEPTHS (default
#                                      0,16384,32768,65536), GPUS_LIST (default "0 0,1 0,1,2,3"), OUT (results file),
#                                      EXTRA_BENCH (more llama-bench arguments, e.g. "-ctk q8_0 -ctv q8_0")
# llama-bench prefills each depth once per test and restores it for the repetitions (state copy), so the times are
# the measured tokens' alone. The KV cache and compute buffer sizes at the largest depth come from its log.
set -uo pipefail
model=$1
bin=${BIN:-$HOME/mx-llama.cpp/build-r712/bin}
depths=${DEPTHS:-0,16384,32768,65536}
out=${OUT:-/dev/stdout}
export LD_LIBRARY_PATH=$bin:${ROCM_LIB:-$HOME/therock-7.12/lib} HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_SPIN=2000000
export GGML_META_TOKEN_GRAPH=1
unset GGML_CUDA_DISABLE_GRAPHS
show() {
    python3 -c 'import sys, json
for l in sys.stdin:
    i = l.find("{\"")
    if i < 0: continue
    r = json.loads(l[i:])
    name = "pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"]
    s = sorted(r["samples_ts"]); med = s[len(s) // 2]
    # a prompt is one forward of n_prompt tokens; generation is one forward per token
    ms = 1000 * (r["n_prompt"] or 1) / med
    print("  %-6s @ d%-6d %8.1f t/s  (%.2f ms per forward, median of %d)" % (name, r["n_depth"], med, ms, len(s)),
          flush=True)'
}
for g in ${GPUS_LIST:-0 0,1 0,1,2,3}; do
    sm=
    [ "$g" != 0 ] && sm="-sm tensor"
    echo "== GPUs $g" | tee -a "$out"
    log=$(mktemp)
    HIP_VISIBLE_DEVICES=$g "$bin/llama-bench" -m "$model" -ngl 99 -fa 1 -lm dio $sm ${EXTRA_BENCH:-} -d "$depths" -p 512 -n 128 -r 3 -o jsonl \
        2> "$log" | show | tee -a "$out"
    HIP_VISIBLE_DEVICES=$g "$bin/llama-bench" -m "$model" -ngl 99 -fa 1 -lm dio $sm ${EXTRA_BENCH:-} -d "$depths" -p 8 -n 0 -r 20 -o jsonl \
        2>> "$log" | show | tee -a "$out"
    grep -E "KV self size|KV buffer size|compute buffer size|RS buffer size" "$log" | sort | uniq -c | tail -8 | tee -a "$out"
    rm -f "$log"
done
