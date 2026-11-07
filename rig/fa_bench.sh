#!/usr/bin/env bash
# Times ruquant's q4_0 KV attention (rq-fattn.cu) alone, at the 27B's shapes (4 KV heads, 6 query heads each, head
# 256) for 1 / 8 / 16 query tokens over 16k and 64k cells, on one GPU of an MI50 box. Seconds per build instead of a
# 64k-token prefill per llama-bench run. Records two profiles (rig/prof.sh): test-backend-ops' own timing, and a
# rocprofv3 kernel trace that splits each call into Q quantization, the attention kernel and the combine.
#   rig/fa_bench.sh BUILD LABEL [VAR=value ...]     BUILD: ~/mx-llama.cpp/build-BUILD/bin
set -uo pipefail
# other people may be using the box's GPUs: run only on an idle one
~/rig/gpu_idle.sh ${GPU:-0} || { echo "GPU ${GPU:-0} is busy, not running"; exit 1; }
build=$1 label=$2
shift 2
B=$HOME/mx-llama.cpp/build-$build/bin
export LD_LIBRARY_PATH=$B:$HOME/therock-7.12/lib HIP_VISIBLE_DEVICES=${GPU:-0}
for v in "$@"; do export "$v"; done
filter="type_K=q4_0,type_V=q4_0"
d=$(PROF_ARGS="--kernel-trace" ~/rig/prof.sh fa-$label-trace "$@" HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -- \
    $B/test-backend-ops perf -o FLASH_ATTN_EXT -p "$filter" -b ROCm0)
echo "trace: $d"
python3 ~/rig/prof_summary.py "$d" "rq_fa|flash_attn_combine" | head -24
# test-backend-ops' timing without the profiler, kept the same way
t=$HOME/profiles/$(date -u +%Y%m%dT%H%M%SZ)-fa-$label-perf
mkdir -p "$t"
{ echo "label: fa-$label-perf"; echo "time: $(date -u +%FT%TZ)"; echo "host: $(hostname)"
  echo "command: $B/test-backend-ops perf -o FLASH_ATTN_EXT -p $filter -b ROCm0"
  echo "settings: $* HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "libggml-hip.so md5: $(md5sum < $B/libggml-hip.so | cut -c1-32)"; } > "$t/meta.txt"
$B/test-backend-ops perf -o FLASH_ATTN_EXT -p "$filter" -b ROCm0 > "$t/run.log" 2>&1
echo "exit: $?" >> "$t/meta.txt"
grep -E "kv=|us/run" "$t/run.log" | sed -E 's/.*kv=([0-9]+),nb=([0-9]+).*: +([0-9]+) runs - +([0-9.]+) us\/run.*/kv \1 nb \2: \4 us/' | grep -E "^kv" | tee "$t/summary.txt"
