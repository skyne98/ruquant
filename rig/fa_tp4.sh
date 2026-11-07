#!/bin/bash
# TP4, build OLD against build NEW alternating old / new / old / new (so drift on a shared host shows up): DFlash2
# after a 64k document (rig/long_ctx_bench.py) and llama-bench tg128 / pp8 at depth 0 and 65536, q4_0 cache. Each
# turn starts only if the GPUs and the host are idle (rig/gpu_idle.sh with host load <= 3: TP4 spins host threads).
#   bash rig/fa_tp4.sh OLD NEW [LABEL]      builds ~/mx-llama.cpp/build-OLD / build-NEW
old=$1 new=$2
export LOAD_MAX=3
d=~/profiles/$(date -u +%Y%m%dT%H%M%SZ)-${3:-fa-tp4}; mkdir -p $d
M=$HOME/models/qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf
D=$HOME/models/dflash2-ost44-h4-l4-c8-g64-rb4.gguf
{ echo "label: ${3:-fa-tp4}"; echo "time: $(date -u +%FT%TZ)"; echo "TP4 (-sm tensor -devd ROCm0), q4_0 cache, builds $old / $new / $old / $new"
  for b in $old $new; do echo "$b libggml-hip.so md5: $(md5sum < ~/mx-llama.cpp/build-$b/bin/libggml-hip.so | cut -c1-32)"; done; } > $d/meta.txt
n=0
for b in $old $new $old $new; do
  n=$((n + 1))
  for i in $(seq 1 30); do ~/rig/gpu_idle.sh 0,1,2,3 > /dev/null && break; sleep 20; done
  { echo "check before $b ($n) at $(date -u +%T):"; ~/rig/gpu_idle.sh 0,1,2,3; } >> $d/meta.txt || { echo "busy before $b ($n), stopping" | tee -a $d/meta.txt; exit 1; }
  B=$HOME/mx-llama.cpp/build-$b/bin
  BIN=$B GPUS=0,1,2,3 EXTRA="-ctk q4_0 -ctv q4_0 -sm tensor -devd ROCm0" LOG=$d/server-$b-$n.log timeout 3600 \
    python3 ~/rig/long_ctx_bench.py $M $D 7 65536 > $d/dflash-$b-$n.txt 2>&1
  for t in "-p 0 -n 128 -r 3 -d 0,65536" "-p 8 -n 0 -r 20 -d 0,65536"; do
    HIP_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=$B:$HOME/therock-7.12/lib HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_SPIN=2000000 GGML_META_TOKEN_GRAPH=1 \
      $B/llama-bench -m $M -ngl 99 -fa 1 -lm dio -sm tensor -ctk q4_0 -ctv q4_0 $t -o jsonl 2>>$d/bench-$b-$n.err | python3 -c 'import sys, json
for l in sys.stdin:
    i = l.find("{\"")
    if i < 0: continue
    r = json.loads(l[i:]); s = sorted(r["samples_ts"]); med = s[len(s) // 2]; n = r["n_prompt"] or 1
    print("  %-6s @ d%-6d %7.1f t/s  %.2f ms per forward" % ("pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"], r["n_depth"], med, 1000 * n / med), flush=True)' >> $d/bench-$b-$n.txt
  done
  echo "host load after $b ($n): $(cat /proc/loadavg)" >> $d/meta.txt
done
for f in $(ls $d/bench-*.txt | sort -t- -k3); do echo "== $(basename $f)"; cat $f; done > $d/results.txt
for f in $d/dflash-*.txt; do echo "== $(basename $f)"; grep -E "^(document|short)" $f; done >> $d/results.txt
echo "exit: 0" >> $d/meta.txt
echo FA-TP4-DONE
