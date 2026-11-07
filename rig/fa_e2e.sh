#!/bin/bash
# ruquant's q4_0 attention, build NEW against build OLD (default build-tm, the published fork), end to end on box 1,
# recorded in ~/profiles (2026-11-06: 8-wave workgroups, two channels per lane):
#   1. test-backend-ops FLASH_ATTN_EXT, all cases, int4 and int8 Q (GPU 2)
#   2. KLD against UD-Q8_K_XL, 16 chunks, 8-token ubatches so attention runs at verify width (GPU 3)
#   3. llama-bench tg128 / pp8 / pp16 at depth 0 and 65536, one GPU (OLD on GPU 0, NEW on GPU 1 at the same time)
#   4. DFlash2 after a 64k-token document (rig/long_ctx_bench.py), one GPU (OLD GPU 0, NEW GPU 1)
#   5. TP4: rig/fa_tp4.sh (old / new / old / new: DFlash2 after 64k, llama-bench at depth 0 and 65536)
#   bash rig/fa_e2e.sh NEW [OLD]
# Old and new run side by side on GPUs 0 and 1 in parts 3 and 4, so the depth-0 numbers are the control for the two
# GPUs' own difference.
new=$1 old=${2:-tm}
# other people may be using the box's GPUs: start only if all four are idle (rig/fa_tp4.sh checks again, host load too)
~/rig/gpu_idle.sh 0,1,2,3 || { echo "GPUs busy, not running"; exit 1; }
d=~/profiles/$(date -u +%Y%m%dT%H%M%SZ)-fa-occ-e2e; mkdir -p $d
M=$HOME/models/qwen3.8-27b-ost44-v2-wg64-rb4-gptq.gguf
D=$HOME/models/dflash2-ost44-h4-l4-c8-g64-rb4.gguf
BN=$HOME/mx-llama.cpp/build-$new/bin BO=$HOME/mx-llama.cpp/build-$old/bin
{ echo "label: fa-occ-e2e"; echo "time: $(date -u +%FT%TZ)"
  echo "new: $BN (libggml-hip.so md5 $(md5sum < $BN/libggml-hip.so | cut -c1-32))"
  echo "old: $BO (libggml-hip.so md5 $(md5sum < $BO/libggml-hip.so | cut -c1-32))"
  echo "q4_0 cache, default settings (RQ_FA_DOT8 unset = 1); parts: tests (GPU 2), KLD -ub 8 (GPU 3), llama-bench one GPU (old GPU 0, new GPU 1),"
  echo "DFlash2 after 64k one GPU (old GPU 0, new GPU 1) and TP4, llama-bench TP4"; echo "gpu check at start:"; ~/rig/gpu_idle.sh 0,1,2,3; } > $d/meta.txt
bench() { # BIN GPUS OUT ARGS...
  local b=$1 g=$2 o=$3; shift 3
  HIP_VISIBLE_DEVICES=$g LD_LIBRARY_PATH=$b:$HOME/therock-7.12/lib HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_SPIN=2000000 GGML_META_TOKEN_GRAPH=1 \
    $b/llama-bench -m $M -ngl 99 -fa 1 -lm dio -ctk q4_0 -ctv q4_0 "$@" -o jsonl 2>>$o.err | python3 -c 'import sys, json
for l in sys.stdin:
    i = l.find("{\"")
    if i < 0: continue
    r = json.loads(l[i:]); s = sorted(r["samples_ts"]); med = s[len(s) // 2]; n = r["n_prompt"] or 1
    print("  %-6s @ d%-6d %7.1f t/s  %.2f ms per forward" % ("pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"], r["n_depth"], med, 1000 * n / med), flush=True)' >> $o
}
( export HIP_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=$BN:$HOME/therock-7.12/lib
  timeout 1800 $BN/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 > $d/tests.log 2>&1
  RQ_FA_DOT8=2 timeout 1800 $BN/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 -p "type_K=q4_0" > $d/tests-int8q.log 2>&1 ) &
( BIN=$BN ROCM_LIB=$HOME/therock-7.12/lib GPUS=3 EXTRA="-ctk q4_0 -ctv q4_0 -ub 8" bash ~/rig/score_box.sh $M "new, kv q4_0, ub 8" > $d/kld-new.txt 2>&1 ) &
for t in "-p 0 -n 128 -r 3 -d 0,65536" "-p 8 -n 0 -r 20 -d 0,65536" "-p 16 -n 0 -r 20 -d 0,65536"; do
  bench $BO 0 $d/bench1-old.txt $t & p1=$!
  bench $BN 1 $d/bench1-new.txt $t & p2=$!
  wait $p1 $p2
done
for which in old new; do
  b=$BO g=0; [ $which = new ] && b=$BN g=1
  BIN=$b GPUS=$g PORT=$((8097 + g)) EXTRA="-ctk q4_0 -ctv q4_0" LOG=$d/server1-$which.log timeout 3600 \
    python3 ~/rig/long_ctx_bench.py $M $D 7 65536 > $d/dflash1-$which.txt 2>&1 &
done
wait
# TP4: rig/fa_tp4.sh, old and new alternating, into its own profile
bash ~/rig/fa_tp4.sh $old $new fa-e2e-tp4 > $d/tp4.log 2>&1
echo "TP4 part: $(grep -E 'profile|busy' $d/tp4.log | tail -1)" >> $d/meta.txt
{ echo "== tests"; grep -h "tests passed" $d/tests.log $d/tests-int8q.log
  echo "== KLD (new)"; cat $d/kld-new.txt
  for f in bench1-old bench1-new; do echo "== $f"; cat $d/$f.txt; done
  for f in dflash1-old dflash1-new; do echo "== $f"; grep -E "^(document|short)" $d/$f.txt; done
  echo "== TP4"; cat $(ls -dt ~/profiles/*fa-e2e-tp4 | head -1)/results.txt; } > $d/results.txt
echo "exit: 0" >> $d/meta.txt
echo E2E-DONE
