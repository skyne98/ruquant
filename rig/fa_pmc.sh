#!/usr/bin/env bash
# Hardware counters of ruquant's q4_0 attention kernel (rq_fa_q4_kernel) for one shape of rig/fa_bench.sh's set, one
# recorded profile (rig/prof.sh) per counter group, on an MI50 box.
#   rig/fa_pmc.sh BUILD LABEL [KV [NB]]       defaults KV 65536, NB 8 (the DFlash2 verify); env GPU (default 0)
set -uo pipefail
# other people may be using the box's GPUs: run only on an idle one
~/rig/gpu_idle.sh ${GPU:-0} || { echo "GPU ${GPU:-0} is busy, not running"; exit 1; }
build=$1 label=$2 kv=${3:-65536} nb=${4:-8}
B=$HOME/mx-llama.cpp/build-$build/bin
export ROCPROFILER_METRICS_PATH=$HOME/tp/rpdefs LD_LIBRARY_PATH=$B:$HOME/therock-7.12/lib
filter="kv=$kv,nb=$nb,.*type_K=q4_0,type_V=q4_0"
i=0
for g in "GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS" "SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_VMEM_RD VALUBusy" \
         "SQC_LDS_IDX_ACTIVE SQ_ACTIVE_INST_VALU SQ_BUSY_CYCLES GRBM_GUI_ACTIVE" "ALUStalledByLDS LDSBankConflict GRBM_GUI_ACTIVE" \
         "RQ_SQ_WAVE_CYCLES RQ_SQ_WAIT_ANY RQ_SQ_WAIT_INST_ANY GRBM_GUI_ACTIVE" "SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT GRBM_GUI_ACTIVE"; do
  d=$(PROF_ARGS="--pmc $g --kernel-include-regex rq_fa_q4" ~/rig/prof.sh fa-$label-kv$kv-nb$nb-pmc$i HIP_VISIBLE_DEVICES=${GPU:-0} -- \
      $B/test-backend-ops perf -o FLASH_ATTN_EXT -p "$filter" -b ROCm0)
  python3 ~/rig/prof_summary.py $d rq_fa_q4 | grep "wgs" | sed -E 's/ +n +[0-9]+//'
  i=$((i + 1))
done
