#!/usr/bin/env bash
# Checks that GPUs of a shared box are idle before a run: other people may be using them. Prints each GPU's load, VRAM
# in use and the processes holding it, and the host's load average; exits 1 if any listed GPU is busy (load above 5%,
# more than 1 GB of VRAM) or the host's 1-minute load is above LOAD_MAX (default 6, half of box 1's cores; TP runs pass
# 3). Box 1's host is shared beyond our container: another tenant's CPU job (load 17 on 12 cores) slowed TP4 decode by
# 15-50% while every GPU looked idle, since TP4 spins host threads on each all-reduce. One-GPU runs hardly notice.
#   rig/gpu_idle.sh [GPUS]      GPUS: comma-separated indices (default all); env LOAD_MAX
set -uo pipefail
want=${1:-all}
busy=0
info=$(rocm-smi --showuse --showmemuse --showmeminfo vram --showpids 2>/dev/null)
n=$(echo "$info" | grep -oE "^GPU\[[0-9]+\]" | sort -u | wc -l)
for ((g = 0; g < n; g++)); do
    if [ "$want" != all ] && [[ ",$want," != *",$g,"* ]]; then continue; fi
    use=$(echo "$info" | grep -E "^GPU\[$g\].*GPU use \(%\)" | grep -oE "[0-9]+$" | head -1)
    used=$(echo "$info" | grep -E "^GPU\[$g\].*VRAM Total Used Memory" | grep -oE "[0-9]+$" | head -1)
    used_gb=$(( ${used:-0} / 1073741824 ))
    pids=$(echo "$info" | awk -v g="$g" '/PID/ && /GPU/ {print}' | grep -cE "\b$g\b" || true)
    state=idle
    if [ "${use:-0}" -gt 5 ] || [ "$used_gb" -ge 1 ]; then state=BUSY; busy=1; fi
    echo "GPU $g: use ${use:-?}%, VRAM ${used_gb} GB, $state"
done
echo "$info" | sed -n '/KFD process information/,/=====/p' | grep -vE "^=+|^$" | head -12
load=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$load" -v m="${LOAD_MAX:-6}" 'BEGIN { exit !(l > m) }'; then
    echo "host load $load (above ${LOAD_MAX:-6}): BUSY"; busy=1
else
    echo "host load $load"
fi
exit $busy
