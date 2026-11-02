#!/usr/bin/env bash
# What is this box? Everything later steps depend on, in one text file.
#   rig/probe.sh > env.txt
section() { printf '\n== %s\n' "$*"; }
run() { printf '$ %s\n' "$*"; "$@" 2>&1 | head -"${LINES_MAX:-60}"; }

section host
run uname -a
run cat /etc/os-release
run id
run nproc
run free -g
run df -h .
section rocm
run ls -d /opt/rocm*
run cat /opt/rocm/.info/version
run hipcc --version
run hipconfig --full
LINES_MAX=400 run rocminfo
section gpus
if command -v rocm-smi >/dev/null; then
    run rocm-smi
    run rocm-smi --showproductname --showvbios --showmeminfo vram
    run rocm-smi --showclocks --showperflevel --showpower --showmaxpower
    run rocm-smi --showtopo
    run rocm-smi --showpids
else # ROCm 7+ replaced rocm-smi with amd-smi
    LINES_MAX=200 run amd-smi static --asic --bus --vbios --limit
    LINES_MAX=200 run amd-smi metric --usage --power --clock --temperature --mem-usage
    run amd-smi topology
    run amd-smi process
fi
section libraries
run ls /opt/rocm/lib/rocblas/library
printf 'rocBLAS gfx906 kernel files: %s\n' "$(ls /opt/rocm/lib/rocblas/library 2>/dev/null | grep -c gfx906)"
run ls /opt/rocm/lib/libhipblas.so /opt/rocm/lib/librccl.so
run which all_reduce_perf
section tools
run cmake --version
run ninja --version
run g++ --version
run python3 -c 'import sys, numpy; print(sys.version, numpy.__version__)'
section network
run curl -sSI --max-time 10 https://huggingface.co
