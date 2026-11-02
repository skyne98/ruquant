#!/usr/bin/env bash
# Builds the rig kit.
#   on the MI50 box:  rig/build.sh           hipcc, gfx906 (ARCH=... to override)
#   here:             rig/build.sh cuda      nvcc, sm_86, to check the harnesses on the 3090
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-hip}
mkdir -p build

if [ "$mode" = hip ]; then
    ARCH=${ARCH:-gfx906}
    CXX=(hipcc -x hip --offload-arch="$ARCH" -O3 -std=c++17)
else
    ARCH=${ARCH:-sm_86}
    CXX=(nvcc -x cu -arch="$ARCH" -O2 -std=c++17)
fi
echo "building with ${CXX[*]}"

# at most JOBS compiles at once (default nproc; a shared box may ask for fewer)
JOBS=${JOBS:-$(nproc)}
pids=()
job() {
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 0.2; done
    "$@" &
    pids+=($!)
}
for b in bench_valu bench_mem bench_atomic bench_p2p bench_skinny; do
    job "${CXX[@]}" "$b.cu" -o "build/$b"
done
job "${CXX[@]}" ../gfx906/test_skinny.cu -o build/test_skinny
# tensor-parallel bench, with RCCL when it's installed
rccl=()
if [ "$mode" = hip ] && [ -f "${ROCM_PATH:-/opt/rocm}/include/rccl/rccl.h" ]; then
    rccl=(-DRQ_RCCL -I"${ROCM_PATH:-/opt/rocm}/include" -L"${ROCM_PATH:-/opt/rocm}/lib" -lrccl)
fi
job "${CXX[@]}" bench_tp.cu "${rccl[@]}" -o build/bench_tp
if [ "$mode" = hip ] && [ -f mi50emu/w4a4_ref.hip ]; then
    # mi50emu's parity runner, with its required flags (see mi50emu README)
    job hipcc -O3 --offload-arch="$ARCH" -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt \
        -fno-gpu-flush-denormals-to-zero mi50emu/w4a4_ref.hip -o build/w4a4_ref
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "some builds failed (see above)"; exit 1; }
ls build
