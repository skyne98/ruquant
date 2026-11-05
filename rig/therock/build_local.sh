#!/usr/bin/env bash
# Builds mx-llama.cpp for gfx906 against TheRock ROCm 7.12 on a workstation instead of the box: inside an Ubuntu 22.04
# container, so the binaries need glibc 2.35 like box 1 (not the host's), then ships build-r712/bin to the box.
#   rig/therock/build_local.sh MX_SRC [HOST]     env THEROCK_DIR (default ~/Experiments/rocm/therock-7.12), JOBS
# MX_SRC is an mx-llama.cpp tree with patches/mx-llama.cpp-ruquant.patch applied; HOST as in rig/box.sh.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
src=$(cd "$1" && pwd)
host=${2:-}
T=${THEROCK_DIR:-$HOME/Experiments/rocm/therock-7.12}
[ -x "$T/lib/llvm/bin/clang++" ] || THEROCK_DIR=$T EXTRACT_ONLY=1 bash "$here/rig/therock/setup.sh"
docker run --rm -v "$src":/src -v "$T":/opt/therock:ro -w /src -e JOBS="${JOBS:-$(nproc)}" ubuntu:22.04 bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq g++ python3-pip git curl > /dev/null
    pip install -q cmake ninja
    export PATH=/opt/therock/bin:/opt/therock/lib/llvm/bin:$PATH ROCM_PATH=/opt/therock HIP_PATH=/opt/therock
    export HIP_DEVICE_LIB_PATH=$(dirname "$(find /opt/therock/lib/llvm -name ockl.bc | head -1)")
    cmake -B build-r712 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
        -DCMAKE_HIP_COMPILER=/opt/therock/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/therock -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
    cmake --build build-r712 -j"$JOBS" --target ggml-hip test-backend-ops llama-server llama-bench llama-perplexity
    chown -R '"$(id -u):$(id -g)"' build-r712'
if [ -n "$host" ]; then
    tar cf - -C "$src/build-r712" bin | "$here/rig/box.sh" "$host" ssh 'mkdir -p ~/mx-llama.cpp/build-r712 && tar xf - -C ~/mx-llama.cpp/build-r712'
fi
echo "built $src/build-r712/bin"
