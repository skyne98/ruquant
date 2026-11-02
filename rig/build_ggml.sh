#!/usr/bin/env bash
# Today's path on the MI50: our patched ggml tree (third_party/llama.cpp) built for HIP, then
# rig/ggml_bench against it. Needs ROCm's hipBLAS + rocBLAS (with gfx906 kernels) and cmake.
# Building the ggml-cuda sources for HIP also proves the RUQUANT hook patch compiles there.
# The vendored tree has no tools/ sources, so no llama-bench: for end to end, point run_all.sh
# at a llama-bench already on the box (LLAMA_BENCH=...).
#   rig/build_ggml.sh            (ARCH=gfx906 by default)
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${ARCH:-gfx906}
gen=$(command -v ninja >/dev/null && echo Ninja || echo "Unix Makefiles")

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S third_party/llama.cpp -B rig/build-ggml -G "$gen" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_HIP=ON -DGPU_TARGETS="$ARCH" -DAMDGPU_TARGETS="$ARCH" \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DGGML_NATIVE=ON -DGGML_CCACHE=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build rig/build-ggml -j"$(nproc)" --target ggml ggml-base ggml-hip

mkdir -p rig/build
g++ -O2 -std=c++17 rig/ggml_bench.cpp -Ithird_party/llama.cpp/ggml/include \
    -Lrig/build-ggml/bin -lggml -lggml-base -Wl,-rpath,"$PWD/rig/build-ggml/bin" -o rig/build/ggml_bench
echo "built rig/build/ggml_bench"
