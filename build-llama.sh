#!/usr/bin/env bash
# Builds our llama.cpp fork into build-llama/: upstream llama.cpp at a pinned commit plus
# patches/llama.cpp-ruquant.patch (the W4A4 / W4A4H ggml types with CPU and CUDA kernels, the q4_0
# KV attention, and the hook that runs src/kernels.rs's CUDA kernels inside ggml's graph). Gives the
# libraries the Rust crate links and the llama.cpp tools, which all run W4A4 GGUFs: llama-perplexity,
# llama-bench, llama-completion, test-backend-ops (and llama-server with SERVER=1). On NVIDIA GPUs
# with INT4 tensor cores (Turing, Ampere, Ada) the W4A4 matmuls and the q4_0 attention's Q.K run on
# them; CUDA_ARCH="75;80;86;89" builds for all of those.
#   ./build-llama.sh                 CUDA if nvcc is found, else HIP if hipcc is, else CPU only
#   BACKEND=cpu|cuda|hip ./build-llama.sh      ARCH=gfx906 (HIP) / CUDA_ARCH=86 (CUDA) to override
#   VULKAN=1 ./build-llama.sh        also build ggml-vulkan
#   SERVER=1 ./build-llama.sh        also build llama-server (accuracy/agentic/serve.sh)
#   ./build-llama.sh --export        after editing build-llama/src: write its diff back to the patch
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=785322d17f2d51e5185cf996a1f4e46d87640841
SRC=build-llama/src
BUILD=build-llama/build
PATCH=$PWD/patches/llama.cpp-ruquant.patch

if [ "${1:-}" = --export ]; then
    git -C $SRC add -N . # include new files
    git -C $SRC diff $COMMIT -- . ':!*.orig' > "$PATCH"
    echo "wrote $PATCH ($(grep -c '^+++ ' "$PATCH") files)"
    exit 0
fi

if [ ! -d $SRC/.git ]; then
    mkdir -p $SRC
    git -C $SRC init -q
    git -C $SRC remote add origin https://github.com/ggml-org/llama.cpp
    git -C $SRC fetch -q --depth 1 origin $COMMIT
    git -C $SRC checkout -q -b ruquant FETCH_HEAD
fi
if git -C $SRC apply --check "$PATCH" 2>/dev/null; then
    git -C $SRC apply "$PATCH"
elif ! git -C $SRC apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patches/llama.cpp-ruquant.patch neither applies nor is applied to $SRC: export or reset it" >&2
    exit 1
fi

backend=${BACKEND:-$(command -v nvcc >/dev/null && echo cuda || (command -v hipcc >/dev/null && echo hip) || echo cpu)}
args=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_CCACHE=OFF -DLLAMA_CURL=OFF
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF)
targets=(ggml ggml-base ggml-cpu llama llama-perplexity llama-bench llama-completion test-backend-ops)
case $backend in
    cuda) args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-native}"); targets+=(ggml-cuda) ;;
    hip)  args+=(-DGGML_HIP=ON -DGPU_TARGETS="${ARCH:-gfx906}" -DAMDGPU_TARGETS="${ARCH:-gfx906}"); targets+=(ggml-hip)
          export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" ;;
    cpu)  ;;
    *) echo "BACKEND must be cpu, cuda or hip" >&2; exit 1 ;;
esac
if [ "${SERVER:-0}" = 1 ]; then
    args+=(-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_MTMD=ON); targets+=(llama-server)
fi
if [ "${VULKAN:-0}" = 1 ]; then # optional extra backend for src/bench.rs
    args+=(-DGGML_VULKAN=ON); targets+=(ggml-vulkan)
fi
gen=()
command -v ninja >/dev/null && gen=(-G Ninja)
cmake -S $SRC -B $BUILD "${gen[@]}" "${args[@]}"
cmake --build $BUILD -j"${JOBS:-$(nproc)}" --target "${targets[@]}"
echo "built $BUILD/bin ($backend)"
