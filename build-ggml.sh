#!/usr/bin/env bash
# Builds our patched ggml tree (third_party/llama.cpp) into build-ggml/.
# The Rust build only links the result; nothing is recompiled by cargo.
set -euo pipefail
: "${CUDA_PATH:?run this from nix develop}"

# our ggml-cuda hook; re-applied if third_party/llama.cpp is ever replaced with a clean copy
cu=third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu
if ! grep -q "RUQUANT patch" "$cu"; then
    patch -d third_party/llama.cpp -p1 < patches/ggml-cuda-custom1-hook.patch
fi

cmake -S third_party/llama.cpp -B build-ggml -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CCACHE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_COMMON=OFF
cmake --build build-ggml -j"${JOBS:-$(nproc)}"
echo "built build-ggml/bin (patched ggml-cuda, CUDA sm native + Vulkan)"
