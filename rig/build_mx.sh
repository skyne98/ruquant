#!/usr/bin/env bash
# Builds mx-llama.cpp (the MI50 fork with DSpark / DFlash2 speculative decoding) with ruquant's W4A4 path, on an
# MI50 box. Run on the box from a copy of this repo's patches/ (rig/box.sh HOST push, or scp the patch):
#   rig/build_mx.sh [DIR]       DIR defaults to ~/mx-llama.cpp; env JOBS (default nproc), ARCH (default gfx906)
# Needs git or a source tarball in DIR, cmake + ninja (pip install --user cmake ninja) and ROCm. The patch carries
# the ROCm 5.7 fixes (relaxed 6.1 gate, hipStreamWaitEvent default argument, duplicate bf16 host symbols).
# Run the binaries with GGML_CUDA_DISABLE_GRAPHS=1 on ROCm 5.7 (graphs are slower there). Models carry their own
# activation clips (RQ_ACT_CLIP / RQ_ACT_CLIP_H override them). Full runtime settings: RESULTS.md.
# RCCL (GGML_HIP_RCCL, needs ROCm's librccl): under -sm tensor, prefill-sized all-reduces go to RCCL and decode-sized
# ones stay on the custom peer-write kernel; TP = 4 prefill +17-19% on box 1 with RCCL's f16 ring.
# Prefill all-reduces (>= 262144 values) go as q4_K blocks by default (all-gather of partials at TP2, reduce-scatter +
# all-gather above): TP2 / TP4 pp2048 +15% / +39% on box 1, a few % KLD on prompts (REPORT-2026-11-02.md, quantized
# prefill all-reduce). GGML_NCCL_AR_WIRE=f16 restores the 16-bit ring; q4_0 / q4_1 / q5_0 / q5_K / q8_0 also work.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-$HOME/mx-llama.cpp}
commit=c4d12e2c0094c9028318bed5bb1a30bb48f8a35b
export PATH=$HOME/.local/bin:/opt/rocm/bin:$PATH
if [ ! -d "$dir" ]; then
    git clone https://github.com/mxxm-t/mx-llama.cpp.git "$dir"
    git -C "$dir" checkout "$commit"
fi
cd "$dir"
if ! grep -q rq_skinny_kernel ggml/src/ggml-cuda/rq-w4a4-skinny.cuh 2>/dev/null; then
    patch -p1 < "$here/patches/mx-llama.cpp-ruquant.patch"
fi
hipcxx=$(ls /opt/rocm/llvm/bin/clang++ /opt/rocm/lib/llvm/bin/clang++ 2>/dev/null | head -1)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS="${ARCH:-gfx906}" -DGPU_TARGETS="${ARCH:-gfx906}" \
    -DCMAKE_HIP_COMPILER="$hipcxx" -DCMAKE_PREFIX_PATH=/opt/rocm -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DGGML_HIP_RCCL="$([ -e /opt/rocm/lib/librccl.so ] && echo ON || echo OFF)"
cmake --build build -j"${JOBS:-$(nproc)}" --target ggml-hip test-backend-ops llama-server llama-cli llama-bench llama-perplexity
echo "built $dir/build/bin; check: HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$dir/build/bin:/opt/rocm/lib $dir/build/bin/test-backend-ops -o MUL_MAT -p type_a=w4a4"
