#!/usr/bin/env bash
# Compile-only gfx906 (MI50) asm for the cost model. Run from the repo root:
#   nix develop .#rocm -c gfx906/build.sh
# Outputs build/*.s; gfx906/model.py reads them. Nothing here runs on a GPU.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

G=../third_party/llama.cpp/ggml
HIP=(hipcc -x hip --offload-arch=gfx906 --offload-device-only -O3 -std=c++17 -S)
GGML=(-DGGML_USE_HIP -DGGML_HIP_NO_MMQ_MFMA -I$G/include -I$G/src -I$G/src/ggml-cuda)

quiet() { "$@" 2>&1 | grep -v "argument unused during compilation" || true; }

# today's MI50 path: q4_0 weights x q8_1 activations (MMVQ for T <= 8, MMQ above)
quiet "${HIP[@]}" "${GGML[@]}" $G/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu -o build/mmq_q4_0.s
quiet "${HIP[@]}" "${GGML[@]}" $G/src/ggml-cuda/mmvq.cu -o build/mmvq.s
quiet "${HIP[@]}" "${GGML[@]}" $G/src/ggml-cuda/quantize.cu -o build/quantize.s

# W4A4 dot8: the presets, plus the variants the presets were picked from
quiet "${HIP[@]}" -ffp-contract=off w4a4_skinny.hip -o build/w4a4_skinny.s
{
    echo '#include <hip/hip_runtime.h>'
    echo '#include "../w4a4_skinny.h"'
    while read -r T R KA TB; do
        echo "template __global__ void w4a4_skinny<$T,$R,$KA,$TB>(const int32_t*, const int32_t*, int32_t*, int, int, int);"
    done <<'EOF'
8 2 4 8
8 4 4 8
16 1 4 16
16 2 4 16
16 2 2 16
16 4 2 16
24 1 1 24
24 2 1 24
24 2 2 12
32 1 1 32
32 2 1 32
32 2 1 16
32 1 2 16
48 1 1 16
48 1 1 24
48 2 1 24
64 1 1 16
64 1 1 32
64 1 2 16
EOF
} > build/sweep.hip
quiet "${HIP[@]}" -ffp-contract=off build/sweep.hip -o build/sweep.s

echo "asm in gfx906/build: $(ls build/*.s | xargs -n1 basename | tr '\n' ' ')"
