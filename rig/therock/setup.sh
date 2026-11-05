#!/usr/bin/env bash
# ROCm 7.12 from AMD's TheRock nightly tarballs for the gfx90X family (gfx906 / gfx908 / gfx90a) in user space, and
# mx-llama.cpp built against it in build-r712, with RCCL. Unlike AMD's packaged 7.2.4 (rig/rocm72), its RCCL has gfx906
# code objects, and the libraries need only glibc 2.27 / GLIBCXX 3.4.22, so they run on Ubuntu 22.04.
#   rig/therock/setup.sh          (on the box; ~5 GB for the slim install, plus the build)
#   env THEROCK_DIR (default ~/therock-7.12), EXTRACT_ONLY=1 (install only, e.g. for rig/therock/build_local.sh)
# The tarball (3.5 GB, 14.9 GB unpacked) is streamed and only what a build and llama.cpp need is kept: no MIOpen,
# FFT, hipBLASLt kernels, Fortran / MLIR, LLVM's own static libraries or test clients (clang's runtime builtins stay:
# every link needs libclang_rt.builtins.a; so does librocroller, which libhipblaslt links).
set -euo pipefail
V=${THEROCK_VERSION:-7.12.0a20260225}
P=${THEROCK_DIR:-$HOME/therock-7.12}
U=https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx90X-dcgpu-$V.tar.gz
if [ ! -x "$P/lib/llvm/bin/clang++" ]; then
    mkdir -p "$P"
    curl -s -f "$U" | tar xzf - -C "$P" \
        --exclude='./share/miopen' --exclude='./clients' --exclude='./share/doc' --exclude='*gtest*' --exclude='*_test*' \
        --exclude='*/test/*' --exclude='*tests/*' --exclude='*-test' --exclude='*-bench*' --exclude='*_bench*' \
        --exclude='./lib/rocfft' --exclude='*libMIOpen*' --exclude='./lib/hipblaslt' --exclude='*librocfft*' \
        --exclude='*migraphx*' --exclude='*hipsparselt*' --exclude='*rocwmma*' --exclude='./share/rocprof*' \
        --exclude='./lib/rocprofiler-sdk' --exclude='./lib/llvm/lib/libclang*.a' --exclude='./lib/llvm/lib/libLLVM*.a' --exclude='./lib/llvm/lib/liblld*.a' --exclude='*flang*' --exclude='*Fortran*' \
        --exclude='*MLIR*' --exclude='*mlir*' --exclude='*rocrand*' --exclude='*hiprand*' \
        --exclude='*-validate' --exclude='./lib/llvm/bin/bbc'
fi
[ "${EXTRACT_ONLY:-0}" = 1 ] && exit 0
export PATH=$HOME/.local/bin:$P/bin:$P/lib/llvm/bin:$PATH ROCM_PATH=$P HIP_PATH=$P
export HIP_DEVICE_LIB_PATH=$(dirname "$(find "$P/lib/llvm" -name ockl.bc | head -1)")
cd ~/mx-llama.cpp
cmake -B build-r712 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_COMPILER="$P/lib/llvm/bin/clang++" -DCMAKE_PREFIX_PATH="$P" -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_OPENSSL=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-r712 -j"${JOBS:-12}" --target ggml-hip test-backend-ops llama-server llama-bench llama-perplexity
echo "run with: LD_LIBRARY_PATH=~/mx-llama.cpp/build-r712/bin:$P/lib GGML_META_TOKEN_GRAPH=1 (no GGML_CUDA_DISABLE_GRAPHS)"
