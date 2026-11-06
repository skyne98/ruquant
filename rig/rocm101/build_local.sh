#!/usr/bin/env bash
# mx-llama.cpp against ROCm 10.1 for gfx906 from mixa3607/ML-gfx906's apt repository (AMD TheRock builds with gfx906
# enabled), on a workstation, run on box 1 (Ubuntu 22.04, no root):
#   rig/rocm101/build_local.sh MX_SRC HOST     env BOX_HOME (default /home/gpu)
# - An Ubuntu 24.04 container installs ROCm 10.1 (runtime, compiler, rocBLAS / hipBLAS, RCCL) and builds MX_SRC into
#   MX_SRC/build-r101 with RCCL.
# - The packages need glibc 2.38, box 1 has 2.35, so the executables get Ubuntu 24.04's loader and libc from
#   BOX_HOME/noble (libc6, libstdc++6, libgcc-s1 unpacked from Ubuntu's archive) as their interpreter and DT_RPATH.
#   Only these binaries see the newer glibc.
# - The runtime libraries go to BOX_HOME/rocm-10.1/lib and the build to ~/mx-llama.cpp/build-r101 on the box. Run with
#   LD_LIBRARY_PATH=~/mx-llama.cpp/build-r101/bin:~/rocm-10.1/lib:~/rocm-10.1/lib/llvm/lib (never with the noble
#   directory in it: the box's own programs would load the wrong libc).
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
src=$(cd "$1" && pwd)
host=$2
H=${BOX_HOME:-/home/gpu}
out=$src/build-r101
mkdir -p "$out"
docker run --rm -v "$src":/src -w /src -e JOBS="${JOBS:-$(nproc)}" -e BOXH="$H" ubuntu:24.04 bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq ca-certificates curl gnupg g++ python3-pip git patchelf > /dev/null
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://s3.arkprojects.space/apt-gfx906/ubuntu/gpg -o /etc/apt/keyrings/apt-gfx906.asc
    printf "Types: deb\nURIs: https://s3.arkprojects.space/apt-gfx906/ubuntu\nSuites: noble\nComponents: main\nArchitectures: amd64\nSigned-By: /etc/apt/keyrings/apt-gfx906.asc\n" > /etc/apt/sources.list.d/gfx906.sources
    apt-get update -qq
    apt-get install -y -qq amdrocm-runtime-dev10.1 amdrocm-llvm-dev10.1 amdrocm-blas-dev10.1 amdrocm-blas10.1-gfx906 \
        amdrocm-rccl-dev10.1 amdrocm-rccl10.1-gfx906 > /dev/null
    pip install -q --break-system-packages cmake ninja
    R=/opt/rocm/core-10.1
    export PATH=$R/bin:$R/lib/llvm/bin:$PATH ROCM_PATH=$R HIP_PATH=$R
    export HIP_DEVICE_LIB_PATH=$(dirname "$(find $R -name ockl.bc | head -1)")
    cmake -B build-r101 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
        -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_OPENSSL=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
    cmake --build build-r101 -j"$JOBS" --target ggml-hip test-backend-ops llama-server llama-bench llama-perplexity
    N=$BOXH/noble/usr/lib/x86_64-linux-gnu
    # every ELF file gets a DT_RPATH with the noble directory first: a library with a RUNPATH resolves its own
    # dependencies (libm, libstdc++) without the executable search path, so they would come from the box
    for f in build-r101/bin/*; do
        [ -f "$f" ] && head -c 4 "$f" | grep -q ELF || continue
        patchelf --force-rpath --set-rpath "\$ORIGIN:$N:$BOXH/rocm-10.1/lib:$BOXH/rocm-10.1/lib/llvm/lib:$BOXH/rocm-10.1/lib/rocm_sysdeps/lib" "$f"
        if patchelf --print-interpreter "$f" > /dev/null 2>&1; then patchelf --set-interpreter $N/ld-linux-x86-64.so.2 "$f"; fi
    done
    # the runtime libraries the build links or dlopens: everything in lib but the compiler, CMake files and static
    # archives; the LLVM shared libraries stay, comgr loads libclang-cpp and libLLVM at run time. The split-arch
    # device code of RCCL and rocBLAS is in .kpack (rccl_lib_gfx906.kpack, blas_lib_gfx906.kpack), next to lib: without
    # it every RCCL collective fails with "invalid device function"
    tar cf build-r101/rocm-10.1-lib.tar -C $R --exclude="./lib/llvm/bin" --exclude="./lib/llvm/include" \
        --exclude="./lib/llvm/share" --exclude="./lib/llvm/libexec" --exclude="./lib/llvm/lib/clang" --exclude="./lib/cmake" \
        --exclude="*.a" ./lib ./.kpack
    chown -R '"$(id -u):$(id -g)"' build-r101'
ls -la "$out/rocm-10.1-lib.tar" | awk '{printf "runtime bundle %.2f GB\n", $5 / 1e9}'
"$here/rig/box.sh" "$host" ssh "mkdir -p ~/rocm-10.1 ~/mx-llama.cpp/build-r101" < /dev/null
"$here/rig/box.sh" "$host" ssh "tar xf - -C ~/rocm-10.1" < "$out/rocm-10.1-lib.tar"
tar cf - -C "$out" bin | "$here/rig/box.sh" "$host" ssh 'tar xf - -C ~/mx-llama.cpp/build-r101'
echo "shipped build-r101 and ~/rocm-10.1"
