#!/bin/bash
# A user-space ROCm 7.2.4 next to the box's /opt/rocm (no root), and mx-llama.cpp built against it in build-r72.
#   rig/rocm72/setup.sh    (run on the box from the ruquant checkout; needs ~3 GB, plus ~2 GB for the build)
# rocBLAS 7.2.4 ships no gfx906 kernels (6.4 dropped them), so its Tensile library dir gets the gfx906 files of
# rocBLAS 6.3.3, which 7.2.4 loads fine (5.7's files segfault in loadPlaceholderLibrary). hipBLASLt and rocSOLVER are
# only imported, never called on this path, so they are stubs (mkstub.py) instead of 2.4 GB of downloads.
set -euo pipefail
V=7.2.4
D=$HOME/rocm-dl; P=$HOME/rocm-$V; R=$P/opt/rocm-$V; here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D" "$P"; cd "$D"
curl -s -O "https://repo.radeon.com/rocm/apt/$V/dists/jammy/main/binary-amd64/Packages.gz"
python3 - "$V" <<'PY'
import gzip, re, sys
pk = {}
for blk in gzip.open("Packages.gz").read().decode().split("\n\n"):
    d = dict(re.findall(r"^([A-Za-z-]+): (.*)$", blk, re.M))
    if "Package" in d: pk[d["Package"]] = d
want = ["rocm-core", "hsa-rocr", "hsa-rocr-dev", "comgr", "hip-runtime-amd", "hip-dev", "hipcc", "rocm-llvm",
        "rocm-device-libs", "rocminfo", "rocprofiler-register", "hipblas", "hipblas-dev", "hipblas-common-dev",
        "rocblas", "rocblas-dev"]
open("urls.txt", "w").write("".join("https://repo.radeon.com/rocm/apt/%s/%s\n" % (sys.argv[1], pk[w]["Filename"]) for w in want))
PY
echo https://repo.radeon.com/rocm/apt/6.3.3/pool/main/r/rocblas/rocblas_4.3.0.60303-74~22.04_amd64.deb >> urls.txt
xargs -n1 -P4 curl -s -O < urls.txt
for f in *70204*.deb; do dpkg-deb -x "$f" "$P"; done
rm -rf old && mkdir old && dpkg-deb -x rocblas_4.3.0.60303-*.deb old
L=$R/lib/rocblas/library; O=old/opt/rocm-6.3.3/lib/rocblas/library
rm -rf "$L" && mkdir "$L" && cp "$O"/*gfx906* "$O"/*fallback.dat "$L"/
sys="$R/lib/libamdhip64.so.7 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /lib/x86_64-linux-gnu/libm.so.6
     /lib/x86_64-linux-gnu/libgcc_s.so.1 /lib/x86_64-linux-gnu/libc.so.6 /lib64/ld-linux-x86-64.so.2"
python3 "$here/mkstub.py" "$R/lib/librocblas.so.5" hipblaslt_stub.c $sys
python3 "$here/mkstub.py" "$R/lib/libhipblas.so.3" rocsolver_stub.c $sys "$R/lib/librocblas.so.5"
gcc -O2 -shared -fPIC -Wl,-soname,libhipblaslt.so.1 hipblaslt_stub.c -o "$R/lib/libhipblaslt.so.1"
gcc -O2 -shared -fPIC -Wl,-soname,librocsolver.so.0 rocsolver_stub.c -o "$R/lib/librocsolver.so.0"
export PATH=$HOME/.local/bin:$R/bin:$R/lib/llvm/bin:$PATH ROCM_PATH=$R HIP_PATH=$R HIP_DEVICE_LIB_PATH=$R/amdgcn/bitcode
cd ~/mx-llama.cpp
cmake -B build-r72 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-r72 -j12 --target ggml-hip test-backend-ops llama-server llama-bench llama-perplexity
echo "run with: LD_LIBRARY_PATH=~/mx-llama.cpp/build-r72/bin:$R/lib ..."
