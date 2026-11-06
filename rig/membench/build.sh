#!/usr/bin/env bash
# Builds rig/membench/stream_bench for gfx906 with TheRock 7.12 in an Ubuntu 22.04 container (glibc of box 1) and ships
# it to the box's ~/rig/membench:   rig/membench/build.sh [HOST]   env THEROCK_DIR (default ~/Experiments/rocm/therock-7.12)
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
T=${THEROCK_DIR:-$HOME/Experiments/rocm/therock-7.12}
docker run --rm -v "$here/rig/membench":/src -v "$T":/opt/therock:ro -w /src ubuntu:22.04 bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq g++ > /dev/null
    export HIP_DEVICE_LIB_PATH=$(dirname "$(find /opt/therock/lib/llvm -name ockl.bc | head -1)")
    /opt/therock/lib/llvm/bin/clang++ -x hip -O3 --offload-arch=gfx906 --rocm-path=/opt/therock -std=c++17 stream_bench.hip \
        -o stream_bench -L/opt/therock/lib -lamdhip64 -Wl,-rpath,/opt/therock/lib
    chown '"$(id -u):$(id -g)"' stream_bench'
if [ -n "${1:-}" ]; then
    "$here/rig/box.sh" "$1" ssh 'mkdir -p ~/rig/membench && cat > ~/rig/membench/stream_bench && chmod +x ~/rig/membench/stream_bench' \
        < "$here/rig/membench/stream_bench"
fi
echo "built rig/membench/stream_bench"
