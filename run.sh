#!/usr/bin/env bash
# Dev runner. Two NixOS specifics:
#  1. ggml is built shared into build-ggml/bin.
#  2. cudatoolkit ships a *stub* libcuda.so; the real driver must be searched first.
set -euo pipefail
export LD_LIBRARY_PATH="$PWD/build-ggml/bin:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
exec ./target/release/ruquant "$@"
