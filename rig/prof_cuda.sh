#!/usr/bin/env bash
# Records one Nsight profile on a local NVIDIA GPU, kept for later study: never overwritten, with what made it.
#   rig/prof_cuda.sh LABEL [VAR=value ...] -- COMMAND [ARGS ...]
#   env PROF_TOOL (nsys, default, or ncu), PROF_ARGS (the tool's options; nsys default: --stats=true with CUDA
#   kernel tracing; ncu default: --set full), PROF_ROOT (default profiles/<gpu name>, e.g. profiles/3090)
# Writes PROF_ROOT/<UTC time>-LABEL.tar.gz (out/: the report and its stats, run.log: the command's output, meta.txt)
# and PROF_ROOT/<UTC time>-LABEL.meta.txt next to it: the command, the VAR=value settings, every RQ_ / GGML_ / CUDA_ /
# LD_ variable, the tool and options, host, GPU, the md5 of the binary's directory's libggml-cuda.so, the exit code.
set -uo pipefail
cd "$(dirname "$0")/.."
label=$1
shift
vars=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    vars+=("$1")
    shift
done
[ "${1:-}" = "--" ] && shift
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
short=$(echo "$gpu" | grep -oE '[0-9]{4}[A-Za-z]*' | head -1)
root=${PROF_ROOT:-profiles/${short:-cuda}}
tool=${PROF_TOOL:-nsys}
name=$(date -u +%Y%m%dT%H%M%SZ)-$label
dir=$(mktemp -d)/$name
mkdir -p "$dir/out" "$root"
bin=$(command -v "$1" || echo "$1")
{
    echo "label: $label"
    echo "time: $(date -u +%FT%TZ)"
    echo "host: $(hostname)"
    echo "gpu: $gpu (driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1))"
    echo "command: $*"
    echo "settings: ${vars[*]:-}"
    echo "binary: $bin"
    lib=$(dirname "$bin")/libggml-cuda.so
    [ -f "$lib" ] && echo "libggml-cuda.so md5: $(md5sum < "$lib" | cut -c1-32)"
    echo "environment:"
    env | grep -E '^(RQ_|GGML_|CUDA_|LD_LIBRARY_PATH)' | sort | sed 's/^/  /'
} > "$dir/meta.txt"
if [ "$tool" = ncu ]; then
    args=${PROF_ARGS:---set full}
    echo "tool: ncu $args" >> "$dir/meta.txt"
    # shellcheck disable=SC2086
    env "${vars[@]}" ncu $args -o "$dir/out/report" -f -- "$@" > "$dir/run.log" 2>&1
else
    args=${PROF_ARGS:---trace=cuda --stats=true}
    echo "tool: nsys profile $args" >> "$dir/meta.txt"
    # shellcheck disable=SC2086
    env "${vars[@]}" nsys profile $args -o "$dir/out/report" -- "$@" > "$dir/run.log" 2>&1
fi
code=$?
echo "exit: $code" >> "$dir/meta.txt"
tar czf "$root/$name.tar.gz" -C "$(dirname "$dir")" "$name"
cp "$dir/meta.txt" "$root/$name.meta.txt"
rm -rf "$(dirname "$dir")"
echo "$root/$name.tar.gz"
exit $code
