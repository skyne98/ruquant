#!/usr/bin/env bash
# Records one rocprofv3 profile on a box, kept for later study: never overwritten, with what made it.
#   rig/prof.sh LABEL [VAR=value ...] -- COMMAND [ARGS ...]
#   env PROF_ROOT (default ~/profiles), ROCPROFV3 (default ~/therock-7.12/bin/rocprofv3), PROF_ARGS (rocprofv3
#   options, default --kernel-trace), ROCPROFILER_METRICS_PATH (extra counter definitions, e.g. ~/tp/rpdefs)
# Writes PROF_ROOT/<UTC time>-LABEL/:
#   out/    rocprofv3's CSV files
#   run.log the command's output
#   meta.txt the command, the VAR=value settings, every RQ_ / GGML_ / HIP_ / HSA_ / ROCPROFILER_ / LD_ variable, the
#            rocprofv3 options, host, GPUs, the md5 of the binary's directory's libggml-hip.so, the exit code
# Pull them to the repo with rig/prof_pull.sh.
set -uo pipefail
label=$1
shift
vars=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    vars+=("$1")
    shift
done
[ "${1:-}" = "--" ] && shift
root=${PROF_ROOT:-$HOME/profiles}
rp=${ROCPROFV3:-$HOME/therock-7.12/bin/rocprofv3}
args=${PROF_ARGS:---kernel-trace}
dir=$root/$(date -u +%Y%m%dT%H%M%SZ)-$label
mkdir -p "$dir/out"
bin=$(command -v "$1" || echo "$1")
{
    echo "label: $label"
    echo "time: $(date -u +%FT%TZ)"
    echo "host: $(hostname)"
    echo "command: $*"
    echo "settings: ${vars[*]:-}"
    echo "rocprofv3: $rp $args"
    echo "binary: $bin"
    lib=$(dirname "$bin")/libggml-hip.so
    [ -f "$lib" ] && echo "libggml-hip.so md5: $(md5sum < "$lib" | cut -c1-32)"
    echo "gpus: $(lspci 2>/dev/null | grep -ciE 'vga|display|3d') ($(grep -o 'gfx[0-9a-z]*' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | sort -u | tr '\n' ' '))"
    echo "environment:"
    env | grep -E '^(RQ_|GGML_|HIP_|HSA_|ROCPROFILER_|LD_LIBRARY_PATH)' | sort | sed 's/^/  /'
} > "$dir/meta.txt"
# shellcheck disable=SC2086
env "${vars[@]}" "$rp" $args --output-format csv -d "$dir/out" -o tr -- "$@" > "$dir/run.log" 2>&1
code=$?
echo "exit: $code" >> "$dir/meta.txt"
echo "$dir"
exit $code
