#!/usr/bin/env bash
# Runs the whole kit and collects everything into one results directory (and a tarball of
# it). Keeps going when a step fails; run.log says which. Most important steps first.
#   rig/run_all.sh [out dir]   NGPU=2 by default; MODEL=... LLAMA_BENCH=... for an end-to-end run
set -uo pipefail
out=$(realpath -m "${1:-$(dirname "$0")/results/$(hostname)-$(date +%Y%m%d-%H%M)}")
cd "$(dirname "$0")"
NGPU=${NGPU:-2}
mkdir -p "$out"
B=build

log() { echo "$*" | tee -a "$out/run.log"; }
step() { # step NAME OUTFILE cmd...
    local name=$1 file=$2
    shift 2
    if [ ! -x "$1" ]; then log "-- $name: skipped ($1 not built)"; return; fi
    log "-- $name: $*"
    local t0=$SECONDS
    "$@" > "$out/$file" 2> "$out/${file%.*}.err"
    log "   exit $? in $((SECONDS - t0)) s -> $file"
}

log "ruquant rig run, $(date -Is), $NGPU GPUs"
./probe.sh > "$out/env.txt" 2>&1

# 1. correctness on real gfx906: dot8 semantics, 64-lane layout, LDS reduction, epilogue
for g in $(seq 0 $((NGPU - 1))); do step "test_skinny gpu$g" "test-gpu$g.txt" $B/test_skinny "$g"; done
# 2. mi50emu parity: the MI50 must produce the spec's bits
for case in parity/*/; do
    [ -d "$case" ] || continue
    name=$(basename "$case")
    for g in $(seq 0 $((NGPU - 1))); do
        work=$(mktemp -d)
        cp "$case"/* "$work"/
        step "parity $name gpu$g" "parity-$name-gpu$g.txt" $B/w4a4_ref "$work" "$g"
        # keep what `parity.py compare` reads: shape, spec outputs, MI50 outputs (not the inputs)
        mkdir -p "$out/parity/$name-gpu$g"
        cp "$work"/shape.txt "$work"/*_spec.* "$work"/*_mi50.* "$out/parity/$name-gpu$g"/ 2>/dev/null
        rm -rf "$work"
    done
done
# 3. the model's hardware assumptions
for g in $(seq 0 $((NGPU - 1))); do step "valu gpu$g" "valu-gpu$g.json" $B/bench_valu "$g"; done
step "mem gpu0" mem-gpu0.json $B/bench_mem 0
[ "$NGPU" -gt 1 ] && step "mem gpu1" mem-gpu1.json $B/bench_mem 1 512
step "atomic gpu0" atomic-gpu0.json $B/bench_atomic 0
[ "$NGPU" -gt 1 ] && step "p2p" p2p.json $B/bench_p2p
[ "$NGPU" -gt 1 ] && step "tp $NGPU gpus" tp.json $B/bench_tp "$NGPU"
# 4. kernels against the model: ours (w4a4) and ggml's (today)
step "skinny gpu0" skinny-gpu0.json $B/bench_skinny 0
step "ggml gpu0" ggml-gpu0.json $B/ggml_bench 0
# 5. end-to-end autoregressive baseline, with a GGUF and a llama-bench already on the box
#    (MODEL=/path/model.gguf LLAMA_BENCH=/path/to/llama-bench; it runs from its own directory,
#    which also holds its ggml backends)
if [ -n "${MODEL:-}" ] && [ -x "${LLAMA_BENCH:-}" ]; then
    lb() ( cd "$(dirname "$LLAMA_BENCH")" && LD_LIBRARY_PATH="$PWD:/opt/rocm/lib:${LD_LIBRARY_PATH:-}" ./"$(basename "$LLAMA_BENCH")" "$@" )
    log "-- llama-bench: $MODEL"
    [ "$NGPU" -gt 1 ] && lb -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -sm layer > "$out/llama-bench.txt" 2> "$out/llama-bench.err" \
        || lb -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -sm none -mg 0 > "$out/llama-bench.txt" 2> "$out/llama-bench.err"
    log "   exit $? -> llama-bench.txt"
fi
if command -v rocm-smi >/dev/null; then rocm-smi --showclocks --showpower --showtemp > "$out/smi-after.txt" 2>&1
elif command -v amd-smi >/dev/null; then amd-smi metric --power --clock --temperature > "$out/smi-after.txt" 2>&1; fi

tar czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
log "done: $out.tar.gz  (bring it back; python3 rig/calibrate.py <dir> turns it into model flags)"
