#!/usr/bin/env bash
# Builds dist/ruquant-rig.tar.gz: the repo (incl. the patched llama.cpp for build_ggml.sh),
# mi50emu's gfx906 parity runner and freshly exported parity vectors. Nothing on the MI50 box
# needs nix or numpy.
#   rig/pack.sh          (MI50EMU=../mi50emu by default)
set -euo pipefail
cd "$(dirname "$0")/.."
MI50EMU=${MI50EMU:-$PWD/../mi50emu}
stage=$(mktemp -d)
dst=$stage/ruquant

mkdir -p "$dst"
git ls-files -co --exclude-standard -z | tar --null -T - -cf - | tar -xf - -C "$dst"
mkdir -p "$dst/rig/mi50emu" "$dst/rig/parity"
cp "$MI50EMU"/gfx906/*.hip* "$dst/rig/mi50emu/"

py=(python3)
python3 -c 'import numpy' 2>/dev/null || py=(nix-shell -p python3Packages.numpy --run)
export_case() { # name M N K
    local cmd="python3 $MI50EMU/tools/parity.py export $dst/rig/parity/$1 --M $2 --N $3 --K $4"
    if [ "${py[0]}" = python3 ]; then $cmd; else "${py[@]}" "$cmd"; fi
}
export_case m333_n4096_k4096 333 4096 4096 # mi50emu's default case
export_case m8_n2560_k6144 8 2560 6144     # a verify step (T=8) through Flash-Next's o_proj

mkdir -p dist
tar czf dist/ruquant-rig.tar.gz -C "$stage" ruquant
rm -rf "$stage"
echo "dist/ruquant-rig.tar.gz ($(du -h dist/ruquant-rig.tar.gz | cut -f1))"
