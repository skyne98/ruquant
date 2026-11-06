#!/usr/bin/env bash
# Copies a box's finished recorded profiles (rig/prof.sh, PROF_ROOT on the box) into the repo, one compressed archive per
# profile, skipping those already here:
#   rig/prof_pull.sh HOST [DEST]      DEST default profiles/box1; env PROF_ROOT (box side, default ~/profiles)
# Each profile becomes DEST/<name>.tar.gz plus DEST/<name>.meta.txt (its meta.txt, readable without unpacking).
# profiles/INDEX.md records what each one is for and what it showed.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
host=$1
dest=${2:-$here/profiles/box1}
root=${PROF_ROOT:-'~/profiles'}
mkdir -p "$dest"
# only finished profiles: rig/prof.sh writes meta.txt's last line, exit:, when the run ends
for name in $("$here/rig/box.sh" "$host" ssh "cd $root && grep -l '^exit:' */meta.txt | cut -d/ -f1" < /dev/null); do
    [ -f "$dest/$name.tar.gz" ] && continue
    "$here/rig/box.sh" "$host" ssh "tar czf - -C $root $name" < /dev/null > "$dest/$name.tar.gz.part"
    mv "$dest/$name.tar.gz.part" "$dest/$name.tar.gz"
    tar xzf "$dest/$name.tar.gz" -O "$name/meta.txt" > "$dest/$name.meta.txt" 2>/dev/null || true
    echo "pulled $name ($(du -h "$dest/$name.tar.gz" | cut -f1))"
done
