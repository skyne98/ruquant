#!/usr/bin/env bash
# Copies a big file to a box through its Cloudflare tunnel, resuming after a dropped connection, and checks the size.
#   rig/upload.sh FILE [user@]HOST DEST
# Reads with O_DIRECT so a 17 GB model doesn't evict what local jobs have in the page cache.
set -uo pipefail
cd "$(dirname "$0")/.."
file=$1
host=$2
dest=$3
size=$(stat -c %s "$file")
for try in 1 2 3 4 5 6; do
    have=$(rig/box.sh "$host" ssh "stat -c %s '$dest.part' 2>/dev/null || echo 0" 2>/dev/null | tail -1)
    have=${have:-0}
    if [ "$have" -eq "$size" ]; then
        rig/box.sh "$host" ssh "mv '$dest.part' '$dest'" && echo "$dest: $size bytes" && exit 0
    fi
    if [ "$have" -gt "$size" ]; then have=0; rig/box.sh "$host" ssh "rm -f '$dest.part'"; fi
    echo "try $try: resuming at $have of $size bytes" >&2
    if [ "$have" -eq 0 ]; then
        dd if="$file" iflag=direct bs=8M status=none | rig/box.sh "$host" ssh "cat > '$dest.part'"
    else
        tail -c +$((have + 1)) "$file" | rig/box.sh "$host" ssh "cat >> '$dest.part'"
    fi
done
echo "$file: upload failed" >&2
exit 1
