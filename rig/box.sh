#!/usr/bin/env bash
# Drives an MI50 box from here, through its Cloudflare tunnel. A quick tunnel's hostname
# changes whenever the owner restarts it, so it's an argument, not ssh config. HOST is
# [user@]hostname, user gpu by default.
#   rig/box.sh HOST ssh [cmd...]   shell, or one command
#   rig/box.sh HOST push           send dist/ruquant-rig.tar.gz and unpack it in ~/ruquant
#   rig/box.sh HOST start CMD...   run CMD in ~/ruquant under nohup (survives a dropped tunnel);
#                                  output goes to ~/ruquant/rig/box.log
#   rig/box.sh HOST log            tail of that log
#   rig/box.sh HOST pull           newest results tarball -> rig/results/, unpacked
set -euo pipefail
cd "$(dirname "$0")/.."
host=${1:?usage: rig/box.sh [user@]HOST ssh|push|start|log|pull}
host=${host#https://}
user=gpu
[[ $host == *@* ]] && user=${host%@*} host=${host#*@}
shift
cf=$(command -v cloudflared || echo "$(nix build --no-link --print-out-paths nixpkgs#cloudflared)/bin/cloudflared")
opts=(-o ProxyCommand="$cf access ssh --hostname $host" -o StrictHostKeyChecking=accept-new
      -o ServerAliveInterval=30 -i "${KEY:-$HOME/.ssh/id_ed25519}")
box() { ssh "${opts[@]}" "$user@$host" "$@"; }

case ${1:-ssh} in
ssh) shift || true; box "$@" ;;
push)
    scp "${opts[@]}" dist/ruquant-rig.tar.gz "$user@$host:"
    box 'rm -rf ruquant && tar xzf ruquant-rig.tar.gz && ls ruquant'
    ;;
start)
    shift
    box "cd ruquant && { nohup bash -c '$*' > rig/box.log 2>&1 < /dev/null & } && echo started"
    ;;
log) box 'tail -n 40 ruquant/rig/box.log' ;;
pull)
    f=$(box 'ls -t ruquant/rig/results/*.tar.gz | head -1')
    mkdir -p rig/results
    scp "${opts[@]}" "$user@$host:$f" rig/results/
    tar xzf "rig/results/$(basename "$f")" -C rig/results
    echo "rig/results/$(basename "$f" .tar.gz)"
    ;;
*) echo "unknown command $1" >&2; exit 2 ;;
esac
