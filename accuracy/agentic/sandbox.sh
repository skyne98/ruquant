#!/usr/bin/env bash
# Runs a command in a bubblewrap sandbox: the whole filesystem read-only, /home and /tmp empty, only RUN (the run's
# directory: work tree, opencode's XDG dirs) writable. Network stays (the agent talks to llama-server on localhost)
# unless NONET=1 (used for the hidden tests).
#   RUN=dir [NONET=1] [SANDBOX_CWD=work] sandbox.sh CMD...
set -euo pipefail
env_dir=${AGENTIC_ENV:?}
net=()
[ "${NONET:-0}" = 1 ] && net=(--unshare-net)
exec bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /home --tmpfs /tmp --bind "$RUN" "$RUN" \
    --unshare-pid --unshare-ipc "${net[@]}" --die-with-parent --chdir "$RUN/${SANDBOX_CWD:-work}" \
    --setenv HOME "$RUN/home" --setenv XDG_CONFIG_HOME "$RUN/xdg/config" --setenv XDG_DATA_HOME "$RUN/xdg/data" \
    --setenv XDG_CACHE_HOME "$RUN/xdg/cache" --setenv XDG_STATE_HOME "$RUN/xdg/state" \
    --setenv PATH "$env_dir/bin:$(dirname "$(readlink -f "$(command -v opencode)")")" --setenv TMPDIR "$RUN/tmp" \
    --setenv OPENCODE_DISABLE_AUTOUPDATE 1 --setenv OPENCODE_DISABLE_LSP_DOWNLOAD 1 \
    "$@"
