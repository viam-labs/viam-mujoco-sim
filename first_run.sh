#!/usr/bin/env bash
# One-time setup, run by viam-server when the module is first installed.
# On Linux, install libEGL if it is missing so cameras can render offscreen
# on the GPU; then install the Python environment.
set -uo pipefail
cd "$(dirname "$0")"

log() { echo "viam-mujoco-sim first_run: $*"; }

if [ "$(uname -s)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
	SUDO=""
	if [ "$(id -u)" != "0" ]; then
		if sudo -n true 2>/dev/null; then SUDO="sudo -n"; else SUDO="unavailable"; fi
	fi
	if [ "$SUDO" = "unavailable" ]; then
		log "WARNING: libEGL.so.1 missing and no root or passwordless sudo; cameras will not render"
	else
		log "installing libegl1"
		export DEBIAN_FRONTEND=noninteractive
		$SUDO apt-get update -qq && $SUDO apt-get install -y -qq libegl1 \
			|| log "WARNING: libegl1 install failed; cameras will not render"
	fi
fi

. env.sh
uv sync --no-dev
