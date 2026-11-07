#!/usr/bin/env bash
# see Makefile for why this gets copied to run.sh
# Adapted from viam-modules/viam-isaac-sim (Apache-2.0); modified for MuJoCo.
set -euo pipefail
cd "$(dirname "$0")"

# offscreen OpenGL for camera rendering: CGL on macOS, EGL (the GPU) on Linux
if [ -z "${MUJOCO_GL:-}" ]; then
	if [ "$(uname -s)" = "Darwin" ]; then export MUJOCO_GL=cgl; else export MUJOCO_GL=egl; fi
fi

. env.sh

exec uv run --no-dev src/main.py "$@"
