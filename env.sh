# common setup for run.sh and first_run.sh
# Adapted from viam-modules/viam-isaac-sim (Apache-2.0); modified for MuJoCo.

# keep the venv in module data so it survives hot reloads
if [ -n "${VIAM_MODULE_DATA:-}" ]; then
	export UV_PROJECT_ENVIRONMENT="$VIAM_MODULE_DATA/venv"
fi

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >& /dev/null; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi
