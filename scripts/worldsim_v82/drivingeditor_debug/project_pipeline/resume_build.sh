#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
source "$BASE/scripts/activate.sh"
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6
unset PIP_EXTRA_INDEX_URL
printf '%s\n' "$$" > "$BASE/logs/setup.pid"
trap 'code=$?; printf "%s\n" "$code" > "$BASE/logs/setup.exit"' EXIT
python -m pip install 'setuptools==75.8.0'
# Remove only the malformed nvdiffrast wheel produced by setuptools 58.
python -m pip uninstall -y UNKNOWN
python -m pip install --no-deps --no-build-isolation "$BASE/deps/nvdiffrast"
python -m pip freeze > "$BASE/logs/environment.freeze.txt"
python -c 'import torch, open3d, quaternion, simple_knn._C, diff_gaussian_rasterization, nvdiffrast.torch; print("CORE_IMPORTS_OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
