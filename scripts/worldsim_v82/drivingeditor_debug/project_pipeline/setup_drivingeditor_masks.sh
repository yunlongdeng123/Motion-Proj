#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
ENV=/root/autodl-tmp/envs/drivingeditor
export PATH="$ENV/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export TMPDIR="$BASE/tmp" PIP_CACHE_DIR="$BASE/cache/pip"
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
unset PIP_EXTRA_INDEX_URL
trap 'code=$?; printf "%s\n" "$code" > "$BASE/logs/mask_setup.exit"' EXIT
# Pin DEVA extras so resolving its dependencies preserves core torch/numpy.
python -m pip install 'numpy==1.26.4' 'opencv-python==4.10.0.84' 'opencv-python-headless==4.10.0.84' 'supervision==0.14.0' 'timm==0.9.16' 'PuLP==2.7.0' pycocotools 'transformers==4.38.2' addict yapf tensorboard hickle gitpython 'gdown<6' hatchling 'gradio==3.50.2' 'gurobipy==12.0.3'
HTTPS_PROXY=http://127.0.0.1:36629 python -m pip install 'thinplate@git+https://github.com/cheind/py-thin-plate-spline'
python -m pip install --no-deps -e "$BASE/DrivingEditor-run/Tracking-Anything-with-DEVA"
if [ ! -d "$BASE/deps/Grounded-Segment-Anything/.git" ]; then
  git -c http.proxy=http://127.0.0.1:36629 clone https://github.com/hkchengrex/Grounded-Segment-Anything.git "$BASE/deps/Grounded-Segment-Anything"
fi
python -m pip install --no-deps -e "$BASE/deps/Grounded-Segment-Anything/segment_anything"
AM_I_DOCKER=False BUILD_WITH_CUDA=True python -m pip install --no-deps --no-build-isolation -e "$BASE/deps/Grounded-Segment-Anything/GroundingDINO"
python -c 'import deva, segment_anything; from deva.ext.automatic_sam import get_sam_model; from groundingdino.models import build_model; print("MASK_IMPORTS_OK")'
python -m pip freeze > "$BASE/logs/environment.freeze.txt"
