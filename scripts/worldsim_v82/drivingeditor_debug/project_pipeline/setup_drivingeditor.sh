#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
ENV=/root/autodl-tmp/envs/drivingeditor
export TMPDIR="$BASE/tmp" CONDA_PKGS_DIRS="$BASE/cache/conda"
export PIP_CACHE_DIR="$BASE/cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$BASE/deps"
printf '%s\n' "$$" > "$BASE/logs/setup.pid"
trap 'code=$?; printf "%s\n" "$code" > "$BASE/logs/setup.exit"' EXIT
if [ ! -x "$ENV/bin/python" ]; then
  /root/miniconda3/bin/conda create -y -p "$ENV" python=3.9 pip
fi
python -m pip install 'pip<26' 'setuptools==75.8.0' wheel 'numpy==1.26.4' ninja
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'numpy==1.26.4' 'opencv-python==4.10.0.84' 'opencv-contrib-python==4.10.0.84' 'kornia<0.8' 'open3d==0.18.0' numpy-quaternion scipy scikit-image scikit-learn tqdm 'imageio[ffmpeg]' trimesh Pillow omegaconf plyfile tensorboardX matplotlib einops pandas rich lpips
if [ ! -d "$BASE/deps/simple-knn/.git" ]; then
  git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$BASE/deps/simple-knn"
fi
python -m pip install --no-build-isolation "$BASE/deps/simple-knn"
if [ ! -d "$BASE/deps/diff-gaussian-rasterization/.git" ]; then
  git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization "$BASE/deps/diff-gaussian-rasterization"
fi
python -m pip install --no-build-isolation "$BASE/deps/diff-gaussian-rasterization"
if [ ! -d "$BASE/deps/nvdiffrast/.git" ]; then
  git clone https://github.com/NVlabs/nvdiffrast "$BASE/deps/nvdiffrast"
fi
python -m pip install --no-build-isolation "$BASE/deps/nvdiffrast"
python -m pip freeze > "$BASE/logs/environment.freeze.txt"
python -c 'import torch, open3d, quaternion, simple_knn._C, diff_gaussian_rasterization, nvdiffrast.torch; print("CORE_IMPORTS_OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
