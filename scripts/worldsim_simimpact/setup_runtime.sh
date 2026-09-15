#!/usr/bin/env bash
set -euo pipefail
export HTTP_PROXY=http://127.0.0.1:41841 HTTPS_PROXY=http://127.0.0.1:41841
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export NO_PROXY=localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn,mirrors.aliyun.com
export no_proxy="$NO_PROXY"
export MAX_JOBS=6 CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 TCNN_CUDA_ARCHITECTURES=86
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH="/root/autodl-tmp/envs/hugsim-impact/bin:/root/autodl-tmp/envs/motionproj/bin:$CUDA_HOME/bin:$PATH"
BASE=/root/autodl-tmp/external/worldsim_simimpact
ENV=/root/autodl-tmp/envs/hugsim-impact
if [ ! -x "$ENV/bin/python" ]; then
  /root/autodl-tmp/envs/nksr-v74/bin/python -m venv --system-site-packages "$ENV"
fi
PY="$ENV/bin/python"
"$PY" /root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1/scripts/link_torch118.py
"$PY" -m pip install --no-deps torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
"$PY" -m pip install 'numpy<2' gymnasium==1.1.1 roma==1.5.2.1 pytorch-lightning==2.2.1 timm==1.0.15 moviepy==2.2.1 positional-encodings==6.0.1 pyogrio rtree ujson pyquaternion jaxtyping ninja rich trimesh plyfile==1.1 mediapy
cd "$BASE"
if [ ! -d HUGSIM_splat ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none --sparse https://github.com/hyzhou404/HUGSIM_splat.git
fi
git -C HUGSIM_splat sparse-checkout init --cone
git -C HUGSIM_splat sparse-checkout set gsplat
git -C HUGSIM_splat submodule update --init --depth 1
"$PY" -m pip install --no-build-isolation --no-deps ./HUGSIM_splat
"$PY" -m pip install --no-build-isolation --no-deps ./HUGSIM/submodules/simple-knn
if [ ! -d tiny-cuda-nn ]; then git clone --depth 1 --recursive --shallow-submodules https://github.com/NVlabs/tiny-cuda-nn.git; fi
"$PY" -m pip install --no-build-isolation --no-deps ./tiny-cuda-nn/bindings/torch
if [ ! -d trajdata ]; then git clone --depth 1 https://github.com/hyzhou404/trajdata.git; fi
"$PY" -m pip install --no-deps ./trajdata
if [ ! -d nuplan-devkit ]; then git clone --depth 1 --branch nuplan-devkit-v1.2 https://github.com/motional/nuplan-devkit.git; fi
"$PY" -m pip install --no-deps ./nuplan-devkit
"$PY" -m pip install --no-deps -e ./HUGSIM/sim -e ./NAVSIM
"$PY" -m pip list --format=freeze > /root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1/runtime_packages.txt
echo SETUP_COMPLETE
