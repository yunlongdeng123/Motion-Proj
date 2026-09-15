#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1
B=/root/autodl-tmp/external/worldsim_simimpact
E=/root/autodl-tmp/envs/splatad-impact
export PATH="$E/bin:/root/autodl-tmp/envs/motionproj/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=6
export TORCH_EXTENSIONS_DIR="$R/torch_extensions"
export HTTP_PROXY=http://127.0.0.1:41841 HTTPS_PROXY=http://127.0.0.1:41841
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=https://pypi.org/simple
"$E/bin/python" -m pip install 'setuptools<70' wheel
"$E/bin/python" -m pip install --no-deps --no-build-isolation -e "$B/neurad-studio" -e "$B/SplatAD_splat" -e "$B/viser-splatad"
"$E/bin/python" "$R/scripts/probe_splatad_dependencies.py"
