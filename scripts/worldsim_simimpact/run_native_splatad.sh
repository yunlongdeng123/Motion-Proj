#!/usr/bin/env bash
set -euo pipefail
N=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1
E=/root/autodl-tmp/envs/splatad-impact
export PATH="$E/bin:/root/autodl-tmp/envs/hugsim-impact/bin:/root/autodl-tmp/envs/motionproj/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=6
export TORCH_EXTENSIONS_DIR="$N/torch_extensions" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HTTP_PROXY=http://127.0.0.1:41841 HTTPS_PROXY=http://127.0.0.1:41841
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$E/bin/python" "$N/scripts/fit_native_splatad.py" "$@"
