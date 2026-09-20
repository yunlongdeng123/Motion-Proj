#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/autodl-tmp/envs/splatad-impact/bin:/root/autodl-tmp/envs/hugsim-impact/bin:/root/autodl-tmp/envs/motionproj/bin:/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1/torch_extensions
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled PYTHONUNBUFFERED=1
exec /root/autodl-tmp/envs/splatad-impact/bin/python /root/autodl-tmp/motion_proj/scripts/worldsim_simimpact/render_sparse_native_sensor_pairs.py
