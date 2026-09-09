#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4
export CUDA_HOME=/root/autodl-tmp/envs/worldsim-v72-pointr
export PATH=/root/autodl-tmp/envs/motionproj/bin:$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions/worldsim_v73
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6
exec /root/autodl-tmp/envs/motionproj/bin/python scripts/check_worldsim_v73_open_charts.py
