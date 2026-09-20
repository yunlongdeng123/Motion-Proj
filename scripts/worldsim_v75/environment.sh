#!/usr/bin/env bash
# source 本文件后执行预检或手动启动；不会自行启动模型。
export PATH=/root/autodl-tmp/envs/worldsim-v75/bin:$PATH
export CUDA_HOME=/root/autodl-tmp/envs/worldsim-v75-tools/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export CPLUS_INCLUDE_PATH=/root/autodl-tmp/envs/worldsim-v75/lib/python3.12/site-packages/nvidia/cu13/include:$CUDA_HOME/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
export LIBRARY_PATH=$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export LOCAL_FILES_ONLY=1
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/torch-extensions-v75
