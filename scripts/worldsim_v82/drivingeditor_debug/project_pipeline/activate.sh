#!/usr/bin/env bash
# Source this file before running the official training entry points.
export DRIVINGEDITOR_ROOT=/root/autodl-tmp/drivingeditor_repro
export PATH="/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_HOME="$DRIVINGEDITOR_ROOT/cache/torch"
export HF_HOME="$DRIVINGEDITOR_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$DRIVINGEDITOR_ROOT/cache"
export TMPDIR="$DRIVINGEDITOR_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$DRIVINGEDITOR_ROOT/cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="$DRIVINGEDITOR_ROOT/cache/matplotlib"
export PYTHONNOUSERSITE=1
