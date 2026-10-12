#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
ENV=/root/autodl-tmp/envs/drivingeditor
export PATH="$ENV/bin:/usr/local/cuda-11.8/bin:$PATH"
export TORCH_HOME="$BASE/cache/torch"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$ENV/bin/python" "$BASE/scripts/run_deva_automatic.py" "$@"
