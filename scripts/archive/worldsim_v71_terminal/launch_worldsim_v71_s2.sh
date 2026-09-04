#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
repo_root="/root/autodl-tmp/motion_proj"
log_root="/root/autodl-tmp/runs/worldsim_v71/launch_logs"

mkdir -p "${log_root}"
cd "${repo_root}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec python scripts/build_worldsim_v71_actor_corpus.py \
  --config configs/worldsim_v71/v71_s2_actor_corpus_v1.yaml \
  --repo-root "${repo_root}" \
  --run-id "${run_id}" \
  > "${log_root}/${run_id}.log" 2>&1
