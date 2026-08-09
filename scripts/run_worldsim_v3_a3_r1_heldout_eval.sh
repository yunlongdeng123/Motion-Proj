#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

run_dir=$1
project=/root/autodl-tmp/motion_proj
if [[ -e $run_dir || -e ${run_dir}.controller.log || -e ${run_dir}.exit_code ]]; then
  echo "refusing to overwrite A3 heldout evaluation: $run_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$run_dir")"

cd "$project" || exit 2
export PYTHONPATH="$project:/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1"
export WANDB_MODE=disabled
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_HOME=/root/autodl-tmp/cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/cache/xdg

/root/autodl-tmp/envs/drivestudio/bin/python \
  scripts/eval_worldsim_v3_a3_r1_heldout.py \
  --run-dir "$run_dir" \
  >"${run_dir}.controller.log" 2>&1
code=$?
printf '%s\n' "$code" >"${run_dir}.exit_code"
exit "$code"
