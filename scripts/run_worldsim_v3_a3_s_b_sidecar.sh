#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

run_dir=$1
project=/root/autodl-tmp/motion_proj
drivestudio=/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1
d2_run=/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T113230Z__a2-d2-formal30k-s0-r1
d2_work=$d2_run/work_dirs/worldsim_v3_a2/scene0230_a2_d2_boundary_residual_formal_s0_i30000
registry=$d2_run/artifacts/evaluations/fixed-d2-boundary-residual/actor_registry.json

if [[ -e $run_dir/sidecar || -e $run_dir/exit_code ]]; then
  echo "refusing to overwrite A3 sidecar run: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir/logs"

cd "$project" || exit 2
export PYTHONPATH="$project:$drivestudio"
set +e
/root/autodl-tmp/envs/drivestudio/bin/python \
  scripts/materialize_worldsim_v3_a3_s_b_sidecar.py \
  --source-config "$d2_work/config.yaml" \
  --checkpoint "$d2_work/checkpoint_final.pth" \
  --registry "$registry" \
  --protocol configs/worldsim_v3/a3_local_refine_protocol_v1.yaml \
  --output-dir "$run_dir/sidecar" \
  --drivestudio-root "$drivestudio" \
  --device cuda:0 \
  --minimum-support-pixels 1 \
  --maximum-candidate-views 60 \
  --first-hit-alpha-threshold 0.5 \
  >"$run_dir/logs/materialize.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$run_dir/exit_code"
exit "$code"
