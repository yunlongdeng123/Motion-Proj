#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [PROTOCOL]" >&2
  exit 2
fi

run_dir=$1
project=/root/autodl-tmp/motion_proj
protocol=${2:-$project/configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml}
if [[ -e $run_dir || -e ${run_dir}.controller.log || -e ${run_dir}.exit_code ]]; then
  echo "refusing to overwrite A4-P5 run: $run_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$run_dir")"
cd "$project" || exit 2

export PYTHONPATH="$project:/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1:${PYTHONPATH:-}"
export TORCH_HOME=/root/autodl-tmp/cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/cache/xdg

log=${run_dir}.controller.log
code=0
/root/autodl-tmp/envs/motionproj/bin/python \
  scripts/run_worldsim_v3_a4_p5_registry.py \
  --run-dir "$run_dir" \
  --protocol "$protocol" >"$log" 2>&1 || code=$?

if [[ $code -eq 0 ]]; then
  /root/autodl-tmp/envs/drivestudio/bin/python \
    scripts/run_worldsim_v3_a4_p5_reload_smoke.py \
    --run-dir "$run_dir" \
    --protocol "$protocol" >>"$log" 2>&1 || code=$?
fi

if [[ $code -eq 0 ]]; then
  /root/autodl-tmp/envs/motionproj/bin/python \
    scripts/aggregate_worldsim_v3_a4_p5.py \
    --run-dir "$run_dir" >>"$log" 2>&1 || code=$?
fi

if [[ $code -eq 0 ]]; then
  /root/autodl-tmp/envs/motionproj/bin/python \
    scripts/audit_worldsim_v3_a4_p5_resume.py \
    --run-dir "$run_dir" >>"$log" 2>&1 || code=$?
fi

if [[ $code -eq 0 ]]; then
  /root/autodl-tmp/envs/motionproj/bin/python \
    scripts/finalize_worldsim_v3_a4_p5.py \
    --run-dir "$run_dir" \
    --protocol "$protocol" >>"$log" 2>&1 || code=$?
fi

printf '%s\n' "$code" >"${run_dir}.exit_code"
exit "$code"
