#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
exec /root/autodl-tmp/envs/worldsim-v72-lidar4d/bin/python scripts/analyze_worldsim_v73_surface_support.py \
  --actor-data "$base/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2" \
  --model "open_charts_r3=$base/WS-V73-Q-V2-01/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3" \
  --model "ray_support_r4=$base/WS-V73-Q-V2-01/20260909T063700Z__open-charts-lidar-ray-support-s7304-r4" \
  --output "$base/WS-V73-M2-SURFACE-SUPPORT-01/20260909T110600Z__open-charts-r3-ray-r4-support-r5"
