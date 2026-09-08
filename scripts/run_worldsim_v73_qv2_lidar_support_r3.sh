#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
# 固定已完成表面的一次诊断；75 DEV全纳入，不调用神经模型或修改任何表面。
/root/autodl-tmp/envs/worldsim-v72-lidar4d/bin/python scripts/analyze_worldsim_v73_surface_support.py \
  --actor-data "$base/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2" \
  --model "mesh_lidar_r2=$base/WS-V73-Q-V2-01/20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2" \
  --model "lidar_r8=$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$base/WS-V73-M2-SURFACE-SUPPORT-01/20260908T235000Z__qv2-lidar-r2-vs-r8-support-r3"
