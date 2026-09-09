#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
python_bin=/root/autodl-tmp/envs/worldsim-v72-lidar4d/bin/python
actors="$base/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2"
joint="$base/WS-V73-Q-V2-01/20260908T221500Z__shared-mesh-full-track-beam-s7304-r1"
lidar="$base/WS-V73-Q-V2-01/20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2"
# r2已有沿束支持统计直接复用；这里仅新增联合r1的固定表面分解。
"$python_bin" scripts/analyze_worldsim_v73_surface_support.py \
  --actor-data "$actors" --model "mesh_joint_r1=$joint" \
  --output "$base/WS-V73-M2-SURFACE-SUPPORT-01/20260909T045000Z__qv2-joint-r1-support-r4"
"$python_bin" scripts/analyze_worldsim_v73_mesh_deformation.py \
  --actor-data "$actors" --model "mesh_joint_r1=$joint" --model "mesh_lidar_r2=$lidar" \
  --output "$base/WS-V73-Q-V2-MESH-DIAGNOSTIC-01/20260909T045000Z__fixed-dev-mesh-deformation-r1"
