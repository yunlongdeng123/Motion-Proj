#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
actors=$base/WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1
out=$base/WS-V73-FINAL-SCENE-01/20260909T135000Z__fixed-r7-r6-external20-r1
# Actor确认的所有保存表面齐全后一次执行；背景固定，不按heldout修改或删面。
/root/autodl-tmp/envs/worldsim-v72-lidar4d/bin/python scripts/evaluate_worldsim_v73_scene_composition.py \
  --scene-data "$base/WS-V73-M4-AV2-SCENE-DATA-01/20260908T031000Z__external20-per-return-build-background-r1" \
  --actor-data "$base/WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1" \
  --model "joint_r7=$actors/joint_r7" --model "lidar_r6=$actors/lidar_r6" \
  --model "lidar_r8=$actors/lidar_r8" --model "native_fusion=$actors/native_fusion" --output "$out"
/root/autodl-tmp/envs/motionproj/bin/python scripts/summarize_worldsim_v73_scene_composition.py \
  --run "$out" --pair lidar_r6=joint_r7 --pair lidar_r8=joint_r7 --pair native_fusion=joint_r7 --output "$out/analysis.json"
