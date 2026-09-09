#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
qv2=$base/WS-V73-Q-V2-01
run=$qv2/20260909T131000Z__open-charts-joint-first-surface-s7304-r7
python_bin=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/ray_support
# 完整final结束后仅执行一次，同表示同物理目标r6为主要联合增量对照。
"$python_bin" scripts/summarize_worldsim_v73_global_results.py --run "$run" \
  --reference "first_surface_r6=$qv2/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6" \
  --reference "closed_joint_r1=$qv2/20260908T221500Z__shared-mesh-full-track-beam-s7304-r1" \
  --reference "lidar_r8=$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$run/analysis.json"
"$python_bin" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python_bin" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" \
  --output "$archive/V73_OPEN_JOINT_R7_TRAINING" --model-label 'Open joint r7'
"$python_bin" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_OPEN_JOINT_R7_PAIRS" --model-label 'Open joint r7' \
  --reference 'first_surface_r6=Matched LiDAR r6' --reference 'closed_joint_r1=Closed joint r1' --reference 'lidar_r8=Narrow R8' \
  --protocol-note 'Same r6 surface/objective/budget; Joint adds native seeds, trainable DPT features and build-depth auxiliary supervision.'
cp "$run/summary.json" "$archive/open_joint_r7_summary.json"
cp "$run/manifest.json" "$archive/open_joint_r7_final_manifest.json"
cp "$run/analysis.json" "$archive/open_joint_r7_analysis.json"
cp "$run/training_analysis.json" "$archive/open_joint_r7_training.json"
