#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
qv2=$base/WS-V73-Q-V2-01
run=$qv2/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6
python_bin=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/ray_support
# 完整final结束后仅执行一次，原同表示r3为主要监督对照。
"$python_bin" scripts/summarize_worldsim_v73_global_results.py --run "$run" \
  --reference "open_charts_r3=$qv2/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3" \
  --reference "ray_support_r4=$qv2/20260909T063700Z__open-charts-lidar-ray-support-s7304-r4" \
  --reference "lidar_r8=$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$run/analysis.json"
"$python_bin" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python_bin" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" \
  --output "$archive/V73_FIRST_SURFACE_R6_TRAINING" --model-label 'First surface r6'
"$python_bin" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_FIRST_SURFACE_R6_PAIRS" --model-label 'First surface r6' \
  --reference 'open_charts_r3=Open r3' --reference 'ray_support_r4=Attraction r4' --reference 'lidar_r8=Narrow R8' \
  --protocol-note 'Same r4 surface, sampling and weight; actual first-depth error when intersected, nearest-surface attraction only on miss.'
cp "$run/summary.json" "$archive/first_surface_r6_summary.json"
cp "$run/manifest.json" "$archive/first_surface_r6_final_manifest.json"
cp "$run/analysis.json" "$archive/first_surface_r6_analysis.json"
cp "$run/training_analysis.json" "$archive/first_surface_r6_training.json"
