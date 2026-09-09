#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
qv2=$base/WS-V73-Q-V2-01
run=$qv2/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3
python_bin=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/open_charts
# 完整final结束后仅执行一次；不生成新的神经推理或曝光20新日志。
"$python_bin" scripts/summarize_worldsim_v73_global_results.py --run "$run" \
  --reference "mesh_lidar_r2=$qv2/20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2" \
  --reference "lidar_r8=$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$run/analysis.json"
"$python_bin" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python_bin" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" \
  --output "$archive/V73_OPEN_CHARTS_lidar_r3_TRAINING" --model-label 'Open charts LiDAR r3'
"$python_bin" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_OPEN_CHARTS_lidar_r3_PAIRS" --model-label 'Open charts LiDAR r3' \
  --reference 'mesh_lidar_r2=Q-v2 closed mesh LiDAR r2' --reference 'lidar_r8=R8 narrow LiDAR patches' \
  --protocol-note 'Same LiDAR data and beam objective; support allocation, local shape, initialization and face budget change together.'
cp "$run/summary.json" "$archive/open_charts_lidar_r3_summary.json"
cp "$run/manifest.json" "$archive/open_charts_lidar_r3_final_manifest.json"
cp "$run/analysis.json" "$archive/open_charts_lidar_r3_analysis.json"
cp "$run/training_analysis.json" "$archive/open_charts_lidar_r3_training.json"
