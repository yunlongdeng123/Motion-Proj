#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
run=/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-GLOBAL-ACTOR-01/20260908T100000Z__population-native-full-track-beam-range-s7304-r14
base=/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-GLOBAL-ACTOR-01
python=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/m2/global
# 只读取本次已完成结果和既有对照，不重新推理或改变正在运行的R12。
"$python" scripts/summarize_worldsim_v73_global_results.py --run "$run" \
  --reference "native_r11=$base/20260907T233000Z__population-native-only-full-track-s7304-r11" \
  --reference "lidar_r8=$base/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$run/analysis.json"
"$python" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" --output "$archive/V73_NATIVE_R14_TRAINING"
"$python" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_NATIVE_R14_PAIRS" --model-label 'R14 native + beam free' \
  --reference 'native_r11=R11 native' --reference 'lidar_r8=R8 LiDAR' \
  --protocol-note 'R14 vs R11 changes free objective; R14 vs R8 shares beam free but changes the learned geometry path. Full-track labels throughout.'
cp "$run/summary.json" "$archive/population_native_r14_summary.json"
cp "$run/manifest.json" "$archive/population_native_r14_manifest.json"
cp "$run/analysis.json" "$archive/population_native_r14_analysis.json"
cp "$run/training_analysis.json" "$archive/population_native_r14_training.json"
