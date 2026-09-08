#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-GLOBAL-ACTOR-01
run=$base/20260908T050000Z__population-joint-full-track-beam-range-s7304-r12
python=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/m2/global
# R12最终评价完成后执行一次；R14−R11继续复用R14已保存的分析。
"$python" scripts/summarize_worldsim_v73_global_results.py --run "$run" \
  --reference "joint_r10=$base/20260907T233000Z__population-joint-full-track-s7304-r10" \
  --reference "native_r14=$base/20260908T100000Z__population-native-full-track-beam-range-s7304-r14" \
  --reference "lidar_r8=$base/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$run/analysis.json"
"$python" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" --output "$archive/V73_JOINT_R12_TRAINING"
"$python" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_JOINT_R12_PAIRS" --model-label 'R12 joint + beam free' \
  --reference 'joint_r10=R10 joint' --reference 'native_r14=R14 native' --reference 'lidar_r8=R8 LiDAR' \
  --protocol-note 'R10 differs in free objective; R14 and R8 share beam free but use different geometry paths. Full-track labels throughout.'
cp "$run/summary.json" "$archive/population_joint_r12_summary.json"
cp "$run/manifest.json" "$archive/population_joint_r12_manifest.json"
cp "$run/analysis.json" "$archive/population_joint_r12_analysis.json"
cp "$run/training_analysis.json" "$archive/population_joint_r12_training.json"
