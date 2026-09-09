#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
qv2=$base/WS-V73-Q-V2-01
run=$qv2/20260909T063700Z__open-charts-lidar-ray-support-s7304-r4
python_bin=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/ray_support
# 完整final结束后仅执行一次，原同表示r3为主要监督对照。
"$python_bin" scripts/summarize_worldsim_v73_global_results.py --run "$run" \
  --reference "open_charts_r3=$qv2/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3" \
  --reference "mesh_lidar_r2=$qv2/20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2" \
  --reference "lidar_r8=$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --output "$run/analysis.json"
"$python_bin" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python_bin" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" \
  --output "$archive/V73_RAY_SUPPORT_R4_TRAINING" --model-label 'Ray support r4'
"$python_bin" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_RAY_SUPPORT_R4_PAIRS" --model-label 'Ray support r4' \
  --reference 'open_charts_r3=Open r3' --reference 'mesh_lidar_r2=Closed r2' --reference 'lidar_r8=Narrow R8' \
  --protocol-note 'Same r3 surface and original losses; extra owned-return attraction adds directional metric and return-observation weighting.'
cp "$run/summary.json" "$archive/ray_support_r4_summary.json"
cp "$run/manifest.json" "$archive/ray_support_r4_final_manifest.json"
cp "$run/analysis.json" "$archive/ray_support_r4_analysis.json"
cp "$run/training_analysis.json" "$archive/ray_support_r4_training.json"
