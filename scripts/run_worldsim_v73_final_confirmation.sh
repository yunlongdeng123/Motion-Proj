#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4
export CUDA_HOME=/root/autodl-tmp/envs/worldsim-v72-pointr
export PATH=/root/autodl-tmp/envs/motionproj/bin:$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions/worldsim_v73
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6
base=/root/autodl-tmp/runs/worldsim_v73
data=$base/WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1
native=$base/WS-V73-M4-AV2-PREFIX-01/20260908T043000Z__external20-28view-prefix-r1
out=$base/WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1
py=/root/autodl-tmp/envs/motionproj/bin/python
# 仅在r7完整30轮与DEV收口后启动；配置已选定，不依据确认结果改模型。
"$py" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="done", "r7尚未完成，不能读取中途checkpoint确认"' \
  "$base/WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7/status.json"
"$py" scripts/evaluate_worldsim_v73_fixed_actors.py --actor-data "$data" --native-run "$native" \
  --checkpoint "$base/WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7/latest.pt" --output "$out/joint_r7"
"$py" scripts/evaluate_worldsim_v73_fixed_actors.py --actor-data "$data" \
  --checkpoint "$base/WS-V73-Q-V2-01/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6/latest.pt" --output "$out/lidar_r6"
"$py" scripts/evaluate_worldsim_v73_fixed_actors.py --actor-data "$data" \
  --checkpoint "$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8/latest.pt" --output "$out/lidar_r8"
"$py" scripts/evaluate_worldsim_v73_fixed_actors.py --actor-data "$data" --native-run "$native" \
  --method native_fusion --output "$out/native_fusion"
"$py" scripts/evaluate_worldsim_v73_fixed_actors.py --actor-data "$data" --method lidar_pca --output "$out/lidar_pca"
"$py" scripts/summarize_worldsim_v73_global_results.py --run "$out/joint_r7" \
  --reference "lidar_r6=$out/lidar_r6" --reference "lidar_r8=$out/lidar_r8" \
  --reference "native_fusion=$out/native_fusion" --reference "lidar_pca=$out/lidar_pca" --output "$out/analysis.json"
"$py" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$out/analysis.json" \
  --output "$out/V73_FINAL_EXTERNAL20_PAIRS" --role external_confirmation --model-label 'Final Joint r7' \
  --reference 'lidar_r6=Matched LiDAR r6' --reference 'lidar_r8=Narrow LiDAR R8' \
  --reference 'native_fusion=Native fusion' --reference 'lidar_pca=LiDAR PCA' \
  --protocol-note 'Fixed weights and build-only inputs; cross-dataset AV2 confirmation, no adaptation or model selection.'
