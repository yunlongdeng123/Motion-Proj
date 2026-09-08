#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
base=/root/autodl-tmp/runs/worldsim_v73
qv2=$base/WS-V73-Q-V2-01
old=$base/WS-V73-M2-GLOBAL-ACTOR-01
python=/root/autodl-tmp/envs/motionproj/bin/python
archive=docs/autoresearch/worldsim_v73/qv2
variant=${1:?Pass joint_r1 or lidar_r2 after its final evaluation completes}
lidar=$qv2/20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2
joint=$qv2/20260908T221500Z__shared-mesh-full-track-beam-s7304-r1
# 完整final结束后各执行一次；仅读保存的结果，不做新日志推理或回算历史比较。
case "$variant" in
  lidar_r2)
    run=$lidar
    refs=(--reference "lidar_r8=$old/20260907T212500Z__population-lidar-track-beam-range-s7304-r8")
    plotrefs=(--reference 'lidar_r8=R8 LiDAR patches')
    label='Q-v2 LiDAR r2'
    note='Same LiDAR observations and beam objective; initialization, shared support and face budget change.'
    ;;
  joint_r1)
    run=$joint
    refs=(--reference "mesh_lidar_r2=$lidar"
          --reference "joint_r12=$old/20260908T050000Z__population-joint-full-track-beam-range-s7304-r12"
          --reference "native_r14=$old/20260908T100000Z__population-native-full-track-beam-range-s7304-r14"
          --reference "lidar_r8=$old/20260907T212500Z__population-lidar-track-beam-range-s7304-r8")
    plotrefs=(--reference 'mesh_lidar_r2=Q-v2 LiDAR r2' --reference 'joint_r12=R12 joint patches'
              --reference 'native_r14=R14 native' --reference 'lidar_r8=R8 LiDAR patches')
    label='Q-v2 joint r1'
    note='All share beam free. r2 removes the whole visual/native path; old methods differ in surface support and budget.'
    ;;
  *) exit 2 ;;
esac
"$python" scripts/summarize_worldsim_v73_global_results.py --run "$run" "${refs[@]}" --output "$run/analysis.json"
"$python" scripts/summarize_worldsim_v73_training.py --run "$run" --output "$run/training_analysis.json"
"$python" scripts/plot_worldsim_v73_training.py --summary "$run/training_analysis.json" --output "$archive/V73_QV2_${variant}_TRAINING" --model-label "$label"
"$python" scripts/plot_worldsim_v73_joint_r10_pairs.py --analysis "$run/analysis.json" \
  --output "$archive/V73_QV2_${variant}_PAIRS" --model-label "$label" "${plotrefs[@]}" --protocol-note "$note"
cp "$run/summary.json" "$archive/shared_mesh_${variant}_summary.json"
cp "$run/manifest.json" "$archive/shared_mesh_${variant}_final_manifest.json"
cp "$run/analysis.json" "$archive/shared_mesh_${variant}_analysis.json"
cp "$run/training_analysis.json" "$archive/shared_mesh_${variant}_training.json"
