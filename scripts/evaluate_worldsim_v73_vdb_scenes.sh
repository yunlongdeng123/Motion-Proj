#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
v73_runs=/root/autodl-tmp/runs/worldsim_v73
for v73_variant in raw carved; do
 if [ "$v73_variant" = raw ]; then
  v73_scene=20260908T054000Z__development-vdbfusion-background-r4
  v73_eval=20260908T054500Z__development-vdb-background-r8
 else
  v73_scene=20260908T054000Z__development-vdbfusion-build-carved-r5
  v73_eval=20260908T054500Z__development-vdb-build-carved-r9
 fi
 /root/autodl-tmp/envs/worldsim-v72-lidar4d/bin/python scripts/evaluate_worldsim_v73_scene_composition.py \
  --scene-data "$v73_runs/WS-V73-M4-SCENE-DATA-01/$v73_scene" \
  --actor-data "$v73_runs/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2" \
  --model "joint_r5=$v73_runs/WS-V73-M2-GLOBAL-ACTOR-01/20260908T012500Z__population-joint-r5-epoch21-resume-r1" \
  --model "r8=$v73_runs/WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8" \
  --model "r9=$v73_runs/WS-V73-M2-GLOBAL-ACTOR-01/20260907T230000Z__population-lidar-track-beam-event-s7304-r9" \
  --model "native_fusion=$v73_runs/WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2" \
  --model "capa_r2=$v73_runs/WS-V73-M2-CAPA-01/20260907T225000Z__population-build-tta-chunked-s7305-r2" \
  --model "adapointr_r2=$v73_runs/WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2" \
  --output "$v73_runs/WS-V73-M4-SCENE-COMPOSITION-01/$v73_eval"
done
