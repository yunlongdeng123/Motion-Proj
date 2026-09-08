#!/bin/bash
set -e
cd /root/autodl-tmp/motion_proj
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4
export CUDA_HOME=/root/autodl-tmp/envs/worldsim-v72-pointr
export PATH=/root/autodl-tmp/envs/motionproj/bin:$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions/worldsim_v73
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6
base=/root/autodl-tmp/runs/worldsim_v73
exec /root/autodl-tmp/envs/motionproj/bin/python scripts/train_worldsim_v73_global_actors.py \
  --native-run "$base/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3" \
  --actor-data "$base/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2" \
  --fit-targets "$base/WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2" \
  --run-id 20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2 \
  --mode lidar_only --query-surface shared_mesh --mesh-level 3 --epochs 30 \
  --completion-init lidar_surface --native-data-weight 0 --fit-label-times all_window \
  --free-weight .5 --free-mode beam_tube_range --free-width-m .03 --free-resolution 32 \
  --event-weight 0 \
  --baseline-results "$base/WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-joint-full-track-s7304-r10/lidar_baseline"
