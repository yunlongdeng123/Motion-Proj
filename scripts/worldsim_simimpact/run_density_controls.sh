#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
R=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1
for scene in scene-0013 scene-0038 scene-0041; do
  for stride in 2 4; do
    /root/autodl-tmp/envs/hugsim-impact/bin/python "$R/scripts/run_native_baseline.py" --scene "$scene" --tag "density_stride${stride}_r1" --collision-sample-stride "$stride" > "$R/${scene}_density${stride}.log" 2>&1
  done
done
