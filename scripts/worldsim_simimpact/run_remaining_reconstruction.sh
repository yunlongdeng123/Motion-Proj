#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
R=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1
for method in vggt omega512 pi3x; do
  /root/autodl-tmp/envs/worldsim-v81/bin/python "$R/scripts/infer_reconstruction.py" --method "$method" > "$R/infer_${method}.log" 2>&1
done
