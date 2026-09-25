#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/external/worldsim_v75/VAD-GS
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
for scene in scene_0230 scene_0255; do
  /root/autodl-tmp/envs/vadgs-v76/bin/python script/v76/generate_sam_prior.py \
    --scene "/root/autodl-tmp/data/v76_vadgs/$scene" --background-only --stop-on-training \
    > "/root/autodl-tmp/data/v76_vadgs/$scene/sam_background.log" 2>&1
done
