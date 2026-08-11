#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/motion_proj
exec /root/autodl-tmp/envs/motionproj/bin/python \
  scripts/run_dr_v2_m5_scene_train.py \
  --run-dir /root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M5-STRESS-3SCENE-01/20260802T193000Z__scene0255-heldout-cache-recovery-s0-r17 \
  --scene-name scene-0255 \
  --scene-index 204 \
  --high-token f4aa30b8d0b44e2381a4abeafbe17642 \
  --boundary-token 80c08b992f1d47359de644be24f491df
