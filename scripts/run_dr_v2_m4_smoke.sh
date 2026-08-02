#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
FRAME_COUNT=${1:?usage: run_dr_v2_m4_smoke.sh FRAME_COUNT OUTPUT_DIR}
OUTPUT_DIR=${2:?usage: run_dr_v2_m4_smoke.sh FRAME_COUNT OUTPUT_DIR}
CHECKPOINT=/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T152252Z__native-train-s0-r8/work_dirs/m3_formal/scene0230_formal_s0/checkpoint_final.pth
REGISTRY=/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T163930Z__formal-checkpoint-recovery-s0-r12/artifacts/actor_registry.json
TOKEN=af663976db5e412e83db033d309c5c29

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:/root/autodl-tmp/third_party/drivestudio" \
WANDB_MODE=disabled OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/autodl-tmp/envs/drivestudio/bin/python scripts/run_dr_v2_m4_pilot.py \
  --checkpoint "$CHECKPOINT" \
  --registry "$REGISTRY" \
  --instance-token "$TOKEN" \
  --output-dir "$OUTPUT_DIR" \
  --frame-count "$FRAME_COUNT" \
  > "${OUTPUT_DIR}.log" 2>&1
