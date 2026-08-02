#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
RUN_DIR=${1:?usage: run_dr_v2_m3_recovery.sh RUN_DIR SOURCE_RUN}
SOURCE_RUN=${2:?usage: run_dr_v2_m3_recovery.sh RUN_DIR SOURCE_RUN}

cd "$PROJECT_ROOT"
bash scripts/run_dr_v2_m3_prepare.sh "$RUN_DIR"
"$PYTHON" scripts/record_dr_v2_m3_asset_reuse.py \
  --run-dir "$RUN_DIR" \
  --source-run "$SOURCE_RUN" \
  > "$RUN_DIR/logs/asset_reuse.log" 2>&1
"$PYTHON" scripts/record_dr_v2_m3_checkpoint_recovery.py \
  --run-dir "$RUN_DIR" \
  --source-run "$SOURCE_RUN" \
  > "$RUN_DIR/logs/checkpoint_recovery.log" 2>&1
