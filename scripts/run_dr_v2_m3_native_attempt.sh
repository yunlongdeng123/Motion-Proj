#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR=${1:?usage: run_dr_v2_m3_native_attempt.sh RUN_DIR SOURCE_ASSET_RUN}
SOURCE_ASSET_RUN=${2:?usage: run_dr_v2_m3_native_attempt.sh RUN_DIR SOURCE_ASSET_RUN}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python

"$PROJECT_ROOT/scripts/run_dr_v2_m3_prepare.sh" "$RUN_DIR"

"$PYTHON" "$PROJECT_ROOT/scripts/record_dr_v2_m3_asset_reuse.py" \
  --run-dir "$RUN_DIR" \
  --source-run "$SOURCE_ASSET_RUN" \
  > "$RUN_DIR/logs/asset_reuse.log" 2>&1

"$PROJECT_ROOT/scripts/run_dr_v2_m3_native_smoke.sh" \
  "$RUN_DIR"
