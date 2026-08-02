#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR=${1:?usage: run_dr_v2_m3_data_pipeline.sh RUN_DIR [WAIT_SESSION]}
WAIT_SESSION=${2:-drv2-m3-r3-extract}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 10
done

"$PYTHON" - "$RUN_DIR/stages/raw_prepare.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"raw preparation stage is absent: {path}")
payload = json.loads(path.read_text())
if payload.get("status") != "done" or payload.get("return_code") != 0:
    raise SystemExit(f"raw preparation did not succeed: {payload}")
PY

"$PYTHON" "$PROJECT_ROOT/scripts/run_dr_v2_m3_data_stage.py" \
  --run-dir "$RUN_DIR" --stage preprocess \
  >> "$RUN_DIR/logs/data_pipeline_wrapper.log" 2>&1

"$PYTHON" "$PROJECT_ROOT/scripts/run_dr_v2_m3_data_stage.py" \
  --run-dir "$RUN_DIR" --stage sky_masks \
  >> "$RUN_DIR/logs/data_pipeline_wrapper.log" 2>&1
