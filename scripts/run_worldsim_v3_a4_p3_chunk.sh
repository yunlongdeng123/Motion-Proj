#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/motion_proj
RUN_ROOT=/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
DRIVE_PYTHON=/root/autodl-tmp/envs/drivestudio/bin/python
DRIVESTUDIO=/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1
PROTOCOL="$PROJECT/configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${1:-"$RUN_ROOT/${STAMP}__a4-p3-chunk-s0-r1"}

export PYTHONPATH="$PROJECT:$DRIVESTUDIO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"

"$PYTHON" "$PROJECT/scripts/run_worldsim_v3_a4_p3_chunk.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL"
"$DRIVE_PYTHON" "$PROJECT/scripts/run_worldsim_v3_a4_p3_worker.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL" --operation source-layout-audit
"$DRIVE_PYTHON" "$PROJECT/scripts/run_worldsim_v3_a4_p3_worker.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL" --operation materialize
"$DRIVE_PYTHON" "$PROJECT/scripts/run_worldsim_v3_a4_p3_worker.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL" --operation reassemble
"$DRIVE_PYTHON" "$PROJECT/scripts/run_worldsim_v3_a4_p3_worker.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL" --operation evaluate
"$DRIVE_PYTHON" "$PROJECT/scripts/run_worldsim_v3_a4_p3_worker.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL" --operation runtime-profile
"$PYTHON" "$PROJECT/scripts/aggregate_worldsim_v3_a4_p3.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL"
"$PYTHON" "$PROJECT/scripts/audit_worldsim_v3_a4_p3_resume.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL"
"$PYTHON" "$PROJECT/scripts/finalize_worldsim_v3_a4_p3.py" \
  --run-dir "$RUN_DIR" --protocol "$PROTOCOL"

printf '%s\n' "$RUN_DIR"
