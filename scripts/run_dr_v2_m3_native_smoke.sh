#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR=${1:?usage: run_dr_v2_m3_native_smoke.sh RUN_DIR [WAIT_SESSION]}
WAIT_SESSION=${2:-}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
SELECTED_TOKEN=af663976db5e412e83db033d309c5c29

if [[ -n "$WAIT_SESSION" ]]; then
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 10
  done
fi

"$PYTHON" - "$RUN_DIR/stages/preprocess.json" "$RUN_DIR/stages/sky_masks.json" <<'PY'
import json
import sys
from pathlib import Path
for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"required data stage missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "done" or payload.get("return_code") != 0:
        raise SystemExit(f"required data stage failed: {payload}")
PY

"$PYTHON" "$PROJECT_ROOT/scripts/run_dr_v2_m3_training.py" \
  --run-dir "$RUN_DIR" --mode profile100 \
  >> "$RUN_DIR/logs/native_smoke_wrapper.log" 2>&1

CHECKPOINT="$RUN_DIR/work_dirs/m3_profile/scene0230_profile100_s0/checkpoint_final.pth"
set +e
/root/autodl-tmp/envs/drivestudio/bin/python \
  "$PROJECT_ROOT/scripts/probe_dr_v2_drivestudio_actor.py" \
  --checkpoint "$CHECKPOINT" \
  --instance-token "$SELECTED_TOKEN" \
  --output "$RUN_DIR/stages/native_actor_mapping_probe.json" \
  >> "$RUN_DIR/logs/native_actor_mapping_probe.log" 2>&1
probe_rc=$?
set -e
printf '%s\n' "$probe_rc" > "$RUN_DIR/stages/native_actor_mapping_probe.rc"
exit "$probe_rc"
