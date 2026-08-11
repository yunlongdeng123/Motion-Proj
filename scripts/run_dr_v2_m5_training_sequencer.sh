#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/motion_proj
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
TASK_ROOT=/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M5-STRESS-3SCENE-01
SEQUENCER_RUN=$TASK_ROOT/20260802T180800Z__training-sequencer-s0-r9
SOURCE_0242=$TASK_ROOT/20260802T180500Z__scene0242-heldout-s0-r5
RECOVERY_0242=$TASK_ROOT/20260802T183000Z__scene0242-training-recovery-s0-r6
RUN_0230=$TASK_ROOT/20260802T183100Z__scene0230-heldout-s0-r7
RUN_0255=$TASK_ROOT/20260802T190500Z__scene0255-heldout-s0-r8
PROTOCOL=$PROJECT/configs/dynamic_editing_v2/m5_protocol_v1.yaml

mkdir -p "$SEQUENCER_RUN/logs"
"$PYTHON" - "$SEQUENCER_RUN/terminal.json" <<'PY'
import datetime, json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "running",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "failure": None,
}, indent=2) + "\n")
PY

on_error() {
  rc=$?
  line=${BASH_LINENO[0]:-unknown}
  "$PYTHON" - "$SEQUENCER_RUN/terminal.json" "$rc" "$line" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "status": "blocked",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "failure": {
        "code": "M5_TRAINING_SEQUENCER_FAILED",
        "return_code": int(sys.argv[2]),
        "shell_line": sys.argv[3],
    },
}, indent=2) + "\n")
os.replace(temporary, path)
PY
  exit "$rc"
}
trap on_error ERR

status_of() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$1"
}

while [[ ! -f "$SOURCE_0242/terminal.json" ]] || [[ "$(status_of "$SOURCE_0242/terminal.json")" == "running" ]]; do
  sleep 30
done
if [[ "$(status_of "$SOURCE_0242/terminal.json")" != "done" ]]; then
  echo "scene-0242 source did not finish" >&2
  exit 20
fi

cd "$PROJECT"
"$PYTHON" scripts/record_dr_v2_m5_training_recovery.py \
  --run-dir "$RECOVERY_0242" \
  --source-run "$SOURCE_0242" \
  --protocol "$PROTOCOL" \
  > "$SEQUENCER_RUN/logs/scene0242_recovery.log" 2>&1

"$PYTHON" scripts/evict_dr_v2_file_cache.py \
  /root/autodl-tmp/data/dynamic_editing_v2 \
  /root/autodl-tmp/runs/dynamic_editing_v2 \
  >> "$SEQUENCER_RUN/logs/cache_advice.log" 2>&1

"$PYTHON" scripts/run_dr_v2_m5_scene_train.py \
  --run-dir "$RUN_0230" \
  --scene-name scene-0230 \
  --scene-index 179 \
  --high-token af663976db5e412e83db033d309c5c29 \
  --boundary-token 18c7f0c5fa6b49449f71c9dbae5c31d4 \
  > "$SEQUENCER_RUN/logs/scene0230.log" 2>&1

"$PYTHON" scripts/evict_dr_v2_file_cache.py \
  /root/autodl-tmp/data/dynamic_editing_v2 \
  /root/autodl-tmp/runs/dynamic_editing_v2 \
  >> "$SEQUENCER_RUN/logs/cache_advice.log" 2>&1

"$PYTHON" scripts/run_dr_v2_m5_scene_train.py \
  --run-dir "$RUN_0255" \
  --scene-name scene-0255 \
  --scene-index 204 \
  --high-token f4aa30b8d0b44e2381a4abeafbe17642 \
  --boundary-token 80c08b992f1d47359de644be24f491df \
  > "$SEQUENCER_RUN/logs/scene0255.log" 2>&1

"$PYTHON" - "$SEQUENCER_RUN/terminal.json" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "status": "done",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "failure": None,
}, indent=2) + "\n")
os.replace(temporary, path)
PY
