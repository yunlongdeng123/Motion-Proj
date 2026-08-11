#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/motion_proj
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
TASK_ROOT=/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M5-STRESS-3SCENE-01
TRAIN_0242=$TASK_ROOT/20260802T183000Z__scene0242-training-recovery-s0-r6
TRAIN_0230=$TASK_ROOT/20260802T183100Z__scene0230-heldout-s0-r7
TRAIN_0255=$TASK_ROOT/20260802T193000Z__scene0255-heldout-cache-recovery-s0-r17
EVAL_SEQUENCER=$TASK_ROOT/20260802T193100Z__evaluation-recovery-s0-r18
SMOKE_1=$TASK_ROOT/20260802T193200Z__scene0242-eval-smoke1-s0-r19
SMOKE_5=$TASK_ROOT/20260802T193300Z__scene0242-eval-smoke5-s0-r20
EVAL_0242=$TASK_ROOT/20260802T194000Z__scene0242-eval-full-s0-r21
EVAL_0230=$TASK_ROOT/20260802T203000Z__scene0230-eval-full-s0-r22
EVAL_0255=$TASK_ROOT/20260802T213000Z__scene0255-eval-full-s0-r23
AGGREGATE=$TASK_ROOT/20260802T223000Z__aggregate-s0-r24
PROTOCOL=$PROJECT/configs/dynamic_editing_v2/m5_protocol_v1.yaml

mkdir -p "$EVAL_SEQUENCER/logs"
"$PYTHON" - "$EVAL_SEQUENCER/terminal.json" <<'PY'
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
  "$PYTHON" - "$EVAL_SEQUENCER/terminal.json" "$rc" "$line" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "status": "blocked",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "failure": {
        "code": "M5_EVALUATION_SEQUENCER_FAILED",
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

while [[ ! -f "$TRAIN_0255/terminal.json" ]] || \
      [[ "$(status_of "$TRAIN_0255/terminal.json")" == "running" ]]; do
  sleep 30
done
if [[ "$(status_of "$TRAIN_0255/terminal.json")" != "done" ]]; then
  echo "M5 scene-0255 recovery training did not finish" >&2
  exit 20
fi

cd "$PROJECT"

run_eval() {
  local run_dir=$1
  local training_run=$2
  local frames=${3:-}
  local log_name=$4
  local command=(
    "$PYTHON" scripts/run_dr_v2_m5_scene_eval.py
    --run-dir "$run_dir"
    --training-run "$training_run"
    --protocol "$PROTOCOL"
  )
  if [[ -n "$frames" ]]; then
    command+=(--frame-count "$frames")
  fi
  "${command[@]}" > "$EVAL_SEQUENCER/logs/$log_name.log" 2>&1
}

# 固定分级验证：两次 smoke 均通过后才允许执行正式评测。
run_eval "$SMOKE_1" "$TRAIN_0242" 1 scene0242_smoke1
run_eval "$SMOKE_5" "$TRAIN_0242" 5 scene0242_smoke5

run_eval "$EVAL_0242" "$TRAIN_0242" "" scene0242_full
"$PYTHON" scripts/evict_dr_v2_file_cache.py \
  /root/autodl-tmp/data/dynamic_editing_v2 \
  /root/autodl-tmp/runs/dynamic_editing_v2 \
  >> "$EVAL_SEQUENCER/logs/cache_advice.log" 2>&1

run_eval "$EVAL_0230" "$TRAIN_0230" "" scene0230_full
"$PYTHON" scripts/evict_dr_v2_file_cache.py \
  /root/autodl-tmp/data/dynamic_editing_v2 \
  /root/autodl-tmp/runs/dynamic_editing_v2 \
  >> "$EVAL_SEQUENCER/logs/cache_advice.log" 2>&1

run_eval "$EVAL_0255" "$TRAIN_0255" "" scene0255_full

"$PYTHON" scripts/finalize_dr_v2_m5.py \
  --run-dir "$AGGREGATE" \
  --protocol "$PROTOCOL" \
  --scene-training "$TRAIN_0230" \
  --scene-training "$TRAIN_0242" \
  --scene-training "$TRAIN_0255" \
  --scene-output "$EVAL_0230/artifacts/stress" \
  --scene-output "$EVAL_0242/artifacts/stress" \
  --scene-output "$EVAL_0255/artifacts/stress" \
  --perception-report "$EVAL_0230/artifacts/perception/report.json" \
  --perception-report "$EVAL_0242/artifacts/perception/report.json" \
  --perception-report "$EVAL_0255/artifacts/perception/report.json" \
  > "$EVAL_SEQUENCER/logs/aggregate.log" 2>&1

"$PYTHON" - "$EVAL_SEQUENCER/terminal.json" <<'PY'
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
