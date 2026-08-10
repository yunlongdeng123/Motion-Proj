#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/motion_proj
CONFIG=${CONFIG:-$PROJECT/configs/worldsim_v33/s1_instance_field_v1.yaml}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 {smoke|formal} ABSOLUTE_NEW_RUN_DIR [HELDOUT_MASK_MANIFEST]" >&2
  exit 2
fi
PHASE=$1
RUN_DIR=$2
EVAL_MANIFEST=${3:-}
if [[ $PHASE != smoke && $PHASE != formal ]]; then
  echo "phase must be smoke or formal" >&2
  exit 2
fi
if [[ $RUN_DIR != /root/autodl-tmp/runs/worldsim_v33/WS-V33-S1-OBJECT-AWARE-GS-01/* ]]; then
  echo "run dir must be under the frozen V3.3 S1 root" >&2
  exit 2
fi
if [[ -e $RUN_DIR ]]; then
  echo "refuse overwrite: $RUN_DIR" >&2
  exit 2
fi
if [[ $PHASE == formal && -z $EVAL_MANIFEST ]]; then
  echo "formal phase requires a heldout mask manifest" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "S1 requires a visible CUDA GPU" >&2
  exit 3
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU preflight failed: compute process already exists" >&2
  exit 3
fi

mkdir -p "$RUN_DIR/logs"
printf '{"status":"running","task_id":"WS-V33-S1-OBJECT-AWARE-GS-01","phase":"%s"}\n' "$PHASE" > "$RUN_DIR/status.json"
failed() {
  code=$?
  printf '{"status":"failed","exit_code":%d,"phase":"%s"}\n' "$code" "$PHASE" > "$RUN_DIR/status.json"
  exit "$code"
}
trap failed ERR
exec > >(tee -a "$RUN_DIR/logs/s1.log") 2>&1

cd "$PROJECT"
ARGS=(--config "$CONFIG" --run-dir "$RUN_DIR" --phase "$PHASE")
if [[ -n $EVAL_MANIFEST ]]; then
  ARGS+=(--eval-mask-manifest "$EVAL_MANIFEST")
fi
/root/autodl-tmp/envs/drivestudio/bin/python scripts/run_worldsim_v33_s1_instance_field.py "${ARGS[@]}"
/root/autodl-tmp/envs/drivestudio/bin/python scripts/finalize_worldsim_v33_s1.py \
  --config "$CONFIG" --run-dir "$RUN_DIR" --phase "$PHASE"

if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU release gate failed: compute process remains" >&2
  exit 3
fi
printf '{"status":"done","task_id":"WS-V33-S1-OBJECT-AWARE-GS-01","phase":"%s"}\n' "$PHASE" > "$RUN_DIR/status.json"
trap - ERR
echo "S1 $PHASE done: $RUN_DIR"
