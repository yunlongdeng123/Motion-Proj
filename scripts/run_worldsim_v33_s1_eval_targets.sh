#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/motion_proj
CONFIG=${CONFIG:-$PROJECT/configs/worldsim_v33/s1_instance_field_v1.yaml}
PARTITION=${PARTITION:-heldout}
if [[ $PARTITION != development && $PARTITION != heldout ]]; then
  echo "PARTITION must be development or heldout" >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 ABSOLUTE_NEW_RUN_DIR" >&2
  exit 2
fi
RUN_DIR=$1
if [[ $RUN_DIR != /root/autodl-tmp/runs/worldsim_v33/WS-V33-S1-OBJECT-AWARE-GS-01/* ]]; then
  echo "run dir must be under the frozen V3.3 S1 root" >&2
  exit 2
fi
if [[ -e $RUN_DIR ]]; then
  echo "refuse overwrite: $RUN_DIR" >&2
  exit 2
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU preflight failed: compute process already exists" >&2
  exit 3
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/artifacts"
printf '{"status":"running","task_id":"WS-V33-S1-OBJECT-AWARE-GS-01","stage":"%s_targets","evaluation_partition":"%s"}\n' "$PARTITION" "$PARTITION" > "$RUN_DIR/status.json"
failed() {
  code=$?
  printf '{"status":"failed","exit_code":%d,"stage":"%s_targets","evaluation_partition":"%s"}\n' "$code" "$PARTITION" "$PARTITION" > "$RUN_DIR/status.json"
  exit "$code"
}
trap failed ERR
exec > >(tee -a "$RUN_DIR/logs/${PARTITION}_targets.log") 2>&1
cd "$PROJECT"
/root/autodl-tmp/envs/drivestudio/bin/python scripts/prepare_worldsim_v33_s1_eval_prompts.py \
  --config "$CONFIG" --output-dir "$RUN_DIR/artifacts/prompts" \
  --partition "$PARTITION"
/root/autodl-tmp/envs/worldsim-v33-sam2/bin/python scripts/build_worldsim_v33_s1_eval_masks.py \
  --config "$CONFIG" \
  --prompt-manifest "$RUN_DIR/artifacts/prompts/prompt_manifest.json" \
  --output-dir "$RUN_DIR/artifacts/masks" \
  --partition "$PARTITION"
/root/autodl-tmp/envs/drivestudio/bin/python scripts/finalize_worldsim_v33_s1_eval_targets.py \
  --config "$CONFIG" --run-dir "$RUN_DIR" --partition "$PARTITION"
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU release gate failed: compute process remains" >&2
  exit 3
fi
printf '{"status":"done","task_id":"WS-V33-S1-OBJECT-AWARE-GS-01","stage":"%s_targets","evaluation_partition":"%s"}\n' "$PARTITION" "$PARTITION" > "$RUN_DIR/status.json"
trap - ERR
echo "S1 $PARTITION targets done: $RUN_DIR"
