#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/motion_proj
CONFIG=${CONFIG:-$PROJECT/configs/worldsim_v32/s1_semantic_lift_v3.yaml}
PROMPT_MANIFEST=${PROMPT_MANIFEST:-/root/autodl-tmp/assets/worldsim_v32/s1_prompt_v3/prompt_manifest.json}

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ABSOLUTE_NEW_RUN_DIR" >&2
  exit 2
fi
RUN_DIR=$1
if [[ $RUN_DIR != /root/autodl-tmp/runs/worldsim_v32/* ]]; then
  echo "run dir must be under /root/autodl-tmp/runs/worldsim_v32" >&2
  exit 2
fi
if [[ -e $RUN_DIR ]]; then
  echo "refuse overwrite: $RUN_DIR" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "S1 requires a visible CUDA GPU; boot a GPU instance first" >&2
  exit 3
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU preflight failed: compute process already exists" >&2
  exit 3
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/artifacts"
exec > >(tee -a "$RUN_DIR/logs/s1.log") 2>&1
printf '{"status":"running","task_id":"WS-V32-S1-SEMANTIC-LIFT-01"}\n' > "$RUN_DIR/status.json"
failed() {
  code=$?
  printf '{"status":"failed","exit_code":%d}\n' "$code" > "$RUN_DIR/status.json"
  exit "$code"
}
trap failed ERR

cd "$PROJECT"
/root/autodl-tmp/envs/drivestudio/bin/python scripts/validate_worldsim_v32_s1.py --config "$CONFIG"
test -f "$PROMPT_MANIFEST"
/root/autodl-tmp/envs/worldsim-v32-sam/bin/python scripts/build_worldsim_v32_sam_masks.py \
  --config "$CONFIG" \
  --prompt-manifest "$PROMPT_MANIFEST" \
  --output-dir "$RUN_DIR/artifacts/sam2"
/root/autodl-tmp/envs/drivestudio/bin/python scripts/lift_worldsim_v32_semantics.py \
  --config "$CONFIG" \
  --mask-manifest "$RUN_DIR/artifacts/sam2/mask_manifest.json" \
  --output-dir "$RUN_DIR/artifacts/semantic_sidecar"
/root/autodl-tmp/envs/drivestudio/bin/python scripts/finalize_worldsim_v32_s1.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR"

printf '{"status":"done","task_id":"WS-V32-S1-SEMANTIC-LIFT-01","sidecar_only":true}\n' > "$RUN_DIR/status.json"
trap - ERR
echo "S1 done: $RUN_DIR"
