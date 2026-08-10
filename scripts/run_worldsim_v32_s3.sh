#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$(realpath -m "$1")
RUN_ROOT=/root/autodl-tmp/runs/worldsim_v32/WS-V32-S3-ASSET-HARVEST-01
case "$RUN_DIR" in
  "$RUN_ROOT"/*) ;;
  *)
    echo "RUN_DIR must be below $RUN_ROOT" >&2
    exit 2
    ;;
esac

PROJECT=/root/autodl-tmp/motion_proj
CONFIG="$PROJECT/configs/worldsim_v32/s3_asset_harvester_v1.yaml"
INPUTS="$RUN_DIR/artifacts/inputs/input_manifest.json"
AH_PYTHON=/root/autodl-tmp/envs/worldsim-v32-asset-harvester/bin/python
MOTION_PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
DRIVESTUDIO_PYTHON=/root/autodl-tmp/envs/drivestudio/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/artifacts"
cd "$PROJECT"

"$AH_PYTHON" scripts/run_worldsim_v32_s3_asset_harvester.py \
  --config "$CONFIG" \
  --input-manifest "$INPUTS" \
  --output-dir "$RUN_DIR/artifacts/asset_harvester" \
  > "$RUN_DIR/logs/inference.log" 2>&1

for SAMPLE in high_support_1view high_support_2view; do
  "$MOTION_PYTHON" scripts/import_worldsim_v32_actor_asset.py \
    --config "$CONFIG" \
    --input-manifest "$INPUTS" \
    --inference-manifest "$RUN_DIR/artifacts/asset_harvester/inference_manifest.json" \
    --sample "$SAMPLE" \
    --output-dir "$RUN_DIR/artifacts/actor_assets/$SAMPLE" \
    > "$RUN_DIR/logs/import_${SAMPLE}.log" 2>&1

  for VIEW_INDEX in 0 1; do
    "$DRIVESTUDIO_PYTHON" scripts/render_worldsim_v32_s3_actor_asset.py \
      --config "$CONFIG" \
      --input-manifest "$INPUTS" \
      --asset-manifest "$RUN_DIR/artifacts/actor_assets/$SAMPLE/actor_asset_manifest.json" \
      --sample high_support_2view \
      --source-view-index "$VIEW_INDEX" \
      --output-dir "$RUN_DIR/artifacts/renders/${SAMPLE}_view${VIEW_INDEX}" \
      > "$RUN_DIR/logs/render_${SAMPLE}_view${VIEW_INDEX}.log" 2>&1
  done
done

"$MOTION_PYTHON" scripts/evaluate_worldsim_v32_s3_actor_asset.py \
  --input-manifest "$INPUTS" \
  --render-manifest "$RUN_DIR/artifacts/renders/high_support_1view_view0/render_manifest.json" \
  --render-manifest "$RUN_DIR/artifacts/renders/high_support_1view_view1/render_manifest.json" \
  --render-manifest "$RUN_DIR/artifacts/renders/high_support_2view_view0/render_manifest.json" \
  --render-manifest "$RUN_DIR/artifacts/renders/high_support_2view_view1/render_manifest.json" \
  --output-dir "$RUN_DIR/artifacts/evaluation" \
  > "$RUN_DIR/logs/evaluation.log" 2>&1

RUN_DIR="$RUN_DIR" "$MOTION_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
path = run_dir / "status.json"
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "running"
payload["stage"] = "s3_finalization"
payload["pipeline_execution"] = "done"
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
