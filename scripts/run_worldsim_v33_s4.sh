#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/motion_proj
PYTHON=/root/autodl-tmp/envs/drivestudio/bin/python
CONFIG=${CONFIG:-$PROJECT/configs/worldsim_v33/s4_spatial_delta_v1.yaml}
ROOT=/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ABSOLUTE_NEW_PACKAGE_RUN_DIR ABSOLUTE_NEW_EVAL_RUN_DIR" >&2
  exit 2
fi
PACKAGE_RUN=$1
EVAL_RUN=$2
for run_dir in "$PACKAGE_RUN" "$EVAL_RUN"; do
  if [[ $run_dir != "$ROOT"/* ]]; then
    echo "run dir must be under $ROOT: $run_dir" >&2
    exit 2
  fi
  if [[ -e $run_dir ]]; then
    echo "refuse overwrite: $run_dir" >&2
    exit 2
  fi
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU preflight failed: compute process already exists" >&2
  exit 3
fi

cd "$PROJECT"
"$PYTHON" scripts/build_worldsim_v33_s4_spatial_delta.py \
  --config "$CONFIG" \
  --run-dir "$PACKAGE_RUN"
PACKAGE_MANIFEST=$PACKAGE_RUN/artifacts/worldsim_asset/package_manifest.json
PACKAGE_SHA=$(
  "$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "$PACKAGE_MANIFEST"
)
"$PYTHON" scripts/evaluate_worldsim_v33_s4_spatial_delta.py \
  --config "$CONFIG" \
  --package-manifest "$PACKAGE_MANIFEST" \
  --package-manifest-sha256 "$PACKAGE_SHA" \
  --run-dir "$EVAL_RUN"

if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "GPU release gate failed: compute process remains" >&2
  exit 3
fi
echo "S4 package and real-render evaluation completed"
