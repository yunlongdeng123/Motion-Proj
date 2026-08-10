#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/root/autodl-tmp/motion_proj"
RUN_ROOT="/root/autodl-tmp/runs/worldsim_v33/WS-V33-S5-SEMANTIC-RENDER-01"
DRIVE_PYTHON="/root/autodl-tmp/envs/drivestudio/bin/python"
HARMONIZER_PYTHON="/root/autodl-tmp/envs/worldsim-v32-asset-harvester/bin/python"
SAM2_PYTHON="/root/autodl-tmp/envs/worldsim-v33-sam2/bin/python"
CONFIG="${REPO_ROOT}/configs/worldsim_v33/s5_semantic_gate_v1.yaml"
RUN_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_NAME}" || "${RUN_NAME}" == */* || "${RUN_NAME}" == .* ]]; then
  echo "--run-name 必须是非隐藏单层目录名" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "config 不存在: ${CONFIG}" >&2
  exit 2
fi
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "拒绝复用已有 S5 run: ${RUN_DIR}" >&2
  exit 3
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU 有活跃 compute process，拒绝启动 S5" >&2
  exit 4
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/artifacts"
cp -- "${CONFIG}" "${RUN_DIR}/config.yaml"
CONFIG="${RUN_DIR}/config.yaml"
STATUS="${RUN_DIR}/status.json"
"${DRIVE_PYTHON}" - "${STATUS}" "${RUN_DIR}" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
path = Path(sys.argv[1])
payload = {
    "schema_version": "worldsim_v33_s5_status_v1",
    "task_id": "WS-V33-S5-SEMANTIC-RENDER-01",
    "state": "running",
    "run_dir": sys.argv[2],
    "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

on_error() {
  local code=$?
  "${DRIVE_PYTHON}" - "${STATUS}" "${code}" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    payload = {"schema_version": "worldsim_v33_s5_status_v1"}
if payload.get("state") != "completed":
    payload.update({
        "state": "failed",
        "exit_code": int(sys.argv[2]),
        "failed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  exit "${code}"
}
trap on_error ERR

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0

"${DRIVE_PYTHON}" scripts/prepare_worldsim_v33_s5_inputs.py \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/logs/01_prepare.log"
INPUT_MANIFEST="${RUN_DIR}/artifacts/inputs/input_manifest.json"
INPUT_SHA="$(sha256sum "${INPUT_MANIFEST}" | awk '{print $1}')"

"${HARMONIZER_PYTHON}" scripts/run_worldsim_v33_s5_harmonizer.py \
  --config "${CONFIG}" \
  --input-manifest "${INPUT_MANIFEST}" \
  --input-manifest-sha "${INPUT_SHA}" \
  --output-dir "${RUN_DIR}/artifacts/harmonizer" \
  2>&1 | tee "${RUN_DIR}/logs/02_harmonizer.log"
HARMONIZER_MANIFEST="${RUN_DIR}/artifacts/harmonizer/harmonizer_manifest.json"
HARMONIZER_SHA="$(sha256sum "${HARMONIZER_MANIFEST}" | awk '{print $1}')"

for _ in $(seq 1 15); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    break
  fi
  sleep 1
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "Harmonizer 退出后 GPU 未释放" >&2
  exit 5
fi

"${SAM2_PYTHON}" scripts/run_worldsim_v33_s5_sam2_detector.py \
  --config "${CONFIG}" \
  --input-manifest "${INPUT_MANIFEST}" \
  --input-manifest-sha "${INPUT_SHA}" \
  --harmonizer-manifest "${HARMONIZER_MANIFEST}" \
  --harmonizer-manifest-sha "${HARMONIZER_SHA}" \
  --output-dir "${RUN_DIR}/artifacts/sam2_detector" \
  2>&1 | tee "${RUN_DIR}/logs/03_sam2_detector.log"
SAM2_MANIFEST="${RUN_DIR}/artifacts/sam2_detector/sam2_detector_manifest.json"
SAM2_SHA="$(sha256sum "${SAM2_MANIFEST}" | awk '{print $1}')"

"${DRIVE_PYTHON}" scripts/finalize_worldsim_v33_s5.py \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --input-manifest "${INPUT_MANIFEST}" \
  --input-manifest-sha "${INPUT_SHA}" \
  --harmonizer-manifest "${HARMONIZER_MANIFEST}" \
  --harmonizer-manifest-sha "${HARMONIZER_SHA}" \
  --sam2-manifest "${SAM2_MANIFEST}" \
  --sam2-manifest-sha "${SAM2_SHA}" \
  2>&1 | tee "${RUN_DIR}/logs/04_finalize.log"

"${DRIVE_PYTHON}" - "${STATUS}" <<'PY'
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("state") != "completed" or not payload.get("accepted"):
    raise SystemExit(f"S5 terminal status 非 accepted completed: {payload}")
print(json.dumps(payload, ensure_ascii=False))
PY
