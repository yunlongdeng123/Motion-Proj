#!/usr/bin/env bash
# Motion-Proj 最小链路 smoke：资产检查 -> 构建 latent 缓存 -> 短训 LoRA -> 评估缓存。
# 默认 1 个样本、2 步训练；参数通过环境变量覆盖，不修改主配置文件。
# 需要 GPU；报告写入 $WORK_DIR/reports/。详见 scripts/README_smoke.md。
#
# 推荐用法：
#   cd /root/autodl-tmp/motion_proj
#   source /root/miniconda3/etc/profile.d/conda.sh && conda activate motionproj
#   bash scripts/smoke_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate motionproj
fi

CONFIG="${CONFIG:-configs/train/motionproj_v1.yaml}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-/root/autodl-tmp/weights/svd-xt}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cache/projection_smoke}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/runs/motionproj_smoke}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
LOG_EVERY="${LOG_EVERY:-1}"
CKPT_EVERY="${CKPT_EVERY:-2}"
RUN_TRAIN="${RUN_TRAIN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
REPORT_DIR="${REPORT_DIR:-$WORK_DIR/reports}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$REPORT_DIR/smoke_$STAMP.md"
LOG_FILE="$REPORT_DIR/smoke_$STAMP.log"

mkdir -p "$REPORT_DIR"

exec > >(tee "$LOG_FILE") 2>&1

status="running"
finish_report() {
  local exit_code=$?
  if [[ "$exit_code" -eq 0 ]]; then
    status="passed"
  else
    status="failed"
  fi

  local cache_count=0
  if [[ -d "$CACHE_DIR" ]]; then
    cache_count="$(find "$CACHE_DIR" -mindepth 2 -maxdepth 2 -name metadata.json 2>/dev/null | wc -l | tr -d ' ')"
  fi

  {
    echo "# Motion-Proj Smoke Report"
    echo ""
    echo "- Status: $status"
    echo "- Timestamp: $STAMP"
    echo "- Project: $ROOT"
    echo "- Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- Config: $CONFIG"
    echo "- Model: $MODEL_PRETRAINED"
    echo "- Cache dir: $CACHE_DIR"
    echo "- Work dir: $WORK_DIR"
    echo "- Cache samples: $cache_count"
    echo "- Train steps: $TRAIN_STEPS"
    echo "- Run train: $RUN_TRAIN"
    echo "- Log: $LOG_FILE"
    echo ""
    echo "## Checkpoints"
    if [[ -d "$WORK_DIR/ckpts" ]]; then
      find "$WORK_DIR/ckpts" -maxdepth 1 -type f -printf "- %p\n" | sort
    else
      echo "- none"
    fi
  } > "$REPORT"

  echo ""
  echo "Smoke report: $REPORT"
  exit "$exit_code"
}
trap finish_report EXIT

OVERRIDES=(
  "model.pretrained=$MODEL_PRETRAINED"
  "paths.cache_dir=$CACHE_DIR"
  "work_dir=$WORK_DIR"
  "paths.ckpt_dir=$WORK_DIR/ckpts"
  "paths.log_dir=$WORK_DIR/logs"
  "cache.store=latent"
  "cache.max_samples=$MAX_SAMPLES"
  "cache.overwrite=true"
  "train.max_steps=$TRAIN_STEPS"
  "train.log_every=$LOG_EVERY"
  "train.ckpt_every=$CKPT_EVERY"
  "train.sample_every=$CKPT_EVERY"
)

echo "== Motion-Proj smoke pipeline =="
echo "config: $CONFIG"
echo "model: $MODEL_PRETRAINED"
echo "cache: $CACHE_DIR"
echo "work:  $WORK_DIR"
echo ""

python scripts/check_assets.py --config "$CONFIG" --require-gpu --min-free-gb "$MIN_FREE_GB" "${OVERRIDES[@]}"

echo ""
echo "== Build latent projection cache =="
python -m motion_proj.cache.build_cache --config "$CONFIG" "${OVERRIDES[@]}"

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo ""
  echo "== Train tiny LoRA smoke run =="
  python -m motion_proj.train.train_motionproj --config "$CONFIG" "${OVERRIDES[@]}"
else
  echo ""
  echo "== Skip train because RUN_TRAIN=$RUN_TRAIN =="
fi

echo ""
echo "== Evaluate cache metadata =="
python -m motion_proj.eval.evaluate --config "$CONFIG" --mode cache "${OVERRIDES[@]}"

echo ""
echo "== Smoke pipeline complete =="
