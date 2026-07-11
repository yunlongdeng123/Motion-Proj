#!/usr/bin/env bash
# Motion-Proj formal training launcher.
# Default action is "start": run asset check -> build latent cache -> train -> eval in background.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/train/motionproj_v1.yaml}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-/root/autodl-tmp/weights/svd-xt}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cache/projection}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/runs/motionproj_v1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

# Empty MAX_SAMPLES means use the config default, currently all available mini clips.
MAX_SAMPLES="${MAX_SAMPLES:-}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
LOG_EVERY="${LOG_EVERY:-20}"
CKPT_EVERY="${CKPT_EVERY:-500}"
SAMPLE_EVERY="${SAMPLE_EVERY:-500}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
CACHE_OVERWRITE="${CACHE_OVERWRITE:-false}"
RUN_CACHE="${RUN_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

LOG_DIR="$WORK_DIR/logs"
CKPT_DIR="$WORK_DIR/ckpts"
REPORT_DIR="$WORK_DIR/reports"
PID_FILE="$WORK_DIR/train_pipeline.pid"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_motionproj_oneclick.sh [start|foreground|tail|status|logs|run]

Common commands:
  bash scripts/train_motionproj_oneclick.sh start       # 后台启动正式链路
  bash scripts/train_motionproj_oneclick.sh tail        # 持续查看最新日志
  bash scripts/train_motionproj_oneclick.sh status      # 查看 PID、最近日志、checkpoint、GPU
  bash scripts/train_motionproj_oneclick.sh foreground  # 前台运行，适合 tmux

Useful overrides:
  TRAIN_STEPS=2000 bash scripts/train_motionproj_oneclick.sh start
  MAX_SAMPLES=10 TRAIN_STEPS=100 bash scripts/train_motionproj_oneclick.sh foreground
  CACHE_OVERWRITE=true bash scripts/train_motionproj_oneclick.sh start
  RUN_CACHE=0 bash scripts/train_motionproj_oneclick.sh start
EOF
}

activate_env() {
  if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate motionproj
  fi
  export HF_ENDPOINT HF_HOME
}

make_overrides() {
  OVERRIDES=(
    "model.pretrained=$MODEL_PRETRAINED"
    "paths.cache_dir=$CACHE_DIR"
    "work_dir=$WORK_DIR"
    "paths.ckpt_dir=$CKPT_DIR"
    "paths.log_dir=$LOG_DIR"
    "cache.store=latent"
    "cache.overwrite=$CACHE_OVERWRITE"
    "train.max_steps=$TRAIN_STEPS"
    "train.log_every=$LOG_EVERY"
    "train.ckpt_every=$CKPT_EVERY"
    "train.sample_every=$SAMPLE_EVERY"
  )
  if [[ -n "$MAX_SAMPLES" ]]; then
    OVERRIDES+=("cache.max_samples=$MAX_SAMPLES")
  fi
}

latest_log() {
  if [[ ! -d "$LOG_DIR" ]]; then
    return 0
  fi
  find "$LOG_DIR" -maxdepth 1 -type f -name 'train_pipeline_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

write_report() {
  local status="$1"
  local exit_code="$2"
  local started_at="$3"
  local log_file="$4"
  local report="$REPORT_DIR/train_report_${started_at}.md"
  local cache_count=0

  mkdir -p "$REPORT_DIR"
  if [[ -d "$CACHE_DIR" ]]; then
    cache_count="$(find "$CACHE_DIR" -mindepth 2 -maxdepth 2 -name metadata.json 2>/dev/null | wc -l | tr -d ' ')"
  fi

  {
    echo "# Motion-Proj Training Report"
    echo ""
    echo "- Status: $status"
    echo "- Exit code: $exit_code"
    echo "- Timestamp: $started_at"
    echo "- Project: $ROOT"
    echo "- Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- Config: $CONFIG"
    echo "- Model: $MODEL_PRETRAINED"
    echo "- Cache dir: $CACHE_DIR"
    echo "- Cache samples: $cache_count"
    echo "- Work dir: $WORK_DIR"
    echo "- Log: $log_file"
    echo "- Train steps: $TRAIN_STEPS"
    echo "- Run cache: $RUN_CACHE"
    echo "- Run train: $RUN_TRAIN"
    echo "- Run eval: $RUN_EVAL"
    echo ""
    echo "## Checkpoints"
    if [[ -d "$CKPT_DIR" ]]; then
      find "$CKPT_DIR" -maxdepth 1 -type f -printf "- %p\n" | sort
    else
      echo "- none"
    fi
  } > "$report"

  echo ""
  echo "Training report: $report"
}

run_pipeline() {
  # 提升为脚本级变量：EXIT trap 在 run_pipeline 返回后才触发，此时函数内的
  # local 作用域已销毁，在 set -u 下引用会报 "unbound variable"（历史 bug）。
  RUN_LOG_FILE="${1:-$LOG_DIR/train_pipeline_$(date +%Y%m%d_%H%M%S).log}"
  RUN_STARTED_AT="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR" "$CACHE_DIR"
  echo "$$" > "$PID_FILE"

  finish() {
    local exit_code=$?
    local status="failed"
    trap - EXIT
    if [[ "$exit_code" -eq 0 ]]; then
      status="passed"
    fi
    # 报告生成失败不应阻断 PID 清理，故用 || true 兜底。
    write_report "$status" "$exit_code" "$RUN_STARTED_AT" "$RUN_LOG_FILE" || \
      echo "WARN: write_report failed (exit $?), continuing cleanup"
    if [[ -f "$PID_FILE" && "$(cat "$PID_FILE" 2>/dev/null)" == "$$" ]]; then
      rm -f "$PID_FILE"
    fi
    exit "$exit_code"
  }
  trap finish EXIT

  activate_env
  make_overrides

  echo "== Motion-Proj formal training pipeline =="
  echo "project: $ROOT"
  echo "config:  $CONFIG"
  echo "model:   $MODEL_PRETRAINED"
  echo "cache:   $CACHE_DIR"
  echo "work:    $WORK_DIR"
  echo "log:     $RUN_LOG_FILE"
  echo "steps:   $TRAIN_STEPS"
  if [[ -n "$MAX_SAMPLES" ]]; then
    echo "samples: $MAX_SAMPLES"
  else
    echo "samples: config default"
  fi
  echo ""
  python -V
  git status --short || true
  echo ""

  python scripts/check_assets.py --config "$CONFIG" --require-gpu --min-free-gb "$MIN_FREE_GB" "${OVERRIDES[@]}"

  if [[ "$RUN_CACHE" == "1" ]]; then
    echo ""
    echo "== Build latent projection cache =="
    python -m motion_proj.cache.build_cache --config "$CONFIG" "${OVERRIDES[@]}"
  else
    echo ""
    echo "== Skip cache build because RUN_CACHE=$RUN_CACHE =="
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    echo ""
    echo "== Train LoRA adapter =="
    python -m motion_proj.train.train_motionproj --config "$CONFIG" "${OVERRIDES[@]}"
  else
    echo ""
    echo "== Skip train because RUN_TRAIN=$RUN_TRAIN =="
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    echo ""
    echo "== Evaluate cache metadata =="
    python -m motion_proj.eval.evaluate --config "$CONFIG" --mode cache "${OVERRIDES[@]}"
  else
    echo ""
    echo "== Skip eval because RUN_EVAL=$RUN_EVAL =="
  fi

  echo ""
  echo "== Formal training pipeline complete =="
}

start_pipeline() {
  mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR"
  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Training pipeline is already running: PID $old_pid"
      echo "Log: $(latest_log)"
      exit 1
    fi
  fi

  local log_file="$LOG_DIR/train_pipeline_$(date +%Y%m%d_%H%M%S).log"
  export CONFIG MODEL_PRETRAINED CACHE_DIR WORK_DIR HF_ENDPOINT HF_HOME
  export MAX_SAMPLES TRAIN_STEPS LOG_EVERY CKPT_EVERY SAMPLE_EVERY MIN_FREE_GB CACHE_OVERWRITE
  export RUN_CACHE RUN_TRAIN RUN_EVAL

  nohup bash "$SCRIPT_PATH" run "$log_file" > "$log_file" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Started Motion-Proj training pipeline."
  echo "PID: $pid"
  echo "Log: $log_file"
  echo "Tail: bash scripts/train_motionproj_oneclick.sh tail"
  echo "Status: bash scripts/train_motionproj_oneclick.sh status"
}

foreground_pipeline() {
  mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REPORT_DIR"
  local log_file="$LOG_DIR/train_pipeline_$(date +%Y%m%d_%H%M%S).log"
  export CONFIG MODEL_PRETRAINED CACHE_DIR WORK_DIR HF_ENDPOINT HF_HOME
  export MAX_SAMPLES TRAIN_STEPS LOG_EVERY CKPT_EVERY SAMPLE_EVERY MIN_FREE_GB CACHE_OVERWRITE
  export RUN_CACHE RUN_TRAIN RUN_EVAL
  bash "$SCRIPT_PATH" run "$log_file" 2>&1 | tee "$log_file"
}

tail_log() {
  local lines="${1:-120}"
  local log_file
  log_file="$(latest_log)"
  if [[ -z "$log_file" ]]; then
    echo "No training log found under $LOG_DIR"
    exit 1
  fi
  echo "Tailing $log_file"
  tail -n "$lines" -F "$log_file"
}

show_logs() {
  if [[ ! -d "$LOG_DIR" ]]; then
    echo "No log dir found: $LOG_DIR"
    exit 0
  fi
  find "$LOG_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s bytes %p\n' | sort
}

show_status() {
  echo "Work dir: $WORK_DIR"
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Status: running"
      echo "PID: $pid"
    else
      echo "Status: not running (stale pid file: ${pid:-empty})"
    fi
  else
    echo "Status: not running"
  fi

  local log_file
  log_file="$(latest_log)"
  if [[ -n "$log_file" ]]; then
    echo "Latest log: $log_file"
    echo ""
    tail -n 50 "$log_file"
  else
    echo "Latest log: none"
  fi

  echo ""
  echo "Checkpoints:"
  if [[ -d "$CKPT_DIR" ]]; then
    find "$CKPT_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s bytes %p\n' | sort || true
  else
    echo "none"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    nvidia-smi
  fi
}

cmd="${1:-start}"
case "$cmd" in
  start)
    start_pipeline
    ;;
  foreground)
    foreground_pipeline
    ;;
  run)
    run_pipeline "${2:-}"
    ;;
  tail)
    tail_log "${2:-120}"
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 2
    ;;
esac
