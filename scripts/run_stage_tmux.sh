#!/usr/bin/env bash
# 为单卡长阶段启动独立 tmux worker、日志管道和五分钟 watchdog。
set -euo pipefail

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "Usage: bash scripts/run_stage_tmux.sh <session> <run-dir> -- <command> [args...]" >&2
  exit 2
fi

session="$1"
run_dir="$2"
shift 3
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "ERROR: tmux session 已存在: $session" >&2
  exit 3
fi
if [[ -e "$run_dir" && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "ERROR: run 目录已存在；显式设置 ALLOW_RESUME=1 才能恢复: $run_dir" >&2
  exit 4
fi
mkdir -p "$run_dir/logs"

printf -v command_q '%q ' "$@"
printf -v root_q '%q' "$root"
printf -v run_q '%q' "$run_dir"
worker="cd $root_q && source /root/miniconda3/etc/profile.d/conda.sh && conda activate motionproj && export HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && exec bash scripts/experiment_worker.sh $run_q $command_q"
tmux new-session -d -s "$session" "$worker"
tmux pipe-pane -o -t "$session" "cat >> $run_q/logs/stdout.log"

watchdog_session="${session}-watchdog"
watchdog="while tmux has-session -t $(printf '%q' "$session") 2>/dev/null; do cd $root_q; source /root/miniconda3/etc/profile.d/conda.sh; conda activate motionproj; python scripts/watchdog.py $run_q >> $run_q/watchdog.jsonl 2>&1 || true; sleep 300; done"
tmux new-session -d -s "$watchdog_session" "$watchdog"

printf 'worker=%s\nwatchdog=%s\nrun_dir=%s\n' "$session" "$watchdog_session" "$run_dir"
