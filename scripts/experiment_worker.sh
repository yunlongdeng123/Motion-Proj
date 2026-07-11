#!/usr/bin/env bash
# 单实例、带 heartbeat 的实验 worker。建议在 tmux 中启动。
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/experiment_worker.sh <run-dir> <command> [args...]" >&2
  exit 2
fi

run_dir="$1"
shift
mkdir -p "$run_dir"
lock_file="$run_dir/worker.lock"
heartbeat="$run_dir/heartbeat.json"
events="$run_dir/events.jsonl"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "已有 worker 持有锁: $lock_file" >&2
  exit 3
fi

heartbeat_loop() {
  while true; do
    printf '{"time":"%s","pid":%d}\n' "$(date --iso-8601=seconds)" "$$" > "$heartbeat.tmp"
    mv "$heartbeat.tmp" "$heartbeat"
    sleep 60
  done
}

heartbeat_loop &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT INT TERM

delays=(60 300 900)
attempt=0
while true; do
  attempt=$((attempt + 1))
  printf '{"time":"%s","event":"attempt","attempt":%d}\n' \
    "$(date --iso-8601=seconds)" "$attempt" >> "$events"
  "$@"
  code=$?
  if [[ $code -eq 0 ]]; then
    printf '{"time":"%s","event":"completed","attempt":%d}\n' "$(date --iso-8601=seconds)" "$attempt" >> "$events"
    exit 0
  fi
  # 2=配置错误，42=NaN，43=训练 OOM；这些都不能以原 run 语义静默重启。
  if [[ $code -eq 2 || $code -eq 42 || $code -eq 43 ]]; then
    printf '{"time":"%s","event":"fatal","attempt":%d,"exit_code":%d}\n' \
      "$(date --iso-8601=seconds)" "$attempt" "$code" >> "$events"
    exit "$code"
  fi
  retry_index=$((attempt - 1))
  if [[ $retry_index -ge ${#delays[@]} ]]; then
    printf '{"time":"%s","event":"failed","attempt":%d,"exit_code":%d}\n' \
      "$(date --iso-8601=seconds)" "$attempt" "$code" >> "$events"
    exit "$code"
  fi
  delay="${delays[$retry_index]}"
  printf '{"time":"%s","event":"retrying","attempt":%d,"exit_code":%d,"delay_seconds":%d}\n' \
    "$(date --iso-8601=seconds)" "$attempt" "$code" "$delay" >> "$events"
  sleep "$delay"
done
