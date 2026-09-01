#!/usr/bin/env bash
set -euo pipefail

readonly S5CMD=/root/autodl-tmp/bin/s5cmd
readonly PYTHON=/root/miniconda3/bin/python
readonly COHORT=/root/autodl-tmp/motion_proj/configs/worldsim_v7/av2_zero_shot_recovery_cohort_v1.json
readonly DATA_ROOT=/root/autodl-tmp/data/av2/sensor/val
readonly STATE_ROOT=/root/autodl-tmp/data/av2/v7_recovery_download_state
readonly DOWNLOAD_LOG=/root/autodl-tmp/data/av2/v7_recovery_download.log
readonly MINIMUM_FREE_BYTES=53687091200

mkdir -p "$DATA_ROOT" "$STATE_ROOT"
exec 9>"$STATE_ROOT/download.lock"
if ! flock -n 9; then
  printf '%s another recovery downloader already owns the lock\n' "$(date --iso-8601=seconds)" >>"$DOWNLOAD_LOG"
  exit 3
fi

mapfile -t log_ids < <(
  "$PYTHON" -c 'import json, pathlib, sys; c=json.loads(pathlib.Path(sys.argv[1]).read_text()); print("\n".join(row["log_id"] for row in c["logs"]))' "$COHORT"
)
if [[ "${#log_ids[@]}" -ne 20 ]]; then
  printf '%s invalid recovery cohort count=%s\n' "$(date --iso-8601=seconds)" "${#log_ids[@]}" >>"$DOWNLOAD_LOG"
  exit 4
fi

printf '%s start logs=%s sequential_logs=true target=%s\n' \
  "$(date --iso-8601=seconds)" "${#log_ids[@]}" "$DATA_ROOT" >>"$DOWNLOAD_LOG"

for index in "${!log_ids[@]}"; do
  log_id="${log_ids[$index]}"
  marker="$STATE_ROOT/$log_id.complete"
  if [[ -f "$marker" ]]; then
    printf '%s skip_complete index=%s log=%s\n' \
      "$(date --iso-8601=seconds)" "$index" "$log_id" >>"$DOWNLOAD_LOG"
    continue
  fi

  available_bytes="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d ' ')"
  if (( available_bytes < MINIMUM_FREE_BYTES )); then
    printf '%s stop_low_space available_bytes=%s\n' \
      "$(date --iso-8601=seconds)" "$available_bytes" >>"$DOWNLOAD_LOG"
    exit 5
  fi

  printf '%s\n' "$log_id" >"$STATE_ROOT/current_log"
  mkdir -p "$DATA_ROOT/$log_id"
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    printf '%s begin index=%s log=%s attempt=%s\n' \
      "$(date --iso-8601=seconds)" "$index" "$log_id" "$attempt" >>"$DOWNLOAD_LOG"
    if "$S5CMD" --numworkers 16 --log error --no-sign-request sync \
      "s3://argoverse/datasets/av2/sensor/val/$log_id/*" \
      "$DATA_ROOT/$log_id/" >>"$DOWNLOAD_LOG" 2>&1; then
      touch "$marker"
      log_bytes="$(du -sb "$DATA_ROOT/$log_id" | cut -f1)"
      printf '%s done index=%s log=%s bytes=%s\n' \
        "$(date --iso-8601=seconds)" "$index" "$log_id" "$log_bytes" >>"$DOWNLOAD_LOG"
      break
    fi
    printf '%s retry index=%s log=%s attempt=%s delay_s=60\n' \
      "$(date --iso-8601=seconds)" "$index" "$log_id" "$attempt" >>"$DOWNLOAD_LOG"
    sleep 60
  done
done

printf '%s\n' ALL_COMPLETE >"$STATE_ROOT/current_log"
touch "$STATE_ROOT/ALL_COMPLETE"
printf '%s all_complete logs=%s\n' "$(date --iso-8601=seconds)" "${#log_ids[@]}" >>"$DOWNLOAD_LOG"
