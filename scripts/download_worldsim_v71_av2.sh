#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/root/autodl-tmp/motion_proj
readonly COHORT="$REPO/configs/worldsim_v71/av2_zero_shot_cohort_v1.json"
readonly DATA=/root/autodl-tmp/data/av2/sensor/val
readonly STATE=/root/autodl-tmp/data/av2/v71_download_state
readonly LOG=/root/autodl-tmp/data/av2/v71_download.log
readonly S5CMD=/root/autodl-tmp/bin/s5cmd

mkdir -p "$DATA" "$STATE"
rm -f "$STATE/ALL_COMPLETE"
mapfile -t LOG_IDS < <(/root/miniconda3/envs/motionproj/bin/python -c \
  'import json,sys; print("\n".join(x["log_id"] for x in json.load(open(sys.argv[1]))["logs"]))' "$COHORT")

for log_id in "${LOG_IDS[@]}"; do
  if [[ -f "$STATE/$log_id.complete" ]]; then
    continue
  fi
  printf '%s\n' "$log_id" > "$STATE/current_log"
  mkdir -p "$DATA/$log_id"
  printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$log_id" >> "$LOG"
  "$S5CMD" --no-sign-request --numworkers 4 --retry-count 20 cp \
    "s3://argoverse/datasets/av2/sensor/val/$log_id/*" "$DATA/$log_id/" >> "$LOG" 2>&1
  touch "$STATE/$log_id.complete"
  printf '[%s] COMPLETE %s\n' "$(date -u +%FT%TZ)" "$log_id" >> "$LOG"
done

touch "$STATE/ALL_COMPLETE"
rm -f "$STATE/current_log"
printf '[%s] ALL_COMPLETE %s logs\n' "$(date -u +%FT%TZ)" "${#LOG_IDS[@]}" >> "$LOG"
