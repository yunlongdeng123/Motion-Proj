#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/motion_proj
source scripts/worldsim_v75/environment.sh
run=/root/autodl-tmp/runs/worldsim_v75/WS-V75-BASELINE-01/20260920-single3090-r1
test ! -e "$run/clean-repeat-seed42.mp4" || exit 2
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/clean-repeat.started_utc"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -lms 200 > "$run/clean-repeat.gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/worldsim_v75/run_baseline.py --execute --blocks 30 --seed 42 \
    --output "$run/clean-repeat-seed42.mp4" > "$run/clean-repeat.log" 2>&1
result=$?
echo "$result" > "$run/clean-repeat.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/clean-repeat.ended_utc"
echo "inference_exit=$result"
exit "$result"
