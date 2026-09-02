#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/autodl-tmp/motion_proj
STATE_ROOT=/root/autodl-tmp/data/av2/v7_recovery_download_state
DOWNLOAD_SCRIPT="$REPO_ROOT/scripts/download_worldsim_v7_av2_recovery.sh"
RUNNER="$REPO_ROOT/scripts/run_worldsim_v7_p6c_sparsity_consistent_external.py"
CONFIG="$REPO_ROOT/configs/worldsim_v7/p6c_sparsity_consistent_selector_v1.yaml"
RUN_ID=20260902T173000Z__sparsity-consistent-s70602-r1
RUN_DIR=/root/autodl-tmp/runs/worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/$RUN_ID
LOCK_PATH=/root/autodl-tmp/data/av2/v7_p6c_external.lock

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "another P6-C external watcher/evaluator holds $LOCK_PATH" >&2
  exit 1
fi

status_file="$RUN_DIR/status.json"
if ! grep -q 'model_frozen_waiting_fresh_av2' "$status_file"; then
  echo "refuse launch: P6-C status is not model_frozen_waiting_fresh_av2" >&2
  cat "$status_file" >&2
  exit 1
fi

last_report=0
while true; do
  completed=$(find "$STATE_ROOT" -maxdepth 1 -type f -name '*.complete' | wc -l)
  if [[ "$completed" -eq 20 && -f "$STATE_ROOT/ALL_COMPLETE" ]]; then
    break
  fi
  if ! pgrep -f "$DOWNLOAD_SCRIPT" >/dev/null; then
    echo "download stopped before readiness: completed=$completed/20" >&2
    exit 1
  fi
  now=$(date +%s)
  if (( now - last_report >= 600 )); then
    current=$(cat "$STATE_ROOT/current_log" 2>/dev/null || true)
    echo "$(date -Iseconds) waiting completed=$completed/20 current=$current"
    last_report=$now
  fi
  sleep 60
done

echo "$(date -Iseconds) ALL_COMPLETE confirmed; starting frozen P6-C external read"
cd "$REPO_ROOT"
export PYTHONPATH=.
exec /root/autodl-tmp/envs/motionproj/bin/python "$RUNNER" \
  --config "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --run-id "$RUN_ID"
