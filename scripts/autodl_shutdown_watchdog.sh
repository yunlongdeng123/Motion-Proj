#!/usr/bin/env bash
set -Eeuo pipefail

HEARTBEAT=/root/autodl-tmp/runs/dynamic_editing_v2/.codex_agent_heartbeat
LOG=/root/autodl-tmp/runs/dynamic_editing_v2/.codex_shutdown_watchdog.log
STALE_SECONDS=${CODEX_WATCHDOG_STALE_SECONDS:-1800}

mkdir -p "$(dirname "$HEARTBEAT")"
touch "$HEARTBEAT"
echo "$(date --iso-8601=seconds) watchdog started stale_seconds=$STALE_SECONDS" >> "$LOG"
while true; do
  now_epoch=$(date +%s)
  heartbeat_epoch=$(stat -c %Y "$HEARTBEAT")
  age=$((now_epoch - heartbeat_epoch))
  if (( age >= STALE_SECONDS )); then
    echo "$(date --iso-8601=seconds) heartbeat stale age=$age; shutting down" >> "$LOG"
    sync
    /usr/bin/shutdown -h now
    exit 0
  fi
  sleep 60
done
