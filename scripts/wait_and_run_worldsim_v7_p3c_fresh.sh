#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/motion_proj
state=/root/autodl-tmp/data/av2/v7_recovery_download_state
p6c_run=/root/autodl-tmp/runs/worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/20260902T173000Z__sparsity-consistent-s70602-r1
p3c_run_id=20260902T231500Z__fresh-visibility-s0-r1
p3c_run=/root/autodl-tmp/runs/worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-FRESH-01/${p3c_run_id}
lock=/root/autodl-tmp/data/av2/v7_p3c_fresh_watcher.lock

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

exec 9>"$lock"
if ! flock -n 9; then
  log "another P3-C fresh watcher owns the lock; exiting"
  exit 0
fi

while true; do
  completed=$(find "$state" -maxdepth 1 -type f -name '*.complete' | wc -l)
  if [[ -f "$p6c_run/summary.json" ]] \
      && grep -q '"status"[[:space:]]*:[[:space:]]*"done"' "$p6c_run/status.json" \
      && ! pgrep -f '[r]un_worldsim_v7_p6c_sparsity_consistent_external.py' >/dev/null; then
    if [[ "$completed" -ne 20 || ! -f "$state/ALL_COMPLETE" ]]; then
      log "P6-C done but download contract incomplete completed=${completed}/20"
      sleep 300
      continue
    fi
    if [[ -e "$p3c_run" ]]; then
      log "P3-C run path already exists; refusing an automatic second read path=${p3c_run}"
      exit 3
    fi
    log "launching frozen fresh P3-C run_id=${p3c_run_id} after P6-C normal completion"
    cd "$repo"
    exec env PYTHONPATH=. /root/autodl-tmp/envs/motionproj/bin/python \
      scripts/run_worldsim_v7_p3c_visibility_certificate.py \
      --config configs/worldsim_v7/p3c_av2_visibility_certificate_fresh_v1.yaml \
      --repo-root "$repo" \
      --run-id "$p3c_run_id"
  fi
  current=$(cat "$state/current_log" 2>/dev/null || true)
  log "waiting for P6-C summary completed=${completed}/20 current=${current:-none}"
  sleep 300
done
