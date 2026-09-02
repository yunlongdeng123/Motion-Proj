#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/motion_proj
state=/root/autodl-tmp/data/av2/v7_p16_download_state
run_id=20260903T183000Z__fresh-av2-literal-first-return-s0-r1

while true; do
  completed=$(find "$state" -maxdepth 1 -type f -name '*.complete' | wc -l)
  printf '%s completed=%s/10\n' "$(date --iso-8601=seconds)" "$completed"
  if [ "$completed" -eq 10 ]; then
    break
  fi
  sleep 120
done

cd "$repo"
exec env PYTHONPATH=. /root/autodl-tmp/envs/motionproj/bin/python \
  scripts/run_worldsim_v7_p23_fresh_av2_first_return_confirmation.py \
  --config configs/worldsim_v7/p23_fresh_av2_literal_first_return_confirmation_v1.yaml \
  --repo-root "$repo" \
  --run-id "$run_id"
