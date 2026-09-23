#!/usr/bin/env bash
set -euo pipefail

# Keep the single GPU occupied without overlapping the in-flight r2 process.
previous_pid=18167
while kill -0 "$previous_pid" 2>/dev/null; do
  sleep 30
done
while true; do
  if free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); then
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib > 22000 )); then
      break
    fi
  fi
  sleep 30
done

cd /root/autodl-tmp/motion_proj
exec env HF_HUB_OFFLINE=1 DRIVEEDITOR_SEQUENTIAL_CFG=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  /root/autodl-tmp/envs/driveeditor/bin/python scripts/run_driveeditor_worldsim_v75_batch.py \
  --source-root /root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor \
  --input-index /root/autodl-tmp/runs/worldsim_v75/WS-V75-FIVE-BASELINES-01/20260923-r1/driveeditor-continuation-r3-fullfirst-index.json \
  --output-root /root/autodl-tmp/runs/worldsim_v75/WS-V75-FIVE-BASELINES-01/20260923-r1/driveeditor-official25-seqcfg-r3 \
  --steps 25 --seed 42 --decoding-t 1
