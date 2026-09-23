#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/runs/worldsim_v75/WS-V75-FIVE-BASELINES-01/20260923-r1
smoke_pid=54303
smoke_root="$root/driveeditor-official25-seqcfg-r4-smoke"
smoke_result="$smoke_root/CFB-LATERAL-ACTOR-02-R4-10S__w06/result.json"

while kill -0 "$smoke_pid" 2>/dev/null; do
  sleep 20
done
if ! test -f "$smoke_result" || ! grep -q '"status": "generation_complete"' "$smoke_result"; then
  echo "r4 smoke failed; refusing to queue the same window without diagnosis" >&2
  exit 1
fi
while true; do
  if free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); then
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib > 22000 )); then
      break
    fi
  fi
  sleep 20
done

cd /root/autodl-tmp/motion_proj
/root/autodl-tmp/envs/driveeditor/bin/python scripts/plan_driveeditor_r9_remaining.py \
  --input-index /root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/driveeditor-r9-all-v2/index.json \
  --visibility-audit "$root/driveeditor-keyframe-visibility.json" \
  --completed-root "$root/driveeditor-official25-seqcfg" \
  --completed-root "$root/driveeditor-official25-seqcfg-r2" \
  --completed-root "$root/driveeditor-official25-seqcfg-r3" \
  --completed-root "$smoke_root" \
  --output "$root/driveeditor-continuation-r5-index.json"

exec env HF_HUB_OFFLINE=1 DRIVEEDITOR_SEQUENTIAL_CFG=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  /root/autodl-tmp/envs/driveeditor/bin/python scripts/run_driveeditor_worldsim_v75_batch.py \
  --source-root /root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor \
  --input-index "$root/driveeditor-continuation-r5-index.json" \
  --output-root "$root/driveeditor-official25-seqcfg-r5" \
  --steps 25 --seed 42 --decoding-t 1
