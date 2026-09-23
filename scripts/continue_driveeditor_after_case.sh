#!/usr/bin/env bash
set -euo pipefail

# Fail-closed, one-shot GPU handoff between two prepared approved cases.
# Usage: continue_driveeditor_after_case.sh prior-pid prior-case next-case next-slug
prior_pid="${1:?prior PID required}"
prior_case="${2:?prior case required}"
next_case="${3:?next case required}"
next_slug="${4:?next slug required}"
[[ "$prior_pid" =~ ^[0-9]+$ ]] || { echo "invalid prior PID" >&2; exit 2; }
[[ "$prior_case" =~ ^CFB-[A-Z0-9-]+$ ]] || { echo "invalid prior case" >&2; exit 2; }
[[ "$next_case" =~ ^CFB-[A-Z0-9-]+$ ]] || { echo "invalid next case" >&2; exit 2; }
[[ "$next_slug" =~ ^[a-z0-9]+$ ]] || { echo "invalid next slug" >&2; exit 2; }

run_root=/root/autodl-tmp/runs/worldsim_v75/WS-V75-FIVE-BASELINES-01/20260923-r1
python=/root/autodl-tmp/envs/driveeditor/bin/python

exec 9>"$run_root/driveeditor-after-$prior_case.lock"
flock -n 9 || { echo "another continuation already holds the lock" >&2; exit 2; }

while kill -0 "$prior_pid" 2>/dev/null; do
  sleep 30
done

"$python" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); case=sys.argv[2]; factual=root/"driveeditor-factual-cases"/case; native=root/"driveeditor-native-cases"/case; iterative=root/"driveeditor-iterative-cases"/case; result=json.loads((iterative/"result.json").read_text()); assert result["status"]=="generation_complete" and result["generated_frames"]==100 and result["native_segments"]==11; assert all(p.is_file() for p in (factual/"factual-reconstruction.mp4", factual/"protocol.json", native/"counterfactual.mp4", native/"protocol.json", native/"paired-auto-metrics.json", iterative/"counterfactual.mp4", iterative/"original-nuscenes.mp4", iterative/"auto-metrics.json", iterative/"detector-metrics.json"))' "$run_root" "$prior_case"

for _ in $(seq 1 60); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "prior case complete and GPU idle; starting $next_case"
    exec bash /root/autodl-tmp/motion_proj/scripts/run_driveeditor_approved_sequence.sh "$next_case" "$next_slug"
  fi
  sleep 10
done

echo "GPU did not become idle after prior case; no next inference started" >&2
exit 3
