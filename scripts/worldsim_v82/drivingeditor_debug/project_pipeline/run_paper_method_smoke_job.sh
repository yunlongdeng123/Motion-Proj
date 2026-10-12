#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
source "$BASE/scripts/activate.sh"
date -Iseconds > "$BASE/logs/paper_method_smoke1000.started"
set +e
python -u "$BASE/scripts/run_paper_smoke.py" \
  --repository "$BASE/DrivingEditor-paper" \
  --scene "$BASE/data/waymo_paper_scenes/smoke_demo_000_020" \
  --output "$BASE/runs/paper_method_smoke1000" \
  --paper-config configs/paper_waymo_reconstruction.yaml \
  --iterations 1000 --including-dynamic
code=$?
set -e
printf '%s\n' "$code" > "$BASE/logs/paper_method_smoke1000.exit"
date -Iseconds > "$BASE/logs/paper_method_smoke1000.finished"
exit "$code"
