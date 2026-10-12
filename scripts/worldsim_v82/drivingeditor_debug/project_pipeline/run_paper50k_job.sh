#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
source "$BASE/scripts/activate.sh"
# Runtime-only recovery for the documented allocator fragmentation failure.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="$BASE/runs/scene0017_000_048_paper_50000_seed0"
SCENE="$BASE/data/waymo_paper_scenes/scene0017_000_048"
exec 9>"$BASE/logs/paper50k_job.lock"
flock -n 9 || { printf '50k job already locked\n' >&2; exit 75; }
test ! -e "$RUN" || { printf 'Run already exists; refusing overwrite: %s\n' "$RUN" >&2; exit 1; }
date -Iseconds > "$BASE/logs/paper50k.started"
printf '%s\n' "$BASHPID" > "$BASE/logs/paper50k.launcher.pid"
finish() { status=$?; printf '%s\n' "$status" > "$BASE/logs/paper50k.exit"; date -Iseconds > "$BASE/logs/paper50k.finished"; }
trap finish EXIT
python "$BASE/scripts/run_paper50k_training.py" \
  --repository "$BASE/DrivingEditor-paper" --scene "$SCENE" --output "$RUN"
