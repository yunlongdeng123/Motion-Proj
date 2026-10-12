#!/usr/bin/env bash
# Full reconstruction entry point; this file does not launch itself.
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
source "$BASE/scripts/activate.sh"
SCENE="$BASE/data/waymo_scenes/0017085_demo_000_020"
RUN="$BASE/runs/0017085_demo_000_020_dynamic_30000"
test -f "$BASE/logs/environment_ready.status"
test -f "$SCENE/points3d.ply"
compgen -G "$SCENE/points3d_dynamic_*.ply" > /dev/null
test -d "$SCENE/dynamic_image/dynamic_image_0"
if [ -e "$RUN" ]; then
  printf 'Refusing to overwrite existing run: %s\n' "$RUN" >&2
  exit 1
fi
cd "$BASE/DrivingEditor-run"
exec python train.py --config configs/waymo_reconstruction.yaml \
  source_path="$SCENE" model_path="$RUN" including_dynamic=True
