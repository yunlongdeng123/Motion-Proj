#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
# Wait only for this task's known setup process, not any unrelated jobs.
SETUP_PID=$(cat "$BASE/logs/setup.pid")
while kill -0 "$SETUP_PID" 2>/dev/null; do sleep 5; done
test "$(cat "$BASE/logs/setup.exit")" = 0
source "$BASE/scripts/activate.sh"
python "$BASE/scripts/verify_drivingeditor_env.py" > "$BASE/logs/environment_verification.log" 2>&1
printf 'core_cuda_verified\n' > "$BASE/logs/core_ready.status"
bash "$BASE/scripts/setup_drivingeditor_masks.sh" > "$BASE/logs/mask_setup.log" 2>&1
python -m pip check > "$BASE/logs/pip_check.log" 2>&1
printf 'ready\n' > "$BASE/logs/environment_ready.status"
