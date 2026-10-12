#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
printf '%s\n' "$$" > "$BASE/logs/formal_demo.pid"
date -Iseconds > "$BASE/logs/formal_demo.started"
trap 'code=$?; printf "%s\n" "$code" > "$BASE/logs/formal_demo.exit"; date -Iseconds > "$BASE/logs/formal_demo.finished"' EXIT
bash "$BASE/scripts/train_demo_30000.sh"
source "$BASE/scripts/activate.sh"
python "$BASE/scripts/verify_formal_completion.py"
