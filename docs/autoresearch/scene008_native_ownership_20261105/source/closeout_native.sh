#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
run="$root/$1"
dest="$root/$1_review"
py=/root/autodl-tmp/envs/drivingeditor/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test "$(cat "$run/status")" = complete
test ! -e "$dest"
"$py" "$root/source/build_native_report.py" "$run" "$dest"
"$py" "$root/source/audit_native.py" "$run" "$dest/data/audit.json"
cp "$root/run_r2/road_diag/"*.json "$dest/data/"
cp "$root/early_control/road_only_1000.json" "$dest/data/road_union_1000.json"
cd "$root"
tar -czf "$1_review.tar.gz" "$1_review"
sha256sum "$1_review.tar.gz" > "$1_review.sha256"
