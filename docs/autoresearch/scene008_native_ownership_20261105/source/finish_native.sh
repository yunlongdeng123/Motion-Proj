#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
run="$root/$1"
base=$(dirname "$root")
dual=$base/scene008_a6_a37_dynamic_20261104_r1
py=/root/autodl-tmp/envs/drivingeditor/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>"$base/gpu_slot.lock"
flock -w 2700 9
test "$(cat "$run/status")" = complete
"$py" "$root/source/inspect_edit.py" --model "$run/formal" --iteration 10000 --frames all --output "$run/edit_native"
"$py" "$root/source/inspect_edit.py" --model "$dual/formal_dual_actor_r1" --iteration 10000 --frames all --output "$run/edit_union"
flock -u 9
bash "$root/source/closeout_native.sh" "$1"
