#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
base=$(dirname "$root")
dual=$base/scene008_a6_a37_dynamic_20261104_r1
py=/root/autodl-tmp/envs/drivingeditor/bin/python
masks=$base/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137/dynamic_actor_assets/actor_3/body_masks
exec 9>"$base/gpu_slot.lock"
flock -n 9
common=(--assignment "$dual/inputs/init_assignment.npz" --a6-masks "$masks")
run="$root/$1"
mkdir "$run"
printf 'running\n' > "$run/status"
trap 'printf "failed at line %d\n" "$LINENO" > "$run/status"' ERR
"$py" "$root/source/train_dual.py" --mode smoke --smoke-seconds "${2:-540}" --output "$run/smoke" "${common[@]}" > "$run/smoke.log" 2>&1
if [ "${2:-540}" = 540 ]; then
    step=$("$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASS_COMPLETE_BOUNDARY" and r["seconds"]>=540; print(r["iteration"])' "$run/smoke/returned.json")
    "$py" "$root/source/train_dual.py" --mode resume --checkpoint "$run/smoke/chkpnt$step.pth" --output "$run/resume" "${common[@]}" > "$run/resume.log" 2>&1
    test -s "$run/resume/restore_audit.json"
    "$py" "$root/source/compare_control.py" --model "$dual/formal_dual_actor_r1" --iteration 1000 --frames pilots --output "$run/control" > "$run/control1k.log" 2>&1
    "$py" "$root/source/train_dual.py" --mode formal --output "$run/formal" "${common[@]}" > "$run/formal.log" 2>&1
    "$py" "$root/source/compare_control.py" --model "$dual/formal_dual_actor_r1" --iteration 10000 --frames all --output "$run/control" > "$run/control10k.log" 2>&1
fi
printf 'complete\n' > "$run/status"
