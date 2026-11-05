#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 || $3 != --confirm-gpu || ( $1 != smoke && $1 != formal ) ]]; then
  printf 'Usage: bash run_gpu.sh {smoke|formal} NEW_OUTPUT_DIR --confirm-gpu\n' >&2
  exit 2
fi

mode=$1
output=$(realpath -m "$2")
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=/root/autodl-tmp/envs/drivingeditor/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec 9>>"$root/../gpu_slot.lock"
flock -n 9 || { printf 'Campaign GPU slot is already reserved.\n' >&2; exit 1; }

fit() { "$python" -B "$root/source/fit_instances.py" "$@" --root "$root" --confirm-gpu; }

if [[ $mode == smoke ]]; then
  fit smoke --out "$output"
  exit 0
fi

# 由操作方选定已经通过的真实 smoke 目录。没有回执则不会启动正式队列。
: "${SCENE008_SMOKE_DIR:?Point SCENE008_SMOKE_DIR at the passing real GPU smoke output.}"
"$python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "PASS" and r["scope"] == "engineering_smoke" and r["restored_b"] and r["restored_c"] and r["d_entry"] and r["settings"] == json.load(open(sys.argv[2]))' "$SCENE008_SMOKE_DIR/receipt.json" "$root/source/settings.json"
mkdir -- "$output"
fit train-b --out "$output/b"
b_checkpoint=$(find "$output/b" -maxdepth 1 -name 'checkpoint*.pt' | sort | tail -1)
fit evaluate --out "$output/eval_b_anchors" --checkpoint "$b_checkpoint" --anchors --groups strong_baseline objects
fit attribute --out "$output/ledger_all49.npz"
fit train-c --out "$output/c" --checkpoint "$b_checkpoint" --ledger "$output/ledger_all49.npz"
c_checkpoint=$(find "$output/c" -maxdepth 1 -name 'checkpoint*.pt' | sort | tail -1)
fit evaluate --out "$output/eval_c_anchors" --checkpoint "$c_checkpoint" --ledger "$output/ledger_all49.npz" --anchors --groups strong_baseline objects replacement
fit attach-pvd --out "$output/d" --checkpoint "$c_checkpoint" --ledger "$output/ledger_all49.npz"
fit evaluate --out "$output/eval_all49" --checkpoint "$output/d/checkpoint000000.pt" --ledger "$output/ledger_all49.npz"
