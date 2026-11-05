#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
exec 9>../gpu_slot.lock
flock -w 1800 9
test -f evaluation/restore.json
flock -u 9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
/root/autodl-tmp/envs/drivingeditor/bin/python source/review.py
tar -czf delivery.tar.gz delivery
sha256sum delivery.tar.gz > delivery.sha256
