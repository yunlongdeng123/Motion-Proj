#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/external/worldsim_v75/VAD-GS
RUN=/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000
PY=/root/autodl-tmp/envs/vadgs-v76/bin/python
mkdir "$RUN/official_test_16k"
trap 'printf "%s\n" "$?" > "$RUN/official_test_16k/exit_code.txt"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
CUDA_VISIBLE_DEVICES= "$PY" script/v76/audit_checkpoint_finite.py \
  --checkpoint "$RUN/trained_model/iteration_16000.pth" --output "$RUN/official_test_16k/checkpoint_finite.json"
"$PY" script/v76/evaluate_official_test.py --config configs/v76/nuscenes_000_repro_exhaustive.yaml \
  --iteration 16000 --output-dir "$RUN/official_test_16k"
"$PY" script/v76/plot_official_test.py --input-dir "$RUN/official_test_16k" \
  --output "$RUN/official_test_16k/frame20.png"
