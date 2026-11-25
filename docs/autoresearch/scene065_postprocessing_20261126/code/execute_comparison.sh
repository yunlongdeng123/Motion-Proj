#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
JOB=/root/autodl-tmp/_wsjob
export PYTHONPATH=/root/autodl-tmp/envs/vace13b_site
PY=/root/miniconda3/bin/python3.10
"$PY" -u "$JOB/stage_official_inpainting.py"
"$PY" -u "$JOB/vace_run_clip.py" --clip 087 --mask-tag w8_swept_cd18 --steps 25 --tone none
"$PY" -u "$JOB/vace_run_clip.py" --clip 141 --mask-tag w8_swept_cd18 --steps 25 --tone none
unset PYTHONPATH
/root/autodl-tmp/envs/driveeditor/bin/python -u "$JOB/run_matched_propainter.py"
printf '\nCOMPARISON_GPU_COMPLETE\n'
