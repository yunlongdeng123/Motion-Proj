#!/usr/bin/env bash
# 完整拟合后运行一次冻结协议；不会等待拟合或自动启动第二个场景。
set -euo pipefail
N=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1
P=/root/autodl-tmp/motion_proj/scripts/worldsim_simimpact
E=/root/autodl-tmp/envs/splatad-impact
FIT="$N/fits/scene-0004/splatad/native-r2"
VAL="$N/validation/scene0004-step030000-full1"
LOOP="$N/closed_loop/scene0004-step030000-full1"
export PATH="$E/bin:/root/autodl-tmp/envs/hugsim-impact/bin:/root/autodl-tmp/envs/motionproj/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$N/torch_extensions" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$E/bin/python" - "$FIT" "$VAL" "$LOOP" <<'PY'
import json,sys
from pathlib import Path
fit,val,loop=map(Path,sys.argv[1:])
assert json.loads((fit/'fit_manifest.json').read_text())['completed'], 'Complete the registered fit first'
assert (fit/'nerfstudio_models/step-000030000.ckpt').is_file()
assert not val.exists() and not loop.exists(), 'Preserve existing evaluation; inspect before any partial-run continuation'
PY
"$E/bin/python" "$P/validate_native_splatad.py" --fit "$FIT" --out "$VAL" --checkpoint-step 30000 --all-lidars --all-cameras
"$E/bin/python" "$P/run_splatad_closed_loop.py" --fit "$FIT" --checkpoint-dir "$VAL/checkpoint" --checkpoint-step 30000 --out "$LOOP" --replay-sensor-timing logged
"$E/bin/python" "$P/audit_native_closed_loop.py" --run "$LOOP" --reference "$N/closed_loop/reference/scene-0004.json"
"$E/bin/python" "$P/export_native_closed_loop.py" --run "$LOOP" --fit "$FIT" --checkpoint-dir "$VAL/checkpoint"
printf '%s\n' "NATIVE_FULL_EVALUATION_COMPLETED $LOOP"
