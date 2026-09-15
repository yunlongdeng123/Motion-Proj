#!/usr/bin/env bash
set -euo pipefail
export SIMIMPACT_RUN_ROOT=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
R="$SIMIMPACT_RUN_ROOT"
E=/root/autodl-tmp/envs/hugsim-impact/bin/python
F=/root/autodl-tmp/envs/worldsim-v81/bin/python
cd "$R/scripts"
"$E" extract_raw_initial.py > "$R/extraction.log" 2>&1
"$E" -c 'import json,os,pathlib; r=pathlib.Path(os.environ["SIMIMPACT_RUN_ROOT"]); d=json.loads((r/"raw_extraction_result.json").read_text()); assert not d["missing"],d["missing"]'
"$E" prepare_reconstruction_inputs.py > "$R/input_preparation.log" 2>&1
"$E" prepare_lidar_reference.py > "$R/reference_preparation.log" 2>&1
mkdir -p "$R/lidar_policy/assets"
ln -s /root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1/lidar_policy/assets/transfuser_seed_0.ckpt "$R/lidar_policy/assets/transfuser_seed_0.ckpt"
"$E" prepare_policy_log_reference.py > "$R/policy_log_reference.log" 2>&1
"$E" prepare_causal_ego_status.py > "$R/causal_status.log" 2>&1
for method in dvgt1 vggt omega512 pi3x; do
  "$F" infer_reconstruction.py --method "$method" > "$R/inference_${method}.log" 2>&1
done
"$E" raycast_lidar_policy_inputs.py > "$R/lidar_raycast.log" 2>&1
"$E" probe_transfuser_lidar.py > "$R/transfuser_probe.log" 2>&1
"$E" simulate_transfuser_pdm.py > "$R/pdm_simulation.log" 2>&1
