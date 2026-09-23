#!/usr/bin/env bash
set -euo pipefail

# Execute one already-prepared, human-approved DriveEditor case on a single GPU.
# Usage: run_driveeditor_approved_sequence.sh CFB-INSERT-02-R4-10S insert02
case_id="${1:?case id required}"
slug="${2:?input/output slug required}"
[[ "$case_id" =~ ^CFB-[A-Z0-9-]+$ ]] || { echo "invalid case id" >&2; exit 2; }
[[ "$slug" =~ ^[a-z0-9]+$ ]] || { echo "invalid slug" >&2; exit 2; }

repo=/root/autodl-tmp/motion_proj
python=/root/autodl-tmp/envs/driveeditor/bin/python
driveeditor=/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor
manifest=/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r9-all-approved
run_root=/root/autodl-tmp/runs/worldsim_v75/WS-V75-FIVE-BASELINES-01/20260923-r1
input_root=/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs
factual_input="$input_root/driveeditor-r9-$slug-factual-native-v1/index.json"
native_input="$input_root/driveeditor-r9-$slug-cf-native-v1/index.json"
iterative_input="$input_root/driveeditor-r9-$slug-iterative-v1/index.json"

for index in "$factual_input" "$native_input" "$iterative_input"; do
  [[ -f "$index" ]] || { echo "missing index: $index" >&2; exit 3; }
  "$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); rows=d.get("cases", []); assert rows and all(r.get("approved_case_id")==sys.argv[2] for r in rows)' "$index" "$case_id"
done

cd "$repo"
export DRIVEEDITOR_SEQUENTIAL_CFG=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

"$python" scripts/run_driveeditor_worldsim_v75_batch.py \
  --source-root "$driveeditor" --input-index "$factual_input" \
  --output-root "$run_root/driveeditor-$slug-factual-native-v1" \
  --steps 25 --seed 42 --decoding-t 1
"$python" scripts/register_driveeditor_paper_native.py \
  --input-index "$factual_input" --run-root "$run_root/driveeditor-$slug-factual-native-v1" \
  --output-root "$run_root/driveeditor-factual-cases" --case-id "$case_id"

"$python" scripts/run_driveeditor_worldsim_v75_batch.py \
  --source-root "$driveeditor" --input-index "$native_input" \
  --output-root "$run_root/driveeditor-$slug-cf-native-v1" \
  --steps 25 --seed 42 --decoding-t 1
"$python" scripts/register_driveeditor_paper_native.py \
  --input-index "$native_input" --run-root "$run_root/driveeditor-$slug-cf-native-v1" \
  --output-root "$run_root/driveeditor-native-cases" --case-id "$case_id"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$python" scripts/compute_cfbench_r9_native_metrics.py \
  --case-id "$case_id" --manifest-root "$manifest" \
  --factual-cases "$run_root/driveeditor-factual-cases" \
  --native-cases "$run_root/driveeditor-native-cases" \
  --geometry-root "$run_root/approved-geometry"

"$python" scripts/run_driveeditor_worldsim_v75_batch.py \
  --source-root "$driveeditor" --input-index "$iterative_input" \
  --output-root "$run_root/driveeditor-$slug-iterative-v1" \
  --paper-iterative --steps 25 --seed 42 --decoding-t 1
"$python" scripts/stitch_driveeditor_r9_iterative.py \
  --manifest-root "$manifest" --input-index "$iterative_input" \
  --window-root "$run_root/driveeditor-$slug-iterative-v1" \
  --output-root "$run_root/driveeditor-iterative-cases" --case-id "$case_id"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$python" scripts/compute_cfbench_r9_video_metrics.py \
  --cases-root "$run_root/driveeditor-iterative-cases" \
  --geometry-root "$run_root/approved-geometry" \
  --mode paper_iterative --case-id "$case_id"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$python" scripts/compute_cfbench_r9_detector_metrics.py \
  --cases-root "$run_root/driveeditor-iterative-cases" \
  --geometry-root "$run_root/approved-geometry" \
  --manifest "$manifest/candidates.json" --sample-step 5 --case-id "$case_id"
