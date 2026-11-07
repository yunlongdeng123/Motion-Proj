"""Launch12 genuinely separate estimator processes; seal predictions before GT."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode=True


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(x,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);args=p.parse_args();root=args.benchmark.resolve()
    source=Path(__file__).resolve().parent;estimator=source/"legal_context_baseline.py";start=time.time()
    patches=read(root/"baseline_inputs/appearance_blind/patch_manifest.json")["patches"]
    tracks=("appearance_blind","geometry_RGB_blind")
    run=root/"baseline_runs/fixed_plane_boundary_median_v1"
    if run.exists():raise FileExistsError("Preserve earlier independent estimator runs")
    run.mkdir(parents=True)
    settings_hash=sha(root/"baseline_inputs/appearance_blind/baseline_settings.json")
    if settings_hash!=sha(root/"baseline_inputs/geometry_RGB_blind/baseline_settings.json"):
        raise ValueError("Tracks must share frozen baseline settings")
    write(run/"schedule.json",dict(method="context_robust_plane_nearest_boundary_median",settings_sha256=settings_hash,
        estimator_sha256=sha(estimator),tracks=list(tracks),patch_ids=[x["patch_id"] for x in patches],
        all_frozen_before_any_reference_evaluation=True,target_GT_access_allowed=False))
    receipts=[]
    for track in tracks:
        for patch in patches:
            work=run/track/patch["patch_id"];work.mkdir(parents=True)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH="",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            cmd=[sys.executable,"-S",str(estimator),"--inputs",str(root/"baseline_inputs"/track),"--patch-id",patch["patch_id"],
                 "--output",str(work/"result"),"--denial-probe",str(root/"evaluator_only/manifest.json")]
            result=subprocess.run(cmd,cwd=work,env=env,text=True,capture_output=True)
            (work/"stdout.txt").write_text(result.stdout);(work/"stderr.txt").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError("Independent estimator failed for "+track+"/"+patch["patch_id"]+": "+result.stderr[-3000:])
            receipt=read(work/"result/receipt.json")
            if not receipt["forbidden_path_probe_denied"] or not receipt["all_allowlisted_input_hashes_verified"]:
                raise ValueError("Estimator I/O audit failed")
            if not receipt["Python_no_site_startup"] or len(receipt["denied_open_attempts"])!=1:
                raise ValueError("Unexpected startup mode or non-allowlisted access attempt")
            receipts.append(dict(track=track,patch_id=patch["patch_id"],path=str(work/"result/receipt.json"),
                receipt_sha256=sha(work/"result/receipt.json"),prediction_sha256=receipt["prediction_sha256"],
                RGB_supported_cells=receipt["RGB_supported_cells"],height_supported_cells=receipt["height_supported_cells"],target_cells=receipt["target_cells"],seconds=receipt["seconds"],
                actual_input_reads=sum(x["kind"]=="allowlisted_input_read" for x in receipt["actual_open_events"])))
            print(result.stdout.strip(),flush=True)
    predictions={}
    for track in tracks:
        arrays={k:[] for k in ("keys","rgb_srgb","height_local_m","RGB_supported","height_supported","normal_local")}
        for patch in patches:
            path=run/track/patch["patch_id"]/"result/prediction.npz"
            with np.load(path,allow_pickle=False) as source_data:
                for key in arrays:arrays[key].append(source_data[key])
        path=run/(track+"_prediction.npz")
        np.savez_compressed(path,**{k:np.concatenate(v) for k,v in arrays.items()})
        predictions[track]=dict(prediction=str(path),prediction_sha256=sha(path),
                               input_manifest_sha256=sha(root/"baseline_inputs"/track/"input_manifest.json"))
    bank=dict(schema="scene008.R1.fixed_baseline_bank.v1",method="context_robust_plane_nearest_boundary_median",settings_sha256=settings_hash,
              estimator_sha256=sha(estimator),candidate_seeds=[42],candidate_count_per_condition=1,tracks=predictions,
              all_predictions_frozen_before_any_GT_evaluation=True,parameter_search=False,reference_or_evaluator_imported_by_estimator=False,
              independent_process_count=len(receipts),patch_process_receipts=receipts,seconds=time.time()-start,
              OS_strong_isolation_certified=False,GPU_used=False,neural_updates=0)
    write(run/"frozen_bank_receipt.json",bank)
    for track,info in predictions.items():
        write(run/(track+"_frozen_candidate.json"),dict(schema="scene008.R1.frozen_candidate.v1",track=track,**info,
              settings_sha256=settings_hash,candidate_seeds=[42],frozen_bank_sha256=sha(run/"frozen_bank_receipt.json"),
              fixed_before_any_GT_evaluation=True))
    print(json.dumps(dict(status="ALL12_INDEPENDENT_ESTIMATORS_FROZEN_NOT_YET_SCORED",seconds=time.time()-start,run=str(run))),flush=True)


if __name__=="__main__":main()
