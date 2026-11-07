"""Separate, once-frozen evaluator for world-grid completion predictions.

Prediction schema: keys int(N,2), rgb_srgb float(N,3), height_local_m float(N).
This evaluator is never an estimator dependency and exposes no sealed targets.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
import road_patch_benchmark as b
from seal_patch_references import linear


def stats(x):
    x=np.asarray(x,float)
    return dict(count=len(x),mean=float(x.mean()) if len(x) else None,
                p90=float(np.quantile(x,.9)) if len(x) else None,
                p99=float(np.quantile(x,.99)) if len(x) else None,max=float(x.max()) if len(x) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True)
    p.add_argument("--prediction",type=Path,required=True);p.add_argument("--frozen-candidate-receipt",type=Path,required=True)
    p.add_argument("--split",choices=("validation","sealed_test"),required=True);p.add_argument("--report",type=Path,required=True)
    args=p.parse_args();root=args.benchmark;frozen=b.read(args.frozen_candidate_receipt)
    digest=b.sha(args.prediction)
    if frozen.get("prediction_sha256")!=digest or not frozen.get("settings_sha256") or "candidate_seeds" not in frozen:
        raise ValueError("Freeze exact prediction/settings/seeds before external evaluation")
    manifest=b.read(root/"public/patch_manifest.json");sealed=b.read(root/"evaluator_only/manifest.json")
    if sealed["patch_manifest_sha256"]!=b.sha(root/"public/patch_manifest.json"):
        raise ValueError("Patch/split manifest changed since sealing")
    consumed=root/"evaluator_only/sealed_test_consumption.json"
    if args.split=="sealed_test":
        if consumed.exists():
            if b.read(consumed)["prediction_sha256"]!=digest:
                raise ValueError("Test GT was already opened for another candidate; no best-of-K on sealed test")
        else:
            b.write(consumed,dict(prediction_sha256=digest,candidate_receipt_sha256=b.sha(args.frozen_candidate_receipt),split="sealed_test"))
    with np.load(args.prediction,allow_pickle=False) as source:
        keys=source["keys"];rgb=source["rgb_srgb"];height=source["height_local_m"]
    if keys.ndim!=2 or keys.shape[1]!=2 or rgb.shape!=(len(keys),3) or height.shape!=(len(keys),):
        raise ValueError("Prediction schema mismatch")
    if not np.isfinite(rgb).all() or not np.isfinite(height).all() or ((rgb<0)|(rgb>1)).any():
        raise ValueError("Nonfinite/out-of-range predictions")
    lookup={tuple(k):i for i,k in enumerate(keys.tolist())}
    if len(lookup)!=len(keys):
        raise ValueError("Duplicate predicted world keys")
    rows=[]
    for patch in manifest["patches"]:
        if patch["split"]!=args.split:
            continue
        with np.load(root/"evaluator_only"/(patch["patch_id"]+".npz"),allow_pickle=False) as source:
            gtkeys=source["keys"];valid=source["stable_radiance_proxy"]
            ids=np.array([lookup.get(tuple(k),-1) for k in gtkeys.tolist()]);available=ids>=0
            score=valid&available
            colour=np.abs(linear(rgb[ids[score]])-source["median_linear_radiance"][score]).mean(1)
            srgb_target=np.where(source["median_linear_radiance"]<=.0031308,source["median_linear_radiance"]*12.92,1.055*np.maximum(source["median_linear_radiance"],0)**(1/2.4)-.055)
            mse=float(np.square(rgb[ids[score]]-srgb_target[score]).mean()) if score.any() else None
            points=source["measured_local_xyz"]
            pointkeys=np.floor(points[:,:2]/.08).astype(int)
            pointids=np.array([lookup.get(tuple(k),-1) for k in pointkeys.tolist()]);pv=pointids>=0
            zerror=np.abs(height[pointids[pv]]-points[pv,2])
            rows.append(dict(patch_id=patch["patch_id"],target_cells=len(gtkeys),reference_stable_cells=int(valid.sum()),
                predicted_target_cells=int(available.sum()),missing_target_cells=int((~available).sum()),scored_cells=int(score.sum()),
                linear_RGB_MAE=stats(colour),sRGB_PSNR=None if mse is None else float(-10*np.log10(max(mse,1e-12))),
                measured_height_error_m=stats(zerror),measured_return_count=len(points),unpredicted_measured_returns=int((~pv).sum()),
                marking_accuracy=None,independent_hidden_GT_claim=False))
    means=[r["linear_RGB_MAE"]["mean"] for r in rows if r["linear_RGB_MAE"]["count"]]
    scored=sum(r["scored_cells"] for r in rows)
    pooled=sum(r["linear_RGB_MAE"]["mean"]*r["scored_cells"] for r in rows if r["scored_cells"])/scored if scored else None
    b.write(args.report,dict(schema="scene008.R1.external_evaluation.v1",split=args.split,prediction_sha256=digest,
          frozen_candidate_receipt_sha256=b.sha(args.frozen_candidate_receipt),patch_manifest_sha256=b.sha(root/"public/patch_manifest.json"),
          per_patch=rows,macro_linear_RGB_MAE=float(np.mean(means)) if means else None,pooled_linear_RGB_MAE=pooled,
          quality_accepted=None,uses_same_scene_radiance_proxy=True,marking_ground_truth_available=False,
          estimator_process_input_isolation_audited=False,no_independent_scene_generalization_claim=True))
    print(json.dumps(dict(status="EXTERNAL_EVALUATION_WRITTEN",split=args.split,patches=len(rows),report=str(args.report))),flush=True)


if __name__=="__main__":
    main()
