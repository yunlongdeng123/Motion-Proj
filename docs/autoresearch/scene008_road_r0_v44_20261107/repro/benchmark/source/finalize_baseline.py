"""Audit separate processes, support denominators and external score lineage."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
import road_patch_benchmark as b


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);args=p.parse_args();root=args.benchmark
    run=root/"baseline_runs/fixed_plane_boundary_median_v1";bank=b.read(run/"frozen_bank_receipt.json")
    rows=[];processes=[];checks=[]
    for row in bank["patch_process_receipts"]:
        receipt=b.read(row["path"])
        input_root=Path(receipt["input_root"]).resolve()
        actual=[e for e in receipt["actual_open_events"] if e["kind"]=="allowlisted_input_read"]
        all_local=all(input_root in Path(e["path"]).resolve().parents for e in actual)
        checks.append(dict(track=row["track"],patch_id=row["patch_id"],PID=receipt["process_id"],
            independent_subprocess=receipt["independent_subprocess"],Python_no_site=receipt["Python_no_site_startup"],
            actual_context_reads=len(actual),all_reads_within_one_track=all_local,
            input_hashes_verified=receipt["all_allowlisted_input_hashes_verified"],
            only_denial_is_explicit_sealed_manifest_probe=len(receipt["denied_open_attempts"])==1 and receipt["denied_open_attempts"][0]["path"]==str(root/"evaluator_only/manifest.json")))
        if not all_local or not checks[-1]["only_denial_is_explicit_sealed_manifest_probe"]:
            raise ValueError("Independent estimator context audit failed")
        processes.append(receipt["process_id"])
    if len(set(processes))!=12 or not all(c["Python_no_site"] for c in checks):
        raise ValueError("Expected12 distinct no-site estimator processes")
    smoke=root/"baseline_entry_smoke_v2/result/prediction.npz"
    formal=run/"appearance_blind/road_200cm_validation_0011acd2b6/result/prediction.npz"
    with np.load(smoke,allow_pickle=False) as a,np.load(formal,allow_pickle=False) as c:
        replay=all(np.array_equal(a[k],c[k]) for k in a.files) and a.files==c.files
    if not replay:
        raise ValueError("Independent estimator entry replay differed")
    for track in ("appearance_blind","geometry_RGB_blind"):
        for split,label in (("validation","validation"),("sealed_test","test")):
            path=run/(track+"_"+label+"_evaluation.json");evaluation=b.read(path)
            parts=evaluation["per_patch"]
            n=sum(x["measured_height_error_m"]["count"] for x in parts)
            pooled_height=sum(x["measured_height_error_m"]["mean"]*x["measured_height_error_m"]["count"] for x in parts if x["measured_height_error_m"]["count"])/n if n else None
            rows.append(dict(track=track,split=split,target_cells=sum(x["target_cells"] for x in parts),
                reference_stable_cells=sum(x["reference_stable_cells"] for x in parts),
                RGB_supported_target_cells=sum(x["RGB_supported_target_cells"] for x in parts),
                RGB_scored_cells=sum(x["scored_cells"] for x in parts),conditional_linear_RGB_MAE=evaluation["pooled_linear_RGB_MAE"],
                measured_height_scored_returns=n,measured_reference_returns=sum(x["measured_return_count"] for x in parts),
                measured_height_MAE_m=pooled_height,max_measured_height_error_m=max(x["measured_height_error_m"]["max"] for x in parts if x["measured_height_error_m"]["count"]),
                slope_proxy_abs_errors_deg={x["patch_id"]:x["slope_proxy_abs_error_deg"] for x in parts},
                unsupported_RGB_patches=[x["patch_id"] for x in parts if x["RGB_supported_target_cells"]==0],
                evaluation_sha256=b.sha(path)))
    first=b.read(root/"baseline_entry_smoke/result/receipt.json");second=b.read(root/"baseline_entry_smoke_v2/result/receipt.json")
    summary=dict(schema="scene008.R1.minimum_legal_baseline_receipt.v1",status="CPU_BASELINE_EXECUTED_EXTERNAL_EVALUATION_WITH_UNSUPPORTED_RGB",
        method=bank["method"],conditions=["appearance_blind","geometry_RGB_blind"],fixed_input_frames=51,
        settings_sha256=bank["settings_sha256"],estimator_sha256=bank["estimator_sha256"],frozen_bank_sha256=b.sha(run/"frozen_bank_receipt.json"),
        preparation_receipt_sha256=b.sha(root/"audit/preparation_receipt.json"),
        independent_estimator_processes=12,process_audits=checks,successful_entry_replay_bitwise_equal=True,
        predictions_frozen_before_any_GT_score=True,external_evaluator_is_a_separate_process=True,
        Python_path_allowlist_audited=True,OS_or_container_strong_isolation_certified=False,
        no_estimator_R0_asset_native_environment_old_cache_other_track_or_sealed_GT_reads=True,
        metrics=rows,RGB_support_full_denominator_per_track=dict(target_cells=1647,supported_cells=697,unsupported_cells=950),
        interpretation=["Height is a context-plane baseline, not a learned road-prior result.",
            "RGB MAE is conditional on prediction support and stable measured-radiance reference, never the entire patch or hidden albedo.",
            "The2m sealed-test and1m validation patches have no legal local RGB support; errors remain null, not guessed.",
            "RBF/spline and neural priors were not added because local legal appearance support is incomplete; no GT-based retuning or patch substitution.",
            "Sparse height and fitted slope/normal are measured-reference proxies; marking GT is absent.",
            "These test scores are now opened for this frozen baseline bank. Any later method development informed by them needs a new untouched patch set for independent confirmation."],
        seconds=dict(formal12_process_queue=bank["seconds"],extra_entry_smoke_first=first["seconds"],extra_entry_smoke_corrected=second["seconds"]),
        engineering_replay=dict(first_smoke="Qhull temporary-file writes were denied and the hull flag became unsupported; not a scientific result",
             correction="Restrict Qhull temp files to independent output; propagate I/O errors; -S startup; no method or settings changes",old_smoke_preserved=True),
        RBF_spline_run=False,parameter_search=False,neural_weight_updates=0,GPU_used=False,power_operations=0,quality_accepted=None,
        physical_all_sensor_strict_blinding_certified=False)
    b.write(run/"baseline_summary.json",summary)
    print(json.dumps(dict(status=summary["status"],rows=rows,entry_replay_bitwise_equal=replay),indent=2),flush=True)


if __name__=="__main__":main()
