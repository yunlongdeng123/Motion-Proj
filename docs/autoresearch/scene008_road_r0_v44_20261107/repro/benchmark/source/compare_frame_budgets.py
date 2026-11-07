"""External budget audit/report, never an estimator dependency.

Verification reads only condition input copies. Final comparison is explicitly
an external evaluator after both prediction banks are frozen; it may open the
already-scored sealed reference and cannot change model inputs or parameters.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
import road_patch_benchmark as b
from seal_patch_references import linear


OLD="fixed_plane_boundary_median_v1"
NEW="full199_plane_boundary_median_development_v1"


def verify(root):
    checks=[]
    for track in ("appearance_blind","geometry_RGB_blind"):
        old=root/"baseline_inputs"/track;new=root/"baseline_inputs/full199"/track
        om=b.read(old/"input_manifest.json");nm=b.read(new/"input_manifest.json")
        if om["raw_ids"]!=list(range(0,199,4))+[198] or nm["raw_ids"]!=list(range(199)):
            raise ValueError("Unexpected deterministic source budgets")
        if b.read(old/"patch_manifest.json")["patches"]!=b.read(new/"patch_manifest.json")["patches"]:
            raise ValueError("Patch, guard or allowed height interval changed")
        oldsettings=b.read(old/"baseline_settings.json");newsettings=b.read(new/"baseline_settings.json")
        metadata={"raw_budget","fixed_before_any_completion_evaluation","fixed_before_own_completion_evaluation","evaluation_role","budget_provenance","predecessor_settings_sha256"}
        if {k:v for k,v in oldsettings.items() if k not in metadata}!={k:v for k,v in newsettings.items() if k not in metadata}:
            raise ValueError("Method or threshold changed")
        for raw in om["raw_ids"]:
            for sub,suffix in (("geometry",".npz"),("camera",".json"),("rgb",".png"),("rgb","_valid.npz"),("guard",".npz")):
                rel=Path(sub)/(f"{raw:03d}"+suffix)
                exact=b.sha(old/rel)==b.sha(new/rel)
                if not exact and sub=="guard":
                    with np.load(old/rel,allow_pickle=False) as a,np.load(new/rel,allow_pickle=False) as c:
                        exact=a.files==c.files and all(np.array_equal(a[k],c[k]) for k in a.files)
                if not exact:raise ValueError("Overlapping source content changed: "+str(rel))
                checks.append(dict(track=track,raw=raw,part=str(rel),identical=True))
    return dict(schema="scene008.R1.frame_budget_only_control.v1",status="PASS_51_TO_199_INPUT_CONTENT_EQUIVALENCE",
        old_budget=51,new_budget=199,only_deterministic_frame_budget_added=True,method_thresholds_patches_guards_height_intervals_unchanged=True,
        overlapping_source_checks=len(checks),all_overlapping_sources_identical=True,checks=checks,
        budget51_provenance="Agent-selected stride4 minimum CPU-entry budget (0,4,...,196 plus198); not a user scientific contract or legal-context maximum",
        frame_selection_uses_opened_errors_or_RGB=False,new_frames=list(sorted(set(range(199))-set(list(range(0,199,4))+[198]))),
        test_role="Already opened by the fixed51 baseline; full199 is same-scene development control, not independent confirmation")


def prediction(path):
    with np.load(path,allow_pickle=False) as a:
        data={k:a[k].copy() for k in a.files}
    data["lookup"]={tuple(k):i for i,k in enumerate(data["keys"].tolist())}
    return data


def aggregate(e):
    p=e["per_patch"];n=sum(x["measured_height_error_m"]["count"] for x in p)
    return dict(target_cells=sum(x["target_cells"] for x in p),stable_reference_cells=sum(x["reference_stable_cells"] for x in p),
        RGB_supported_target_cells=sum(x["RGB_supported_target_cells"] for x in p),RGB_scored_cells=sum(x["scored_cells"] for x in p),
        conditional_linear_RGB_MAE=e["pooled_linear_RGB_MAE"],
        measured_height_scored_returns=n,measured_reference_returns=sum(x["measured_return_count"] for x in p),
        measured_height_MAE_m=sum(x["measured_height_error_m"]["mean"]*x["measured_height_error_m"]["count"] for x in p if x["measured_height_error_m"]["count"])/n if n else None,
        unsupported_RGB_patches=[x["patch_id"] for x in p if x["RGB_supported_target_cells"]==0],
        slope_proxy_abs_error_deg={x["patch_id"]:x["slope_proxy_abs_error_deg"] for x in p})


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);p.add_argument("--verify-only",action="store_true")
    args=p.parse_args();root=args.benchmark
    verification_path=root/"audit/full199_budget_extension_verification.json"
    if not verification_path.exists():b.write(verification_path,verify(root))
    if args.verify_only:
        print({"status":"PASS_51_TO_199_INPUT_CONTENT_EQUIVALENCE","checks":b.read(verification_path)["overlapping_source_checks"]});return
    oldrun=root/"baseline_runs"/OLD;newrun=root/"baseline_runs"/NEW
    oldbank=b.read(oldrun/"frozen_bank_receipt.json");newbank=b.read(newrun/"frozen_bank_receipt.json")
    if oldbank["estimator_sha256"]!=newbank["estimator_sha256"] or newbank["evaluation_role"]!="same_scene_development_control_after_51_test_open":
        raise ValueError("Estimator changed or development role missing")
    process_checks=[]
    for row in newbank["patch_process_receipts"]:
        r=b.read(row["path"]);inputroot=Path(r["input_root"]).resolve()
        actual=[x for x in r["actual_open_events"] if x["kind"]=="allowlisted_input_read"]
        passed=r["Python_no_site_startup"] and r["all_allowlisted_input_hashes_verified"] and len(r["denied_open_attempts"])==1 and all(inputroot in Path(x["path"]).resolve().parents for x in actual)
        if not passed:raise ValueError("Full199 independent estimator I/O audit failed")
        process_checks.append(dict(track=r["track"],patch_id=r["patch_id"],PID=r["process_id"],actual_input_reads=len(actual),passed=True))
    if len({x["PID"] for x in process_checks})!=12:raise ValueError("Expected12 new independent processes")
    comparisons=[]
    for track in ("appearance_blind","geometry_RGB_blind"):
        a=prediction(oldrun/(track+"_prediction.npz"));c=prediction(newrun/(track+"_prediction.npz"))
        for split,label in (("validation","validation"),("sealed_test","test")):
            oldevaluation=b.read(oldrun/(track+"_"+label+"_evaluation.json"));newevaluation=b.read(newrun/(track+"_"+label+"_evaluation.json"))
            common_old=[];common_new=[];newly=0;lost=0
            for patch in b.read(root/"public/patch_manifest.json")["patches"]:
                if patch["split"]!=split:continue
                with np.load(root/"evaluator_only"/(patch["patch_id"]+".npz"),allow_pickle=False) as ref:
                    for j,key in enumerate(ref["keys"].tolist()):
                        oldid=a["lookup"].get(tuple(key));newid=c["lookup"].get(tuple(key))
                        oldok=oldid is not None and bool(a["RGB_supported"][oldid]);newok=newid is not None and bool(c["RGB_supported"][newid])
                        newly+=int(newok and not oldok);lost+=int(oldok and not newok)
                        if oldok and newok and ref["stable_radiance_proxy"][j]:
                            target=ref["median_linear_radiance"][j]
                            common_old.append(float(np.abs(linear(a["rgb_srgb"][oldid])-target).mean()))
                            common_new.append(float(np.abs(linear(c["rgb_srgb"][newid])-target).mean()))
            comparisons.append(dict(track=track,split=split,budget51=aggregate(oldevaluation),budget199=aggregate(newevaluation),
                newly_RGB_supported_target_cells=newly,lost_RGB_supported_target_cells=lost,
                common_supported_stable_cells=len(common_old),common_denominator_RGB_MAE_51=float(np.mean(common_old)) if common_old else None,
                common_denominator_RGB_MAE_199=float(np.mean(common_new)) if common_new else None,
                conditional_MAE_denominators_may_differ=True))
    result=dict(schema="scene008.R1.public_budget_comparison.v1",status="FULL199_FIXED_METHOD_DEVELOPMENT_CONTROL_COMPLETE",
        budget51_provenance=b.read(verification_path)["budget51_provenance"],
        correction="Unsupported color in the51-frame run never established absence of legal color context across all199.",
        old_budget_contract_is_scientific_maximum=False,full199_source_selection="All raw0..198, deterministic, no source chosen by RGB or opened errors",
        method_thresholds_estimator_code_patch_guard_height_interval_unchanged=True,input_equivalence_receipt_sha256=b.sha(verification_path),
        estimator_sha256=newbank["estimator_sha256"],old_bank_sha256=b.sha(oldrun/"frozen_bank_receipt.json"),new_bank_sha256=b.sha(newrun/"frozen_bank_receipt.json"),
        independent_estimator_processes=12,process_I_O_audits=process_checks,OS_strong_isolation_certified=False,
        evaluation_role="same_scene_development_control_after_51_test_open",independent_confirmation=False,
        all_new_predictions_frozen_before_own_external_evaluation=True,old51_outputs_and_test_opening_records_preserved=True,
        comparisons=comparisons,CPU_seconds=dict(old51_queue=oldbank["seconds"],full199_queue=newbank["seconds"],full199_input_copy_and_guard_preparation=b.read(root/"audit/baseline_input_preparation_full199.json")["seconds"]),
        scope=dict(RGB="FRONT only",blinded_sources=199,other_RGB_cameras_audited=False,LiDAR_sensors=5,returns=2),
        unsupported_scope="Only this fixed local4m estimator, guards/prisms, source qualification and FRONT declaration; never proof of all-camera or universally absent context",
        physical_all_sensor_strict_blinding_certified=False,marking_GT_available=False,independent_normal_or_albedo_GT=False,
        new_scientific_assumptions_added=False,neural_updates=0,GPU_used=False,power_operations=0,
        next_claim_boundary="Any later learned-prior gain must use the full199 baseline and matching support denominators; future independent confirmation requires untouched patches.")
    b.write(newrun/"public_summary.json",result)
    print({"status":result["status"],"comparisons":comparisons,"CPU_seconds":result["CPU_seconds"]})


if __name__=="__main__":main()
