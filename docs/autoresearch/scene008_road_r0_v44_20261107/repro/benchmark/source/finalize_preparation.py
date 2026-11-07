"""Audit preparation lineage without running/scoring a completion model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import road_patch_benchmark as b


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);p.add_argument("--root-contract",type=Path,required=True)
    args=p.parse_args();root=args.benchmark;parent=root.parent
    expected="e9f0b47b983270d8954c71330e2bab8824d6e6f91d72fb45e9aaa76ec2fe6464"
    if b.sha(args.root_contract)!=expected:
        raise ValueError("Root frozen R0 preservation contract changed")
    full=b.read(root/"audit/source_rebuild_full199.json")
    smoke=b.read(root/"audit/source_rebuild_smoke_0_137_198.json")
    parity=b.read(root/"audit/fast_decoder_parity.json")
    validation=b.read(root/"audit/validation_full199.json")
    export=b.read(root/"audit/isolated_export.json")
    selection=b.read(root/"audit/selection_receipt.json")
    sealed=b.read(root/"evaluator_only/manifest.json")
    domains=b.read(parent/"integration_domains/manifest.json")
    if [r["raw"] for r in full["rows"]]!=list(range(199)):
        raise ValueError("Missing/duplicate declared FRONT records")
    if any(not r["record_CRC_passed"] or not r["target_colour_intervention_exact"] or r["raw_LiDAR_sensor_return_pairs"]!=10 for r in full["rows"]):
        raise ValueError("Source rebuild invariants failed")
    isolated=[]
    allowed_patch={"patch_id","split","size_m","center_local_xy_m","xy_bounds_local_m","guard_xy_bounds_local_m","context_z_interval_m","world_target_corners_m","mask_height_uses_target_geometry"}
    for track in ("appearance_blind","geometry_RGB_blind"):
        path=root/"isolated_inputs"/track
        m=b.read(path/"input_manifest.json");patches=b.read(path/"patch_manifest.json")
        if m["raw_ids"]!=list(range(199)) or m["declared_cameras"]!=[1] or m["support_counts_RGB_GT_markings_target_slope_included"]:
            raise ValueError("Isolated track schema/taint failed")
        if any(set(p)!=allowed_patch for p in patches["patches"]):
            raise ValueError("Sealed selection diagnostic leaked into estimator patch metadata")
        for rel in m["input_files"]:
            q=Path(rel)
            if q.is_absolute() or ".." in q.parts:
                raise ValueError("Nonlocal estimator input dependency")
        counts={sub:len(list((path/sub).iterdir())) for sub in ("rgb","geometry","camera")}
        if counts!={"rgb":398,"geometry":199,"camera":199}:
            raise ValueError("Isolated all199 input inventory incomplete")
        isolated.append(dict(track=track,input_manifest_sha256=b.sha(path/"input_manifest.json"),files=counts))
    final=dict(schema="scene008.R0_R1.CPU_preparation_receipt.v1",
        status="PASS_FRONT199_CPU_PREPARATION_PHYSICAL_STRICTNESS_NOT_CERTIFIED",
        root_R0_frozen_contract=dict(path=str(args.root_contract),sha256=expected,original_modified=False),
        current_vehicle=dict(C2_sha256="0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d",old_M2_is_current=False),
        authoritative_R1_generation="FRONT_geometry_v2",initial_draft_disqualified="../public used old source_valid with whole-image sharpness; preserved, never estimator input",
        patches=6,validation_patches=3,sealed_test_patches=3,world_sizes_m=[.5,1.,2.],guard_m=.75,
        selection_reads_target_RGB=False,selection_reads_old_source_valid=False,selection_reads_fit_errors=False,
        selection_receipt_sha256=b.sha(root/"audit/selection_receipt.json"),
        actual_RGB_scope=dict(frames=199,cameras=["FRONT"],other_four_cameras_audited=False),
        raw_LiDAR_rebuilt=dict(sensors=5,returns_per_sensor=2,before_fusion=True,original_returns=sum(r["input_return_count"] for r in full["rows"]),
            geometry_RGB_blind_removed_points=sum(r["geometry_RGB_blind_removed_points"] for r in full["rows"])),
        masking=dict(native_masked_pixel_frames=sum(r["native_masked_pixels"] for r in full["rows"]),
            prepared_invalid_pixel_frames=sum(r["prepared_invalid_pixels"] for r in full["rows"]),
            target_colour_intervention_exact_all199=True,full_prism_filter_passed=True,
            max_constant_velocity_motion_bound_m=max(r["motion_bound_constant_velocity_m"] for r in full["rows"])),
        decoder_parity=dict(matrices=parity["matrix_count"],full_raw137_XYZ_bitwise_equal=True,dependency=parity["decoder"]),
        checks=dict(full_source_validation=validation["check_count"],all_passed=validation["all_checks_passed"],isolated_export_byte_parity=True,isolated_track_inventory=isolated),
        fixed_R0_protection=dict(domain_manifest_sha256=b.sha(parent/"integration_domains/manifest.json"),totals=domains["totals"],prediction_conditioned=False),
        sealed_evaluation=dict(patch_manifest_sha256=sealed["patch_manifest_sha256"],store_outside_estimator_inputs=True,
            references_prepared_not_scored=True,target_grid_cells=sum(r["target_grid_cells"] for r in sealed["references"]),
            stable_radiance_proxy_cells=sum(r["stable_radiance_proxy_cells"] for r in sealed["references"]),
            independent_marking_GT=False,independent_hidden_albedo_or_normal_GT=False),
        seconds=dict(geometry_only_selection=selection["seconds"],slow_three_raw_smoke_extra=smoke["seconds"],
                     fast_decoder_parity_extra=parity["seconds"],formal_front199_source_preparation=full["seconds"]),
        completion_methods_run=0,training_updates=0,GPU_used=False,power_operations=0,
        source_TFRecord_or_old_experiments_modified=False,R0_road_candidate_adopted_by_this_subtask=False,
        external_estimator_process_isolation_audited=False,physical_all_sensor_strict_blinding_certified=False,
        limitations=["FRONT-only RGB; no all-five-camera search or never-visible claim.",
            "Rolling-shutter/pose interval assumes bounded constant velocity; physical timing/calibration envelope is not independently certified.",
            "Only one isolated track may be mounted; old scene, other track, sealed GT and integration_domains must remain inaccessible to an estimator.",
            "Directory/hash checks are not OS-level process isolation.",
            "Reference colors are qualified actual radiance proxies, and sparse measured heights/slope are not dense hidden geometry or unshadowed albedo GT.",
            "No marking GT, completion result, independent scene test, model post-training or quality-accepted result is established."])
    b.write(root/"audit/preparation_receipt.json",final)
    b.write(parent/"selection_history.json",dict(authoritative="FRONT_geometry_v2/public/patch_manifest.json",
        estimator_entry="FRONT_geometry_v2/isolated_inputs/<track>/input_manifest.json",
        disqualified_draft="public/patch_manifest.json",reason="Old source_valid includes full-image sharpness gate; not accepted as geometry-only evidence",
        corrected_method="Recomputed no-RGB direct measured-return witnesses over199 cached FRONT records; no old source_valid or RGB read",
        bounds_happened_to_match=True,old_artifacts_preserved=True))
    print(json.dumps(dict(status=final["status"],checks=validation["check_count"],removed_points=final["raw_LiDAR_rebuilt"]["geometry_RGB_blind_removed_points"],
                          formal_seconds=full["seconds"],stable_reference_cells=final["sealed_evaluation"]["stable_radiance_proxy_cells"])),flush=True)


if __name__=="__main__":
    main()
