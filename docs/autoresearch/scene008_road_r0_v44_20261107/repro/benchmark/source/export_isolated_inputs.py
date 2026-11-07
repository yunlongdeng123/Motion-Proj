"""Copy a single clean track with no sealed diagnostics or sibling data.

Selection support counts and reference labels are deliberately not exported.
The final directory is still subject to actual process/mount isolation audit.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
import road_patch_benchmark as b


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args();root=args.output
    full=b.read(root/"audit/source_rebuild_full199.json")
    if not full["complete_FRONT199"] or full["all_five_RGB_cameras_audited"]:
        raise ValueError("Expected explicitly FRONT-only all199 preparation")
    original=b.read(root/"public/patch_manifest.json")
    patch_keys=("patch_id","split","size_m","center_local_xy_m","xy_bounds_local_m","guard_xy_bounds_local_m","context_z_interval_m","world_target_corners_m","mask_height_uses_target_geometry")
    clean=dict(schema="scene008.R1.estimator_patch_inputs.v1",coordinate_anchor=original["coordinate_anchor"],
               coordinate_system=original["coordinate_system"],guard_m=original["guard_m"],
               declared_cameras=[1],raw_ids=list(range(199)),world_grid_cell_m=.08,
               patches=[{k:p[k] for k in patch_keys} for p in original["patches"]],
               frozen_selection_manifest_sha256=b.sha(root/"public/patch_manifest.json"),
               evaluator_support_counts_or_labels_included=False,strict_all_sensor_blinding_certified=False)
    exports=[]
    for track in ("appearance_blind","geometry_RGB_blind"):
        source=root/"public/inputs/full199"/track;dest=root/"isolated_inputs"/track
        if dest.exists():
            raise FileExistsError("Keep prior isolated export immutable")
        dest.mkdir(parents=True)
        source_manifest=b.read(source/"input_manifest.json")
        for sub in ("rgb","geometry","camera"):
            (dest/sub).mkdir()
            for file in sorted((source/sub).iterdir()):
                if file.is_symlink() or not file.is_file():
                    raise ValueError("No links, only explicit known input files")
                rel=str(file.relative_to(source))
                expected=source_manifest["input_files"][rel]
                if b.sha(file)!=expected["sha256"]:
                    raise ValueError("Source input changed before isolation export")
                shutil.copyfile(file,dest/sub/file.name)
                if b.sha(dest/sub/file.name)!=expected["sha256"]:
                    raise ValueError("Export byte parity failed")
        b.write(dest/"patch_manifest.json",clean)
        files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=b.sha(p)) for p in sorted(dest.rglob("*")) if p.is_file()}
        input_manifest=dict(schema="scene008.R1.isolated_input_track.v2",track=track,declared_cameras=[1],raw_ids=list(range(199)),
                            input_files=files,only_this_directory_may_be_mounted=True,
                            support_counts_RGB_GT_markings_target_slope_included=False,
                            other_track_evaluator_original_scene_access_allowed=False,
                            physical_all_sensor_strict_blinding_certified=False,
                            estimator_process_access_audited=False)
        b.write(dest/"input_manifest.json",input_manifest)
        exports.append(dict(track=track,directory=str(dest),input_manifest_sha256=b.sha(dest/"input_manifest.json"),
                            files=len(files),bytes=sum(v["bytes"] for v in files.values()),
                            source_export_byte_parity=True))
    b.write(root/"audit/isolated_export.json",dict(status="PASS_FRONT199_SELF_CONTAINED_TRACK_EXPORT",tracks=exports,
             source_rebuild_receipt_sha256=b.sha(root/"audit/source_rebuild_full199.json"),
             sealed_target_support_counts_and_labels_removed=True,old_checkpoint_or_cache_conditioning_allowed=False,
             actual_process_filesystem_isolation_audited=False,physical_strictness_certified=False,GPU_used=False))
    print({"status":"PASS_FRONT199_SELF_CONTAINED_TRACK_EXPORT","tracks":len(exports)})


if __name__=="__main__":
    main()
