"""Evaluator-only reference preparation, after patch/split hashes are frozen.

Never imported by an estimator. No reference RGB values are printed or put in
public inputs. These measured-radiance references are not hidden-road albedo GT.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
import road_patch_benchmark as b


def linear(x):
    x=np.clip(x,0,1)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--geometry",type=Path,default=b.GEOMETRY)
    p.add_argument("--measured-pool",type=Path,required=True)
    args=p.parse_args();root=args.output
    manifest=b.read(root/"public/patch_manifest.json")
    frozen_hash=b.sha(root/"public/patch_manifest.json")
    destination=root/"evaluator_only"
    if destination.exists():
        raise FileExistsError("Keep sealed references immutable")
    destination.mkdir(parents=True)
    # Only this evaluator process ever opens the unblinded appearance arrays.
    with np.load(args.geometry,allow_pickle=False) as source:
        xyz=source["local_xyz"];keys=source["keys"];source_raw=source["source_raw"]
        valid=source["source_valid"].astype(bool);rgb=source["source_rgb"].astype(np.float32)
    with np.load(args.measured_pool,allow_pickle=False) as source:
        pool=source["local_xyz"]
        pool_raw=source["source_raw"] if "source_raw" in source.files else np.full(len(pool),-1,np.int16)
    rows=[]
    for patch in manifest["patches"]:
        keep=b.inside_xy(xyz,patch["xy_bounds_local_m"])
        qvalid=valid[:,keep];qrgb=rgb[:,keep];qxyz=xyz[keep];qkeys=keys[keep]
        qlinear=linear(qrgb)
        # This fixed reference rule is not a candidate score or selection gate.
        masked=np.where(qvalid[:,:,None],qlinear,np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            median=np.nanmedian(masked,axis=0)
            spread=np.nanquantile(np.abs(masked-median[None]),.90,axis=0).max(1)
        counts=qvalid.sum(0)
        first=np.where(qvalid,source_raw[:,None],999).min(0)
        last=np.where(qvalid,source_raw[:,None],-1).max(0)
        stable=(counts>=3)&(last-first>=6)&(spread<=.05)&np.isfinite(median).all(1)
        point_keep=b.inside_xy(pool,patch["xy_bounds_local_m"])
        points=pool[point_keep].astype(float)
        # Slope is a sealed geometric proxy, never a target label for selection.
        if len(points)>=3:
            design=np.c_[points[:,:2],np.ones(len(points))]
            plane=np.linalg.lstsq(design,points[:,2],rcond=None)[0]
            slope_deg=float(np.degrees(np.arctan(np.linalg.norm(plane[:2]))))
        else:
            plane=np.full(3,np.nan);slope_deg=None
        path=destination/(patch["patch_id"]+".npz")
        np.savez_compressed(path,keys=qkeys,reference_query_local_xyz=qxyz,source_raw=source_raw,
                            source_valid=qvalid,source_sRGB=qrgb,median_linear_radiance=np.nan_to_num(median,nan=0),
                            stable_radiance_proxy=stable,measured_local_xyz=points.astype(np.float32),
                            measured_source_raw=pool_raw[point_keep],marking_label_available=np.array(False))
        rows.append(dict(patch_id=patch["patch_id"],split=patch["split"],size_m=patch["size_m"],
                         target_grid_cells=int(keep.sum()),geometry_reference_returns=len(points),
                         RGB_ge3_witness_cells=int((counts>=3).sum()),stable_radiance_proxy_cells=int(stable.sum()),
                         qualified_cell_frame_records=int(qvalid.sum()),measured_slope_proxy_deg=slope_deg,
                         independent_normal_GT=False,marking_GT_available=False,
                         material_stratum="unannotated; no asphalt/marking coverage claim",
                         reference_sha256=b.sha(path)))
    b.write(destination/"manifest.json",dict(schema="scene008.R1.sealed_reference.v1",status="SEALED_EVALUATOR_REFERENCES_PREPARED_NOT_SCORED",
            patch_manifest_sha256=frozen_hash,selection_precedes_RGB_open=True,source_reference_sha256=b.sha(args.geometry),
            source_measured_pool_sha256=b.sha(args.measured_pool),references=rows,
            reference_is_independent_hidden_GT=False,radiance_proxy_is_unshadowed_albedo=False,
            stable_reference_rule="Median linear RGB; >=3 geometric FRONT witnesses; span>=6 raw; p90 max-channel absolute deviation<=.05. Fixed after geometry-only selection and before completion.",
            selection_or_estimator_access_allowed=False,OS_level_input_isolation_audited=False,
            limitations=["Single-scene FRONT-only reference.","World grid radiance is interpolated from actual qualified images, not dense physical texture truth.","Direct measured height is real sparse LiDAR; fitted slope/normal is a proxy.","No independent marking labels are available, so marking accuracy is unscored.","Unstable/unknown cells remain in denominators and are not invented GT."]))
    print(json.dumps(dict(status="SEALED_EVALUATOR_REFERENCES_PREPARED_NOT_SCORED",patches=len(rows),target_RGB_printed=False,split_or_target_bounds_changed=False)),flush=True)


if __name__=="__main__":
    main()
