"""CPU-only current49 real-observation material control on immutable V2 geometry.

One pre-existing q75-cluster/median rule, without threshold search, rendering,
calibration, hidden GT, or neighbour colour filling. Pilot render outputs are
read only to describe diagnostic cohorts and never select or fit material.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.spatial import cKDTree

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
RUN = CAM / "scene008_road_r0_v44_20261107_r1"
EXPECTED_V2_SHA = "6112be03108b5a9b1a6deb2f605ef850dcd8ba5018346fa822993177815807a2"
COLOUR_KEYS = ("reference_linear", "reference_srgb", "rgb")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            h.update(block)
    return h.hexdigest()


def ahash(a):
    a = np.asarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)


def srgb(x):
    return np.where(x <= .0031308, x*12.92, 1.055*np.maximum(x, 1e-10)**(1/2.4)-.055)


def describe(x):
    x = np.asarray(x, np.float64)
    return dict(count=int(x.size), mean=float(x.mean()) if x.size else None,
        p50=float(np.quantile(x, .5)) if x.size else None,
        p95=float(np.quantile(x, .95)) if x.size else None,
        maximum=float(x.max()) if x.size else None)


def colour_delta(x):
    x = np.asarray(x)
    return dict(records=len(x), signed_mean_RGB=x.mean(0).tolist() if len(x) else None,
        absolute_RGB_MAE=float(np.abs(x).mean()) if len(x) else None,
        abs_channel_p95=float(np.quantile(np.abs(x), .95)) if len(x) else None)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def current_cluster(a):
    raws = a["source_raw"].astype(int)
    subset = (raws >= 89) & (raws <= 137)
    assert subset.sum() == 49 and np.array_equal(raws[subset], np.arange(89, 138))
    valid = a["source_valid"][subset].astype(bool)
    rgb = a["source_rgb"][subset]
    assert np.isfinite(rgb[valid]).all(), "Qualified source contains NaN/inf"
    # Invalid/unknown NaNs cannot enter a statistic, even as zero evidence.
    observed = linear(np.where(valid[:, :, None], rgb, 0)).astype(np.float32)
    luma = observed @ np.array([.2126, .7152, .0722], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q75 = np.nanquantile(np.where(valid, luma, np.nan), .75, axis=0)
        upper = valid & (luma >= q75[None]-.025) & (luma <= q75[None]+.06)
        provisional = np.nanmedian(np.where(upper[:, :, None], observed, np.nan), axis=0)
    finite_reference = np.isfinite(provisional).all(1)
    provisional = np.nan_to_num(provisional, nan=0).astype(np.float32)
    residual = np.abs(observed-provisional[None]).max(2)
    agree = valid & (residual < .035)
    count = agree.sum(0).astype(np.int16)
    first = np.where(agree, raws[subset, None], 999).min(0).astype(np.int16)
    last = np.where(agree, raws[subset, None], -1).max(0).astype(np.int16)
    span = np.where(count > 0, last-first, 0).astype(np.int16)
    finite_range = finite_reference & (provisional.min(1) > .012) & (provisional.max(1) < .88)
    stable = (count >= 3) & (span >= 6) & finite_range
    best = raws[subset][np.argmin(np.where(agree, residual, np.inf), axis=0)].astype(np.int16)
    best[~stable] = -1
    return dict(raws=raws[subset], valid=valid, rgb=rgb, upper=upper, agree=agree,
        reference_linear=provisional, reference_srgb=srgb(provisional).astype(np.float32),
        stable=stable, count=count, span=span, best_source_raw=best,
        valid_count=valid.sum(0).astype(np.int16))


def cohort(a, c, mask, new):
    old = a["reference_srgb"]
    valid = c["valid"] & mask[None]
    n = c["valid_count"][mask]
    selected_delta = new[mask]-old[mask]
    return dict(cells=int(mask.sum()), qualified_current49_records=int(valid.sum()),
        current49_source_count=dict(zero=int((n==0).sum()), one=int((n==1).sum()),
            two=int((n==2).sum()), ge3=int((n>=3).sum())),
        current49_stable_cells=int((mask & c["stable"]).sum()),
        old199_minus_current49_observed=colour_delta(np.broadcast_to(old,c["rgb"].shape)[valid]-c["rgb"][valid]),
        candidate_minus_current49_observed=colour_delta(np.broadcast_to(new,c["rgb"].shape)[valid]-c["rgb"][valid]),
        candidate_minus_old199=colour_delta(selected_delta))


def pilot_cohorts(a, c, new, pilot):
    """Nearest-cell projection is diagnostic only, not a simulated rasterizer."""
    result = []
    active_front = a["active"].astype(bool) & (a["local_xyz"][:,0] > 19.75)
    for raw in (105,113,130,137):
        obs = load(CAM/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        extension = load(pilot/f"{raw:03d}/extension_delete_float.npz")
        baseline = load(pilot/f"{raw:03d}/baseline_delete_float.npz")
        meta = json.loads((CAM/f"scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44/{raw:03d}/metrics.json").read_text())["actual_camera_pose"]
        target = obs["rgb"].astype(np.float32).transpose(2,0,1)
        if target.max() > 1: target /= 255
        bad = (extension["extension_alpha_T"].squeeze() > .9) & obs["road"].astype(bool)
        yy,xx = np.nonzero(bad)
        homogeneous = np.c_[a["model_xyz"], np.ones(len(a["keys"]))]
        projected = homogeneous @ np.asarray(meta["full_projection"])
        uv = (projected[:,:2]/projected[:,3:4]+1)*np.array([480,320])/2-.5
        usable = active_front & np.isfinite(uv).all(1) & (projected[:,3] > 0)
        ids = np.flatnonzero(usable)
        distance,index = cKDTree(uv[usable]).query(np.c_[xx,yy])
        chosen = ids[index]
        rawidx = int(np.flatnonzero(c["raws"] == raw)[0])
        valid = c["valid"][rawidx,chosen]
        unique = np.unique(chosen)
        near_colour = a["reference_srgb"][chosen]
        near_new = new[chosen]
        row = dict(raw=raw, pixels=int(bad.sum()),
            domain="Saved V2 extension alphaT>.9 intersect fixed original observed road; posthoc diagnostic only",
            GT_mean=target[:,bad].mean(1).tolist() if bad.any() else None,
            V2_render_mean=extension["rgb"][:,bad].mean(1).tolist() if bad.any() else None,
            V44_RGB_MAE=float(np.abs(baseline["rgb"][:,bad]-target[:,bad]).mean()) if bad.any() else None,
            V2_RGB_MAE=float(np.abs(extension["rgb"][:,bad]-target[:,bad]).mean()) if bad.any() else None,
            nearest_projected_cell_distance_px=describe(distance), nearest_cells_unique=len(unique),
            nearest_cell_current49_stable_pixel_fraction=float(c["stable"][chosen].mean()) if len(chosen) else None,
            nearest_cell_current_raw_qualified_fraction=float(valid.mean()) if len(chosen) else None,
            nearest_cell_old_reference_mean=near_colour.mean(0).tolist() if len(chosen) else None,
            nearest_cell_candidate_mean=near_new.mean(0).tolist() if len(chosen) else None,
            nearest_cell_candidate_delta=colour_delta(near_new-near_colour),
            nearest_cell_current_source_count=describe(c["valid_count"][chosen]),
            nearest_cell_current_raw_colour_mean=c["rgb"][rawidx,chosen[valid]].mean(0).tolist() if valid.any() else None,
            nearest_cell_current_raw_cache_minus_ref=colour_delta(c["rgb"][rawidx,chosen[valid]]-near_colour[valid]),
            nearest_cell_candidate_is_not_rendered_result=True)
        result.append(row)
        print(json.dumps(dict(pilot_material_cohort=row)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",type=Path,default=RUN/"surface/frontfill_v2/road_reference_199.npz")
    p.add_argument("--output",type=Path,default=RUN/"surface/material_clip49")
    p.add_argument("--pilot",type=Path,default=RUN/"results/pilot_v2")
    p.add_argument("--analyze-only",action="store_true")
    args=p.parse_args();started=time.time();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"analysis.json").exists() or (args.output/"road_reference_clip49.npz").exists():
        raise FileExistsError("Preserve prior material control; choose a fresh output")
    assert sha(args.asset)==EXPECTED_V2_SHA
    a=load(args.asset);original_hashes={k:ahash(v) for k,v in a.items()}
    c=current_cluster(a)
    front=a["active"].astype(bool)&(a["local_xyz"][:,0]>19.75)
    override=front & c["stable"]
    new=a["reference_srgb"].copy();new[override]=c["reference_srgb"][override]
    cohorts=dict(all_candidates=np.ones(len(front),bool),all_active=a["active"].astype(bool),
        active_front=front,front_current49_override=override,front_full199_fallback=front&~override)
    if "candidate_origin" in a:
        cohorts["active_front_added_v2"]=front&(a["candidate_origin"]==1)
    analysis=dict(schema="R0_CURRENT49_REAL_MATERIAL_ANALYSIS_V1",input_asset=str(args.asset),
        input_sha256=EXPECTED_V2_SHA,source_raws=c["raws"].tolist(),
        rule="Unchanged linear q75 luma cluster[-.025,+.06], channel median, maxRGB agreement<.035, >=3 agreeing raws, span>=6, linear channels(.012,.88)",
        RGB_encoding="source_rgb is encoded sRGB; exactly once sRGB->linear and inverse conversion",
        geometry_active_and_original_noncolour_arrays_unchanged=True,
        cohorts={name:cohort(a,c,mask,new) for name,mask in cohorts.items()},
        pilot_posthoc_cohorts=pilot_cohorts(a,c,new,args.pilot),
        limits=["Current49 are build inputs, not held-out validation.",
            "Nearest projected cells describe source support, not exact raster weights or a rendered candidate prediction.",
            "No hidden-road truth, no deleted/rendered/vehicle pixel used as a colour source.",
            "Full199 fallback stays exact where current49 lacks stable qualified witnesses."],
        CPU_only=True,GPU_used=False,torch_imported=False,colour_residual_fit=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/"analysis.json",analysis)
    print(json.dumps(dict(active_front=analysis["cohorts"]["active_front"])),flush=True)
    if args.analyze_only:return
    payload=dict(a)
    payload["reference_srgb"]=new;payload["rgb"]=a["rgb"].copy();payload["rgb"][override]=new[override]
    payload["reference_linear"]=a["reference_linear"].copy();payload["reference_linear"][override]=c["reference_linear"][override]
    payload.update(material49_override=override,material49_stable=c["stable"],
        material49_source_raw=c["raws"].astype(np.int16),material49_source_upper=c["upper"],
        material49_source_agree=c["agree"],material49_valid_count=c["valid_count"],
        material49_agree_count=c["count"],material49_temporal_span=c["span"],
        material49_best_source_raw=c["best_source_raw"],
        material_source_choice=np.where(override,1,0).astype(np.uint8))
    target=args.output/"road_reference_clip49.npz";np.savez_compressed(target,**payload)
    b=load(target)
    unchanged={k:ahash(b[k])==v for k,v in original_hashes.items() if k not in COLOUR_KEYS}
    assert all(unchanged.values())
    assert all(np.array_equal(b[k][~override],a[k][~override]) for k in COLOUR_KEYS)
    assert np.array_equal(b["source_valid"],a["source_valid"])
    assert (~c["upper"]|c["valid"]).all() and (~c["agree"]|c["valid"]).all()
    assert sha(args.asset)==EXPECTED_V2_SHA
    receipt=dict(schema="R0_CURRENT49_REAL_MATERIAL_CONTROL_V1",asset=str(target),asset_sha256=sha(target),
        input_asset=str(args.asset),input_sha256=EXPECTED_V2_SHA,source_script_sha256=sha(__file__),
        candidates=len(front),active=int(a["active"].sum()),active_front=int(front.sum()),
        overridden_front_cells=int(override.sum()),fallback_front_cells=int((front&~override).sum()),
        changed_original_arrays=list(COLOUR_KEYS),unchanged_original_array_hash_checks=unchanged,
        original_array_sha256=original_hashes,
        source_policy="Only immutable source_valid current89..137 witnesses; same fixed stable rule; exact full199 proxy fallback; only active x>19.75 cells can change material",
        provenance="material49_source_upper identifies the actual median source cohort; source_agree verifies stability; original source_agree/confidence remain the full199 provenance",
        known_current49_is_build_context=True,heldout_claim=False,learned_road_prior_claim=False,
        hidden_GT_used=False,render_RGB_used_to_fit=False,geometry_updates=0,active_mask_updates=0,
        optimizer_updates=0,colour_residual_fit=False,GPU_used=False,torch_imported=False,
        original_asset_unchanged=True,shutdown_performed=False,seconds=time.time()-started)
    write(args.output/"receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("asset","asset_sha256","active_front","overridden_front_cells","fallback_front_cells","seconds")}),flush=True)


if __name__=="__main__":main()
