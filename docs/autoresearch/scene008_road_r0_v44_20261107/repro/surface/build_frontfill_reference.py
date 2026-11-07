"""Build a measured-support road asset from immutable FRONT199 CPU caches.

This script imports no renderer/model/checkpoint or torch. Query heights are
interpolated from real pooled ground returns, not dense LiDAR measurements.
Only geometrically qualified, guarded, box-clear RGB samples enter statistics.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transform(x, matrix):
    return x @ matrix[:3, :3].T + matrix[:3, 3]


def linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def srgb(x):
    return np.where(x <= .0031308, x * 12.92, 1.055 * np.maximum(x, 1e-10) ** (1 / 2.4) - .055)


def count(x):
    return int(np.count_nonzero(x))


def bounds(x, selected=None):
    a = x if selected is None else x[selected]
    return [a.min(0).tolist(), a.max(0).tolist()] if len(a) else None


def near_ground(f, plane):
    """Exact original expanded-grid near-return rule; no new inference."""
    p = f["cloud"]
    good = ((np.abs(p @ plane[:3] + plane[3]) <= .15)
            & (p[:, 0] >= .5) & (p[:, 0] <= 90) & (np.abs(p[:, 1]) <= 8))
    for c, h, axes in zip(f["box_center"], f["box_half"], f["box_axes"]):
        ids = np.flatnonzero(good)
        good[ids] &= ~(np.abs((p[ids] - c) @ axes) <= h).all(1)
    return good & (p[:, 0] < 3)


def measured_pool(full, plane):
    chunks, raws, kinds, indices, frame_sources = [], [], [], [], []
    for raw in range(199):
        path = full / "frames" / f"{raw:03d}.npz"
        with np.load(path, allow_pickle=False) as f:
            near = np.flatnonzero(near_ground(f, plane))
            for kind, xyz, source_index in (
                (0, f["ground_local"], np.arange(len(f["ground_local"]), dtype=np.int32)),
                (1, f["cloud"][near], near.astype(np.int32)),
            ):
                chunks.append(xyz)
                raws.append(np.full(len(xyz), raw, np.int16))
                kinds.append(np.full(len(xyz), kind, np.uint8))
                indices.append(source_index)
            frame_sources.append(dict(raw=raw, path=str(path), sha256=sha256(path),
                                      ground_local_count=len(f["ground_local"]), near_extra_count=len(near)))
    points = np.concatenate(chunks)
    source_raw = np.concatenate(raws)
    source_kind = np.concatenate(kinds)
    source_index = np.concatenate(indices)
    _, first = np.unique(np.floor(points[:, :2] / .1).astype(int), axis=0, return_index=True)
    return (points[first], source_raw[first], source_kind[first], source_index[first],
            len(points), frame_sources)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    args = p.parse_args()
    started = time.time()
    base, out = args.base, args.output
    out.mkdir(parents=True, exist_ok=True)
    target = out / "road_reference_199.npz"
    if target.exists() or (out / "receipt.json").exists():
        raise FileExistsError("Preserve a built asset; choose a fresh output directory")
    paths = dict(
        evidence=args.evidence,
        legacy_domain=base / "shadow_a5_a37_199_20261103_r4/inputs/road_reference.npz",
        legacy_current=base / "scene008_native_road1k_20261105_r1/inputs/road_reference.npz",
        bundle=base / "r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json",
        extraction=base / "r15_gn11_fullclip_cpu_20261026/extraction.json",
    )
    sources_before = {k: dict(path=str(v), bytes=v.stat().st_size, sha256=sha256(v)) for k, v in paths.items()}
    meta = json.loads(paths["bundle"].read_text())
    anchor = np.asarray(meta["coordinate_anchor"], float)
    world_to_model = np.asarray(meta["world_to_model"], float)
    plane = np.asarray(json.loads(Path(meta["original_readiness"]).read_text())
                       ["SGCM_geometry"]["ground_fit"]["plane_anchor"], float)
    with np.load(paths["evidence"], allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    with np.load(paths["legacy_domain"], allow_pickle=False) as f:
        legacy = {k: f[k] for k in ("keys", "domain", "domain_a5", "domain_a37")}
    with np.load(paths["legacy_current"], allow_pickle=False) as f:
        current_xyz, current_active = f["local_xyz"], f["active"].astype(bool)
    original_keys = legacy["keys"]
    assert np.array_equal(a["keys"][:len(original_keys)], original_keys)
    for name in ("domain", "domain_a5", "domain_a37"):
        legacy[name] = np.r_[legacy[name], np.zeros(len(a["keys"])-len(original_keys), bool)]
    assert np.array_equal(a["source_raw"], np.arange(199))
    keys = a["keys"].astype(np.int64)
    xyz = a["local_xyz"].astype(np.float64)
    normals = a["normal_local"].astype(np.float64)
    cell = float(a["cell_m"])
    n = len(keys)
    assert n == 53082 and abs(cell - .08) < 1e-12
    assert np.max(np.abs((keys + .5) * cell - xyz[:, :2])) < 2e-6

    pool, pool_raw, pool_kind, pool_index, total_returns, frame_sources = measured_pool(
        base / "r15_gn11_fullclip_cpu_20261026", plane)
    distances, nearest = cKDTree(pool[:, :2]).query(xyz[:, :2], k=8, distance_upper_bound=.8)
    finite = np.isfinite(distances)
    safe = np.minimum(nearest, len(pool) - 1)
    support_ids = np.where(finite, nearest, -1).astype(np.int32)
    support_count = finite.sum(1).astype(np.uint8)
    residual = np.abs(((pool[safe].astype(float) - xyz[:, None]) * normals[:, None]).sum(2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        height_residual = np.nanmedian(np.where(finite, residual, np.nan), axis=1)
    geometry_valid = (a["surface_supported"].astype(bool) & (support_count >= 3)
                      & np.isfinite(xyz).all(1) & np.isfinite(normals).all(1)
                      & np.isfinite(height_residual) & (height_residual <= .3))
    geometry_confidence = (np.minimum(support_count / 8, 1)
                           * np.exp(-np.nan_to_num(height_residual, nan=100) / .15)
                           * geometry_valid).astype(np.float32)
    np.savez_compressed(out / "measured_ground_support_pool.npz", local_xyz=pool,
                        source_raw=pool_raw, source_kind=pool_kind, source_index=pool_index,
                        coordinate_anchor=anchor, world_to_model=world_to_model, dedup_cell_m=.1)

    # The source cache stores samples even where its geometry/guard gate failed.
    # Mask FIRST: such pixels must never influence reference statistics or delivery.
    valid = a["source_valid"].astype(bool)
    observed = linear(np.where(valid[:, :, None], a["source_rgb"], 0)).astype(np.float32)
    luma = observed @ np.array([.2126, .7152, .0722], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q75 = np.nanquantile(np.where(valid, luma, np.nan), .75, axis=0)
        upper = valid & (luma >= q75[None] - .025) & (luma <= q75[None] + .06)
        provisional = np.nanmedian(np.where(upper[:, :, None], observed, np.nan), axis=0)
    provisional = np.nan_to_num(provisional, nan=0).astype(np.float32)
    agree = valid & (np.abs(observed - provisional[None]).max(2) < .035)
    agree_count = agree.sum(0).astype(np.int16)
    valid_count = valid.sum(0).astype(np.int16)
    source_raw = a["source_raw"].astype(np.int16)
    first = np.where(agree, source_raw[:, None], 999).min(0).astype(np.int16)
    last = np.where(agree, source_raw[:, None], -1).max(0).astype(np.int16)
    first[agree_count == 0] = -1
    temporal_span = np.where(agree_count > 0, last - first, 0).astype(np.int16)
    colour_range_valid = (provisional.min(1) > .012) & (provisional.max(1) < .88)
    active = geometry_valid & (agree_count >= 3) & (temporal_span >= 6) & colour_range_valid
    colour_valid = active.copy()
    reference = np.where(active[:, None], provisional, 0).astype(np.float32)
    photometric_residual = np.abs(observed - provisional[None]).max(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agreement_residual = np.nanmedian(np.where(agree, photometric_residual, np.nan), axis=0)
    photo_confidence = (np.minimum(agree_count / 8, 1) * np.minimum(temporal_span / 12, 1)
                        * np.exp(-np.nan_to_num(agreement_residual, nan=100) / .035)
                        * active).astype(np.float32)
    best = np.argmin(np.where(agree, photometric_residual, np.inf), axis=0).astype(np.int16)
    best[~active] = -1
    # Retain accepted samples only; NaN means unavailable, never black evidence.
    accepted_rgb = np.where(valid[:, :, None], a["source_rgb"], np.nan).astype(np.float32)

    lo, hi = keys.min(0), keys.max(0)
    shape = hi - lo + 1
    ij = keys - lo
    grid = np.full(tuple(shape), -1, np.int32)
    grid[tuple(ij.T)] = np.arange(n, dtype=np.int32)
    chart = np.zeros(tuple(shape), bool)
    chart[tuple(ij.T)] = active
    interior = binary_erosion(chart)[tuple(ij.T)]
    previous_distance, _ = cKDTree(current_xyz[current_active, :2]).query(xyz[:, :2])
    state = np.zeros(n, np.uint8)
    state[valid_count > 0] = 1
    state[geometry_valid & (valid_count > 0)] = 2
    state[active] = 3
    world = transform(xyz, anchor)
    model = transform(world, world_to_model)
    normal_world = normals @ anchor[:3, :3].T
    payload = dict(
        keys=keys, local_xyz=xyz.astype(np.float32), world_xyz=world, model_xyz=model,
        normal_local=normals.astype(np.float32), normal_world=normal_world.astype(np.float32),
        gaussian_sigma_m=a["gaussian_sigma_m"].astype(np.float32), cell_m=np.float64(cell),
        reference_linear=reference, reference_srgb=srgb(reference).astype(np.float32),
        rgb=srgb(reference).astype(np.float32), active=active, real_active=active,
        measured=active, inferred=np.zeros(n, bool), colour_valid=colour_valid,
        supported=geometry_valid, surface_supported=a["surface_supported"].astype(bool),
        geometry_measured_support=geometry_valid, actual_return_support_count=support_count,
        support_pool_ids=support_ids, support_distances_m=distances.astype(np.float32),
        support_height_residual_m=height_residual.astype(np.float32),
        geometry_confidence=geometry_confidence, photometric_confidence=photo_confidence,
        confidence=(geometry_confidence * photo_confidence).astype(np.float32),
        source_valid=valid, source_agree=agree, source_valid_count=valid_count,
        agree_count=agree_count, source_raw=source_raw, source_rgb=accepted_rgb,
        first_clean_source=first, last_clean_source=last, best_source_raw=best,
        temporal_span_frames=temporal_span, agreement_residual_linear=agreement_residual.astype(np.float32),
        colour_range_valid=colour_range_valid, evidence_state=state,
        appearance_provenance=np.where(active, 1, 0).astype(np.uint8),
        geometry_provenance=np.where(geometry_valid, 1, 0).astype(np.uint8),
        unknown=~active, route_interior=interior, domain=np.ones(n, bool),
        legacy_domain=legacy["domain"].astype(bool), domain_a5=legacy["domain_a5"].astype(bool),
        domain_a37=legacy["domain_a37"].astype(bool),
        legacy_current_nearest_active_xy_m=previous_distance.astype(np.float32),
        grid_index=grid, grid_keys_min=lo, grid_shape_xy=shape,
        coordinate_anchor=anchor, world_to_model=world_to_model,
        meter_to_model=np.float64(meta["meter_to_model"]),
        candidate_origin=a["candidate_origin"].astype(np.uint8),
    )
    np.savez_compressed(target, **payload)
    del observed, accepted_rgb, luma
    sources_after = {k: sha256(v) for k, v in paths.items()}
    assert all(sources_before[k]["sha256"] == sources_after[k] for k in paths)
    checks = dict(
        source_files_sha_unchanged=True,
        source_valid_exactly_matches_immutable_cache=bool(np.array_equal(valid, a["source_valid"])),
        source_agree_subset_of_valid=bool((~agree | valid).all()),
        active_requires_three_stable_views=bool((agree_count[active] >= 3).all()),
        active_requires_temporal_span6=bool((temporal_span[active] >= 6).all()),
        active_requires_geometry=bool(geometry_valid[active].all()),
        active_reference_finite=bool(np.isfinite(reference[active]).all()),
        no_colour_interpolation=True, no_inferred_cells=True,
        world_transform_error_m=float(np.max(np.abs(world - a["world_xyz"]))),
        model_transform_error=float(np.max(np.abs(model - a["model_xyz"]))),
        grid_lookup_exact=bool(np.array_equal(grid[tuple(ij.T)], np.arange(n))),
    )
    with np.load(target, allow_pickle=False) as f:
        checks["saved_active_exact"] = bool(np.array_equal(f["active"], active))
        checks["invalid_source_rgb_is_nan"] = bool(np.isnan(f["source_rgb"][~valid]).all())
        checks["accepted_source_rgb_finite"] = bool(np.isfinite(f["source_rgb"][valid]).all())
    assert all(v for v in checks.values() if isinstance(v, bool))
    assert checks["world_transform_error_m"] < 5e-6 and checks["model_transform_error"] < 5e-7
    write(out / "verification.json", checks)
    source_count_summary = {str(k): count(valid_count == k) for k in range(3)}
    source_count_summary["ge3"] = count(valid_count >= 3)
    clean_count_summary = {str(k): count(agree_count == k) for k in range(3)}
    clean_count_summary["ge3"] = count(agree_count >= 3)
    receipt = dict(
        schema="R0_MEASURED_SUPPORT_FRONT199_ROAD_FRONTFILL_V2", created_utc=datetime.now(timezone.utc).isoformat(),
        asset=str(target), asset_sha256=sha256(target), asset_bytes=target.stat().st_size,
        CPU_only=True, GPU_used=False, torch_imported=False, checkpoint_loads=0, optimizer_updates=0,
        original_assets_modified=False, source_camera="FRONT", source_raw_range=[0, 198],
        all199_are_build_inputs=True, heldout_scene_claim=False,
        candidates=n, cell_m=cell, candidate_bounds_local_m=bounds(xyz), active_bounds_local_m=bounds(xyz, active),
        geometry_supported=count(geometry_valid), cached_surface_supported=count(a["surface_supported"]),
        active=count(active), unknown=count(~active), inferred=0, source_count_histogram=source_count_summary,
        agreeing_source_count_histogram=clean_count_summary, accepted_point_frame_records=count(valid),
        agreeing_point_frame_records=count(agree),
        active_outside_current_xmax=count(active & (xyz[:, 0] > current_xyz[current_active, 0].max())),
        active_in_legacy_domain=count(active & legacy["domain"]),
        active_in_a37_domain=count(active & legacy["domain_a37"]),
        legacy_current_bounds_local_m=bounds(current_xyz, current_active),
        denominator_reconciliation=dict(full_expanded_candidates=45769, legacy_2m_domain_cells=count(legacy["domain"]),
            added_front_rectangle_cells=n-45769, complete_frontfill_candidates=n,
            interpretation="45769 is the original fixed 2.8m halo; 36790 is its a5/a37 sweep+2m domain subset. V2 adds7313 previously absent front-rectangle cells by existing x/y bounds, with unchanged true-source rules. This is a candidate-domain completeness fix; no raw137 ROI or threshold tuning."),
        geometry_rule="Original expanded8cm height/normal, >=3 nearest of8 deduplicated real ground returns within0.8m and median tangent-normal support residual<=0.3m; height remains measured-support interpolation, not per-cell LiDAR.",
        appearance_rule="Only cached source_valid geometric triangle+FRONT guard+3D box-clear samples. Linear upper cluster q75[-0.025,+0.06], maxRGB agreement<0.035, >=3 agreeing raws, span>=6, reference channels in(0.012,0.88). Fixed for entire candidate grid.",
        sigma_rule="Unchanged expanded8cm physical sigma=[0.052,0.052,0.015] metres; renderer multiplies by meter_to_model exactly once.",
        unknown_policy="Unknown includes unsupported geometry, no accepted FRONT source, unstable colour or insufficient count/span; unknown is never a never-visible claim. No neighbour RGB filling.",
        provenance=dict(appearance_0="unknown/no repair authority", appearance_1="stable real FRONT multi-view colour proxy",
            geometry_0="unknown", geometry_1="height/normal interpolation supported by actual pooled LiDAR returns",
            pool_source_kind_0="frame ground_local index", pool_source_kind_1="frame cloud near-ground reclassification index"),
        coordinates=dict(contract="local anchor metres -> coordinate_anchor -> world metres -> world_to_model -> model units",
            meter_to_model=float(meta["meter_to_model"]), coordinate_anchor=anchor.tolist(), world_to_model=world_to_model.tolist(),
            grid_order="grid_index[ix,iy], keys=grid_keys_min+[ix,iy]; missing candidate=-1"),
        support_pool=dict(path=str(out / "measured_ground_support_pool.npz"), sha256=sha256(out / "measured_ground_support_pool.npz"),
            points=len(pool), original_return_records=total_returns, dedup_xy_m=.1, source_frames=frame_sources),
        sources=sources_before, verification=checks, seconds=time.time()-started,
        limits=["RGB is a real-view clean-road proxy, not independently observed shadow-free GT.",
                "No surround-camera search; no negative visibility conclusion.",
                "No raw137 ROI-specific fitting, no rendered residual colour, no native/model feature colour.",
                "This asset does not by itself authorize replacing already-reviewed current-road cells."],
    )
    write(out / "receipt.json", receipt)
    write(out / "asset_interface.json", dict(schema=receipt["schema"], asset=str(target), sha256=receipt["asset_sha256"],
          arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in payload.items()},
          coordinates=receipt["coordinates"], rules={k: receipt[k] for k in ("geometry_rule", "appearance_rule", "sigma_rule", "unknown_policy")},
          source_camera="FRONT", source_raw_range=[0, 198], heldout_scene_claim=False))
    print(json.dumps({k: receipt[k] for k in ("asset", "asset_sha256", "candidates", "geometry_supported", "active", "unknown", "active_outside_current_xmax", "active_in_a37_domain", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
