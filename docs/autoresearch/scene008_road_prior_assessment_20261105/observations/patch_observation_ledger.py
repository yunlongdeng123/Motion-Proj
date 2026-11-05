"""CPU reprojects a small predefined patch through 199 cached FRONT records.

No raw JPEG/range decoding and no model/checkpoint import. Three cached PNGs
are decoded only after geometric witnesses are found, for evidence examples.
This does not test the other four cameras and never proves negative visibility.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
sys.dont_write_bytecode = True


def project(points, K, w):
    camera = points @ w[:3, :3].T + w[:3, 3]
    image = camera @ K.T
    return image[:, :2] / np.maximum(camera[:, 2:], 1e-8) - .5, camera[:, 2]


def count(mask):
    return int(np.count_nonzero(mask))


def sample(a, uv, order=1, cval=0.):
    return map_coordinates(a.astype(float), [uv[:, 1], uv[:, 0]], order=order,
                           mode="constant", cval=cval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    started = time.time()
    base, run = args.base, args.run
    out = run / "observations"
    jsonpath = out / "patch137_observation_ledger.json"
    if jsonpath.exists():
        raise FileExistsError("Preserve earlier patch audit")
    full = base / "r15_gn11_fullclip_cpu_20261026"
    source = base / "r15_gn11_stage1_v41_20261023/tools"
    sys.path.insert(0, str(source))
    from v41_dense_geometry import measured_patch_evidence, ray_box_blocked

    with np.load(run/"coverage/137_world_maps.npz") as m, np.load(run/"coverage/137_masks.npz") as a:
        group_masks = {name:a[name].astype(bool) for name in ("fixed_roi", "under_tail_band")}
        geometry_valid = m["geometry_valid"].astype(bool)
        pick = (group_masks["fixed_roi"] | group_masks["under_tail_band"]) & geometry_valid
        pixels_yx = np.argwhere(pick)
        xyz = m["local_xyz"][pick].astype(float)
        members = {name:mask[pick] for name, mask in group_masks.items()}
        groups = {name:dict(pixel_denominator=count(mask), inherited_geometry_valid=count(mask & geometry_valid),
                           inherited_geometry_unknown=count(mask & ~geometry_valid)) for name,mask in group_masks.items()}

    # Coverage geometry is a ray/surface intersection, not a new LiDAR return.
    # Check physical support against the pooled actual LiDAR points separately.
    with np.load(full/"full199_ground_pool.npz") as a:
        ground = a["local_xyz"]
    distances, _ = cKDTree(ground[:, :2]).query(xyz[:, :2], k=8, distance_upper_bound=.8)
    physical_support = np.isfinite(distances).sum(1) >= 3
    with np.load(full/"full199_initialization.npz") as a:
        _, near = cKDTree(a["local_xyz"][:, :2]).query(xyz[:, :2])
        normals = a["normal_local"][near].astype(float)
        height_delta = np.abs(xyz[:, 2]-a["local_xyz"][near, 2])
    # Height confidence is conservative and explicitly retained in the ledger.
    height_consistent = height_delta <= .3
    geometry_confident = physical_support & height_consistent

    matches = {}
    for label, path, cell in (
        ("global20cm", full/"full199_initialization.npz", .2),
        ("expanded8cm", base/"shadow_a5_a37_199_20261103/inputs/expanded/evidence.npz", .08)):
        with np.load(path) as a:
            lookup = {tuple(k):i for i,k in enumerate(a["keys"])}
            ids = np.array([lookup.get(tuple(k), -1) for k in np.floor(xyz[:, :2]/cell).astype(int)])
            safe = np.maximum(ids, 0)
            valid = a["source_valid"][:, safe] & (ids >= 0)[None]
            delta = np.abs(xyz[:, 2]-a["local_xyz"][safe, 2])
            rec = dict(path=str(path), cell_m=cell,
                contract="Same XY-containing cached cell, not exact query-point RGB. Direct reprojection below is the stronger test.",
                matched_queries=count(ids >= 0), matched_with_any_sample=count(valid.any(0)),
                matched_cell_height_delta_max_m=float(delta[ids >= 0].max()) if (ids >= 0).any() else None,
                groups={name:dict(queries=count(mask), matching_cell=count(mask & (ids >= 0)),
                    matching_cell_with_sample=count(mask & valid.any(0)),
                    matching_cell_ge3_samples=count(mask & (valid.sum(0) >= 3))) for name,mask in members.items()})
            if "eligible" in a.files:
                rec["matching_old_eligible_cells_queries"] = count((ids >= 0) & a["eligible"][safe])
            matches[label] = rec

    N = len(xyz)
    keys = ("in_front_fov", "guarded_2d", "box_blocked", "nearest_return_confirms_ground",
            "nearest_return_in_front", "nearest_return_conflict", "triangle_evidence", "triangle_conflict",
            "source_quality_eligible", "geometric_RGB_qualified")
    flags = {key:np.zeros((199, N), bool) for key in keys}
    quality_rows = json.loads((full/"extraction.json").read_text())["frames"]
    sharpness = np.array([r["sharpness"] for r in quality_rows])
    quality = np.minimum(sharpness/max(np.median(sharpness[89:138]), 1e-9), 1.)
    frame_rows = []
    for raw in range(199):
        with np.load(full/"frames"/f"{raw:03d}.npz") as f:
            K, w = f["K"], f["w2l"]
            uv, depth = project(xyz, K, w)
            origin = np.linalg.inv(w)[:3, 3]
            ray = origin-xyz
            length = np.linalg.norm(ray, axis=1)
            cosine = np.abs((ray/np.maximum(length[:, None], 1e-9)*normals).sum(1))
            area = K[0, 0]*K[1, 1]*.2**2*cosine/np.maximum(depth**2, 1e-8)
            inside = (depth>.1) & (uv[:, 0]>=0) & (uv[:, 0]<479) & (uv[:, 1]>=0) & (uv[:, 1]<319)
            guarded = inside & (sample(f["guard"], uv, cval=1.)>0)
            boxes = [dict(center=c, half=h, axes=ax) for c,h,ax in zip(f["box_center"], f["box_half"], f["box_axes"])]
            blocked = inside & ray_box_blocked(xyz, origin, boxes)
            source_quality = (cosine >= .12) & (area >= .25) & (quality[raw] >= .2)
            cloud, ground_mask = f["cloud"], f["ground"]
            cuv, cdepth = project(cloud, K, w)
            near_distance, nearest = cKDTree(cuv).query(uv)
            direct = inside & (near_distance<=1.5)
            confirms = direct & ground_mask[nearest] & (np.abs(cdepth[nearest]-depth)<=.3)
            in_front = direct & (cdepth[nearest]<depth-.3)
            conflict = direct & (~ground_mask[nearest] | (np.abs(cdepth[nearest]-depth)>.3))
            candidate = geometry_confident & inside & ~guarded & ~blocked & source_quality
            ids = np.flatnonzero(candidate)
            triangle, tri_conflict = np.zeros(N,bool), np.zeros(N,bool)
            if len(ids):
                ok, _, _, cf = measured_patch_evidence(cloud, ground_mask, K, w, xyz[ids], uv[ids], normals[ids])
                triangle[ids], tri_conflict[ids] = ok, cf
            values = dict(in_front_fov=inside, guarded_2d=guarded, box_blocked=blocked,
                nearest_return_confirms_ground=confirms, nearest_return_in_front=in_front,
                nearest_return_conflict=conflict, triangle_evidence=triangle, triangle_conflict=tri_conflict,
                source_quality_eligible=inside & source_quality, geometric_RGB_qualified=candidate & triangle)
            for key,value in values.items():
                flags[key][raw] = value
            frame_rows.append(dict(raw=raw, **{k:count(v) for k,v in values.items()}))
        if raw%50==0:
            print(json.dumps(dict(raw=raw, queried_points=N, elapsed_s=time.time()-started)), flush=True)

    qualified = flags["geometric_RGB_qualified"]
    # Explicitly unresolved despite rejection: conservative guard, sparse depth,
    # surface/normal uncertainty, quality gate or untested camera are not proof.
    definitely_blocked = flags["box_blocked"] | flags["nearest_return_in_front"]
    unresolved_in_fov = flags["in_front_fov"] & ~definitely_blocked & ~qualified
    group_results = {}
    for name,member in members.items():
        q = qualified[:, member]
        all_out = ~flags["in_front_fov"][:, member].any(0)
        every_front_opportunity_blocked = (definitely_blocked[:, member] | ~flags["in_front_fov"][:, member]).all(0) & ~all_out
        group_results[name] = dict(**groups[name],
            query_points_with_actual_pool_support=count(member & physical_support),
            query_points_height_consistent=count(member & height_consistent),
            points_with_geometric_RGB_witness=count(q.any(0)),
            points_with_ge3_geometric_RGB_witnesses=count(q.sum(0)>=3),
            points_with_no_qualified_FRONT_RGB_witness=count(~q.any(0)),
            qualified_point_frame_records=count(q),
            witness_frames=np.flatnonzero(q.any(1)).tolist(),
            all199_FRONT_out_of_fov_points=count(all_out),
            all_in_fov_FRONT_opportunities_evidence_blocked_points=count(every_front_opportunity_blocked),
            points_with_unresolved_FRONT_opportunity=count(unresolved_in_fov[:, member].any(0)),
            local_bounds_m=[xyz[member].min(0).tolist(), xyz[member].max(0).tolist()])

    witnesses = []
    ranked = np.argsort(-qualified.sum(1), kind="stable")
    for raw in ranked[:3]:
        if not qualified[raw].any():
            continue
        raw = int(raw)
        with np.load(full/"frames"/f"{raw:03d}.npz") as f:
            uv, _ = project(xyz, f["K"], f["w2l"])
        image = np.asarray(Image.open(full/"rgb"/f"{raw:03d}.png"), float)/255
        selected = np.flatnonzero(qualified[raw])
        selected = selected[np.linspace(0,len(selected)-1,min(6,len(selected))).astype(int)]
        rows = []
        for idx in selected:
            rgb = [float(sample(image[:,:,ch], uv[idx:idx+1])[0]) for ch in range(3)]
            rows.append(dict(query_index=int(idx), source_raw=raw,
                source_uv=uv[idx].tolist(), target137_pixel_xy=pixels_yx[idx][::-1].tolist(),
                query_local_xyz_m=xyz[idx].tolist(), sampled_sRGB=rgb))
        witnesses.append(dict(raw=raw, qualified_query_points=count(qualified[raw]), samples=rows,
            caveat="Geometric RGB witness under the cached conservative rules; unshadowed/photometrically clean status is not asserted by this audit."))

    arrays_path = out/"patch137_observation_ledger.npz"
    np.savez_compressed(arrays_path, local_xyz=xyz, target137_pixels_yx=pixels_yx,
        physical_support=physical_support, height_consistent=height_consistent,
        height_delta_m=height_delta, **{f"member_{k}":v for k,v in members.items()}, **flags)
    result = dict(schema="SMALL_PATCH_FRONT199_EVIDENCE_LEDGER_V1", CPU_only=True,
        target_raw=137, source_camera="FRONT", source_frames=199, queried_points=N,
        original_user_570_pixels_reproduced=False,
        ROI_contract="Independent fixed ROI[230,204,270,222] and tail-band[230,204,266,224]; supplied pixel coordinates of the user's 570-pixel patch were unavailable.",
        groups=group_results, cell_level_lookup=matches, frame_rows=frame_rows, RGB_witness_examples=witnesses,
        ledger_npz=str(arrays_path), seconds=time.time()-started,
        geometry_warning="Query XYZ comes from a measured-support ray/surface interpolation. Height consistency and pool support do not convert it into an actual dense LiDAR observation.",
        inference_limits=["No accepted source is not equivalent to never visible.",
            "A coarse box occlusion is an annotation-based proxy; sparse depth leaves unresolved visibility.",
            "All199x5-camera RGB has not been searched. Other camera visibility is unknown.",
            "Near/far timestamp camera and distortion/rolling-shutter calibration can change marginal visibility.",
            "Current-layer absence is a scope/representation condition, independent of whether real RGB exists."],
        original_raw_decodes=0, cached_RGB_PNGs_decoded=len(witnesses), GPU_used=False,
        optimizer_updates=0, original_assets_modified=False)
    jsonpath.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(jsonpath), groups=group_results), indent=2), flush=True)


if __name__ == "__main__":
    main()
