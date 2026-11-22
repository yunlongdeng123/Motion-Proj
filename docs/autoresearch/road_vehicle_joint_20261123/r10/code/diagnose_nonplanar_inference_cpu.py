"""Read-only inference diagnostics. This does not authorize geometry or render assets."""
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, cKDTree


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def pin(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0}
    return dict(count=int(x.size), min=float(x.min()), mean=float(x.mean()),
                median=float(np.median(x)), p05=float(np.quantile(x, .05)),
                p95=float(np.quantile(x, .95)), max=float(x.max()))


def design(xy, centre):
    x, y = (xy - centre).T
    return np.stack([np.ones(len(x)), x, y, x*x, x*y, y*y], -1)


def normals(xy, centre, b):
    x, y = (xy - centre).T
    n = np.c_[-b[1]-2*b[3]*x-b[4]*y, -b[2]-b[4]*x-2*b[5]*y, np.ones(len(x))]
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def medoids(X, ids, origin):
    cell = np.floor((X[ids, :2] - origin) / .1).astype(int)
    order = np.lexsort((cell[:, 1], cell[:, 0]))
    ids, cell = ids[order], cell[order]
    _, starts = np.unique(cell, axis=0, return_index=True)
    starts = np.sort(starts)
    selected = []
    for lo, hi in zip(starts, np.r_[starts[1:], len(ids)]):
        q = ids[lo:hi]
        mid = np.median(X[q], axis=0)
        selected.append(q[np.argmin(np.linalg.norm(X[q]-mid, axis=1))])
    return np.asarray(selected, np.int64)


def fit(A, z, mass=None):
    mass = np.ones(len(z)) if mass is None else np.asarray(mass, np.float64)
    use = mass > 0
    A, z, mass = A[use], z[use], mass[use]
    if np.linalg.matrix_rank(A) < 6:
        return None, None
    b = np.linalg.lstsq(A*np.sqrt(mass[:, None]), z*np.sqrt(mass), rcond=None)[0]
    for _ in range(8):
        r = A@b-z
        w = mass*np.minimum(1., .03/np.maximum(abs(r), 1e-12))
        b = np.linalg.lstsq(A*np.sqrt(w[:, None]), z*np.sqrt(w), rcond=None)[0]
    return b, w


def neighbourhood(P, raw, Xtrain, rawtrain, query):
    tree = cKDTree(P[:, :2])
    nearest, _ = tree.query(query)
    out = {"nearest_medoid_m": nearest}
    all_distance, _ = cKDTree(Xtrain[:, :2]).query(query)
    out["nearest_training_point_m"] = all_distance
    for radius in [.8, 1.5]:
        count = np.zeros(len(query), np.int16)
        for r in np.unique(rawtrain):
            d, _ = cKDTree(Xtrain[rawtrain == r, :2]).query(query)
            count += d <= radius
        out[f"distinct_frames_within_{radius:g}m"] = count
    for radius in [1., 2.]:
        neighbours = tree.query_ball_point(query, radius)
        quadrants = np.zeros(len(query), np.uint8)
        gap = np.full(len(query), 360., np.float64)
        count = np.fromiter((len(q) for q in neighbours), np.int32)
        for i, ids in enumerate(neighbours):
            if not ids:
                continue
            delta = P[ids, :2]-query[i]
            quadrants[i] = len(np.unique((delta[:, 0] >= 0)*2 + (delta[:, 1] >= 0)))
            angles = np.sort(np.arctan2(delta[:, 1], delta[:, 0]))
            gap[i] = np.degrees(np.diff(np.r_[angles, angles[0]+2*np.pi]).max())
        out[f"support_quadrants_within_{radius:g}m"] = quadrants
        out[f"largest_angular_gap_within_{radius:g}m_deg"] = gap
        out[f"medoid_count_within_{radius:g}m"] = count
    return out


def summarize_maps(maps, mask):
    return {k: stats(v[mask]) for k, v in maps.items()}


def sample(m, uv, valid):
    H, W = m.shape
    u = np.floor(np.nan_to_num(uv[..., 0])).astype(int)
    v = np.floor(np.nan_to_num(uv[..., 1])).astype(int)
    good = valid & (u >= 0) & (v >= 0) & (u+1 < W) & (v+1 < H)
    u, v = np.clip(u, 0, W-2), np.clip(v, 0, H-2)
    x, y = uv[..., 0]-u, uv[..., 1]-v
    out = ((1-x)*(1-y)*m[v, u] + x*(1-y)*m[v, u+1] +
           (1-x)*y*m[v+1, u] + x*y*m[v+1, u+1])
    return np.where(good, out, np.nan), good


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    C, out = args.campaign_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    R10 = C/"road_underbody_local_20261123_r10"
    strict = R10/"audit/local_surface_geometry_v1"
    contract = read(strict/"geometry_qualification_contract.json")
    diagnostic_contract = dict(schema="R10.nonplanar_inference_diagnostic_contract.v1",
        complete=True, decision="DIAGNOSTIC_ONLY_NOT_APPROVED", written_before_diagnostics=True,
        strict_contract=pin(strict/"geometry_qualification_contract.json"),
        train_raw=contract["train_raw"], heldout_raw=contract["heldout_raw"],
        fit="Identical strict 0.1m measured medoids and IRLS8 quadratic profile; no extra points added",
        cluster_bootstrap="512 resamples of the 9 training-frame clusters on fixed measured medoids; repeated clusters weight their original rows; seed20261123; no heldout fitting",
        leave_one_frame_out="refit medoid selection on remaining strict training points for each of9 frames",
        uncertainty="Conditional sensitivity under this quadratic/profile and proxy-filtered measurements; not a calibrated coverage probability or evidence of hidden surface truth",
        inference_candidate="Only missing-parent texels within global measured XY convex hull; no PASS threshold registered or inferred geometry activated",
        C0_C1="Compare raw quadratic profile with actual supported parent boundary height and normal; no silent correction or C1 claim",
        original_denominators="red357/692 and focused1923/3288 remain fixed; rasterization is perspective-correct actual XYZ, not old perview material masks",
        renderable_authorized=False, GPU_forward=0, updates=0)
    assert not (out/"diagnostic_contract.json").exists()
    write(out/"diagnostic_contract.json", diagnostic_contract)

    measured = load(strict/"strict_measurements.npz")
    s = load(strict/"local_surface_geometry.npz")
    gridpath = C/"road_shared_appearance_20261122_r7/shared_grid_geometry.npz"
    g = load(gridpath)
    X, raw = measured["world_xyz"], measured["source_raw"]
    train = measured["used_for_fit"]
    centre = s["fit_XY_centre"]
    origin = np.asarray(contract["XY_fit_context_bbox"])[0]
    selected = medoids(X, np.flatnonzero(train), origin)
    P, selected_raw = X[selected], raw[selected]
    A = design(P[:, :2], centre)
    b, weights = fit(A, P[:, 2])
    coeff_error = float(abs(b-s["fit_coefficients"]).max())
    assert coeff_error < 1e-12
    world = s["world_xyz"]
    shape = world.shape[:2]
    Q = world[..., :2].reshape(-1, 2)
    AQ = design(Q, centre)
    hull = ConvexHull(P[:, :2])
    margin = -(Q@hull.equations[:, :2].T+hull.equations[:, 2]).max(-1)
    inside = margin >= -1e-10
    factors = neighbourhood(P, selected_raw, X[train], raw[train], Q)
    factors["convex_hull_margin_m"] = margin
    residual = A@b-P[:, 2]
    inv = np.linalg.pinv(A.T@(weights[:, None]*A))
    leverage = np.einsum("ni,ij,nj->n", AQ, inv, AQ)
    train_leverage = np.einsum("ni,ij,nj->n", A, inv, A)
    sigma2 = float(np.sum(weights*residual**2)/max(1, np.sum(weights)-6))
    factors["regression_leverage"] = leverage
    factors["leverage_over_max_training"] = leverage/train_leverage.max()
    factors["conditional_height_mean_SE_m"] = np.sqrt(np.maximum(0, sigma2*leverage))
    factors["conditional_new_point_SD_m"] = np.sqrt(np.maximum(0, sigma2*(1+leverage)))
    rng = np.random.default_rng(20261123)
    trains = np.asarray(contract["train_raw"])
    bootstrap = []
    deficient = 0
    for _ in range(512):
        draw = rng.integers(0, len(trains), size=len(trains))
        counts = np.bincount(draw, minlength=len(trains))
        mass = np.array([counts[np.flatnonzero(trains == r)[0]] for r in selected_raw])
        bb, _ = fit(A, P[:, 2], mass)
        if bb is None:
            deficient += 1
        else:
            bootstrap.append(bb)
    bootstrap = np.asarray(bootstrap)
    bs_std = np.zeros(len(Q)); bs_lo = np.zeros(len(Q)); bs_hi = np.zeros(len(Q))
    for lo in range(0, len(Q), 4096):
        pred = AQ[lo:lo+4096]@bootstrap.T
        bs_std[lo:lo+4096] = pred.std(-1, ddof=1)
        bs_lo[lo:lo+4096] = np.quantile(pred, .025, axis=-1)
        bs_hi[lo:lo+4096] = np.quantile(pred, .975, axis=-1)
    factors["frame_cluster_bootstrap_SD_m"] = bs_std
    factors["frame_cluster_bootstrap_95pct_width_m"] = bs_hi-bs_lo
    loo = []
    for r in trains:
        ids = medoids(X, np.flatnonzero(train & (raw != r)), origin)
        bb, _ = fit(design(X[ids, :2], centre), X[ids, 2])
        assert bb is not None
        loo.append(bb)
    loo = np.asarray(loo)
    loo_pred = AQ@loo.T
    factors["leave_one_frame_out_height_span_m"] = np.ptp(loo_pred, axis=-1)
    factors["leave_one_frame_out_max_delta_m"] = np.max(abs(loo_pred-(AQ@b)[:, None]), axis=-1)
    factors = {k: v.reshape(shape) for k, v in factors.items()}
    original = s["original_geometry_supported"].astype(bool)
    direct = s["geometry_valid"].astype(bool)
    inference = inside.reshape(shape) & ~original & ~direct
    u0, v0, u1, v1 = s["bounds_uvxy"].astype(int)
    parentxyz = g["world_xyz"][v0:v1, u0:u1]
    parentnormal = g["normal_world"][v0:v1, u0:u1]
    parent_boundary = original & binary_dilation(~original)
    missing_boundary = ~original & binary_dilation(original)
    height_delta = world[..., 2]-parentxyz[..., 2]
    nn = np.linalg.norm(parentnormal, axis=-1)
    normal_good = parent_boundary & np.isfinite(parentnormal).all(-1) & (nn > 1e-8)
    dot = np.sum(s["normal_world"]*parentnormal/np.maximum(nn[..., None], 1e-12), -1)
    angle = np.degrees(np.arccos(np.clip(abs(dot), 0., 1.)))
    boundary = dict(complete=True, decision="NOT_C0_C1_MATCHED_NOT_RENDERABLE",
        supported_boundary_vertices=int(parent_boundary.sum()),
        raw_profile_parent_height_abs_delta_m=stats(abs(height_delta[parent_boundary])),
        raw_profile_parent_normal_delta_deg=stats(angle[normal_good]),
        normal_usable_vertices=int(normal_good.sum()),
        requirement="Any approved appended geometry must reuse actual parent boundary XYZ and quantify discrete first-derivative/normal agreement. This raw profile has not been boundary-corrected.",
        parent_geometry_preserved_byteexact=True)
    write(out/"parent_boundary_C0_C1.json", boundary)
    np.savez_compressed(out/"inference_candidate_DIAGNOSTIC_ONLY.npz",
        profile_world_xyz=world, profile_normal_world=s["normal_world"], uv_global=s["uv_global"],
        original_parent_world_xyz=parentxyz, original_parent_normal_world=parentnormal,
        direct_witness_mask=direct, original_geometry_supported=original,
        inferred_only_candidate_mask=inference, global_hull_interpolation_mask=inside.reshape(shape),
        renderable_mask=np.zeros(shape, bool), geometry_confidence_is_calibrated_probability=np.asarray(False),
        parent_supported_boundary_mask=parent_boundary, parent_missing_boundary_mask=missing_boundary,
        parent_boundary_height_delta_m=height_delta,
        parent_boundary_normal_delta_deg=angle,
        bootstrap_coefficients=bootstrap, leave_one_frame_out_coefficients=loo,
        fit_coefficients=b, fit_XY_centre=centre,
        bounds_uvxy=s["bounds_uvxy"], spacing_m=s["spacing_m"],
        **factors)
    print(json.dumps(dict(stage="inference_diagnostics_ready", direct=int(direct.sum()),
        missing_parent=int((~original).sum()), inside_hull=int(inside.sum()),
        inferred_unapproved=int(inference.sum()), boundary=boundary)), flush=True)

    spec = importlib.util.spec_from_file_location("local_surface_projection", R10/"code/local_surface_projection.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    target_rows = []
    for r in [86, 110]:
        cp = C/f"road_shared_appearance_20261122_r7/projection_cache_v1/{r:03d}/view_cache.npz"
        view = load(cp)
        old = C/f"road_vehicle_joint_dggt_20261123_r8/residual_rgb_groups_v1/CPU_attribution_v2_focus/{r:03d}"
        meta, small = read(old/"attribution.json"), load(old/"ROI_attribution.npz")
        x0, y0, x1, y1 = meta["ROI_xyxy"]
        E, K = view["world_to_camera"], view["camera_K"]
        HW = view["projection_valid"].shape
        coverage = {}
        for name, xyz, valid in [
            ("strict_direct_witness", world, direct),
            ("actual_parent_geometry", parentxyz, original),
            ("quadratic_full_hull_DIAGNOSTIC", world, inside.reshape(shape)),
            ("quadratic_inferred_only_DIAGNOSTIC", world, inference)]:
            projection = mod.raster_surface(xyz, s["uv_global"], valid, E, K, HW[::-1])
            coverage[name] = projection
        q = coverage["quadratic_full_hull_DIAGNOSTIC"]
        localuv = q["atlas_uv_global"]-np.array([u0, v0])
        infer_fraction, bounded = sample(inference.astype(float), localuv, q["projection_valid"])
        sampled = {k: sample(v, localuv, q["projection_valid"])[0] for k, v in factors.items()}
        payload = {name+"_projected": value["projection_valid"] for name, value in coverage.items()}
        payload["quadratic_diagnostic_atlas_uv_global"] = q["atlas_uv_global"]
        payload["inferred_fraction_DIAGNOSTIC"] = infer_fraction
        rows = {}
        for region in ["red_diagnostic", "focused_residual_candidate", "focused_ground_envelope", "original_proxy_lower_edge_ROI"]:
            mask = np.zeros(HW, bool)
            mask[y0:y1, x0:x1] = small[region]
            denominator = int(mask.sum())
            valid = mask & q["projection_valid"] & bounded
            partition = dict(
                outside_proposed_quadratic_projection=int((mask & ~q["projection_valid"]).sum()),
                projection_bilinear_boundary=int((mask & q["projection_valid"] & ~bounded).sum()),
                parent_supported_all4=int((valid & (infer_fraction <= 1e-9)).sum()),
                inferred_missing_parent_all4_unapproved=int((valid & (infer_fraction >= 1-1e-9)).sum()),
                mixed_parent_inferred_quad_unapproved=int((valid & (infer_fraction > 1e-9) & (infer_fraction < 1-1e-9)).sum()))
            assert sum(partition.values()) == denominator
            rows[region] = dict(fixed_pixels=denominator,
                projected_support_counts={k:int((mask & v["projection_valid"]).sum()) for k,v in coverage.items()},
                proposed_profile_query_partition=partition,
                factors_on_proposed_projection={k:stats(v[valid]) for k,v in sampled.items()},
                quadratic_projection_is_not_geometry_approval=True)
            payload[region] = mask
        np.savez_compressed(out/f"{r:03d}_target_coverage_DIAGNOSTIC.npz", **payload)
        row = dict(raw=r, regions=rows, camera=pin(cp), target_definition=pin(old/"ROI_attribution.npz"))
        target_rows.append(row)
        write(out/f"{r:03d}_target_coverage.json", row)
        print(json.dumps(dict(stage="target_coverage", raw=r, regions=rows)), flush=True)
    result = dict(schema="R10.neighbourhood_constrained_nonplanar_inference_diagnostic.v1",
        complete=True, decision="NOT_APPROVED_FOR_RENDERING", renderable_texels=0,
        original_strict_certificate=pin(strict/"certificate.json"), strict_mask_unchanged=True,
        direct_witness_texels=int(direct.sum()), parent_supported_texels=int(original.sum()),
        parent_missing_texels=int((~original).sum()), global_hull_interpolation_texels=int(inside.sum()),
        inferred_only_unapproved_texels=int(inference.sum()), fit_coefficient_replay_max_abs=coeff_error,
        measured_training_points=int(train.sum()), measured_training_medoids=len(selected),
        training_medoids_by_frame={str(r):int((selected_raw==r).sum()) for r in trains},
        weighted_residual_sigma_m=float(np.sqrt(sigma2)),
        bootstrap_valid_replicates=len(bootstrap), bootstrap_rank_deficient=deficient,
        uncertainty_is_not_calibrated_probability=True, heldout_used_for_fit=False,
        regions={"all_patch":summarize_maps(factors, np.ones(shape,bool)),
                 "parent_missing_global_interpolation_unapproved":summarize_maps(factors, inference),
                 "strict_direct_witness":summarize_maps(factors, direct)},
        parent_boundary=boundary, target_rows=target_rows,
        candidate=pin(out/"inference_candidate_DIAGNOSTIC_ONLY.npz"),
        diagnostic_contract=pin(out/"diagnostic_contract.json"), parent_geometry=pin(gridpath),
        original_geometry_and_models_modified=False, source_RGB_used_for_geometry_fit=False,
        new_GPU_forward=0, updates=0, elapsed_seconds=time.time()-start)
    write(out/"inference_statistics.json", result)
    print(json.dumps(dict(complete=True, receipt=pin(out/"inference_statistics.json"))), flush=True)


if __name__ == "__main__":
    main()
