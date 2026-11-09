"""Inherited fixed-chart C2 depth fit with explicit vehicle/source dependencies.

The original prior association, physical pitch, five IRLS solves, sigmas, retained
support and SH0 material initialization are unchanged. No parameter search,
additional training, frame-specific repair or identity inference is introduced.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
from PIL import Image
try:
    from .v44_vehicle_prior import read_json, frame_sources, frame_mask_path, frame_target
except ImportError:
    from v44_vehicle_prior import read_json, frame_sources, frame_mask_path, frame_target


def load_surface_helpers(deps_source):
    """Load only caller-declared source modules; never rely on legacy defaults."""
    roots = [Path(path).resolve() for path in deps_source]; modules = {}; provenance = {}
    saved_path = sys.path.copy()
    try:
        sys.path[:0] = [str(path) for path in roots]
        for name in ("fit_surface", "full_pvd"):
            paths = list(dict.fromkeys(path/(name+".py") for path in roots if (path/(name+".py")).is_file()))
            if len(paths) != 1: raise ValueError(f"deps-source must locate one {name}.py")
            path = paths[0]; digest = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location("_v44_dep_"+name+"_"+digest[:12], path)
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
            modules[name] = module; provenance[name] = dict(path=str(path), sha256=digest)
    finally:
        sys.path[:] = saved_path
    return modules["fit_surface"], modules["full_pvd"], provenance


def fit_vehicle_surface(root, bank, mask_root, track_id, actor_id, deps_source, *,
                        prior_dir=None, pvd_points=None, output=None, camera_index=0):
    from scipy.spatial import cKDTree
    from scipy.sparse import eye
    from scipy.sparse.linalg import spsolve
    root, bank, mask_root = map(Path, (root, bank, mask_root)); start = time.monotonic()
    prior_dir = root/"revealed_prior" if prior_dir is None else Path(prior_dir)
    pvd_points = prior_dir/"pvd_seed000/candidate_points_actor_native.npz" if pvd_points is None else Path(pvd_points)
    out = root/"revealed_surface" if output is None else Path(output)
    contract = read_json(prior_dir/"input_contract.json")
    if contract.get("track_id", track_id) != track_id or contract.get("actor_id", actor_id) != actor_id:
        raise ValueError("prior contract and requested vehicle identity differ")
    solver, prior_helpers, provenance = load_surface_helpers(deps_source)
    tensor_basis, knots_for, curvature, nearest3_pitch = (getattr(solver, key) for key in
                                                        ("tensor_basis", "knots_for", "curvature", "nearest3_pitch"))
    with np.load(prior_dir/"observations.npz", allow_pickle=False) as observations:
        xyz = observations["sample_xyz_local_m"].copy(); raws = observations["sample_raw_frame"].copy()
        observed_rgb = observations["sample_rgb"].copy(); box = observations["size_lwh_m"].copy()
    with np.load(pvd_points, allow_pickle=False) as data:
        pvd = {key: data[key].copy() for key in data.files}
    mothers = pvd["points_actor_native_m"]; patch = prior_helpers.pca_point_patches(mothers)
    if len(patch["sigma"]) != len(mothers): raise ValueError("inherited PVD patch/source topology differs")
    receipt, frames = frame_sources(bank); by_raw = {row["raw"]: row for row in frames}
    areas = {row["raw"]: int((np.array(Image.open(frame_mask_path(mask_root, row))) == actor_id).sum()) for row in frames}
    reference = max(areas, key=lambda raw: (areas[raw], -raw))
    if areas[reference] == 0: raise ValueError("requested vehicle has no observed chart mask")
    row = by_raw[reference]; obs = read_json(row["observation_path"]); body = frame_target(obs, track_id)
    lines = (bank/"calib"/(row["id"]+".txt")).read_text().splitlines()
    cal = {line.split(":")[0]: np.fromstring(line.split(":")[1], sep=" ") for line in lines}
    K = cal[f"P{camera_index}"].reshape(3, 4)[:, :3]; cam_ego = np.eye(4)
    cam_ego[:3] = cal[f"Tr_velo_to_cam_{camera_index}"].reshape(3, 4)
    camera = cam_ego @ np.linalg.inv(np.array(obs["ego_world"]).reshape(4, 4)) @ np.array(body["world_from_actor"]).reshape(4, 4)
    im = np.array(Image.open(bank/obs["image_path"]).convert("RGB")); h, w = im.shape[:2]
    mask = np.array(Image.fromarray(np.uint8(np.array(Image.open(frame_mask_path(mask_root, row))) == actor_id)).resize((w, h), Image.Resampling.NEAREST)) > 0
    yy, xx = np.where(mask); xy = np.column_stack((xx, yy)).astype(float)
    mc = mothers @ camera[:3, :3].T + camera[:3, 3]
    muv = (mc @ K.T)[:, :2]/mc[:, 2, None] - .5; positive = mc[:, 2] > .1
    distance, index = cKDTree(muv[positive]).query(xy, k=4); depths = mc[positive, 2][index]
    weights = 1/np.maximum(distance, .5)**2; prior_z = (depths*weights).sum(1)/weights.sum(1)
    measured = xyz @ camera[:3, :3].T + camera[:3, 3]
    uv = (measured @ K.T)[:, :2]/measured[:, 2, None] - .5; ij = np.floor(uv+.5).astype(int)
    fullmap = np.full((h, w), -1, np.int32); fullmap[yy, xx] = np.arange(len(xy))
    inside = (measured[:, 2] > .1) & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    ids = np.flatnonzero(inside); ids = ids[fullmap[ij[ids, 1], ij[ids, 0]] >= 0]
    keys = ij[ids, 1]*w+ij[ids, 0]; order = np.lexsort((ids, measured[ids, 2], keys))
    ids, keys = ids[order], keys[order]; ids = ids[np.r_[True, keys[1:] != keys[:-1]]]
    prior_index = fullmap[ij[ids, 1], ij[ids, 0]]; good = abs(measured[ids, 2]-prior_z[prior_index]) <= .75; accepted = ids[good]
    heldout = np.asarray(receipt.get("holdout_raws", []))
    if not len(accepted) or np.isin(raws[accepted], heldout).any(): raise ValueError("no valid declared-source measurement association")
    perframe = [nearest3_pitch(xyz[raws == raw]) for raw in sorted(set(raws)) if (raws == raw).sum() >= 4]
    pitch = max(nearest3_pitch(mothers), float(np.median(perframe)))
    zref = float(np.median(prior_z)); tu = knots_for(xy[:, 0].min(), xy[:, 0].max(), K[0, 0]*pitch/zref)
    tv = knots_for(xy[:, 1].min(), xy[:, 1].max(), K[1, 1]*pitch/zref)
    Bp = tensor_basis(xy, tu, tv); Bm = tensor_basis(uv[accepted], tu, tv)
    D = curvature(len(tu)-4, len(tv)-4); n = Bp.shape[1]; coef = np.zeros(n)
    terms = [(Bp, prior_z-zref, .20), (Bm, measured[accepted, 2]-zref, .05), (D, np.zeros(D.shape[0]), .05)]; trace = []
    for _ in range(5):
        H = eye(n, format="csc")*(1e-8/n); rhs = np.zeros(n)
        for B, y, sigma in terms:
            residual = B@coef-y; robust = np.minimum(1., sigma/np.maximum(abs(residual), 1e-15))
            weight = robust/(len(y)*sigma*sigma); H = H+B.T@B.multiply(weight[:, None]); rhs += B.T@(weight*y)
        coef = spsolve(H.tocsc(), rhs)
        if not np.isfinite(coef).all(): raise ValueError("nonfinite inherited depth fit")
        trace.append(float(abs(Bm@coef+zref-measured[accepted, 2]).mean()))
    z = zref+Bp@coef
    if not np.all(z > .1): raise ValueError("nonpositive inherited chart depth")
    rays = np.column_stack((xy+.5, np.ones(len(xy)))) @ np.linalg.inv(K).T
    local = (rays*z[:, None]-camera[:3, 3]) @ np.linalg.inv(camera[:3, :3]).T; chart_rgb = im[yy, xx]/255
    mp = np.floor(muv+.5).astype(int)
    visible = (mp[:, 0] >= 0) & (mp[:, 0] < w) & (mp[:, 1] >= 0) & (mp[:, 1] < h) & positive
    covered = np.zeros(len(mothers), bool); covered[visible] = mask[mp[visible, 1], mp[visible, 0]]; retained = ~covered
    retained_xyz = mothers[retained]; retained_rgb = observed_rgb[cKDTree(xyz).query(retained_xyz)[1]]
    positions = np.concatenate((retained_xyz, local)).astype(np.float32); N = len(positions); cut = len(retained_xyz)
    sigma_chart = (.6*z/K[0, 0]).astype(np.float32); sigma = np.r_[patch["sigma"][retained], sigma_chart].astype(np.float32)
    normals = np.concatenate((patch["normal"][retained], np.tile(-camera[2, :3], (len(local), 1)))); normals /= np.linalg.norm(normals, axis=1)[:, None]
    rgb = np.concatenate((retained_rgb, chart_rgb)).astype(np.float32); sh = np.zeros((N, 16, 3), np.float32); sh[:, 0] = (rgb-.5)/np.float32(.28209479177387814)
    arrays = dict(xyz=positions, normal=normals.astype(np.float32), sigma=sigma, thickness=sigma.copy(),
                  opacity=np.r_[np.full(cut, .995), np.full(len(local), .99)].astype(np.float32), box=box.astype(np.float32),
                  knn_spacing=sigma/np.float32(.6), normal_radius=3*sigma, normal_confidence=np.ones(N, np.float32),
                  generated=np.r_[pvd["generated_mask"][retained], np.zeros(len(local), bool)],
                  conditioning=np.r_[~pvd["generated_mask"][retained], np.zeros(len(local), bool)], normal_valid=np.ones(N, bool),
                  out_of_box=(abs(positions) > box/2).any(1), conditioning_held=np.zeros(N, bool), source_index=np.arange(N),
                  conditioning_observation_index=np.full(N, -1, np.int64),
                  conditioning_raw_frame=np.r_[np.full(cut, -1), np.full(len(local), reference)].astype(np.int64),
                  conditioning_cloud_row=np.full(N, -1, np.int64), representation_kind=np.r_[np.zeros(cut), np.ones(len(local))].astype(np.int64),
                  resample_reference_pixel_xy=np.r_[np.zeros((cut, 2)), xy].astype(np.float32), rgb=rgb, canonical_sh=sh,
                  colour_inferred=np.r_[np.ones(cut, bool), np.zeros(len(local), bool)],
                  metadata_json=np.asarray(json.dumps(dict(object="explicit vehicle source", track_id=track_id, actor_id=actor_id,
                    reference_raw=reference, canonical_material="fixed SH0 reference chart plus inferred PVD retained support",
                    geometry="same C2 robust depth spline; explicit source bindings"))))
    out.mkdir(parents=True, exist_ok=False); np.savez_compressed(out/"initial_asset.npz", **arrays)
    np.savez_compressed(out/"posterior.npz", control_delta_z_m=coef, z_reference_m=zref, u_knots_pixels=tu, v_knots_pixels=tv,
                        reference_K=K, reference_camera_from_local_m=camera, reference_raw=np.array(reference),
                        accepted_sample_ids=accepted, accepted_sample_raw=raws[accepted], query_depth_m=z, query_local_xyz=local)
    report = dict(status="PASS_NEW_OBJECT_C2_DEPTH_FIT", reference_raw=reference, observed_chart_pixels=len(local), retained_prior_rows=cut,
                  total_rows=N, control_shape=[len(tu)-4, len(tv)-4], physical_control_pitch_m=pitch, accepted_measurements=len(accepted),
                  association_uncertain=len(ids)-len(accepted), BUILD_measurements=len(xyz), IRLS_training_camera_depth_mae_trace=trace,
                  steps=5, prior_sigma_m=.20, measurement_sigma_m=.05, curvature_sigma_m=.05, colour_fit_steps=0,
                  holdout_used=bool(np.isin(raws[accepted], heldout).any()), seconds=time.monotonic()-start,
                  track_id=track_id, actor_id=actor_id, bank=str(bank), mask_root=str(mask_root),
                  observations=str(prior_dir/"observations.npz"), pvd_points=str(pvd_points), dependencies=provenance,
                  method="inherited C2 prior/measurement association and fixed 5-step robust depth fit")
    (out/"receipt.json").write_text(json.dumps(report, indent=2)); return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "bank", "mask-root"): ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--track-id", required=True); ap.add_argument("--actor-id", type=int, required=True)
    ap.add_argument("--deps-source", type=Path, required=True, action="append")
    ap.add_argument("--prior-dir", type=Path); ap.add_argument("--pvd-points", type=Path); ap.add_argument("--output", type=Path)
    ap.add_argument("--camera-index", type=int, default=0); args = ap.parse_args()
    print(json.dumps(fit_vehicle_surface(args.root, args.bank, args.mask_root, args.track_id, args.actor_id,
                                         args.deps_source, prior_dir=args.prior_dir, pvd_points=args.pvd_points,
                                         output=args.output, camera_index=args.camera_index)))


if __name__ == "__main__": main()
