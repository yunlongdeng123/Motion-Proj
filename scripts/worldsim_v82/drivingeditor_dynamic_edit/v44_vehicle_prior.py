"""Inherited whole-box PVD input preparation with explicit bank/identity bindings.

No prior generation/training is performed. Native PLY nx stores model_time; it
is matched to receipt frames, never converted using a frame interval or offset.
The conditioning count, coordinate/color math and normalization are inherited.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def model_time_frame_indices(times, frames, *, tolerance=None):
    """Nearest receipt.model_time, allowing only declared/float32 storage error."""
    times = np.asarray(times, dtype=float)
    reference = np.asarray([row["model_time"] for row in frames], dtype=float)
    if times.ndim != 1 or not len(reference) or not np.isfinite(times).all() or not np.isfinite(reference).all():
        raise ValueError("finite PLY times and nonempty receipt model_time rows are required")
    if len(np.unique(reference)) != len(reference) or len(np.unique(reference.astype(np.float32))) != len(reference):
        raise ValueError("receipt model times must retain distinct frame identity in float32 PLY")
    order = np.argsort(reference); sorted_time = reference[order]
    right = np.clip(np.searchsorted(sorted_time, times), 0, len(reference)-1)
    left = np.maximum(right-1, 0); dl = abs(times-sorted_time[left]); dr = abs(times-sorted_time[right])
    equal = np.isclose(dl, dr, rtol=0, atol=np.finfo(float).eps * np.maximum(1., abs(times))*4)
    if ((left != right) & equal).any(): raise ValueError("PLY model_time is ambiguous between receipt frames")
    chosen = np.where(dl < dr, left, right); error = np.minimum(dl, dr)
    if tolerance is None: tolerance = np.finfo(np.float32).eps * max(1., float(abs(reference).max())) * 4
    if not np.isfinite(tolerance) or tolerance < 0 or (error > tolerance).any():
        raise ValueError("PLY model_time is not a stored receipt-frame time")
    return order[chosen]


def frame_sources(bank):
    """Bind receipt rows to actual observation metadata; no filename-prefix rule."""
    bank = Path(bank); receipt = read_json(bank/"receipt.json"); observations = {}
    for path in sorted((bank/"observations").glob("*.json")):
        obs = read_json(path); raw = obs["raw"]
        if raw in observations: raise ValueError("duplicate observation raw IDs")
        observations[raw] = path
    rows = []
    for source in receipt["frames"]:
        row = dict(source); row["observation_path"] = observations[row["raw"]]
        if read_json(row["observation_path"])["timestamp_micros"] != row["timestamp_micros"]:
            raise ValueError("receipt/observation timestamp mismatch")
        rows.append(row)
    if len({row["raw"] for row in rows}) != len(rows): raise ValueError("duplicate receipt raw IDs")
    return receipt, rows


def frame_mask_path(mask_root, row):
    return Path(mask_root)/Path(row["observation_path"]).stem/"labels.png"


def frame_target(obs, track_id):
    matches = [body for body in obs["boxes"] if body.get("track", body.get("track_id")) == track_id]
    if len(matches) != 1: raise ValueError("requested track needs unique same-frame physical geometry")
    return matches[0]


def normalization_contract(size, template):
    size = np.asarray(size, float); rotation = np.asarray(template["native_to_training_raw_rotation"])
    span = np.asarray(template["training_mean_full_bbox_span_xyz"]); mean = np.asarray(template["training_mean_xyz"])
    center = np.asarray(template["training_mean_full_bbox_center_xyz"]); std = template["training_scalar_std"]
    if size.shape != (3,) or (size <= 0).any() or not np.isfinite(size).all(): raise ValueError("invalid physical size")
    scale = np.diag(size[[1, 2, 0]]/span); matrix = np.eye(4)
    matrix[:3, :3] = rotation.T @ scale * std; matrix[:3, 3] = rotation.T @ scale @ (mean-center)
    return dict(schema_version="pvd.fixed_wholebox.new_object.v1", whole_box_lwh_m=size.tolist(),
                partial_bounds_scaled=False, partial_bounds_centered=False, pvd_to_actor_native_4x4=matrix.tolist(),
                training_mean_xyz=mean.tolist(), training_scalar_std=std, training_mean_full_bbox_span_xyz=span.tolist(),
                training_mean_full_bbox_center_xyz=center.tolist(), native_to_training_raw_rotation=rotation.tolist(),
                normalization_source="same fixed whole-box and official PointFlow car/train stats as current method",
                front_sign_assumption=template["front_sign_assumption"])


def prepare_vehicle_prior(root, bank, mask_root, track_id, actor_id, pvd_base, *, output=None, time_tolerance=None):
    from plyfile import PlyData
    root, bank, mask_root, pvd_base = map(Path, (root, bank, mask_root, pvd_base))
    out = root/"revealed_prior" if output is None else Path(output)
    identity = read_json(root/"identity_contract.json")
    sources = [row for row in identity["source_actor_rows"] if row["actor_id"] == actor_id and row["track_id"] == track_id]
    if len(sources) != 1: raise ValueError("actor ID and raw track ID do not match native source identity")
    source = Path(sources[0]["ply"]); source = source if source.is_absolute() else root/source
    v = PlyData.read(source)["vertex"].data
    world = np.column_stack([v[key] for key in ("x", "y", "z")])
    if not len(world): raise ValueError("native actor point source is empty")
    receipt, frames = frame_sources(bank)
    row_index = model_time_frame_indices(v["nx"], frames, tolerance=time_tolerance)
    raws = np.asarray([frames[i]["raw"] for i in row_index], dtype=np.int64)
    xyz = np.empty_like(world); rgb = np.zeros_like(world); valid = np.zeros(len(world), bool); sizes = []; counts = []
    for index in sorted(set(row_index.tolist()), key=lambda i: frames[i]["raw"]):
        row = frames[index]; raw = row["raw"]; ids = np.flatnonzero(row_index == index)
        obs = read_json(row["observation_path"]); body = frame_target(obs, track_id); sizes.append(body["dimensions"])
        labels = np.array(Image.open(frame_mask_path(mask_root, row)))
        if not (labels == actor_id).any(): raise ValueError("native source frame has no matching accepted actor mask")
        inv = np.linalg.inv(np.asarray(body["world_from_actor"]).reshape(4, 4)); xyz[ids] = world[ids] @ inv[:3, :3].T + inv[:3, 3]
        camera = np.asarray(obs["camera_pose_world"]).reshape(4, 4) @ np.asarray(obs["camera_extrinsic"]).reshape(4, 4)
        inv = np.linalg.inv(camera); cam = world[ids] @ inv[:3, :3].T + inv[:3, 3]
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = obs["camera_intrinsic_Brown"]
        x, y = -cam[:, 1]/cam[:, 0], -cam[:, 2]/cam[:, 0]; r = x*x+y*y; radial = 1+k1*r+k2*r*r+k3*r*r*r
        uv = np.c_[fx*(x*radial+2*p1*x*y+p2*(r+2*x*x))+cx, fy*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+cy]
        im = np.array(Image.open(bank/obs["image_path"]).convert("RGB"))
        pixels = np.clip(uv.astype(int), [0, 0], [im.shape[1]-1, im.shape[0]-1])
        rgb[ids] = im[pixels[:, 1], pixels[:, 0]]/255; valid[ids] = True
        counts.append(dict(raw=int(raw), frame_id=row["id"], model_time=row["model_time"], points=len(ids)))
    size = np.median(sizes, axis=0); out.mkdir(parents=True, exist_ok=False); point_path = out/"observations.npz"
    np.savez_compressed(point_path, observed_points_actor_native_m=xyz, size_lwh_m=size, raw_frame=raws,
                        sample_xyz_local_m=xyz, sample_raw_frame=raws, sample_rgb=rgb, sample_rgb_valid=valid,
                        sample_source_ply_row=np.arange(len(xyz)))
    norm = normalization_contract(size, read_json(pvd_base/"normalization_contract.json"))
    (out/"normalization_contract.json").write_text(json.dumps(norm, indent=2))
    contract = dict(points_npz=str(point_path), points_sha256=hashlib.sha256(point_path.read_bytes()).hexdigest(),
                    conditioning_count=200, scope="explicit actor/track real source measurements", track_id=track_id,
                    actor_id=actor_id, frames=counts, heldout_excluded=receipt.get("holdout_raws", []),
                    bank=str(bank), mask_root=str(mask_root), source_ply=str(source), pvd_base=str(pvd_base),
                    time_mapping="nearest actual receipt.frames.model_time; float32 storage error only")
    (out/"input_contract.json").write_text(json.dumps(contract, indent=2)); return contract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "bank", "mask-root", "pvd-base"): ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--track-id", required=True); ap.add_argument("--actor-id", type=int, required=True)
    ap.add_argument("--output", type=Path); ap.add_argument("--time-tolerance", type=float)
    args = ap.parse_args()
    result = prepare_vehicle_prior(args.root, args.bank, args.mask_root, args.track_id, args.actor_id,
                                   args.pvd_base, output=args.output, time_tolerance=args.time_tolerance)
    print(json.dumps(result))


if __name__ == "__main__": main()
