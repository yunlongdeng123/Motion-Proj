"""Read-only CPU adapter for exported Waymo observation banks.

Layout: observations/*.json, prompts/*.json, and asset paths explicitly stored
in each observation. A prep root containing inputs/ is also accepted. RGB stays
native. All actual in-image returns are retained, including multiple surfaces in
one source pixel. Depth/ray-entry use Waymo camera-forward X in metres; camera
axes are X forward, Y left, Z up. No other OBB is treated as an opaque surface.

API: load_waymo_evidence(bank_path, frame_key, target_track, ...). target_track is
a same-frame prompt/track mapping, or an opaque track ID resolved in that frame.
Complete caller-supplied geometry is accepted with its inferred/source metadata
retained. Missing tracks are never inferred here. Index arrays address returned
N rows; source_indices maps them back to original binary LiDAR rows. The sixth
column of the default six-float export is retained as encoded source_pixel_index
(TOP range-image pixel/return encoding; -1 for other sensors). It is not sensor ID.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from PIL import Image


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _se3(value, name):
    matrix = np.asarray(value, dtype=float).reshape(4, 4)
    r = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-5)):
        raise ValueError(f"{name} must be a finite rigid SE3")
    return matrix.copy()


@lru_cache(maxsize=8)
def _index(signature):
    index = {}
    for filename, _, _ in signature:
        path = Path(filename); obs = _read(path)
        keys = [path.stem, obs.get("raw"), obs.get("frame_key"), obs.get("id")]
        if obs.get("image_path"): keys.append(Path(obs["image_path"]).stem)
        for key in {str(k) for k in keys if k is not None}:
            index.setdefault(key, set()).add(filename)
    return index


def _observation_path(bank, frame_key, explicit):
    if explicit is not None:
        p = Path(explicit); return p if p.is_absolute() else bank / p
    key = str(frame_key)
    if Path(key).name != key or "/" in key or "\\" in key:
        raise ValueError("frame_key is an identifier, not a relative path")
    direct = bank / "observations" / (key + ".json")
    if direct.is_file() and (not isinstance(frame_key, int) or _read(direct).get("raw") == frame_key):
        return direct
    signature = tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)
                      for p in sorted((bank / "observations").glob("*.json")))
    matches = _index(signature).get(key, set())
    if len(matches) != 1:
        raise KeyError(f"frame_key {frame_key!r} has {len(matches)} observation matches")
    return Path(next(iter(matches)))


def _target(bank, obs_path, obs, value):
    target = {"track_id": value} if isinstance(value, str) else dict(value)
    track = target.get("track_id", target.get("track"))
    if not isinstance(track, str) or not track:
        raise ValueError("target_track requires an explicit nonempty track ID")
    if target.get("track_id", track) != target.get("track", track):
        raise ValueError("conflicting track identifiers")
    if "world_from_actor" not in target or "dimensions" not in target:
        prompt_path = bank / "prompts" / obs_path.name
        rows = _read(prompt_path) if prompt_path.is_file() else []
        matches = [row for row in rows if row.get("track_id", row.get("track")) == track]
        if not matches:
            matches = [row for row in obs.get("boxes", []) if row.get("track_id", row.get("track")) == track]
        if len(matches) != 1:
            raise KeyError("target has no unique same-frame geometry; supply explicit geometry")
        target = {**matches[0], **target}; origin = "same_frame"
    else:
        origin = "explicit"
    for key in ("raw", "timestamp_micros"):
        if key in target and key in obs and target[key] != obs[key]:
            raise ValueError("target geometry is bound to a different observation")
    pose = _se3(target["world_from_actor"], "world_from_actor")
    dimensions = np.asarray(target["dimensions"], dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("target dimensions must be finite positive length,width,height")
    return {**target, "track_id": track}, pose, dimensions, origin


def _ray_volume(rays, camera_to_world, world_from_actor, dimensions):
    local = np.linalg.inv(world_from_actor) @ camera_to_world
    direction = rays @ local[:3, :3].T; origin = local[:3, 3]; half = dimensions / 2
    parallel = abs(direction) < 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.divide(-half - origin, direction, out=np.full_like(direction, -np.inf), where=~parallel)
        b = np.divide(half - origin, direction, out=np.full_like(direction, np.inf), where=~parallel)
    near = np.maximum(np.minimum(a, b).max(1), 0.); far = np.maximum(a, b).min(1)
    hit = (np.isfinite(rays).all(1) & ~(parallel & (abs(origin) > half)).any(1)
           & (far >= near) & (far > 0))
    return np.where(hit, near, np.nan), np.where(hit, far, np.nan)


def own_volume_depth(evidence, output_shape, *, undistort_iterations=8):
    """Optional query-grid entry/exit depth; this is geometry, not visibility."""
    shape = tuple(output_shape)
    if len(shape) != 2 or any(isinstance(x, bool) or int(x) != x or x <= 0 for x in shape):
        raise ValueError("output_shape must contain positive integer height,width")
    shape = tuple(int(x) for x in shape)
    if type(undistort_iterations) is not int or undistort_iterations < 1:
        raise ValueError("undistort_iterations must be a positive integer")
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = evidence["intrinsics_brown"]
    yy, xx = np.indices(shape); h, w = evidence["image_shape"]
    xd = ((xx + .5) * w / shape[1] - cx) / fx; yd = ((yy + .5) * h / shape[0] - cy) / fy
    x, y = xd.copy(), yd.copy()
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for _ in range(undistort_iterations):
            r = x*x + y*y; radial = 1 + k1*r + k2*r*r + k3*r*r*r
            x, y = ((xd - 2*p1*x*y - p2*(r + 2*x*x)) / radial,
                    (yd - p1*(r + 2*y*y) - 2*p2*x*y) / radial)
    rays = np.stack((np.ones_like(x), -x, -y), axis=-1).reshape(-1, 3)
    near, far = _ray_volume(rays, evidence["camera_to_world"], evidence["world_from_actor"], evidence["dimensions"])
    return near.reshape(shape), far.reshape(shape)


def load_waymo_evidence(bank_path, frame_key, target_track, *, observation_path=None,
                        foreground_margin=0.5, output_shape=None, lidar_columns=6,
                        source_pixel_column=5, membership_tolerance=1e-6,
                        min_depth=1e-6, verify_timestamps=True):
    """Load source RGB/returns and target evidence; never deduplicate/project-zbuffer.

    ray_near/ray_far are N per-return exact-ray depths. foreground_indices and
    own_indices address returned projected_uv/depth; source_indices recovers raw
    file rows. Optional output_shape adds dense own_near/far and visibility_kwargs.
    Own membership supplies geometric source support, not optical visibility GT.
    """
    bank = Path(bank_path).resolve()
    if not (bank / "observations").is_dir() and (bank / "inputs" / "observations").is_dir(): bank /= "inputs"
    obs_path = _observation_path(bank, frame_key, observation_path); obs = _read(obs_path)
    target, actor, dimensions, origin = _target(bank, obs_path, obs, target_track)
    intrinsics = np.asarray(obs["camera_intrinsic_Brown"], dtype=float)
    if intrinsics.shape != (9,) or not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
        raise ValueError("native Brown intrinsics must be finite fx,fy,cx,cy,k1,k2,p1,p2,k3")
    size = np.asarray(obs["camera_size"])
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any() or (size != np.floor(size)).any():
        raise ValueError("camera_size must be native positive integer width,height")
    shape = (int(size[1]), int(size[0])); image_path = bank / obs["image_path"]; lidar_path = bank / obs["lidar_path"]
    with Image.open(image_path) as image: rgb = np.array(image.convert("RGB"))
    if rgb.shape != (*shape, 3): raise ValueError("RGB dimensions differ from native camera_size")
    if type(lidar_columns) is not int or lidar_columns < 3: raise ValueError("lidar_columns must be >=3")
    raw = np.fromfile(lidar_path, dtype="<f4")
    if raw.size % lidar_columns: raise ValueError("LiDAR bytes do not match declared float32 row width")
    raw = raw.reshape(-1, lidar_columns)
    if source_pixel_column is not None and (type(source_pixel_column) is not int or not 0 <= source_pixel_column < lidar_columns):
        raise ValueError("source_pixel_column must address the declared LiDAR row")
    if not all(np.isfinite(x) and x >= 0 for x in (foreground_margin, membership_tolerance, min_depth)):
        raise ValueError("depth margin/tolerances must be finite and nonnegative")
    ego = _se3(obs["ego_world"], "ego_world"); image_pose = _se3(obs["camera_pose_world"], "camera_pose_world")
    extrinsic = _se3(obs["camera_extrinsic"], "camera_extrinsic"); camera = image_pose @ extrinsic
    transform = np.linalg.inv(camera) @ ego; points = raw[:, :3].astype(float)
    pc = points @ transform[:3, :3].T + transform[:3, 3]; z = pc[:, 0]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x, y = -pc[:, 1] / z, -pc[:, 2] / z; r = x*x + y*y; radial = 1 + k1*r + k2*r*r + k3*r*r*r
        uv = np.c_[fx*(x*radial + 2*p1*x*y + p2*(r + 2*x*x)) + cx,
                   fy*(y*radial + p1*(r + 2*y*y) + 2*p2*x*y) + cy]
    valid = (np.isfinite(pc).all(1) & np.isfinite(uv).all(1) & (z > min_depth)
             & (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0]))
    indices = np.flatnonzero(valid); pc, uv, z = pc[valid], uv[valid], z[valid]
    local = np.linalg.inv(actor) @ ego; actor_points = points[valid] @ local[:3, :3].T + local[:3, 3]
    own = (abs(actor_points) <= dimensions/2 + membership_tolerance).all(1)
    near, far = _ray_volume(pc / z[:, None], camera, actor, dimensions)
    own_indices = np.flatnonzero(own); foreground = np.flatnonzero(~own & np.isfinite(near) & (z < near - foreground_margin))
    timestamp = obs.get("timestamp_micros"); records = []
    if verify_timestamps:
        timestamp_path = bank / "timestamp" / (Path(obs["image_path"]).stem + ".txt")
        if timestamp_path.is_file() and int(timestamp_path.read_text().strip()) != timestamp: raise ValueError("image timestamp mismatch")
        for name in ("receipt.json", "source_records.json"):
            path = bank / name
            if not path.is_file(): continue
            data = _read(path); rows = data.get("frames", []) if isinstance(data, dict) else data
            matches = [row for row in rows if row.get("raw") == obs.get("raw")]
            if len(matches) > 1: raise ValueError("duplicate source frame records")
            for row in matches:
                if row.get("timestamp_micros") != timestamp: raise ValueError("source timestamp mismatch")
                if "lidar_points" in row and row["lidar_points"] != len(raw): raise ValueError("LiDAR source count mismatch")
                records.append({"table": name, "record": row})
    evidence = {"rgb": rgb, "image_shape": shape, "intrinsics_brown": intrinsics.copy(),
                "intrinsics": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]),
                "projected_uv": uv, "depth": z, "own_source": own,
                "ray_near": near, "ray_far": far, "foreground_indices": foreground, "own_indices": own_indices,
                "foreground_uv": uv[foreground], "foreground_depth": z[foreground], "own_supported_uv": uv[own_indices],
                "source_indices": indices, "native_pixel_xy": np.floor(uv).astype(np.int64),
                "source_pixel_index": None if source_pixel_column is None else raw[valid, source_pixel_column].copy(),
                "lidar_rows": raw[valid].copy(), "source_total_points": len(raw), "camera_xyz": pc,
                "points_ego": points[valid], "actor_points": actor_points,
                "rejected_source_indices": np.flatnonzero(~valid), "camera_to_world": camera,
                "ego_to_world": ego, "camera_from_ego": transform, "world_from_actor": actor, "dimensions": dimensions.copy(),
                "target_track": target, "target_geometry_origin": origin, "frame_key": frame_key,
                "raw": obs.get("raw"), "timestamp_micros": timestamp,
                "source": {"bank": str(bank), "observation": str(obs_path.resolve()), "image": str(image_path.resolve()),
                           "lidar": str(lidar_path.resolve()), "lidar_dtype": "<f4", "lidar_columns": lidar_columns,
                           "source_pixel_column": source_pixel_column, "records": records,
                           "depth_convention": "camera_forward_X_metres", "projection_policy": "all_actual_returns_no_zbuffer_no_neighbor_min"}}
    if output_shape is not None:
        evidence["own_near"], evidence["own_far"] = own_volume_depth(evidence, output_shape)
        evidence["visibility_kwargs"] = {key: evidence[key] for key in ("projected_uv", "depth", "own_source", "image_shape", "own_near", "own_far")}
        evidence["visibility_kwargs"]["output_shape"] = tuple(output_shape)
    return evidence
