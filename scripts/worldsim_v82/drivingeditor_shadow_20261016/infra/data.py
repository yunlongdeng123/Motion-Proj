"""真实 Waymo 框/track 的 CPU 刚体实例输入；无分割、模型或伪造点云。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


WAYMO_LABEL_SOURCE = "https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto"
EXISTING_WINDOWS = {17: (0, 48), 8: (23, 71), 29: (69, 117)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_new(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rigid_matrix(value: Any, name: str = "pose") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.size == 16, f"{name}: real 4x4 pose required")
    matrix = matrix.reshape(4, 4)
    require(np.isfinite(matrix).all(), f"{name}: nonfinite matrix")
    require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0), f"{name}: invalid homogeneous row")
    rotation = matrix[:3, :3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0), f"{name}: rotation must be proper rigid")
    return matrix


def similarity_parts(value: Any) -> tuple[np.ndarray, float, np.ndarray]:
    """原 loader transform_matrix 的 sR/t；局部坐标缩放一次，pose 保持刚体。"""
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.size == 16, "normalization: real 4x4 transform required")
    matrix = matrix.reshape(4, 4)
    require(np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0),
            "normalization: invalid homogeneous transform")
    scale = float(np.linalg.norm(matrix[:3, :3], axis=0).mean())
    require(np.isfinite(scale) and scale > 0, "normalization: positive world scale required")
    rotation = matrix[:3, :3] / scale
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0),
            "normalization must be a uniform proper similarity, not shear or anisotropic scale")
    return rotation, scale, matrix[:3, 3].copy()


def transform_xyz(xyz: Any, matrix: Any) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), "Real finite Nx3 points required")
    value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return points @ value[:3, :3].T + value[:3, 3]


@dataclass(frozen=True)
class WaymoBox:
    raw_track_id: str
    center_vehicle_m: np.ndarray
    size_lwh_m: np.ndarray
    heading_rad: float
    source_label: Mapping[str, Any]

    @classmethod
    def from_label(cls, value: Mapping[str, Any]) -> "WaymoBox | None":
        """仅接受 laser_labels 的原始字段；无基于点云/2D 框估算的替代。"""
        label = value.get("source_label", value)
        require(isinstance(label, Mapping), "Missing real Waymo source_label")
        kind = label.get("type")
        require(kind is not None, "Missing genuine Waymo label type")
        if kind not in (1, "TYPE_VEHICLE"):
            return None
        track = label.get("id")
        require(isinstance(track, str) and bool(track.strip()), "Missing genuine Waymo raw track ID")
        if value.get("raw_track_id") is not None:
            require(value["raw_track_id"] == track, "Audit/raw label track ID mismatch")
        box = label.get("box")
        require(isinstance(box, Mapping), "Missing genuine Waymo 3D box; no point-cloud fallback")
        keys = ("center_x", "center_y", "center_z", "length", "width", "height", "heading")
        require(all(key in box for key in keys), "Incomplete genuine Waymo box center/dimensions/heading")
        numbers = np.array([box[key] for key in keys], dtype=np.float64)
        require(np.isfinite(numbers).all() and np.all(numbers[3:6] > 0), "Invalid real Waymo dimensions/heading")
        return cls(track, numbers[:3], numbers[3:6], float(numbers[6]), dict(label))

    @property
    def vehicle_from_local_m(self) -> np.ndarray:
        cosine, sine = np.cos(self.heading_rad), np.sin(self.heading_rad)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
        pose[:3, 3] = self.center_vehicle_m
        return pose

    def local_xyz_m(self, vehicle_xyz_m: Any) -> np.ndarray:
        xyz = np.asarray(vehicle_xyz_m, dtype=np.float64)
        return (xyz - self.center_vehicle_m) @ self.vehicle_from_local_m[:3, :3]

    def contains_vehicle(self, vehicle_xyz_m: Any, padding_m: float = 0) -> np.ndarray:
        require(np.isfinite(padding_m) and padding_m >= 0, "Box padding must be explicit finite nonnegative meters")
        return np.all(np.abs(self.local_xyz_m(vehicle_xyz_m)) <= self.size_lwh_m / 2 + padding_m, axis=1)

    def normalized_pose(self, world_from_vehicle: Any, normalization: Any) -> np.ndarray:
        world_from_local = rigid_matrix(world_from_vehicle) @ self.vehicle_from_local_m
        rotation, scale, shift = similarity_parts(normalization)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation @ world_from_local[:3, :3]
        pose[:3, 3] = scale * (rotation @ world_from_local[:3, 3]) + shift
        return pose


@dataclass(frozen=True)
class FrameAnnotations:
    frame_id: str
    source_index: int
    timestamp_micros: int
    model_time: float
    world_from_vehicle: np.ndarray
    boxes: tuple[WaymoBox, ...]
    annotation_source: str

    @classmethod
    def from_audit(cls, audit: Mapping[str, Any], row: Mapping[str, Any], pose: Any,
                   annotation_source: str) -> "FrameAnnotations":
        name = str(row["id"])
        require(audit.get("frame_id", name) == name, f"Raw/prepared frame mismatch: {name}")
        source_index = int(row["source_index"])
        if "source_index" in audit:
            require(audit["source_index"] == source_index, f"Raw source index mismatch: {name}")
        timestamp = audit.get("timestamp_micros", audit.get("frame_timestamp_micros"))
        require(type(timestamp) is int and timestamp > 0, f"Actual raw timestamp missing: {name}")
        if row.get("timestamp_micros") is not None:
            require(int(row["timestamp_micros"]) == timestamp, f"Raw/prepared timestamps differ: {name}")
        pose = rigid_matrix(pose, f"frame.pose/{name}")
        raw_pose = audit.get("frame_pose_world_from_vehicle")
        if raw_pose is not None:
            require(np.allclose(pose, rigid_matrix(raw_pose), atol=1e-7, rtol=0), f"Raw/prepared ego pose differs: {name}")
        require("laser_labels" in audit and isinstance(audit["laser_labels"], list),
                f"Real laser_labels absent: {name}; no synthetic boxes")
        boxes = tuple(box for item in audit["laser_labels"] if (box := WaymoBox.from_label(item)) is not None)
        require(len({box.raw_track_id for box in boxes}) == len(boxes), f"Duplicate raw track annotation: {name}")
        return cls(name, source_index, timestamp, float(row["model_time"]), pose, boxes, annotation_source)


def assign_lidar_to_boxes(vehicle_xyz_m: Any, boxes: Sequence[WaymoBox],
                          padding_m: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """-1=背景，-2=多框歧义；所有重叠点从每个实例拒绝，仍从背景排除。"""
    xyz = np.asarray(vehicle_xyz_m, dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all(), "Missing/invalid real LiDAR XYZ")
    owner = np.full(len(xyz), -1, dtype=np.int32)
    count = np.zeros(len(xyz), dtype=np.int32)
    for index, box in enumerate(boxes):
        inside = box.contains_vehicle(xyz, padding_m)
        first = inside & (count == 0)
        owner[first] = index
        count[inside] += 1
    owner[count > 1] = -2
    return owner, count > 0


def filter_static_seed(world_xyz_m: Any, model_times: Any, frames: Sequence[FrameAnnotations],
                       instance_ids: Sequence[str], padding_m: float = 0,
                       time_tolerance: float = 2e-6) -> tuple[np.ndarray, dict[str, Any]]:
    """按真实 seed 点的时间筛 OBB 并集，保留原采样/顺序，禁止 nearest-mask 替代。"""
    xyz = np.asarray(world_xyz_m, dtype=np.float64)
    times = np.asarray(model_times, dtype=np.float64).reshape(-1)
    require(xyz.shape == (len(times), 3) and len(xyz) > 0 and np.isfinite(xyz).all() and np.isfinite(times).all(),
            "Existing finite real static seed and per-point model timestamps required")
    require(bool(frames), "Actual frame annotations required for background exclusion")
    frame_times = np.array([frame.model_time for frame in frames])
    require(np.all(np.diff(frame_times) > 0), "Frame model times must be strictly increasing")
    nearest = np.searchsorted(frame_times, times)
    nearest = np.clip(nearest, 0, len(frames) - 1)
    previous = np.maximum(nearest - 1, 0)
    choose_previous = np.abs(times - frame_times[previous]) <= np.abs(times - frame_times[nearest])
    nearest[choose_previous] = previous[choose_previous]
    require(np.all(np.abs(times - frame_times[nearest]) <= time_tolerance),
            "Static seed point time has no exact source frame; refuse cross-frame or temporal-union fallback")
    selected = set(instance_ids)
    keep = np.ones(len(xyz), dtype=bool)
    counts = []
    for index, frame in enumerate(frames):
        point_indices = np.flatnonzero(nearest == index)
        boxes = [box for box in frame.boxes if box.raw_track_id in selected]
        if len(point_indices) and boxes:
            vehicle_xyz = transform_xyz(xyz[point_indices], np.linalg.inv(frame.world_from_vehicle))
            _, excluded = assign_lidar_to_boxes(vehicle_xyz, boxes, padding_m)
            keep[point_indices[excluded]] = False
        counts.append({"frame_id": frame.frame_id, "seed_points": int(len(point_indices)),
                       "excluded_vehicle_points": int(np.count_nonzero(~keep[point_indices]))})
    return keep, {"policy": "per-point source-time selected oriented-box union, including ambiguous overlaps",
                  "padding_m": float(padding_m), "input_points": int(len(xyz)), "retained_points": int(keep.sum()),
                  "excluded_points": int((~keep).sum()), "sampling_and_order_reused": True, "frames": counts}


def initial_scales(local_xyz: Any, fallback_m: float, model_scale: float) -> np.ndarray:
    """仅尺度参数在单点时使用显式先验；位置始终来自真实 LiDAR。"""
    xyz = np.asarray(local_xyz, dtype=np.float64)
    require(fallback_m > 0 and np.isfinite(fallback_m), "Scale initialization prior must be explicit positive meters")
    if len(xyz) < 2:
        return np.full((len(xyz), 3), fallback_m * model_scale, dtype=np.float32)
    from scipy.spatial import cKDTree
    distances, _ = cKDTree(xyz).query(xyz, k=min(4, len(xyz)), workers=1)
    squared = np.mean(distances[:, 1:] ** 2, axis=1)
    distance = np.sqrt(np.maximum(squared, (1e-3 * model_scale) ** 2))
    return np.repeat(distance[:, None], 3, axis=1).astype(np.float32)


def make_instance_bundle(frames: Sequence[FrameAnnotations], vehicle_clouds: Sequence[np.ndarray],
                         normalization: Any, instance_ids: Sequence[str] | None = None,
                         point_rgb: Sequence[np.ndarray] | None = None,
                         point_color_known: Sequence[np.ndarray] | None = None,
                         mask_known: Any | None = None, front_visible: Any | None = None,
                         padding_m: float = 0, scale_prior_m: float = .02) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    require(len(frames) > 0 and len(frames) == len(vehicle_clouds), "Actual frame/cloud sequence missing")
    _, model_scale, _ = similarity_parts(normalization)
    model_times = np.array([frame.model_time for frame in frames], dtype=np.float64)
    physical_micros = np.array([frame.timestamp_micros for frame in frames], dtype=np.int64)
    require(np.all(np.diff(model_times) > 0) and np.all(np.diff(physical_micros) > 0), "Actual model/physical times must increase")
    available = sorted({box.raw_track_id for frame in frames for box in frame.boxes})
    explicit = instance_ids is not None
    ids = list(instance_ids) if explicit else available
    require(len(ids) > 0 and len(set(ids)) == len(ids) and all(item in available for item in ids),
            "Real selected vehicle tracks required; unknown IDs and placeholder actors refused")
    index_by_id = {track: index for index, track in enumerate(ids)}
    length, count = len(frames), len(ids)
    poses = np.broadcast_to(np.eye(4), (length, count, 4, 4)).copy()
    valid = np.zeros((length, count), dtype=bool)
    box_size = np.zeros((length, count, 3), dtype=np.float32)
    known = np.zeros((length, count), dtype=bool) if mask_known is None else np.asarray(mask_known, dtype=bool)
    visibility = np.full((length, count), -1, dtype=np.int8) if front_visible is None else np.asarray(front_visible, dtype=np.int8)
    require(known.shape == visibility.shape == (length, count) and np.isin(visibility, [-1, 0, 1]).all(),
            "Separate mask-known/FRONT-visibility table required")
    chunks = {track: {key: [] for key in ("xyz", "rgb", "color_known", "frame", "lidar")} for track in ids}
    frame_reports = []
    for frame_index, (frame, cloud) in enumerate(zip(frames, vehicle_clouds)):
        cloud = np.asarray(cloud)
        require(cloud.ndim == 2 and cloud.shape[1] >= 3 and len(cloud) > 0 and np.isfinite(cloud).all(),
                f"Missing/invalid real LiDAR: {frame.frame_id}; no fallback cloud")
        rgb = np.zeros((len(cloud), 3), dtype=np.float32) if point_rgb is None else np.asarray(point_rgb[frame_index], dtype=np.float32)
        color_known = np.zeros(len(cloud), dtype=bool) if point_color_known is None else np.asarray(point_color_known[frame_index], dtype=bool)
        require(rgb.shape == (len(cloud), 3) and color_known.shape == (len(cloud),)
                and np.isfinite(rgb).all() and np.all((rgb >= 0) & (rgb <= 1)), "Real projected RGB must be finite Nx3 in [0,1]")
        boxes = sorted((box for box in frame.boxes if box.raw_track_id in index_by_id), key=lambda box: index_by_id[box.raw_track_id])
        owner, vehicle_union = assign_lidar_to_boxes(cloud[:, :3], boxes, padding_m)
        for box_index, box in enumerate(boxes):
            k = index_by_id[box.raw_track_id]
            valid[frame_index, k] = True
            poses[frame_index, k] = box.normalized_pose(frame.world_from_vehicle, normalization)
            box_size[frame_index, k] = box.size_lwh_m * model_scale
            selected = np.flatnonzero(owner == box_index)
            instance = chunks[box.raw_track_id]
            instance["xyz"].append((box.local_xyz_m(cloud[selected, :3]) * model_scale).astype(np.float32))
            instance["rgb"].append(rgb[selected])
            instance["color_known"].append(color_known[selected])
            instance["frame"].append(np.full(len(selected), frame_index, dtype=np.int32))
            instance["lidar"].append(selected.astype(np.int32))
        frame_reports.append({"frame_id": frame.frame_id, "real_lidar_points": int(len(cloud)),
                              "vehicle_union_points": int(vehicle_union.sum()), "ambiguous_overlap_points_rejected": int((owner == -2).sum()),
                              "assigned_instance_points": int((owner >= 0).sum())})
    arrays = {key: [] for key in ("local_xyz", "rgb", "color_known", "scale", "point_ids", "source_frame_indices", "source_lidar_indices")}
    per_instance = []
    for k, track in enumerate(ids):
        instance = chunks[track]
        xyz = np.concatenate(instance["xyz"], axis=0)
        if explicit:
            require(len(xyz) > 0, f"Requested real vehicle track has no unambiguous LiDAR: {track}; no template fallback")
        per_instance.append(len(xyz))
        arrays["local_xyz"].append(xyz)
        arrays["rgb"].append(np.concatenate(instance["rgb"]))
        arrays["color_known"].append(np.concatenate(instance["color_known"]))
        arrays["scale"].append(initial_scales(xyz, scale_prior_m, model_scale))
        arrays["point_ids"].append(np.full(len(xyz), k, dtype=np.int32))
        arrays["source_frame_indices"].append(np.concatenate(instance["frame"]))
        arrays["source_lidar_indices"].append(np.concatenate(instance["lidar"]))
    require(sum(per_instance) > 0, "No genuine unambiguous vehicle LiDAR; geometry readiness refused")
    bundle = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    bundle.update(instance_ids=np.asarray(ids, dtype=str), timestamps=model_times,
                  physical_timestamps=physical_micros.astype(np.float64) / 1e6, raw_timestamp_micros=physical_micros,
                  source_indices=np.array([frame.source_index for frame in frames], dtype=np.int32),
                  frame_ids=np.array([frame.frame_id for frame in frames], dtype=str),
                  poses=poses, valid=valid, box_size=box_size, mask_known=known, front_visible=visibility,
                  points_per_instance=np.asarray(per_instance, dtype=np.int64), instance_initialized=np.asarray(per_instance) > 0,
                  normalization=np.asarray(normalization, dtype=np.float64).reshape(4, 4), world_scale=np.array(model_scale))
    trajectories = []
    for k, track in enumerate(ids):
        observations = []
        for t, frame in enumerate(frames):
            box = next((box for box in frame.boxes if box.raw_track_id == track), None)
            observations.append({"frame_id": frame.frame_id, "source_index": frame.source_index, "timestamp_micros": frame.timestamp_micros,
                                 "model_time": frame.model_time, "frame_valid": bool(valid[t, k]), "mask_known": bool(known[t, k]),
                                 "front_visibility": {-1: "unknown", 0: "not_visible", 1: "visible"}[int(visibility[t, k])],
                                 "pose_normalized_world_from_local": poses[t, k].tolist() if box else None,
                                 "box_size_lwh_m": box.size_lwh_m.tolist() if box else None,
                                 "source_label": dict(box.source_label) if box else None})
        trajectories.append({"instance_index": k, "raw_track_id": track, "real_lidar_points": int(per_instance[k]), "frames": observations})
    report = {"schema": "waymo_rigid_instances_v1", "instance_count": count, "real_instance_points": int(sum(per_instance)),
              "vehicle_selection": "all genuine TYPE_VEHICLE tracks, stationary included" if not explicit else "explicit genuine raw track IDs",
              "motion_filter": "none", "overlap_policy": "reject ambiguous point from all instances; exclude full box union from background",
              "box_padding_m": float(padding_m), "world_scale": model_scale,
              "units": {"local_xyz": "normalized model length", "poses": "rigid normalized world from scaled local; rotation unscaled",
                        "box_size": "normalized length in length,width,height order", "timestamps": "centered loader model times",
                        "physical_timestamps": "Waymo raw microseconds / 1e6", "raw_boxes": "vehicle-frame meters/radians"},
              "frame_valid_policy": "genuine laser annotation present; independent of FRONT visibility and mask completeness",
              "invalid_pose_policy": "identity storage sentinel; valid=false prevents rendering, no extrapolated trajectory",
              "scale_initializer": {"method": "sqrt mean of up to three real local nearest-neighbor squared distances", "singleton_prior_m": scale_prior_m,
                                    "min_neighbor_distance_m": .001},
              "zero_lidar_instances": [track for track, n in zip(ids, per_instance) if not n],
              "mask_known_pairs": int(known.sum()), "mask_complete_frames": int(np.all(known, axis=1).sum()),
              "mask_coverage_policy": "all selected track/frame 2D masks actually reviewed; absent laser annotations never imply known-empty FRONT mask",
              "waymo_box_semantics_source": WAYMO_LABEL_SOURCE, "frames": frame_reports, "trajectories": trajectories}
    return bundle, report


_PLY_DTYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
               "char": "i1", "int8": "i1", "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
               "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}


def read_seed_ply(path: str | Path) -> dict[str, np.ndarray]:
    """读取原 DrivingEditor PLY，不依赖 Open3D；保留 time/nx 与原颜色。"""
    path = Path(path)
    require(path.is_file(), "Existing real points3d.ply missing; seed regeneration refused")
    properties, count, current, format_name = [], None, None, None
    with path.open("rb") as stream:
        require(stream.readline().strip() == b"ply", "Invalid real static PLY header")
        while True:
            line = stream.readline()
            require(bool(line), "Incomplete real static PLY header")
            tokens = line.decode("ascii").strip().split()
            if not tokens or tokens[0] in ("comment", "obj_info"):
                continue
            if tokens[0] == "end_header":
                break
            if tokens[0] == "format":
                format_name = tokens[1]
            elif tokens[0] == "element":
                current = tokens[1]
                if current == "vertex":
                    count = int(tokens[2])
                elif count is None:
                    require(int(tokens[2]) == 0, "PLY vertices must be first nonempty element")
            elif tokens[0] == "property" and current == "vertex":
                require(len(tokens) == 3 and tokens[1] in _PLY_DTYPES, "Unsupported static PLY vertex property")
                properties.append((tokens[2], _PLY_DTYPES[tokens[1]]))
        require(count is not None and count > 0 and format_name in ("binary_little_endian", "ascii"), "Real static PLY vertices missing")
        dtype = np.dtype([(name, "<" + kind) for name, kind in properties])
        if format_name == "binary_little_endian":
            vertices = np.fromfile(stream, dtype=dtype, count=count)
            require(len(vertices) == count, "Static PLY vertex payload incomplete")
        else:
            raw = np.loadtxt(stream, max_rows=count, ndmin=2)
            require(raw.shape == (count, len(properties)), "Static ASCII PLY payload invalid")
            vertices = np.empty(count, dtype=dtype)
            for column, (name, _) in enumerate(properties):
                vertices[name] = raw[:, column]
    names = set(vertices.dtype.names)
    require({"x", "y", "z"}.issubset(names) and ("time" in names or "nx" in names), "PLY XYZ and real source model timestamps required")
    xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    times = vertices["time" if "time" in names else "nx"].astype(np.float64)
    if "time" in names and "nx" in names:
        require(np.allclose(vertices["time"], vertices["nx"], rtol=0, atol=1e-7), "Static PLY time/nx disagree")
    colors = np.column_stack([vertices[key] for key in ("red", "green", "blue")]).astype(np.float32) / 255 if {"red", "green", "blue"}.issubset(names) else np.zeros((count, 3), dtype=np.float32)
    require(np.isfinite(xyz).all() and np.isfinite(times).all(), "Nonfinite real static PLY")
    return {"xyz_world_m": xyz, "rgb": colors, "time": times}


def read_calibration(path: str | Path, camera_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = {line.split(":", 1)[0]: np.fromstring(line.split(":", 1)[1], sep=" ")
              for line in Path(path).read_text(encoding="utf-8").splitlines() if ":" in line}
    projection, transform = values.get(f"P{camera_index}"), values.get(f"Tr_velo_to_cam_{camera_index}")
    require(projection is not None and projection.size == 12 and transform is not None and transform.size == 12,
            "Actual named camera calibration missing")
    camera_from_vehicle = np.eye(4)
    camera_from_vehicle[:3] = transform.reshape(3, 4)
    require(np.isfinite(projection).all(), "Nonfinite actual camera intrinsics")
    return projection.reshape(3, 4), rigid_matrix(camera_from_vehicle, "Tr_velo_to_cam")


def project_point_rgb(cloud: Any, rgb: Any, projection: Any, camera_from_vehicle: Any,
                      occlusion_tolerance_m: float = .5) -> tuple[np.ndarray, np.ndarray]:
    """实际 FRONT 图像着色；使用原 CPU 适配的 2x2 深度可见性规则。"""
    xyz = np.asarray(cloud)[:, :3]
    image = np.asarray(rgb)
    require(image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8, "Actual uint8 RGB image required")
    camera = transform_xyz(xyz, camera_from_vehicle)
    positive = np.flatnonzero(camera[:, 2] > .01)
    output = np.zeros((len(xyz), 3), dtype=np.float32)
    known = np.zeros(len(xyz), dtype=bool)
    if not len(positive):
        return output, known
    uvw = np.column_stack((camera[positive], np.ones(len(positive)))) @ np.asarray(projection).reshape(3, 4).T
    require(np.all(uvw[:, 2] > 0), "Actual pinhole projection has inconsistent positive depth")
    uv = np.floor(uvw[:, :2] / uvw[:, 2:3] + .5).astype(np.int64)
    height, width = image.shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    selected, uv = positive[inside], uv[inside]
    if not len(selected):
        return output, known
    bins = (uv[:, 1] // 2) * ((width + 1) // 2) + uv[:, 0] // 2
    nearest = np.full(((height + 1) // 2) * ((width + 1) // 2), np.inf)
    np.minimum.at(nearest, bins, camera[selected, 2])
    visible = camera[selected, 2] <= nearest[bins] + occlusion_tolerance_m
    output[selected[visible]] = image[uv[visible, 1], uv[visible, 0]].astype(np.float32) / 255
    known[selected[visible]] = True
    return output, known


def read_prepared_manifest(scene: str | Path, strict_existing_scene: bool = True) -> dict[str, Any]:
    scene = Path(scene)
    path = scene / "preparation_manifest.json"
    require(path.is_file(), "Actual prepared scene manifest missing")
    manifest = read_json(path)
    rows = manifest.get("frames")
    require(isinstance(rows, list) and len(rows) >= 2, "Actual prepared frame rows missing")
    names = [row.get("id") for row in rows]
    require(all(isinstance(name, str) and Path(name).name == name and name.isalnum() for name in names)
            and len(set(names)) == len(names), "Prepared frame IDs invalid or duplicated")
    indices = [row.get("source_index") for row in rows]
    require(all(type(index) is int for index in indices) and indices == list(range(indices[0], indices[0] + len(indices))),
            "Actual contiguous raw source-frame mapping required")
    interval = manifest.get("model_frame_interval")
    require(isinstance(interval, (int, float)) and interval > 0, "Actual loader model_frame_interval missing")
    expected_times = np.arange(len(rows), dtype=np.float64) * interval - interval * (len(rows) - 1) / 2
    for row, model_time in zip(rows, expected_times):
        require(row.get("model_time") is not None and np.isclose(row["model_time"], model_time, atol=1e-12, rtol=0),
                "Prepared model timestamp differs from actual loader convention")
    if strict_existing_scene:
        scene_index = manifest.get("author_file_index", manifest.get("global_training_index"))
        require(scene_index in EXISTING_WINDOWS, "Only the three explicitly declared existing scenes are supported")
        require((indices[0], indices[-1]) == EXISTING_WINDOWS[scene_index] and len(rows) == 49 and interval == .02,
                "Exact existing scene raw49 window/model times required")
    require(manifest.get("camera", "FRONT") == "FRONT" and manifest.get("camera_index", 0) == 0, "Actual FRONT source required")
    return manifest


def load_frame_annotations(scene: Path, manifest: Mapping[str, Any], annotations: Path | None = None,
                           tfrecord: Path | None = None, pb_root: Path | None = None,
                           crc_library: Path | None = None, waymo_reader: Path | None = None) -> tuple[list[FrameAnnotations], dict[str, Any]]:
    """原始 TFRecord 只读 labels/pose；LiDAR 始终复用已准备的真实 Nx6 文件。"""
    require((annotations is None) != (tfrecord is None), "Provide exactly one real annotation source: audit path or TFRecord")
    rows = manifest["frames"]
    audits: dict[str, Mapping[str, Any]] = {}
    provenance: dict[str, Any]
    if annotations is not None:
        require(annotations.exists(), "Real annotation path missing")
        if annotations.is_dir():
            for row in rows:
                path = annotations / (row["id"] + ".json")
                require(path.is_file(), f"Genuine per-frame track/box audit missing: {path}")
                audits[row["id"]] = read_json(path)
        else:
            value = read_json(annotations)
            records = value.get("frames") if isinstance(value, Mapping) else value
            require(isinstance(records, list), "Consolidated actual box audit must contain frame list")
            require(all(isinstance(item, Mapping) and isinstance(item.get("frame_id"), str) for item in records), "Actual audit frame IDs missing")
            audits = {item["frame_id"]: item for item in records}
            require(len(audits) == len(records), "Duplicate actual audit frame IDs")
        require(set(audits) == {row["id"] for row in rows}, "Annotation/frame window differs from prepared input")
        provenance = {"kind": "real_waymo_laser_labels_audit", "path": str(annotations.resolve()), "raw_read": False}
    else:
        import importlib.util
        import sys
        from google.protobuf.json_format import MessageToDict
        require(tfrecord.is_file(), "Real raw TFRecord missing; no annotation fallback")
        declared_raw = manifest.get("raw_path")
        if declared_raw:
            require(tfrecord.name == Path(declared_raw).name, "Raw TFRecord identity differs from prepared manifest")
        expected_context = manifest.get("segment_context_name") or tfrecord.name.removeprefix("segment-").removesuffix("_with_camera_labels.tfrecord")
        helper = waymo_reader or (Path(__file__).resolve().parents[1] / "drivingeditor_debug/followup_20261013/red_preparation/waymo_numpy_cpu.py")
        require(helper.is_file(), "Existing pure protobuf/NumPy Waymo reader missing")
        spec = importlib.util.spec_from_file_location("waymo_instances_cpu_reader", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset_pb2 = module.protobuf(pb_root or module.PB_ROOT)
        by_index = {row["source_index"]: row for row in rows}
        context_name = None
        for source_index, payload, _ in module.records(tfrecord, rows[-1]["source_index"], crc_library or module.CRC_LIBRARY):
            if source_index not in by_index:
                continue
            frame = dataset_pb2.Frame.FromString(payload)
            if context_name is None:
                context_name = frame.context.name
            require(bool(context_name) and frame.context.name == context_name, "Actual raw segment context changed")
            require(context_name == expected_context, "Actual raw context differs from original prepared source identity")
            row = by_index[source_index]
            labels = [MessageToDict(label, preserving_proto_field_name=True) for label in frame.laser_labels]
            audits[row["id"]] = {"frame_id": row["id"], "source_index": source_index, "timestamp_micros": int(frame.timestamp_micros),
                                  "frame_pose_world_from_vehicle": list(frame.pose.transform), "laser_labels": labels}
        require(len(audits) == len(rows), "Actual raw did not cover the declared prepared frame window")
        require(not any(name == "torch" or name.startswith("tensorflow") for name in sys.modules), "CPU label reader imported a model framework")
        provenance = {"kind": "real_waymo_laser_labels_TFRecord", "path": str(tfrecord.resolve()), "context_name": context_name,
                      "raw_read": True, "LiDAR_reextracted": False, "reader": str(helper)}
    frames = []
    for row in rows:
        name = row["id"]
        pose_path, timestamp_path = scene / "pose" / (name + ".txt"), scene / "timestamp" / (name + ".txt")
        require(pose_path.is_file() and timestamp_path.is_file(), f"Actual prepared pose/raw timestamp missing: {name}")
        frame = FrameAnnotations.from_audit(audits[name], row, np.loadtxt(pose_path), str(annotations or tfrecord))
        require(int(timestamp_path.read_text(encoding="ascii").strip()) == frame.timestamp_micros,
                f"Actual prepared/raw file timestamp differs: {name}")
        frames.append(frame)
    require(np.all(np.diff([frame.timestamp_micros for frame in frames]) > 0), "Actual physical timestamps must strictly increase")
    return frames, provenance


def _actual_body_from_selection(mask_scene: Path, frame_id: str, labels: np.ndarray,
                                 row: Mapping[str, Any]) -> np.ndarray:
    """只解析已审查的真实 DEVA/SAM；GT 框不生成或修补 2D mask。"""
    require(row.get("reviewed") is True and bool(str(row.get("review_note") or "").strip()), "Actual per-frame mask review required")
    override = row.get("body_override")
    if override and override.get("visible_ownership_resolution"):
        resolution = override["visible_ownership_resolution"]
        require(resolution.get("root_visual_conflict_reviewed") is True
                and resolution.get("method") == "keep_original_foreground_DEVA_owner_only_on_overlap"
                and resolution.get("GT_boxes_used") is False,
                "Explicit visually checked categorical foreground ownership required")
        exclude_ids = resolution.get("foreground_DEVA_instance_ids")
        require(isinstance(exclude_ids, list) and exclude_ids
                and all(type(i) is int and i > 0 and np.any(labels == i) for i in exclude_ids),
                "Ownership must reference genuine existing foreground DEVA pixels")
        original_row = dict(row)
        original_row["body_override"] = {k: v for k, v in override.items() if k != "visible_ownership_resolution"}
        raw_body = _actual_body_from_selection(mask_scene, frame_id, labels, original_row)
        foreground = np.isin(labels, exclude_ids)
        removed = int(np.count_nonzero(raw_body & foreground))
        require(removed == resolution.get("removed_overlap_pixels"), "Recorded ownership overlap changed")
        # A recorded annotation composition: add no new positive pixel and
        # preserve raw SAM/DEVA files. This never changes any Gaussian order.
        return raw_body & ~foreground
    visibility, ids = row.get("visibility"), row.get("instance_ids")
    require(visibility in ("visible", "not_visible") and isinstance(ids, list) and len(ids) == len(set(ids)),
            "Reviewed mask requires explicit FRONT visibility and unique real DEVA IDs")
    require(all(type(item) is int and 1 <= item <= 65535 for item in ids), "Actual DEVA mask IDs invalid")
    override = row.get("body_override")
    if override is None:
        require(bool(ids) == (visibility == "visible"), "Actual mask IDs/visibility disagree")
        require(all(np.any(labels == item) for item in ids), f"Selected actual DEVA ID absent: {frame_id}")
        return np.isin(labels, ids)
    from PIL import Image
    require(visibility == "visible" and override.get("source") == "cached_SAM_vit_h_real_inference"
            and override.get("root_engineering_reviewed") is True and bool(str(override.get("review_note") or "").strip()),
            "SAM mask must be actual reviewed cached-model output")
    receipt_path, candidate_path = Path(override.get("execution_receipt", "")), Path(override.get("path", ""))
    require(receipt_path.is_file() and candidate_path.is_file(), "Actual SAM execution receipt/candidate missing")
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "REAL_SAM_CANDIDATES_EXPORTED_REVIEW_PENDING" and receipt.get("returncode") == 0
            and receipt.get("scene") == str(mask_scene) and receipt.get("model") == "cached_SAM_vit_h"
            and receipt.get("GT_boxes_used") is False and receipt.get("source_DEVA_modified") is False,
            "SAM evidence must show real inference without GT-box mask fabrication")
    actual_frames = [item for item in receipt.get("frames", []) if item.get("frame_id") == frame_id]
    require(len(actual_frames) == 1 and actual_frames[0].get("prompt", {}).get("GT_boxes_used") is False, "Actual SAM frame/prompt receipt differs")
    candidates = [item for item in actual_frames[0].get("model_output_candidates", [])
                  if item.get("candidate_index") == override.get("candidate_index") and item.get("path") == str(candidate_path)]
    require(len(candidates) == 1, "SAM mask must be exact real recorded model candidate")
    actual = np.array(Image.open(candidate_path))
    require(actual.shape == labels.shape and actual.dtype == np.uint8 and set(np.unique(actual)).issubset({0, 255})
            and bool(actual.any()) and int((actual > 0).sum()) == candidates[0].get("area"), "Actual SAM binary candidate pixels/area invalid")
    return actual > 0


def reviewed_mask_tables(scene: Path, frames: Sequence[FrameAnnotations], instance_ids: Sequence[str],
                          selection_path: Path | None, output_masks: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from PIL import Image
    known = np.zeros((len(frames), len(instance_ids)), dtype=bool)
    visible = np.full(known.shape, -1, dtype=np.int8)
    if selection_path is None:
        return known, visible, {"selection": None, "status": "unknown_all_pairs", "known_pairs": 0, "unknown_pairs": int(known.size),
                                "mask_generation": "none; GT boxes never used to fabricate 2D masks"}
    require(selection_path.is_file(), "Explicit reviewed mask selection path missing")
    selected = read_json(selection_path)
    require(selected.get("status") == "visually_reviewed" and isinstance(selected.get("actors"), list), "Actual visually reviewed track-to-mask selection required")
    require(selected.get("GT_boxes_used_to_create_masks", False) is False, "GT-box-fabricated 2D masks refused")
    mask_scene = Path(selected.get("scene", ""))
    require(mask_scene.is_dir(), "Recorded actual mask source scene is missing")
    tracks = {}
    frame_names = {frame.frame_id for frame in frames}
    for actor in selected["actors"]:
        track = actor.get("raw_track_id")
        require(isinstance(track, str) and track not in tracks, "Reviewed mask track identity missing/duplicated")
        rows = actor.get("frames")
        require(isinstance(rows, list) and all(isinstance(row, Mapping) and row.get("frame_id") in frame_names for row in rows),
                "Reviewed masks contain missing/wrong raw frame IDs")
        by_frame = {row["frame_id"]: row for row in rows}
        require(len(by_frame) == len(rows), "Duplicate reviewed mask frame")
        tracks[track] = by_frame
    output_masks.mkdir(parents=True, exist_ok=False)
    areas = []
    require(len(instance_ids) < 65535, "Categorical PNG instance table exceeds real uint16 labels")
    for t, frame in enumerate(frames):
        image_path = scene / "image/image_0" / (frame.frame_id + ".png")
        with Image.open(image_path) as image:
            shape = (image.height, image.width)
            source_rgb = np.array(image.convert("RGB")) if mask_scene.resolve() != scene.resolve() else None
        label_path = mask_scene / "mask/mask_0/Annotations" / (frame.frame_id + ".png")
        applicable = [(k, tracks[track][frame.frame_id]) for k, track in enumerate(instance_ids)
                      if track in tracks and frame.frame_id in tracks[track]]
        categorical = np.zeros(shape, dtype=np.uint16)
        if applicable:
            require(label_path.is_file(), f"Real DEVA PNG missing: {frame.frame_id}")
            labels = np.array(Image.open(label_path))
            require(labels.shape == shape and labels.dtype in (np.uint8, np.uint16, np.int32), "Actual DEVA label dimensions/type invalid")
            if source_rgb is not None:
                original_rgb = np.array(Image.open(mask_scene / "image/image_0" / (frame.frame_id + ".png")).convert("RGB"))
                require(np.array_equal(source_rgb, original_rgb), "Inherited reviewed masks require exact actual RGB equality for this frame")
            for k, row in applicable:
                body = _actual_body_from_selection(mask_scene, frame.frame_id, labels, row)
                require(not np.any(categorical[body]), f"Reviewed real instance masks overlap: {frame.frame_id}")
                categorical[body] = k + 1
                known[t, k] = True
                visible[t, k] = int(row["visibility"] == "visible")
        Image.fromarray(categorical, mode="I;16").save(output_masks / (frame.frame_id + ".png"))
        areas.append({"frame_id": frame.frame_id, "known_instances": int(known[t].sum()), "reviewed_body_pixels": int(np.count_nonzero(categorical))})
    return known, visible, {"selection": str(selection_path.resolve()), "source_scene": str(mask_scene), "status": "real_reviewed_partial_mapping",
                            "known_pairs": int(known.sum()), "unknown_pairs": int((~known).sum()), "categorical_label": "instance index + 1; zero=unassigned",
                            "zero_label_is_not_a_known_background_when_mapping_incomplete": True,
                            "mask_generation": "reviewed DEVA IDs / real reviewed SAM candidates; optional explicitly recorded 2D foreground ownership on overlap; no GT-box masks", "frames": areas}


def prepare_scene(scene: str | Path, output: str | Path, *, annotations: str | Path | None = None,
                  tfrecord: str | Path | None = None, pb_root: str | Path | None = None,
                  crc_library: str | Path | None = None, waymo_reader: str | Path | None = None, normalization_path: str | Path | None = None,
                  selection_path: str | Path | None = None, instance_ids: Sequence[str] | None = None,
                  padding_m: float = 0, scale_prior_m: float = .02,
                  strict_existing_scene: bool = True) -> dict[str, Any]:
    import sys
    import os
    import time
    from datetime import datetime, timezone
    from PIL import Image
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    scene, output = Path(scene).resolve(), Path(output).resolve()
    require(output != scene and scene not in output.parents, "Isolated new output must be outside existing prepared scene")
    require(not output.exists(), "No-clobber: isolated instance output already exists")
    require(not any(name == "torch" or name.startswith("tensorflow") for name in sys.modules), "CPU preparation must not import Torch/TF")
    manifest = read_prepared_manifest(scene, strict_existing_scene)
    normalize_path = Path(normalization_path) if normalization_path else scene / "transform_matrix.txt"
    require(normalize_path.is_file(), "Actual original loader normalization transform missing; do not assume scene scale")
    normalization = np.loadtxt(normalize_path)
    _, model_scale, _ = similarity_parts(normalization)
    frames, source = load_frame_annotations(scene, manifest, Path(annotations) if annotations else None,
                                             Path(tfrecord) if tfrecord else None, Path(pb_root) if pb_root else None,
                                             Path(crc_library) if crc_library else None, Path(waymo_reader) if waymo_reader else None)
    ids = list(instance_ids) if instance_ids is not None else sorted({box.raw_track_id for frame in frames for box in frame.boxes})
    require(ids, "Actual scene contains no annotated vehicles; no fake actor created")
    seed = read_seed_ply(scene / "points3d.ply")
    expected_points = manifest.get("num_points", 600000 if strict_existing_scene else len(seed["time"]))
    require(len(seed["time"]) == expected_points and (not strict_existing_scene or expected_points == 600000),
            "Actual unchanged 600k static seed required for these scenes")
    # 所有必需真实数据存在后，才创建隔离输出；失败保留 partial 便于审查。
    clouds, colors, color_known = [], [], []
    for frame in frames:
        cloud_path = scene / "velodyne" / (frame.frame_id + ".bin")
        image_path = scene / "image/image_0" / (frame.frame_id + ".png")
        require(cloud_path.is_file() and image_path.is_file(), f"Actual LiDAR/RGB missing: {frame.frame_id}")
        cloud = np.fromfile(cloud_path, dtype=np.float32)
        require(cloud.size > 0 and cloud.size % 6 == 0 and np.isfinite(cloud).all(), f"Actual Nx6 LiDAR invalid: {frame.frame_id}")
        cloud = cloud.reshape(-1, 6)
        projection, camera_from_vehicle = read_calibration(scene / "calib" / (frame.frame_id + ".txt"))
        rgb = np.array(Image.open(image_path).convert("RGB"))
        require(not strict_existing_scene or rgb.shape == (1280, 1920, 3), "Actual FRONT RGB dimensions differ from existing scene contract")
        measured_rgb, measured_known = project_point_rgb(cloud, rgb, projection, camera_from_vehicle)
        clouds.append(cloud)
        colors.append(measured_rgb)
        color_known.append(measured_known)
    output.mkdir(parents=True, exist_ok=False)
    masks_known, visibility, mask_report = reviewed_mask_tables(scene, frames, ids, Path(selection_path) if selection_path else None, output / "instance_masks")
    bundle, report = make_instance_bundle(frames, clouds, normalization, instance_ids, colors, color_known,
                                           masks_known, visibility, padding_m, scale_prior_m)
    keep, background_report = filter_static_seed(seed["xyz_world_m"], seed["time"], frames, ids, padding_m)
    require(keep.any(), "Box exclusion removed all genuine static seed points")
    np.savez_compressed(output / "instances.npz", **bundle)
    np.savez_compressed(output / "background.npz", xyz=transform_xyz(seed["xyz_world_m"][keep], normalization).astype(np.float32),
                        rgb=seed["rgb"][keep], time=seed["time"][keep].astype(np.float32), seed_indices=np.flatnonzero(keep).astype(np.int32))
    trajectories = report.pop("trajectories")
    write_json_new(output / "trajectories.json", {"schema": "waymo_rigid_instances_v1", "source_scene": str(scene), "annotation_source": source,
                                                "frames": [{"frame_id": frame.frame_id, "source_index": frame.source_index, "timestamp_micros": frame.timestamp_micros,
                                                            "model_time": frame.model_time, "frame_pose_world_from_vehicle": frame.world_from_vehicle.tolist()} for frame in frames],
                                                "trajectories": trajectories})
    report.update(source_scene=str(scene), output=str(output), annotation_source=source,
                  source_raw_window=[frames[0].source_index, frames[-1].source_index], original_normalization_path=str(normalize_path.resolve()),
                  masks=mask_report, background=background_report, model_interval=float(manifest["model_frame_interval"]),
                  actual_physical_interval_seconds_median=float(np.median(np.diff([frame.timestamp_micros for frame in frames])) / 1e6),
                  source_static_seed_modified=False, old_assets_modified=False, GPU_used=False,
                  Torch_imported=False, TF_imported=False, geometry_ready=True,
                  execution={"started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
                             "wall_seconds": time.monotonic() - started, "interpreter": sys.executable, "pid": os.getpid(),
                             "command": sys.argv, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                             "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
                  full_mask_supervision_ready=bool(np.all(masks_known)),
                  rgb_depth_available_independent_of_mask_mapping=True, training_started=False,
                  supervised_mask_guard="Only reviewed mask_known=true pairs; categorical zero is unknown for incomplete mapping; retain full in/out opacity on known masks")
    require(not any(name == "torch" or name.startswith("tensorflow") for name in sys.modules), "CPU preparation unexpectedly imported Torch/TF")
    write_json_new(output / "preparation_report.json", report)
    return report
