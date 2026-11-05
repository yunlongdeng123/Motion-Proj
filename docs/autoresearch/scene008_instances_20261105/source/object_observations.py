"""Scene008 CPU observation bank from raw LiDAR, reviewed FRONT masks and native cameras."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


CAMPAIGN = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
SCENE = "10094743350625019937_3420_000_3440_000"
RAW_RECORD = Path("/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/training") / (
    f"segment-{SCENE}_with_camera_labels.tfrecord"
)
RAW_READER = Path("/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101")
NATIVE = CAMPAIGN / "scene008_a37_only_B10k_20261103_r1/formal10k"
PREPARED = CAMPAIGN / (
    "nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137"
)
REVIEW = CAMPAIGN / "r15_gn11_stage1_v41_20261023_dense_r7/frames"
ROAD = CAMPAIGN / "scene008_a37_only_B10k_road_20261103_r3/inputs/views"
WORLD_BUNDLE = CAMPAIGN / "r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json"
PVD = CAMPAIGN / "scene008_a37_C2_20261103_r1/inputs/actor_surface_pvd_seed0.npz"
PVD_GEOMETRY = PVD.with_name("a6_geometry_minimal.npz")
TRACKS = {6: "7MxwK9zTRZ9g5lc_ZC0z6A", 37: "vBOasuXpjI7h7d-X_VEQdA"}
ACTORS = (6, 37)
RAW_IDS = tuple(range(89, 138))
SHAPE = (320, 480)
VOXEL_M = 0.05


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as contents:
        return dict(contents)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


class ObservationBank:
    """Fixed raw-indexed supervision and two object-local initialization assets."""

    def __init__(self, root):
        self.root = Path(root)
        self.receipt = read_json(self.root / "receipt.json")
        self.raw_ids = tuple(self.receipt["native_raw_ids"])
        if self.receipt["status"] != "PASS" or self.raw_ids != RAW_IDS or self.receipt["actors"] != list(ACTORS):
            raise ValueError("Invalid or incomplete Scene008 observation bank")

    def frame(self, raw):
        if raw not in self.raw_ids:
            raise KeyError(raw)
        return read_npz(self.root / "frames" / f"{raw:03d}.npz")

    def asset(self, actor):
        if actor not in ACTORS:
            raise KeyError(actor)
        return read_npz(self.root / f"a{actor}.npz")


def transform(xyz, matrix):
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def pose(box):
    c, s = np.cos(box.heading), np.sin(box.heading)
    mat = np.eye(4)
    mat[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    mat[:3, 3] = [box.center_x, box.center_y, box.center_z]
    return mat, np.array([box.length, box.width, box.height])


def projected_returns(world, K, world_to_camera):
    camera = transform(world, world_to_camera)
    depth = camera[:, 2]
    uv = (camera @ K.T)[:, :2] / np.maximum(depth[:, None], 0.01)
    x, y = np.floor(uv).astype(np.int32).T
    h, w = SHAPE
    valid = (depth > 0.1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return depth, np.clip(x, 0, w - 1), np.clip(y, 0, h - 1), valid


def nearest_map(depth, x, y, rows):
    result = np.full(SHAPE, np.inf, np.float32)
    np.minimum.at(result, (y[rows], x[rows]), depth[rows])
    return result


def box_rays(K, camera_from_world, world_from_local, lwh, pad_m):
    """Entry and exit in camera-z metres for rays through the annotated cuboid."""
    yy, xx = np.indices(SHAPE)
    rays = np.stack((xx + 0.5, yy + 0.5, np.ones_like(xx)), axis=-1) @ np.linalg.inv(K).T
    world_from_camera = np.linalg.inv(camera_from_world)
    local_from_camera = np.linalg.inv(world_from_local) @ world_from_camera
    local_rays = rays @ local_from_camera[:3, :3].T
    center = local_from_camera[:3, 3]
    half = lwh / 2 + pad_m
    parallel = np.abs(local_rays) < 1e-12
    safe = np.where(parallel, 1.0, local_rays)
    low = (-half - center) / safe
    high = (half - center) / safe
    enter = np.maximum(np.where(parallel, -np.inf, np.minimum(low, high)).max(axis=-1), 0.1)
    leave = np.where(parallel, np.inf, np.maximum(low, high)).min(axis=-1)
    intersects = (leave >= enter) & ~(parallel & (np.abs(center) > half)).any(axis=-1)
    return np.where(intersects, enter, np.inf), np.where(intersects, leave, np.inf)


def associations(lidar, labels, ids):
    """Associate returns to one candidate box, guarding against road and overlap."""
    boxes = {label.id: pose(label.box) for label in labels
             if min(label.box.length, label.box.width, label.box.height) > 0}
    result = []
    for track in ids:
        if track not in boxes:
            result.append(None)
            continue
        ego_from_local, size = boxes[track]
        center = ego_from_local[:3, 3]
        local = (lidar - center) @ ego_from_local[:3, :3]
        rows = np.flatnonzero((np.abs(local) <= size / 2 + 1e-5).all(axis=1)
                              & (local[:, 2] > -size[2] / 2 + 0.10))
        conflict = np.zeros(len(rows), bool)
        points = lidar[rows]
        for other, (other_pose, other_size) in boxes.items():
            if other == track:
                continue
            if np.linalg.norm(other_pose[:3, 3] - center) > np.linalg.norm((size + other_size) / 2 + .03):
                continue
            other_local = (points - other_pose[:3, 3]) @ other_pose[:3, :3]
            conflict |= (np.abs(other_local) <= other_size / 2 + 0.03).all(axis=1)
        rows = rows[~conflict]
        result.append((ego_from_local, size, rows, local[rows].astype(np.float32)))
    return result


def native_frame(raw, camera, native_time, ego, world, associated, model_from_world, scale):
    import cv2
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion, minimum_filter

    fid = f"0008{raw:03d}"
    calibration = dict((line.split(":", 1)[0], np.fromstring(line.split(":", 1)[1], sep=" "))
                       for line in (PREPARED / "calib" / f"{fid}.txt").read_text().splitlines() if ":" in line)
    camera_from_ego = np.eye(4)
    camera_from_ego[:3] = calibration["Tr_velo_to_cam_0"].reshape(3, 4)
    camera_from_world = camera_from_ego @ np.linalg.inv(ego)
    K = calibration["P0"].reshape(3, 4)[:, :3] / np.array([[4], [4], [1]])
    native_K = np.array([[camera["fx"] / 4, 0, camera["cx"] / 4],
                         [0, camera["fy"] / 4, camera["cy"] / 4], [0, 0, 1]])
    if not np.allclose(K, native_K, atol=1e-6, rtol=0):
        raise ValueError(f"raw{raw}: raw/native camera intrinsics disagree")
    model_from_camera = np.eye(4)
    model_from_camera[:3, :3] = camera["rotation"]
    model_from_camera[:3, 3] = camera["position"]
    expected_camera = model_from_world @ np.linalg.inv(camera_from_world)
    expected_camera[:3, :3] /= scale
    camera_error = np.max(np.abs(model_from_camera - expected_camera))
    if camera_error > 1e-5:
        raise ValueError(f"raw{raw}: raw/native camera transform mismatch {camera_error}")
    image = np.asarray(Image.open(PREPARED / "image/image_0" / f"{fid}.png").convert("RGB"))
    rgb = cv2.resize(image / 255.0, SHAPE[::-1]).astype(np.float32)
    with np.load(REVIEW / f"{fid}.npz", allow_pickle=False) as reference:
        labels = reference["actor_labels"]
    for group in (4, 5):
        mask_path = PREPARED / f"dynamic_actor_assets/actor_{group - 1}/body_masks" / f"{fid}.png"
        reviewed = np.asarray(Image.open(mask_path).resize(SHAPE[::-1], Image.Resampling.NEAREST)) > 0
        if not np.array_equal(reviewed, labels == group):
            raise ValueError(f"raw{raw}: reviewed mask mismatch for label {group}")
    masks = np.stack([labels == 4, labels == 5])
    body = np.stack([binary_erosion(mask, iterations=1) & ~binary_dilation(masks[1 - k], iterations=1)
                     for k, mask in enumerate(masks)])
    depth, x, y, projected = projected_returns(world, K, camera_from_world)
    full_depth = nearest_map(depth, x, y, np.flatnonzero(projected))
    nearest = minimum_filter(full_depth, size=3, mode="constant", cval=np.inf)
    colors, measurements, fronts, backs, exact_fronts = [], [], [], [], []
    poses = np.stack([ego @ item[0] for item in associated])
    sizes = np.stack([item[1] for item in associated])
    for k, (_, size, rows, _) in enumerate(associated):
        all_front, _ = box_rays(K, camera_from_world, poses[k], size, 0)
        front, back = box_rays(K, camera_from_world, poses[k], size, .18)
        image_xy = (y[rows], x[rows])
        known = projected[rows] & body[k][image_xy] & (depth[rows] <= nearest[image_xy] + .20)
        color = np.zeros((len(rows), 3), np.float32)
        color[known] = rgb[y[rows[known]], x[rows[known]]]
        colors.append((color, known))
        measurements.append(nearest_map(depth, x, y, rows[known]))
        fronts.append(front)
        backs.append(back)
        exact_fronts.append(all_front)
    fronts, backs, exact_fronts = np.stack(fronts), np.stack(backs), np.stack(exact_fronts)
    # A distant unassociated return cannot veto the reviewed silhouette. A strictly
    # nearer measured first return makes that pixel ambiguous.
    visible = body & ~(full_depth[None] + .35 < fronts)
    measured = np.stack(measurements)
    depth_valid = visible & np.isfinite(measured)
    road = read_npz(ROAD / f"{raw:03d}.npz")
    reviewed_target = masks.any(axis=0)
    if not np.array_equal(road["actor_labels"], labels):
        raise ValueError(f"raw{raw}: road label binding mismatch")
    visible_road = road["valid"] & ~reviewed_target & ~road["other_vehicle_guard"] & (labels == 0)
    repair_road = observed_road_edit(road)
    protect = road["protect_mask"] & ~binary_dilation(reviewed_target, iterations=2)
    protect |= (labels > 0) & ~reviewed_target & ~binary_dilation(reviewed_target, iterations=2)
    exact_hit = np.isfinite(exact_fronts)
    near_target = np.isfinite(fronts)
    # Use an actual reviewed foreground return for strict occlusion. Cuboid overlap
    # without measured depth order remains a separate, weaker flag.
    occluded = near_target & ~masks & body[::-1] & (measured[::-1] + .15 < fronts)
    collar = np.stack([binary_dilation(hit, iterations=6) for hit in near_target])
    missed_box = collar & ~near_target
    through_box = near_target & body[::-1] & (measured[::-1] > backs + .25) & np.isfinite(measured[::-1])
    possible_occluded = near_target & ~masks & body[::-1] & ~through_box
    free = (missed_box | through_box) & ~occluded & ~possible_occluded & ~np.stack(
        [binary_dilation(mask, iterations=2) for mask in masks])
    unknown = ~(visible | occluded | free)
    evidence = dict(raw=np.int16(raw), rgb=rgb, visible=visible, occluded=occluded, free=free, unknown=unknown,
                    possible_occluded=possible_occluded, visible_reviewed=masks, actor_labels=labels,
                    depth_m=np.where(depth_valid, measured, 0).astype(np.float32), depth_valid=depth_valid,
                    scene_depth_m=np.where(np.isfinite(full_depth), full_depth, 0).astype(np.float32),
                    box_near_m=np.where(near_target, fronts, 0).astype(np.float32),
                    box_far_m=np.where(near_target, backs, 0).astype(np.float32),
                    box_candidate=exact_hit, protect=protect, road=visible_road, road_edit=repair_road,
                    road_edit_rgb=road["reference"], road_edit_depth_m=road["depth_m"].astype(np.float32),
                    K=K, model_to_camera=np.linalg.inv(model_from_camera),
                    world_to_camera=camera_from_world, model_from_local=model_from_world[None] @ poses,
                    world_from_local=poses, box_lwh_m=sizes, model_time=np.float32(native_time))
    if (visible & occluded).any() or (visible & free).any() or (occluded & free).any():
        raise ValueError(f"raw{raw}: contradictory ray labels")
    return evidence, colors, float(camera_error)


def observed_road_edit(road):
    return (road["repair_mask"] & road["valid"] & ~road["inferred"] & ~road["inferred_fine"]
            & ~road["weak_measured"])


def fuse(samples):
    """Use one real return as each anchor and retain exact per-sample provenance."""
    xyz, rgb, valid, raw, row = (np.concatenate(values) for values in zip(*samples))
    voxels, group = np.unique(np.floor(xyz / VOXEL_M).astype(np.int32), axis=0, return_inverse=True)
    frame_pairs = np.unique(np.column_stack((group, raw)), axis=0)
    support = np.bincount(frame_pairs[:, 0], minlength=len(voxels))
    count = np.bincount(group, minlength=len(voxels))
    centroid = np.column_stack([np.bincount(group, weights=coord, minlength=len(voxels)) / count for coord in xyz.T])
    distances = np.linalg.norm(xyz - centroid[group], axis=1)
    order = np.lexsort((row, raw, distances, ~valid, group))
    start = np.r_[True, np.diff(group[order]) != 0]
    representatives = order[start]
    # A reviewed LiDAR association is sufficient. Geometry-only anchors need
    # support in multiple distinct raw frames after transformation to object space.
    accepted = valid[representatives] | (support >= 2)
    reps = representatives[accepted]
    if not len(reps) or not valid[reps].any():
        raise ValueError("No measured and reviewed anchors for object")
    anchor = np.full(len(voxels), -1, np.int32)
    anchor[accepted] = np.arange(len(reps), dtype=np.int32)
    return dict(xyz_local_m=xyz[reps], rgb=rgb[reps], rgb_valid=valid[reps],
                source_raw_frame=raw[reps], source_cloud_row=row[reps],
                source_frame_count=support[accepted].astype(np.int16), voxel_key=voxels[accepted],
                sample_xyz_local_m=xyz, sample_rgb=rgb, sample_rgb_valid=valid, sample_raw_frame=raw,
                sample_source_cloud_row=row, sample_anchor=anchor[group])


def build(output):
    # The decoder checks no CUDA/model framework imports. Keep that environment
    # change out of the import path used by the GPU fitting and evaluation CLIs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import cv2

    cv2.setNumThreads(1)
    sys.path.insert(0, str(RAW_READER))
    import waymo_numpy_cpu as decoder
    from fast_range_matrix import matrix

    decoder.matrix = matrix
    pb = decoder.protobuf()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    bundle = read_json(WORLD_BUNDLE)
    model_from_world = np.asarray(bundle["world_to_model"])
    scale = float(bundle["meter_to_model"])
    with np.load(PVD_GEOMETRY, allow_pickle=False) as pvd_geometry:
        pvd_poses = pvd_geometry["model_from_canonical"]
    cameras = {int(row["img_name"][-3:]): row for row in read_json(NATIVE / "cameras.json")}
    binding = read_json(NATIVE / "actual_scene_binding.json")
    expected_frames = [f"image_0/0008{raw:03d}" for raw in RAW_IDS]
    if binding["frame_ids"] != expected_frames or sorted(cameras) != list(RAW_IDS):
        raise ValueError("Unexpected native49 camera IDs")
    times = dict(zip(RAW_IDS, binding["timestamps"], strict=True))
    samples = [[], []]
    sizes = [[], []]
    frame_summary = {}
    point_counts = []
    track_frames = [[], []]
    for raw, payload, _ in decoder.records(RAW_RECORD, 198):
        frame = pb.Frame.FromString(payload)
        if frame.context.name != SCENE:
            raise ValueError(f"Wrong TFRecord scene at raw{raw}")
        ego = np.asarray(frame.pose.transform).reshape(4, 4)
        lidar, _ = decoder.lidar(frame, pb)
        xyz = lidar[:, :3].astype(np.float64)
        associated = associations(xyz, frame.laser_labels, tuple(TRACKS.values()))
        if raw in times:
            if any(item is None for item in associated):
                raise ValueError(f"Missing raw track at native49 raw{raw}")
            view, colors, camera_error = native_frame(raw, cameras[raw], times[raw], ego, transform(xyz, ego),
                                                      associated, model_from_world, scale)
            save(output / "frames" / f"{raw:03d}.npz", **view)
            stats = {key: np.count_nonzero(view[key], axis=(1, 2)).tolist()
                     for key in ("visible", "occluded", "possible_occluded", "free", "depth_valid")}
            pvd_pose_error = float(np.abs(view["model_from_local"][0] - pvd_poses[raw]).max())
            if pvd_pose_error > 1e-7:
                raise ValueError(f"raw{raw}: fixed PVD transform differs by {pvd_pose_error} model units")
            frame_summary[str(raw)] = dict(stats, camera_error=camera_error,
                                           pvd_pose_error=pvd_pose_error, protect=int(view["protect"].sum()),
                                           road=int(view["road"].sum()), road_edit=int(view["road_edit"].sum()))
        point_counts.append([0 if item is None else len(item[2]) for item in associated])
        for k, item in enumerate(associated):
            if item is None:
                continue
            _, size, rows, local = item
            track_frames[k].append(raw)
            sizes[k].append(size)
            rgb, known = colors[k] if raw in times else (np.zeros_like(local), np.zeros(len(local), bool))
            samples[k].append((local, rgb, known, np.full(len(rows), raw, np.int16), rows.astype(np.int32)))
        if raw % 25 == 0:
            print(f"raw {raw}: decoded and associated", flush=True)
    asset_stats = {}
    for k, actor in enumerate(ACTORS):
        asset = fuse(samples[k])
        asset["box_lwh_m"] = np.median(sizes[k], axis=0).astype(np.float32)
        asset["track_id"] = np.asarray(TRACKS[actor])
        path = output / f"a{actor}.npz"
        save(path, **asset)
        asset_stats[f"a{actor}"] = dict(anchors=len(asset["xyz_local_m"]), rgb_anchors=int(asset["rgb_valid"].sum()),
                                       samples=len(asset["sample_xyz_local_m"]), frames=len(track_frames[k]),
                                       ranges_m=[asset["xyz_local_m"].min(axis=0).tolist(),
                                                 asset["xyz_local_m"].max(axis=0).tolist()],
                                       sha256=sha256(path))
    if len(frame_summary) != len(RAW_IDS):
        raise ValueError("Missing native49 observations")
    hashes = {path.name: sha256(path) for path in sorted((output / "frames").glob("*.npz"))}
    receipt = dict(status="PASS", schema=1, scene=SCENE, actors=list(ACTORS), native_raw_ids=list(RAW_IDS),
                   lidar_raw_ids=list(range(len(point_counts))), assets=asset_stats, frame_summary=frame_summary,
                   associated_points_by_raw=point_counts, frame_hashes=hashes, pvd_path=str(PVD), pvd_sha256=sha256(PVD),
                   source_record=str(RAW_RECORD), source_record_sha256=sha256(RAW_RECORD), camera_binding=str(NATIVE),
                   meter_to_model=scale, world_bundle_sha256=sha256(WORLD_BUNDLE), gpu_used=False,
                   source_sha256=sha256(Path(__file__)), pvd_geometry_sha256=sha256(PVD_GEOMETRY),
                   masks="Reviewed per-instance FRONT silhouettes in raw89–137; RGB from same-frame prepared raw FRONT.",
                   occlusion="Reviewed other actor with measured nearer camera-z; possible cuboid overlap is separate.",
                   free="Local collar outside padded 3D envelope or a reviewed return beyond the complete envelope.",
                   road="Fixed R3 road references. Edit evaluation excludes inferred, inferred_fine and weak_measured.",
                   geometry="Exact 5-LiDAR 2-return decode with CRC in raw0–198. Annotated OBB, bottom 10 cm guard, "
                            "and exclusion of other annotated boxes. Keep a real representative per 5 cm voxel with "
                            "reviewed native49 RGB evidence or support from multiple object-aligned raw frames. "
                            "Geometry-only points are associations, not per-point semantic ground truth.",
                   appearance="No RGB labels outside native49. Unknown surfaces and PVD-generated rows have no measured RGB.")
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(asset_stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    build(options.output)
