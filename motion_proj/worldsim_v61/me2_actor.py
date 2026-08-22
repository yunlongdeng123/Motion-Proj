"""WorldSim V6.1 ME-2：actor 控制输入、canonical asset 与占据评估。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from motion_proj.worldsim_v61.me1_oracle import obb_intersects
from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN, load_lidar


class ME2ActorError(RuntimeError):
    """ME-2 actor 输入或 geometry contract 失败。"""


def _box_corners(size_lwh: np.ndarray) -> np.ndarray:
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=np.float64)
    return signs * np.asarray(size_lwh, dtype=np.float64)[None, :] / 2.0


def actor_state(
    instances: Mapping[str, Any], actor_id: int, frame: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """读取一个 native actor 的 local-to-global pose、LWH 与 class。"""
    info = instances[str(int(actor_id))]
    annotations = info["frame_annotations"]
    frames = [int(value) for value in annotations["frame_idx"]]
    try:
        index = frames.index(int(frame))
    except ValueError as error:
        raise ME2ActorError(f"actor {actor_id} 在 frame {frame} 不活跃") from error
    return (
        np.asarray(annotations["obj_to_world"][index], dtype=np.float64),
        np.asarray(annotations["box_size"][index], dtype=np.float64),
        str(info["class_name"]),
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _inverse_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - matrix[:3, 3]) @ matrix[:3, :3]


def _fixed_sample(points: np.ndarray, count: int) -> np.ndarray:
    """按输入顺序均匀取固定数量；不足时确定性循环，不添加噪声。"""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ME2ActorError("actor control point 为空")
    if values.shape[0] >= int(count):
        indices = np.linspace(0, values.shape[0] - 1, num=int(count), dtype=np.int64)
    else:
        indices = np.arange(int(count), dtype=np.int64) % values.shape[0]
    return values[indices].astype(np.float32)


def _project_points(
    points_global: np.ndarray, t_global_camera: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = _inverse_transform_points(points_global, t_global_camera)
    valid = camera[:, 2] > 0.1
    pixels = np.full((camera.shape[0], 2), np.nan, dtype=np.float64)
    projected = camera[valid] @ intrinsics.T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def actor_projection_mask(
    pose: np.ndarray,
    size_lwh: np.ndarray,
    t_global_camera: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把 native oriented box 投到 camera，要求八角都在相机前方。"""
    corners_global = _transform_points(_box_corners(size_lwh), pose)
    pixels, valid = _project_points(corners_global, t_global_camera, intrinsics)
    if not np.all(valid):
        raise ME2ActorError("target actor box 有角点位于相机后方")
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if not np.any(mask):
        raise ME2ActorError("target actor 投影为空")
    return mask.astype(bool), pixels


def _write_actor_rgba(
    image_path: Path, mask: np.ndarray, output_path: Path, border_fraction: float
) -> dict[str, Any]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != mask.shape:
        raise ME2ActorError(f"image/mask shape 漂移: {image.shape[:2]} != {mask.shape}")
    ys, xs = np.nonzero(mask)
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    border = int(round(float(border_fraction) * span))
    left = max(0, int(xs.min()) - border)
    right = min(image.shape[1], int(xs.max()) + border + 1)
    top = max(0, int(ys.min()) - border)
    bottom = min(image.shape[0], int(ys.max()) + border + 1)
    rgba = np.concatenate((image, (mask[..., None] * 255).astype(np.uint8)), axis=2)
    crop = rgba[top:bottom, left:right]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop, mode="RGBA").save(output_path)
    return {
        "source_image": str(image_path),
        "source_shape_hw": [int(image.shape[0]), int(image.shape[1])],
        "crop_ltrb": [left, top, right, bottom],
        "alpha_pixels": int(mask.sum()),
        "crop_shape_hw": [int(crop.shape[0]), int(crop.shape[1])],
    }


def build_actor_controls(
    *,
    scene: str,
    scene_root: Path,
    target_frame: int,
    actor_id: int,
    source_frames: Iterable[int],
    method_grid_path: Path,
    output_dir: Path,
    camera_id: int,
    point_count: int,
    voxel_count: int,
    lidar_record_width: int,
    box_margin_m: float,
    crop_border_fraction: float,
) -> dict[str, Any]:
    """只用 raw image、O_method 与 native metadata 构造四臂输入。"""
    scene_root = scene_root.resolve()
    instances_path = scene_root / "instances/instances_info.json"
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    pose, size_lwh, class_name = actor_state(instances, actor_id, target_frame)
    image_path = scene_root / f"images/{int(target_frame):03d}_{int(camera_id)}.jpg"
    t_global_camera = np.loadtxt(
        scene_root / f"extrinsics/{int(target_frame):03d}_{int(camera_id)}.txt"
    )
    intrinsics_values = np.loadtxt(scene_root / f"intrinsics/{int(camera_id)}.txt").reshape(-1)
    fx, fy, cx, cy = intrinsics_values[:4]
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    with Image.open(image_path) as image:
        width, height = image.size
    native_mask, projected_pixels = actor_projection_mask(
        pose,
        size_lwh,
        t_global_camera,
        intrinsics,
        width=width,
        height=height,
    )
    image_audit = _write_actor_rgba(
        image_path,
        native_mask,
        output_dir / "actor_rgba.png",
        crop_border_fraction,
    )

    observed_local: list[np.ndarray] = []
    observed_counts: list[dict[str, int]] = []
    for source_frame in source_frames:
        source_pose, source_size, _ = actor_state(instances, actor_id, int(source_frame))
        lidar_points = load_lidar(
            scene_root / f"lidar/{int(source_frame):03d}.bin", lidar_record_width
        )
        t_global_lidar = np.loadtxt(scene_root / f"lidar_pose/{int(source_frame):03d}.txt")
        points_global = _transform_points(lidar_points, t_global_lidar)
        points_actor = _inverse_transform_points(points_global, source_pose)
        inside = np.all(
            np.abs(points_actor)
            <= source_size[None, :] / 2.0 + float(box_margin_m) + 1e-12,
            axis=1,
        )
        observed_local.append(points_actor[inside])
        observed_counts.append(
            {"frame": int(source_frame), "raw": int(lidar_points.shape[0]), "actor": int(inside.sum())}
        )
    observed = np.concatenate(observed_local, axis=0)
    canonical_half_span = float(np.max(size_lwh) / 2.0)
    # Native actor 是 x=length, y=width, z=height；Omni 的 bbox/point/voxel
    # 契约是 x=length, y=height, z=width，因此三类控制必须共用 LHW 轴序。
    observed_omni = observed[:, [0, 2, 1]]
    point_control = _fixed_sample(observed_omni / canonical_half_span, int(point_count))

    method_grid = np.load(method_grid_path, allow_pickle=False)
    actor_indices = np.asarray(method_grid["actor_voxel_indices"], dtype=np.int64)
    actor_ids = np.asarray(method_grid["actor_instance_ids"], dtype=np.int64)
    selected_indices = actor_indices[actor_ids == int(actor_id)]
    if selected_indices.shape[0] == 0:
        raise ME2ActorError(f"O_method 缺少 target actor {actor_id}")
    grid_origin = np.asarray(method_grid["grid_origin_m"], dtype=np.float64)
    voxel_size = float(method_grid["voxel_size_m"])
    points_target_lidar = grid_origin + (selected_indices.astype(np.float64) + 0.5) * voxel_size
    t_global_target_lidar = np.loadtxt(scene_root / f"lidar_pose/{int(target_frame):03d}.txt")
    points_global = _transform_points(points_target_lidar, t_global_target_lidar)
    points_actor = _inverse_transform_points(points_global, pose)
    points_omni = points_actor[:, [0, 2, 1]]
    voxel_control = _fixed_sample(points_omni / canonical_half_span, int(voxel_count))
    bbox_control = np.asarray(
        [size_lwh[0], size_lwh[2], size_lwh[1]], dtype=np.float32
    ) / float(np.max(size_lwh))

    controls_path = output_dir / "controls.npz"
    np.savez_compressed(
        controls_path,
        bbox=bbox_control.reshape(1, 3),
        point=point_control,
        voxel=voxel_control,
        native_size_lwh=size_lwh.astype(np.float64),
        native_actor_pose=pose.astype(np.float64),
        projected_box_pixels=projected_pixels.astype(np.float64),
    )
    return {
        "schema_version": "worldsim_v61.me2_actor_input.v1",
        "scene": scene,
        "target_frame": int(target_frame),
        "actor_id": int(actor_id),
        "class_name": class_name,
        "native_size_lwh": size_lwh.tolist(),
        "canonical_half_span_m": canonical_half_span,
        "method_source_frames": [int(value) for value in source_frames],
        "observed_point_counts": observed_counts,
        "observed_actor_point_count": int(observed.shape[0]),
        "point_control_count": int(point_control.shape[0]),
        "method_actor_voxel_count": int(selected_indices.shape[0]),
        "voxel_control_count": int(voxel_control.shape[0]),
        "bbox_control_lhw": bbox_control.tolist(),
        "omni_control_coordinate_order": "length_height_width",
        "image": image_audit,
        "actor_rgba_path": str(output_dir / "actor_rgba.png"),
        "controls_path": str(controls_path),
    }


def _deterministic_surface_points(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> np.ndarray:
    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ) / 2.0
    valid = areas > 0
    triangles = triangles[valid]
    areas = areas[valid]
    if triangles.shape[0] == 0 or not np.isfinite(areas).all():
        raise ME2ActorError("mesh 没有有限非退化 surface")
    rng = np.random.default_rng(int(seed))
    face_indices = rng.choice(triangles.shape[0], size=int(count), replace=True, p=areas / areas.sum())
    selected = triangles[face_indices]
    uv = rng.random((int(count), 2))
    root = np.sqrt(uv[:, :1])
    return (
        (1.0 - root) * selected[:, 0]
        + root * (1.0 - uv[:, 1:2]) * selected[:, 1]
        + root * uv[:, 1:2] * selected[:, 2]
    )


def canonicalize_mesh(
    mesh_path: Path, native_size_lwh: np.ndarray, surface_count: int, seed: int
) -> tuple[trimesh.Trimesh, np.ndarray, dict[str, Any]]:
    """按 aspect-only 轴置换与单一 uniform scale 对齐，不做 anisotropic warp。"""
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not vertices.size or not faces.size or not np.isfinite(vertices).all():
        raise ME2ActorError(f"mesh 为空或非有限: {mesh_path}")
    components = mesh.split(only_watertight=False)
    face_counts = [int(np.asarray(component.faces).shape[0]) for component in components]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    centered = vertices - center[None, :]
    extents = centered.max(axis=0) - centered.min(axis=0)
    if np.any(extents <= 0):
        raise ME2ActorError("mesh extent 退化")
    native = np.asarray(native_size_lwh, dtype=np.float64)
    candidates: list[tuple[float, tuple[int, int, int], float, np.ndarray]] = []
    for permutation in itertools.permutations(range(3)):
        permuted = extents[list(permutation)]
        log_ratios = np.log(native / permuted)
        scale = float(np.exp(log_ratios.mean()))
        ratios = permuted * scale / native
        cost = float(np.mean(np.square(np.log(ratios))))
        candidates.append((cost, permutation, scale, ratios))
    cost, permutation, scale, extent_ratios = min(candidates, key=lambda row: (row[0], row[1]))
    aligned_vertices = centered[:, list(permutation)] * scale
    aligned = trimesh.Trimesh(vertices=aligned_vertices, faces=faces, process=False)
    surface = _deterministic_surface_points(
        np.asarray(aligned.vertices), np.asarray(aligned.faces), int(surface_count), int(seed)
    )
    audit = {
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "mesh_watertight": bool(mesh.is_watertight),
        "connected_components": len(components),
        "largest_component_face_fraction": max(face_counts) / max(sum(face_counts), 1),
        "source_extents": extents.tolist(),
        "axis_permutation_to_lwh": list(permutation),
        "uniform_scale_m": scale,
        "canonical_extents_lwh_m": np.asarray(aligned.extents, dtype=float).tolist(),
        "canonical_extent_ratios": np.asarray(extent_ratios, dtype=float).tolist(),
        "aspect_log_mse": cost,
        "anisotropic_scale_used": False,
    }
    return aligned, surface, audit


def prepare_compiled_asset(
    *,
    mesh_path: Path,
    native_size_lwh: np.ndarray,
    surface_count: int,
    seed: int,
    mesh_output_path: Path,
    surface_output_path: Path,
    audit_output_path: Path,
    mean_rgb: np.ndarray,
) -> dict[str, Any]:
    """落盘 canonical mesh、surface Gaussians 与对齐审计。"""
    aligned, surface, audit = canonicalize_mesh(
        mesh_path, native_size_lwh, int(surface_count), int(seed)
    )
    mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
    surface_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(mesh_output_path)
    rgb = np.broadcast_to(
        np.asarray(mean_rgb, dtype=np.float32).reshape(1, 3), (surface.shape[0], 3)
    ).copy()
    np.savez_compressed(
        surface_output_path,
        positions_lwh_m=surface.astype(np.float32),
        rgb=rgb,
        opacity=np.ones((surface.shape[0], 1), dtype=np.float32),
        scale_m=np.full((surface.shape[0], 1), 0.1, dtype=np.float32),
    )
    audit_output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "canonical_mesh_path": str(mesh_output_path),
        "surface_gaussians_path": str(surface_output_path),
        "canonical_audit_path": str(audit_output_path),
        "surface_gaussian_count": int(surface.shape[0]),
        "mesh_audit": audit,
    }


def _grid_values(path: Path) -> dict[str, np.ndarray | float]:
    values = np.load(path, allow_pickle=False)
    semantics = np.asarray(values["static_semantics"], dtype=np.uint8).copy()
    actor_indices = np.asarray(values["actor_voxel_indices"], dtype=np.int64)
    actor_ids = np.asarray(values["actor_instance_ids"], dtype=np.int64)
    semantics[actor_indices[:, 0], actor_indices[:, 1], actor_indices[:, 2]] = OCCUPIED
    return {
        "semantics": semantics,
        "actor_indices": actor_indices,
        "actor_ids": actor_ids,
        "origin": np.asarray(values["grid_origin_m"], dtype=np.float64),
        "voxel_size": float(values["voxel_size_m"]),
    }


def _neighbor_support(semantics: np.ndarray, indices: np.ndarray) -> np.ndarray:
    supported = np.zeros(indices.shape[0], dtype=bool)
    shape = np.asarray(semantics.shape, dtype=np.int64)
    for delta in itertools.product((-1, 0, 1), repeat=3):
        query = indices + np.asarray(delta, dtype=np.int64)[None, :]
        valid = np.all((query >= 0) & (query < shape[None, :]), axis=1)
        if np.any(valid):
            q = query[valid]
            supported[valid] |= semantics[q[:, 0], q[:, 1], q[:, 2]] == OCCUPIED
    return supported


def _interpolate_pose(
    pose0: np.ndarray, pose1: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    interpolation = Slerp(
        [0.0, 1.0], Rotation.from_matrix([pose0[:3, :3], pose1[:3, :3]])
    )
    rotation = interpolation([float(alpha)]).as_matrix()[0]
    center = (1.0 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]
    return center, rotation


def swept_collision_audit(
    *,
    scene_root: Path,
    target_frame: int,
    actor_id: int,
    generated_size_lwh: np.ndarray,
    interpolation_samples: int,
    legal_pairs: set[frozenset[int]],
) -> dict[str, Any]:
    instances = json.loads(
        (scene_root / "instances/instances_info.json").read_text(encoding="utf-8")
    )
    frame_instances = json.loads(
        (scene_root / "instances/frame_instances.json").read_text(encoding="utf-8")
    )
    first, second = int(target_frame), int(target_frame) + 1
    pose0, _, _ = actor_state(instances, actor_id, first)
    pose1, _, _ = actor_state(instances, actor_id, second)
    others = sorted(
        (set(int(value) for value in frame_instances[str(first)])
        & set(int(value) for value in frame_instances[str(second)]))
        - {int(actor_id)}
    )
    collisions: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    pair_checks = 0
    for other_id in others:
        other_pose0, other_size0, _ = actor_state(instances, other_id, first)
        other_pose1, other_size1, _ = actor_state(instances, other_id, second)
        for alpha in np.linspace(0.0, 1.0, int(interpolation_samples)):
            center, rotation = _interpolate_pose(pose0, pose1, float(alpha))
            other_center, other_rotation = _interpolate_pose(
                other_pose0, other_pose1, float(alpha)
            )
            other_size = (1.0 - alpha) * other_size0 + alpha * other_size1
            pair_checks += 1
            if not obb_intersects(
                center,
                rotation,
                generated_size_lwh,
                other_center,
                other_rotation,
                other_size,
            ):
                continue
            row = {
                "actor_id": int(actor_id),
                "other_actor_id": int(other_id),
                "segment": [first, second],
                "alpha": float(alpha),
            }
            if frozenset((int(actor_id), int(other_id))) in legal_pairs:
                filtered.append({**row, "relation": "truck_trailer_hitch"})
            else:
                collisions.append(row)
            break
    return {
        "pair_checks": pair_checks,
        "legal_relation_filtered_contacts": filtered,
        "unfiltered_collisions": collisions,
        "passed": not collisions,
    }


def _silhouette_factors(
    surface_local: np.ndarray,
    actor_pose: np.ndarray,
    t_global_camera: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    native_width: int,
    native_height: int,
) -> dict[str, float | int]:
    global_points = _transform_points(surface_local, actor_pose)
    pixels, valid = _project_points(global_points, t_global_camera, intrinsics)
    pixels = pixels[valid]
    pixels[:, 0] *= mask.shape[1] / float(native_width)
    pixels[:, 1] *= mask.shape[0] / float(native_height)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < mask.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < mask.shape[0])
    )
    pixels = pixels[inside]
    silhouette = np.zeros(mask.shape, dtype=np.uint8)
    if pixels.shape[0] >= 3:
        hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
        cv2.fillConvexPoly(silhouette, hull, 1)
    silhouette = silhouette.astype(bool)
    intersection = int(np.count_nonzero(silhouette & mask))
    union = int(np.count_nonzero(silhouette | mask))
    return {
        "projected_surface_point_count": int(pixels.shape[0]),
        "silhouette_pixel_count": int(silhouette.sum()),
        "hole_coverage": intersection / max(int(mask.sum()), 1),
        "silhouette_precision": intersection / max(int(silhouette.sum()), 1),
        "silhouette_iou": intersection / max(union, 1),
    }


def evaluate_mesh(
    *,
    mesh_path: Path,
    scene_root: Path,
    scene: str,
    target_frame: int,
    actor_id: int,
    evidence_grid_path: Path,
    case_mask: np.ndarray,
    thresholds: Mapping[str, Any],
    interpolation_samples: int,
    legal_pairs: set[frozenset[int]],
    surface_count: int,
    seed: int,
    camera_id: int,
    prepared_asset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """对 method/eval tier 使用同一命名因子，不把结果压成 scalar。"""
    instances = json.loads(
        (scene_root / "instances/instances_info.json").read_text(encoding="utf-8")
    )
    actor_pose, native_size, class_name = actor_state(instances, actor_id, target_frame)
    if prepared_asset is None:
        _, surface_local, mesh_audit = canonicalize_mesh(
            mesh_path, native_size, int(surface_count), int(seed)
        )
    else:
        surface_values = np.load(prepared_asset["surface_gaussians_path"], allow_pickle=False)
        surface_local = np.asarray(surface_values["positions_lwh_m"], dtype=np.float64)
        mesh_audit = dict(prepared_asset["mesh_audit"])
    grid = _grid_values(evidence_grid_path)
    t_global_lidar = np.loadtxt(scene_root / f"lidar_pose/{int(target_frame):03d}.txt")
    points_global = _transform_points(surface_local, actor_pose)
    points_lidar = _inverse_transform_points(points_global, t_global_lidar)
    origin = np.asarray(grid["origin"], dtype=np.float64)
    voxel_size = float(grid["voxel_size"])
    indices = np.floor((points_lidar - origin[None, :]) / voxel_size).astype(np.int64)
    shape = np.asarray(np.asarray(grid["semantics"]).shape, dtype=np.int64)
    inside = np.all((indices >= 0) & (indices < shape[None, :]), axis=1)
    indices = np.unique(indices[inside], axis=0)
    if indices.shape[0] == 0:
        raise ME2ActorError("canonical mesh 没有落入 SceneIR-O grid")
    semantics = np.asarray(grid["semantics"], dtype=np.uint8)
    states = semantics[indices[:, 0], indices[:, 1], indices[:, 2]]
    support = _neighbor_support(semantics, indices)
    actor_indices = np.asarray(grid["actor_indices"], dtype=np.int64)
    actor_ids = np.asarray(grid["actor_ids"], dtype=np.int64)
    target_indices = actor_indices[actor_ids == int(actor_id)]
    candidate_set = {tuple(row) for row in indices.tolist()}
    covered_native = np.asarray(
        [
            any(
                tuple((row + np.asarray(delta, dtype=np.int64)).tolist()) in candidate_set
                for delta in itertools.product((-1, 0, 1), repeat=3)
            )
            for row in target_indices
        ],
        dtype=bool,
    )

    collision = swept_collision_audit(
        scene_root=scene_root,
        target_frame=int(target_frame),
        actor_id=int(actor_id),
        generated_size_lwh=np.asarray(mesh_audit["canonical_extents_lwh_m"], dtype=np.float64),
        interpolation_samples=int(interpolation_samples),
        legal_pairs=legal_pairs,
    )
    t_global_camera = np.loadtxt(
        scene_root / f"extrinsics/{int(target_frame):03d}_{int(camera_id)}.txt"
    )
    intrinsics_values = np.loadtxt(scene_root / f"intrinsics/{int(camera_id)}.txt").reshape(-1)
    fx, fy, cx, cy = intrinsics_values[:4]
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    with Image.open(scene_root / f"images/{int(target_frame):03d}_{int(camera_id)}.jpg") as image:
        native_width, native_height = image.size
    silhouette = _silhouette_factors(
        surface_local,
        actor_pose,
        t_global_camera,
        intrinsics,
        case_mask,
        native_width,
        native_height,
    )
    factors = {
        "candidate_voxel_count": int(indices.shape[0]),
        "candidate_inside_grid_fraction": float(np.mean(inside)),
        "free_space_conflict_count": int(np.count_nonzero(states == FREE)),
        "free_space_conflict": float(np.mean(states == FREE)),
        "unknown_fraction": float(np.mean(states == UNKNOWN)),
        "observed_surface_support": float(np.mean(support)),
        "native_actor_surface_coverage": float(np.mean(covered_native)) if covered_native.size else 0.0,
        "collision": collision,
        "ground_contact": "UNKNOWN",
        "ground_contact_reason": "no_independent_ground_plane_in_frozen_me2_evidence",
        "photo_semantic": {
            **silhouette,
            "semantic_class": class_name,
            "semantic_status": "UNKNOWN",
        },
        "mesh": mesh_audit,
    }
    extent_ratios = np.asarray(mesh_audit["canonical_extent_ratios"], dtype=np.float64)
    checks = {
        "nonempty_closed_surface_proxy": int(indices.shape[0]) > 0,
        "largest_component_not_catastrophic": mesh_audit["largest_component_face_fraction"]
        >= float(thresholds["minimum_largest_component_face_fraction"]),
        "extent_not_catastrophic": bool(
            np.all(extent_ratios >= float(thresholds["minimum_extent_ratio"]))
            and np.all(extent_ratios <= float(thresholds["maximum_extent_ratio"]))
        ),
        "free_space_conflict_zero": factors["free_space_conflict_count"]
        == int(thresholds["require_free_space_conflict_count"]),
        "observed_surface_supported": factors["observed_surface_support"]
        >= float(thresholds["minimum_observed_surface_support"]),
        "native_actor_coverage": factors["native_actor_surface_coverage"]
        >= float(thresholds["minimum_native_actor_surface_coverage"]),
        "swept_collision_zero": len(collision["unfiltered_collisions"])
        == int(thresholds["require_swept_collision_count"]),
        "ground_contact_pass_or_unknown": factors["ground_contact"]
        in set(thresholds["ground_contact_allowed"]),
        "photo_not_catastrophic": silhouette["hole_coverage"]
        >= float(thresholds["minimum_hole_coverage"]),
        "semantic_not_catastrophic": silhouette["silhouette_iou"]
        >= float(thresholds["minimum_silhouette_iou"]),
    }
    return {
        "schema_version": "worldsim_v61.me2_actor_geometry_factors.v1",
        "scene": scene,
        "target_frame": int(target_frame),
        "actor_id": int(actor_id),
        "factors": factors,
        "checks": checks,
        "passed": all(checks.values()),
    }
