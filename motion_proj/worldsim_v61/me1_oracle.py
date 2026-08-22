"""WorldSim V6.1 ME-1：Oracle Occupancy proposal 编译与独立评估。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from scipy.ndimage import maximum_filter
from scipy.spatial.transform import Rotation, Slerp

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN, sha256_file


TASK_ID = "WS-V61-ME1-ORACLE-OCC-PROPOSAL-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class ME1ExperimentError(RuntimeError):
    """ME-1 输入、编译或评估合同失败。"""


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://"):
        raise ME1ExperimentError("只接受 runs URI")
    relative = Path(uri.removeprefix("runs://"))
    if ".." in relative.parts:
        raise ME1ExperimentError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def _verify_files(root: Path, files: Mapping[str, str]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ME1ExperimentError(f"冻结源漂移: {path}")


def fuse_two_factors(left: str, right: str) -> str:
    """沿用 R10 的保守双因子融合。"""
    if left == right == "ACCEPT":
        return "ACCEPT"
    if left == right == "REJECT":
        return "REJECT"
    return "ABSTAIN"


def method_decision(photo: str, method_gate_passed: bool) -> str:
    """O2 只消费冻结外观因子与 O_method 编译门。"""
    if photo == "REJECT":
        return "REJECT"
    if photo == "ACCEPT" and method_gate_passed:
        return "ACCEPT"
    return "ABSTAIN"


def occupancy_gate(factors: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    """独立 O_eval 的命名 geometry factors，不压成 scalar。"""
    return bool(
        factors["free_space_conflict"] <= thresholds["maximum_free_space_conflict"]
        and factors["observed_surface_support"] >= thresholds["minimum_observed_surface_support"]
        and factors["unknown_volume_fraction"] <= thresholds["maximum_unknown_volume_fraction"]
        and factors["projected_surface_coverage"] >= thresholds["minimum_projected_surface_coverage"]
        and factors["method_eval_depth_overlap"] >= thresholds["minimum_method_eval_depth_overlap"]
        and factors["median_relative_depth_error"] <= thresholds["maximum_median_relative_depth_error"]
    )


def obb_intersects(
    center_a: np.ndarray,
    rotation_a: np.ndarray,
    size_a: np.ndarray,
    center_b: np.ndarray,
    rotation_b: np.ndarray,
    size_b: np.ndarray,
) -> bool:
    """用 15 个 separating axes 判断两个三维 oriented boxes 是否穿透。"""
    half_a = np.asarray(size_a, dtype=np.float64) / 2.0
    half_b = np.asarray(size_b, dtype=np.float64) / 2.0
    axes_a = np.asarray(rotation_a, dtype=np.float64)
    axes_b = np.asarray(rotation_b, dtype=np.float64)
    delta = np.asarray(center_b, dtype=np.float64) - np.asarray(center_a, dtype=np.float64)
    candidate_axes = [axes_a[:, index] for index in range(3)]
    candidate_axes.extend(axes_b[:, index] for index in range(3))
    candidate_axes.extend(
        np.cross(axes_a[:, left], axes_b[:, right])
        for left in range(3)
        for right in range(3)
    )
    contact_tolerance = 1e-9
    for raw_axis in candidate_axes:
        norm = float(np.linalg.norm(raw_axis))
        if norm <= 1e-10:
            continue
        axis = raw_axis / norm
        radius_a = float(np.dot(half_a, np.abs(axes_a.T @ axis)))
        radius_b = float(np.dot(half_b, np.abs(axes_b.T @ axis)))
        if abs(float(np.dot(delta, axis))) >= radius_a + radius_b - contact_tolerance:
            return False
    return True


def _load_grid(path: Path) -> dict[str, np.ndarray | float]:
    values = np.load(path, allow_pickle=False)
    semantics = np.asarray(values["static_semantics"], dtype=np.uint8).copy()
    actor_grid = np.zeros(semantics.shape, dtype=np.int32)
    actor_indices = np.asarray(values["actor_voxel_indices"], dtype=np.int64)
    actor_ids = np.asarray(values["actor_instance_ids"], dtype=np.int32)
    semantics[actor_indices[:, 0], actor_indices[:, 1], actor_indices[:, 2]] = OCCUPIED
    actor_grid[actor_indices[:, 0], actor_indices[:, 1], actor_indices[:, 2]] = actor_ids
    return {
        "semantics": semantics,
        "actor_grid": actor_grid,
        "origin": np.asarray(values["grid_origin_m"], dtype=np.float64),
        "voxel_size": float(values["voxel_size_m"]),
    }


def _camera_contract(
    scene_root: Path, frame: int, projection: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    camera_id = int(projection["camera_id"])
    t_global_lidar = np.loadtxt(scene_root / f"lidar_pose/{frame:03d}.txt")
    t_global_camera = np.loadtxt(scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt")
    t_lidar_camera = np.linalg.inv(t_global_lidar) @ t_global_camera
    values = np.loadtxt(scene_root / f"intrinsics/{camera_id}.txt").reshape(-1)
    fx, fy, cx, cy = values[:4]
    width, height = int(projection["width"]), int(projection["height"])
    scale_x = width / float(projection["native_width"])
    scale_y = height / float(projection["native_height"])
    intrinsics = np.asarray(
        [[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return t_lidar_camera, intrinsics


def _raycast(
    grid: Mapping[str, Any],
    t_lidar_camera: np.ndarray,
    intrinsics: np.ndarray,
    projection: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """在 GPU 上以半 voxel 步长抽取首个闭合 occupied cell。"""
    device = torch.device("cuda")
    height, width = int(projection["height"]), int(projection["width"])
    ys, xs = np.indices((height, width), dtype=np.float32)
    directions_camera = np.stack(
        (
            (xs - intrinsics[0, 2]) / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(xs),
        ),
        axis=-1,
    ).reshape(-1, 3)
    directions_lidar = directions_camera @ t_lidar_camera[:3, :3].T
    directions_lidar /= np.linalg.norm(directions_lidar, axis=1, keepdims=True)
    semantics = np.asarray(grid["semantics"], dtype=np.uint8)
    actor_grid = np.asarray(grid["actor_grid"], dtype=np.int32)
    semantics_gpu = torch.from_numpy(semantics.astype(np.int16)).to(device)
    origin_gpu = torch.as_tensor(grid["origin"], dtype=torch.float32, device=device)
    camera_origin_gpu = torch.as_tensor(t_lidar_camera[:3, 3], dtype=torch.float32, device=device)
    shape_gpu = torch.as_tensor(semantics.shape, dtype=torch.long, device=device)
    step = float(projection["step_m"])
    distances = torch.arange(
        float(projection["near_m"]), float(projection["far_m"]), step, device=device
    )
    depth = torch.full((height * width,), float("nan"), dtype=torch.float32, device=device)
    linear = torch.full((height * width,), -1, dtype=torch.long, device=device)
    batch_size = int(projection["ray_batch_size"])
    stride_yz = int(semantics.shape[1] * semantics.shape[2])
    stride_z = int(semantics.shape[2])
    for start in range(0, directions_lidar.shape[0], batch_size):
        directions = torch.as_tensor(
            directions_lidar[start : start + batch_size], dtype=torch.float32, device=device
        )
        points = camera_origin_gpu[None, None, :] + directions[:, None, :] * distances[None, :, None]
        indices = torch.floor(
            (points - origin_gpu[None, None, :]) / float(grid["voxel_size"])
        ).long()
        valid = torch.all((indices >= 0) & (indices < shape_gpu[None, None, :]), dim=-1)
        clipped = torch.minimum(
            torch.maximum(indices, torch.zeros_like(indices)), shape_gpu[None, None, :] - 1
        )
        occupied = valid & (
            semantics_gpu[clipped[..., 0], clipped[..., 1], clipped[..., 2]] == int(OCCUPIED)
        )
        has_hit = torch.any(occupied, dim=1)
        first = torch.argmax(occupied.to(torch.int8), dim=1)
        row = torch.arange(directions.shape[0], device=device)
        first_indices = clipped[row, first]
        local_depth = distances[first]
        local_linear = (
            first_indices[:, 0] * stride_yz + first_indices[:, 1] * stride_z + first_indices[:, 2]
        )
        local_depth[~has_hit] = float("nan")
        local_linear[~has_hit] = -1
        depth[start : start + directions.shape[0]] = local_depth
        linear[start : start + directions.shape[0]] = local_linear
    depth_cpu = depth.reshape(height, width).cpu().numpy()
    linear_cpu = linear.reshape(height, width).cpu().numpy()
    actor = np.zeros_like(linear_cpu, dtype=np.int32)
    valid = linear_cpu >= 0
    actor[valid] = actor_grid.reshape(-1)[linear_cpu[valid]]
    return {"depth_m": depth_cpu, "voxel_linear": linear_cpu, "actor_instance_id": actor}


def _actor_state(instances: Mapping[str, Any], actor_id: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
    info = instances[str(actor_id)]
    annotations = info["frame_annotations"]
    frames = [int(value) for value in annotations["frame_idx"]]
    index = frames.index(int(frame))
    pose = np.asarray(annotations["obj_to_world"][index], dtype=np.float64)
    size = np.asarray(annotations["box_size"][index], dtype=np.float64)
    return pose, size


def _interpolate_actor(
    instances: Mapping[str, Any], actor_id: int, first: int, second: int, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose0, size0 = _actor_state(instances, actor_id, first)
    pose1, size1 = _actor_state(instances, actor_id, second)
    interpolation = Slerp([0.0, 1.0], Rotation.from_matrix([pose0[:3, :3], pose1[:3, :3]]))
    rotation = interpolation([float(alpha)]).as_matrix()[0]
    center = (1.0 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]
    size = (1.0 - alpha) * size0 + alpha * size1
    return center, rotation, size


def _actor_motion_contract(
    scene_root: Path, target_frame: int, actor_ids: list[int], interpolation_samples: int
) -> dict[str, Any]:
    instances = json.loads((scene_root / "instances/instances_info.json").read_text(encoding="utf-8"))
    frame_instances = json.loads((scene_root / "instances/frame_instances.json").read_text(encoding="utf-8"))
    lifecycle_exact = True
    rigid = True
    canonical_size_exact = True
    segments: dict[int, tuple[int, int]] = {}
    for actor_id in actor_ids:
        annotations = instances[str(actor_id)]["frame_annotations"]
        frames = [int(value) for value in annotations["frame_idx"]]
        registered = sorted(int(frame) for frame, values in frame_instances.items() if actor_id in [int(v) for v in values])
        lifecycle_exact &= frames == registered
        poses = [np.asarray(value, dtype=np.float64) for value in annotations["obj_to_world"]]
        rigid &= all(
            np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0)
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0.0)
            and np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6, rtol=0.0)
            for pose in poses
        )
        sizes = np.asarray(annotations["box_size"], dtype=np.float64)
        canonical_size_exact &= bool(np.allclose(sizes, sizes[0], atol=1e-9, rtol=0.0))
        if target_frame + 1 in frames:
            segments[actor_id] = (target_frame, target_frame + 1)
        elif target_frame - 1 in frames:
            segments[actor_id] = (target_frame - 1, target_frame)

    collisions: list[dict[str, Any]] = []
    pair_checks = 0
    alphas = np.linspace(0.0, 1.0, int(interpolation_samples))
    for actor_id, (first, second) in sorted(segments.items()):
        other_ids = sorted(set(int(value) for value in frame_instances[str(first)]) & set(int(value) for value in frame_instances[str(second)]))
        for other_id in other_ids:
            if other_id == actor_id:
                continue
            for alpha in alphas:
                left = _interpolate_actor(instances, actor_id, first, second, float(alpha))
                right = _interpolate_actor(instances, other_id, first, second, float(alpha))
                pair_checks += 1
                if obb_intersects(*left, *right):
                    collisions.append(
                        {
                            "actor_id": actor_id,
                            "other_actor_id": other_id,
                            "segment": [first, second],
                            "alpha": float(alpha),
                        }
                    )
                    break
    return {
        "actor_instance_ids": actor_ids,
        "native_lifecycle_exact": bool(lifecycle_exact),
        "native_poses_rigid": bool(rigid),
        "canonical_size_exact": bool(canonical_size_exact),
        "trajectory_segment_count": len(segments),
        "all_actors_have_trajectory_segment": len(segments) == len(actor_ids),
        "swept_interpolation_samples": int(interpolation_samples),
        "actor_actor_pair_checks": pair_checks,
        "actor_actor_swept_collisions": collisions,
        "passed": bool(
            actor_ids
            and lifecycle_exact
            and rigid
            and canonical_size_exact
            and len(segments) == len(actor_ids)
            and not collisions
        ),
    }


def _method_factors(
    mask: np.ndarray,
    raycast: Mapping[str, np.ndarray],
    case: Mapping[str, Any],
    minimum_coverage: float,
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    linear = np.asarray(raycast["voxel_linear"], dtype=np.int64)
    valid = linear >= 0
    candidates = np.unique(linear[mask & valid])
    actor_ids = sorted(
        int(value)
        for value in np.unique(np.asarray(raycast["actor_instance_id"], dtype=np.int32)[mask & valid])
        if int(value) > 0
    )
    coverage = float(np.mean(valid[mask]))
    needs_actor = case["hole_type"] == "actor_removal_hole"
    factors = {
        "projected_surface_coverage": coverage,
        "candidate_occupied_voxel_count": int(candidates.size),
        "free_space_conflict": 0.0,
        "unknown_volume_fraction": 0.0,
        "occupied_surface_support": 1.0 if candidates.size else 0.0,
        "collision_body_closure": bool(candidates.size),
        "depth_consistency": 1.0 if candidates.size else 0.0,
        "native_actor_identity_count": len(actor_ids),
        "actor_identity_required": needs_actor,
    }
    factors["passed"] = bool(
        candidates.size
        and coverage >= minimum_coverage
        and (not needs_actor or actor_ids)
        and factors["collision_body_closure"]
    )
    return factors, candidates, actor_ids


def _independent_eval_factors(
    mask: np.ndarray,
    candidate_linear: np.ndarray,
    method_raycast: Mapping[str, np.ndarray],
    eval_grid: Mapping[str, Any],
    eval_raycast: Mapping[str, np.ndarray],
    dilation_voxels: int,
) -> dict[str, Any]:
    semantics = np.asarray(eval_grid["semantics"], dtype=np.uint8)
    flat = semantics.reshape(-1)
    candidate_labels = flat[candidate_linear]
    support = maximum_filter(
        semantics == OCCUPIED, size=2 * int(dilation_voxels) + 1, mode="constant"
    ).reshape(-1)
    method_depth = np.asarray(method_raycast["depth_m"], dtype=np.float32)
    eval_depth = np.asarray(eval_raycast["depth_m"], dtype=np.float32)
    method_valid = np.isfinite(method_depth)
    eval_valid = np.isfinite(eval_depth)
    overlap = mask & method_valid & eval_valid
    relative = np.abs(method_depth[overlap] - eval_depth[overlap]) / np.maximum(eval_depth[overlap], 1.0)
    return {
        "free_space_conflict": float(np.mean(candidate_labels == FREE)) if candidate_labels.size else 1.0,
        "observed_surface_support": float(np.mean(support[candidate_linear])) if candidate_linear.size else 0.0,
        "unknown_volume_fraction": float(np.mean(candidate_labels == UNKNOWN)) if candidate_labels.size else 1.0,
        "projected_surface_coverage": float(np.mean(eval_valid[mask])),
        "method_eval_depth_overlap": float(np.mean((method_valid & eval_valid)[mask])),
        "median_relative_depth_error": float(np.median(relative)) if relative.size else math.inf,
        "candidate_voxel_count": int(candidate_linear.size),
        "eval_evidence_identity": "O_eval_hidden_disjoint_raw_lidar_plus_native_actor_contract",
    }


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    decisions = [row["decisions"][arm] for row in rows]
    accepted = [row for row in rows if row["decisions"][arm] == "ACCEPT"]
    return {
        "schema_version": "worldsim_v61.me1_arm_summary.v1",
        "arm": arm,
        "denominator": len(rows),
        "accept_count": decisions.count("ACCEPT"),
        "abstain_count": decisions.count("ABSTAIN"),
        "reject_count": decisions.count("REJECT"),
        "false_safe_count": sum(bool(row["false_safe"][arm]) for row in rows),
        "accepted_mask_pixels": sum(int(row["mask_pixel_count"]) for row in accepted),
        "accepted_case_ids": [row["case_id"] for row in accepted],
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise ME1ExperimentError("正式 ME-1 run 要求干净工作树")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise ME1ExperimentError("ME-1 task_id 漂移")
    if not torch.cuda.is_available():
        raise ME1ExperimentError("ME-1 GPU ray compiler 需要 CUDA")

    me0_root = _resolve_runs_uri(config["sources"]["me0_run"])
    b0_root = Path(config["sources"]["b0_2d_run"])
    r9_root = Path(config["sources"]["r9_cross_run"])
    r10_root = Path(config["sources"]["r10_run"])
    _verify_files(me0_root, config["sources"]["me0_files"])
    _verify_files(b0_root, config["sources"]["b0_2d_files"])
    _verify_files(r9_root, config["sources"]["r9_cross_files"])
    _verify_files(r10_root, config["sources"]["r10_files"])
    if not json.loads((me0_root / "ME0_GATE.json").read_text(encoding="utf-8"))["passed"]:
        raise ME1ExperimentError("ME-0 未通过")

    cases = _read_jsonl(r9_root / "CASES.jsonl")
    if len(cases) != int(config["cohort"]["expected_case_count"]):
        raise ME1ExperimentError("28-case denominator 漂移")
    b0_rows = {row["case_id"]: row for row in _read_jsonl(b0_root / "verifier_worker/PER_CASE_ARMS.jsonl")}
    r9_rows = {row["case_id"]: row for row in _read_jsonl(r9_root / "verifier_worker/PER_CASE_ARMS.jsonl")}
    r10_rows = {row["case_id"]: row for row in _read_jsonl(r10_root / "FACTORIZED_DECISIONS.jsonl")}
    case_ids = {row["case_id"] for row in cases}
    if case_ids != set(b0_rows) or case_ids != set(r9_rows) or case_ids != set(r10_rows):
        raise ME1ExperimentError("baseline case identity 漂移")

    run_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__oracle-occ-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.cuda.set_device(int(config["resources"]["gpu"]))
        torch.cuda.reset_peak_memory_stats()
        projection_index: list[dict[str, Any]] = []
        method_rays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        method_grids: dict[tuple[str, int], dict[str, Any]] = {}
        unit_keys = sorted({(row["scene"], int(row["frame_index"])) for row in cases})

        # 第一阶段只读取 O_method：编译决策在任何 O_eval tensor 进入内存前固化。
        for scene, frame in unit_keys:
            scene_root = Path(config["raw_evidence"][scene])
            t_lidar_camera, intrinsics = _camera_contract(scene_root, frame, config["projection"])
            grid = _load_grid(me0_root / f"evidence/{scene}/f{frame:03d}/O_method.npz")
            rays = _raycast(grid, t_lidar_camera, intrinsics, config["projection"])
            method_grids[(scene, frame)] = grid
            method_rays[(scene, frame)] = rays
            relative = Path(f"projections/{scene}/f{frame:03d}/O_method.npz")
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **rays)
            projection_index.append(
                {
                    "scene": scene,
                    "frame_index": frame,
                    "tier": "O_method",
                    "path": str(relative),
                    "sha256": sha256_file(path),
                    "visible_surface_pixels": int(np.sum(rays["voxel_linear"] >= 0)),
                }
            )

        method_rows: list[dict[str, Any]] = []
        proposal_root = run_dir / "compiled_proposals"
        proposal_root.mkdir()
        motion_cache: dict[tuple[str, int, tuple[int, ...]], dict[str, Any]] = {}
        for case in cases:
            case_id = case["case_id"]
            key = (case["scene"], int(case["frame_index"]))
            payload = np.load(r9_root / "verifier_inputs" / f"{case_id}.npz", allow_pickle=False)
            mask = np.asarray(payload["mask"], dtype=bool)
            factors, candidate_linear, actor_ids = _method_factors(
                mask,
                method_rays[key],
                case,
                float(config["method_gate"]["minimum_surface_coverage"]),
            )
            proposal_path = r9_root / "cross_frontend_reconstruction_proposals" / f"{case_id}__repeat1.npy"
            proposal = np.load(proposal_path, allow_pickle=False)
            hit_linear = method_rays[key]["voxel_linear"]
            selected = mask & (hit_linear >= 0)
            selected_linear = hit_linear[selected]
            selected_rgb = proposal[selected]
            order = np.argsort(selected_linear, kind="stable")
            sorted_linear = selected_linear[order]
            first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]] if sorted_linear.size else np.asarray([], dtype=bool)
            unique_linear = sorted_linear[first]
            unique_rgb = selected_rgb[order][first]
            relative = Path(f"compiled_proposals/{case_id}.npz")
            compiled_path = run_dir / relative
            np.savez_compressed(
                compiled_path,
                occupied_voxel_linear=unique_linear.astype(np.int64),
                appearance_rgb_uint8=unique_rgb.astype(np.uint8),
                actor_instance_ids=np.asarray(actor_ids, dtype=np.int32),
            )
            if set(unique_linear.tolist()) != set(candidate_linear.tolist()):
                raise ME1ExperimentError(f"surface attachment 与 candidate volume 漂移: {case_id}")

            b0_decision = fuse_two_factors(b0_rows[case_id]["P1"]["decision"], b0_rows[case_id]["P2"]["decision"])
            b1_decision = r10_rows[case_id]["overall_decision"]
            o1_decision = (
                "ACCEPT" if b1_decision == "ACCEPT" and factors["passed"]
                else "REJECT" if b1_decision == "REJECT"
                else "ABSTAIN"
            )
            o2_decision = method_decision(r9_rows[case_id]["P1"]["decision"], bool(factors["passed"]))
            motion = None
            o3_decision = o2_decision
            if case["hole_type"] == "actor_removal_hole":
                cache_key = (case["scene"], int(case["frame_index"]), tuple(actor_ids))
                if cache_key not in motion_cache:
                    motion_cache[cache_key] = _actor_motion_contract(
                        Path(config["raw_evidence"][case["scene"]]),
                        int(case["frame_index"]),
                        actor_ids,
                        int(config["runtime_4d"]["interpolation_samples"]),
                    )
                motion = motion_cache[cache_key]
                if o2_decision == "ACCEPT" and not motion["passed"]:
                    o3_decision = "REJECT" if motion["actor_actor_swept_collisions"] else "ABSTAIN"
            method_rows.append(
                {
                    "schema_version": "worldsim_v61.me1_method_decision.v1",
                    "case_id": case_id,
                    "scene": case["scene"],
                    "frame_index": int(case["frame_index"]),
                    "frontend": case["frontend"],
                    "hole_type": case["hole_type"],
                    "mask_pixel_count": int(mask.sum()),
                    "method_geometry_factors": factors,
                    "native_4d_contract": motion,
                    "compiled_proposal_path": str(relative),
                    "compiled_proposal_sha256": sha256_file(compiled_path),
                    "decision_inputs": ["frozen_R9_P1_photo", "O_method", "native_actor_metadata"],
                    "decisions": {
                        "B0-2D": b0_decision,
                        "B1-R10": b1_decision,
                        "O1-GATE": o1_decision,
                        "O2-OCC-GEOMETRY": o2_decision,
                        "O3-OCC-4D": o3_decision,
                    },
                }
            )
        _write_jsonl(run_dir / "METHOD_DECISIONS.jsonl", method_rows)
        method_decisions_sha256 = sha256_file(run_dir / "METHOD_DECISIONS.jsonl")

        # 第二阶段才读取 O_eval；它只产生 truth/false-safe，不修改任何 arm decision。
        eval_rays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        eval_grids: dict[tuple[str, int], dict[str, Any]] = {}
        for scene, frame in unit_keys:
            scene_root = Path(config["raw_evidence"][scene])
            t_lidar_camera, intrinsics = _camera_contract(scene_root, frame, config["projection"])
            grid = _load_grid(me0_root / f"evidence/{scene}/f{frame:03d}/O_eval.npz")
            rays = _raycast(grid, t_lidar_camera, intrinsics, config["projection"])
            eval_grids[(scene, frame)] = grid
            eval_rays[(scene, frame)] = rays
            relative = Path(f"projections/{scene}/f{frame:03d}/O_eval.npz")
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **rays)
            projection_index.append(
                {
                    "scene": scene,
                    "frame_index": frame,
                    "tier": "O_eval",
                    "path": str(relative),
                    "sha256": sha256_file(path),
                    "visible_surface_pixels": int(np.sum(rays["voxel_linear"] >= 0)),
                }
            )

        def score(row: dict[str, Any]) -> dict[str, Any]:
            case_id = row["case_id"]
            key = (row["scene"], int(row["frame_index"]))
            payload = np.load(r9_root / "verifier_inputs" / f"{case_id}.npz", allow_pickle=False)
            mask = np.asarray(payload["mask"], dtype=bool)
            compiled = np.load(run_dir / row["compiled_proposal_path"], allow_pickle=False)
            candidate_linear = np.asarray(compiled["occupied_voxel_linear"], dtype=np.int64)
            factors = _independent_eval_factors(
                mask,
                candidate_linear,
                method_rays[key],
                eval_grids[key],
                eval_rays[key],
                int(config["projection"]["observed_support_dilation_voxels"]),
            )
            factors["passed"] = occupancy_gate(factors, config["independent_eval_gate"])
            decisions = dict(row["decisions"])
            false_safe = {arm: decision == "ACCEPT" and not factors["passed"] for arm, decision in decisions.items()}
            return {
                **row,
                "independent_eval_geometry_factors": factors,
                "false_safe": false_safe,
                "decision_artifact_frozen_before_eval_sha256": method_decisions_sha256,
            }

        workers = min(int(config["resources"]["maximum_cpu_workers"]), len(method_rows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            final_rows = list(executor.map(score, method_rows))
        _write_jsonl(run_dir / "PER_CASE.jsonl", final_rows)
        _write_jsonl(run_dir / "PROJECTION_INDEX.jsonl", projection_index)

        arms = ["B0-2D", "B1-R10", "O1-GATE", "O2-OCC-GEOMETRY", "O3-OCC-4D"]
        arm_summaries = [_arm_summary(final_rows, arm) for arm in arms]
        total_mask_pixels = sum(int(row["mask_pixel_count"]) for row in final_rows)
        for summary in arm_summaries:
            summary["accept_coverage"] = summary["accept_count"] / len(final_rows)
            summary["accepted_mask_area_yield"] = summary["accepted_mask_pixels"] / total_mask_pixels
        _write_jsonl(run_dir / "ARM_SUMMARIES.jsonl", arm_summaries)
        summary_index = {row["arm"]: row for row in arm_summaries}
        primary_name = config["primary_gate"]["arm"]
        primary = summary_index[primary_name]
        expected_original = set(config["cohort"]["expected_r10_accept_ids"])
        primary_accepted = set(primary["accepted_case_ids"])
        new_cases = [row for row in final_rows if row["case_id"] in primary_accepted - expected_original]
        new_actor = sum(row["hole_type"] == "actor_removal_hole" for row in new_cases)
        new_static = sum(row["hole_type"] in {"missing_route_support", "disocclusion"} for row in new_cases)
        checks = {
            "case_denominator_exact": len(final_rows) == int(config["cohort"]["expected_case_count"]),
            "B0_is_frozen_big_lama": json.loads((b0_root / "SUMMARY.json").read_text())["selected_generator"] == "big_lama",
            "B1_exact_r10_counts": (
                summary_index["B1-R10"]["accept_count"] == 3
                and summary_index["B1-R10"]["abstain_count"] == 7
                and summary_index["B1-R10"]["reject_count"] == 18
            ),
            "O1_does_not_increase_coverage": summary_index["O1-GATE"]["accept_count"] <= 3,
            "method_decisions_frozen_before_eval": all(
                row["decision_artifact_frozen_before_eval_sha256"] == method_decisions_sha256 for row in final_rows
            ),
            "primary_minimum_accepted_cases": primary["accept_count"] >= int(config["primary_gate"]["minimum_accepted_cases"]),
            "primary_zero_false_safe": primary["false_safe_count"] <= int(config["primary_gate"]["maximum_false_safe_count"]),
            "all_r10_accepts_retained": expected_original <= primary_accepted,
            "minimum_new_actor_cases": new_actor >= int(config["primary_gate"]["minimum_new_actor_cases"]),
            "minimum_new_static_or_disocclusion_cases": new_static >= int(config["primary_gate"]["minimum_new_static_or_disocclusion_cases"]),
            "minimum_accepted_mask_area_yield": primary["accepted_mask_area_yield"] >= float(config["primary_gate"]["minimum_accepted_mask_area_yield"]),
            "no_training": True,
            "no_confirmation_read": True,
        }
        elapsed = time.monotonic() - started
        peak_gpu_gib = torch.cuda.max_memory_allocated() / 1024**3
        checks["wall_within_budget"] = elapsed <= float(config["resources"]["maximum_wall_seconds"])
        checks["gpu_memory_within_budget"] = peak_gpu_gib <= float(config["resources"]["maximum_gpu_memory_gib"])
        checks["disk_free_within_budget"] = free_gib >= float(config["resources"]["minimum_disk_free_gib"])
        gate = {
            "schema_version": "worldsim_v61.me1_gate.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "primary_arm": primary_name,
            "checks": checks,
            "passed": all(checks.values()),
            "new_actor_case_count": new_actor,
            "new_static_or_disocclusion_case_count": new_static,
        }
        _write_json(run_dir / "ME1_GATE.json", gate)
        metrics = {
            "schema_version": "worldsim_v61.me1_metrics.v1",
            "task_id": TASK_ID,
            "arm_summaries": arm_summaries,
            "total_mask_pixels": total_mask_pixels,
            "primary_improvement_cases_over_r10": primary["accept_count"] - 3,
            "primary_improvement_percentage_points": 100.0 * (primary["accept_count"] - 3) / len(final_rows),
        }
        _write_json(run_dir / "METRICS.json", metrics)
        resource = {
            "schema_version": "worldsim_v61.me1_resource_audit.v1",
            "wall_seconds": elapsed,
            "gpu_name": torch.cuda.get_device_name(),
            "peak_gpu_memory_gib": peak_gpu_gib,
            "gpu_raycast_tiers": 8,
            "maximum_cpu_workers": workers,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "model_inference_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "RESOURCE_AUDIT.json", resource)
        summary = {
            "schema_version": "worldsim_v61.me1_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "source_commit": source_commit,
            "status": "done" if gate["passed"] else "rejected",
            "hypothesis_outcome": "accepted_oracle_upper_bound" if gate["passed"] else "rejected_oracle_upper_bound",
            "primary_arm": primary_name,
            "primary_accept_count": primary["accept_count"],
            "primary_false_safe_count": primary["false_safe_count"],
            "primary_accepted_mask_area_yield": primary["accepted_mask_area_yield"],
            "stop_model_integration": not gate["passed"],
            "failure_ledger_delta": "none" if gate["passed"] else "required",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        manifest = {
            "schema_version": "worldsim_v61.me1_manifest.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_artifacts": {
                "me0_run": str(me0_root),
                "b0_2d_run": str(b0_root),
                "r9_cross_run": str(r9_root),
                "r10_run": str(r10_root),
            },
            "artifacts": {
                name: sha256_file(run_dir / name)
                for name in (
                    "METHOD_DECISIONS.jsonl",
                    "PER_CASE.jsonl",
                    "PROJECTION_INDEX.jsonl",
                    "ARM_SUMMARIES.jsonl",
                    "ME1_GATE.json",
                    "METRICS.json",
                    "RESOURCE_AUDIT.json",
                    "SUMMARY.json",
                )
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        terminal = {
            "schema_version": "worldsim_v61.me1_terminal.v1",
            "task_id": TASK_ID,
            "status": summary["status"],
            "canonical": bool(gate["passed"]),
            "run_uri": f"run://worldsim_v61/{TASK_ID}/{run_dir.name}",
        }
        _write_json(run_dir / "TERMINAL.json", terminal)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.me1_terminal.v1",
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
