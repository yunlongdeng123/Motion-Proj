"""WorldSim V3 A1 最小 LiDAR provenance 的纯函数。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from motion_proj.worldsim_v3.calibration_endpoints import (
    coverage_status,
    summarize_distribution,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lidar_provenance_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("LiDAR provenance schema_version must be 1")
    if payload.get("audit_version") != "A1-LP0-v1":
        raise ValueError("audit_version must be A1-LP0-v1")
    raw = payload.get("raw_inputs")
    runtime = payload.get("runtime_initialization")
    depth = payload.get("initial_depth_residual")
    boundary = payload.get("scope_boundary")
    if not all(isinstance(row, Mapping) for row in (raw, runtime, depth, boundary)):
        raise ValueError("LiDAR provenance contract has missing sections")
    if raw.get("lidar_scan_format") != "float32_xyzi" or raw.get(
        "bytes_per_point"
    ) != 16:
        raise ValueError("raw LiDAR format must remain float32 XYZI")
    if runtime.get("seed") != 0:
        raise ValueError("A1 paired initialization seed must remain zero")
    if not runtime.get("require_recorded_lidar_actor_tensor_match"):
        raise ValueError("recorded LiDAR/actor tensor match must be required")
    if runtime.get("exact_background_reproduction") != "report_not_gate":
        raise ValueError("exact background reproduction must remain report_not_gate")
    if depth.get("split") != "heldout_test":
        raise ValueError("initial depth residual split must be heldout_test")
    for key in (
        "lidar_valid_minimum_meters",
        "rendered_valid_minimum_meters",
        "minimum_rendered_opacity",
        "minimum_valid_points",
        "minimum_coverage",
    ):
        if float(depth.get(key, 0)) <= 0:
            raise ValueError(f"initial_depth_residual {key} must be positive")
    deferred = set(boundary.get("deferred_to_a2", []))
    required_deferred = {
        "post_densification_per_gaussian_ancestry",
        "parent_child_lineage",
        "split_clone_source_chain",
    }
    if deferred != required_deferred:
        raise ValueError("A2 ancestry boundary was changed")


def raw_lidar_block_contract(
    scan_path: Path,
    pose_path: Path,
    *,
    frame: int,
    bytes_per_point: int = 16,
) -> dict[str, Any]:
    if not scan_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"missing LiDAR block for frame {frame}")
    scan_bytes = scan_path.stat().st_size
    if scan_bytes % bytes_per_point:
        raise ValueError(f"LiDAR scan byte count is misaligned: {scan_path}")
    pose = np.loadtxt(pose_path)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"invalid LiDAR pose: {pose_path}")
    return {
        "frame": int(frame),
        "scan_path": str(scan_path),
        "scan_bytes": int(scan_bytes),
        "scan_sha256": sha256_file(scan_path),
        "raw_point_count": int(scan_bytes // bytes_per_point),
        "pose_path": str(pose_path),
        "pose_bytes": int(pose_path.stat().st_size),
        "pose_sha256": sha256_file(pose_path),
    }


def actor_input_mapping(
    instance_samples: Mapping[str, Mapping[str, Any]],
    registry_actors: list[Mapping[str, Any]],
    selected_actors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_true_id: dict[str, Mapping[str, Any]] = {}
    for actor in registry_actors:
        key = str(actor.get("processed_true_instance_id"))
        if key in by_true_id:
            raise ValueError(f"duplicate processed actor ID: {key}")
        by_true_id[key] = actor
    roles_by_model: dict[int, list[str]] = {}
    for role, actor in selected_actors.items():
        if actor and actor.get("availability") == "available":
            roles_by_model.setdefault(int(actor["rigid_model_index"]), []).append(role)

    result: list[dict[str, Any]] = []
    for true_id, sample in sorted(
        instance_samples.items(), key=lambda item: int(item[0])
    ):
        actor = by_true_id.get(str(true_id))
        if actor is None:
            raise ValueError(f"sampled actor {true_id} is absent from registry")
        model_index = int(actor["rigid_model_index"])
        visible_frames = int(sample["visible_frame_count"])
        if visible_frames != int(actor["processed_frame_count"]):
            raise ValueError(f"visible frame mismatch for actor {true_id}")
        point_count = int(sample["point_count"])
        result.append(
            {
                "processed_true_instance_id": int(true_id),
                "rigid_model_index": model_index,
                "instance_token": actor["instance_token"],
                "class_name": actor["class_name"],
                "selected_roles": sorted(roles_by_model.get(model_index, [])),
                "node_type": sample["node_type"],
                "input_point_count": point_count,
                "initial_gaussian_count": point_count,
                "points_sha256": sample["points_sha256"],
                "colors_sha256": sample["colors_sha256"],
                "visible_frame_count": visible_frames,
            }
        )
    if len(result) != len(registry_actors):
        raise ValueError("sampled actor count does not match registry")
    return result


def compare_initialization_provenance(
    source: Mapping[str, Any], reproduced: Mapping[str, Any]
) -> dict[str, Any]:
    source_background = source["background_lidar_sample"]
    reproduced_background = reproduced["background_lidar_sample"]
    background_lidar_fields = (
        "point_count",
        "points_sha256",
        "colors_sha256",
        "times_sha256",
    )
    background_lidar_exact = all(
        source_background.get(key) == reproduced_background.get(key)
        for key in background_lidar_fields
    )
    source_instances = source["instance_lidar_samples"]
    reproduced_instances = reproduced["instance_lidar_samples"]
    instance_fields = (
        "node_type",
        "point_count",
        "points_sha256",
        "colors_sha256",
        "visible_frame_count",
    )
    actor_inputs_exact = set(source_instances) == set(reproduced_instances) and all(
        all(source_instances[key].get(field) == reproduced_instances[key].get(field)
            for field in instance_fields)
        for key in source_instances
    )
    source_counts = source["initialized_gaussians"]
    reproduced_counts = reproduced["initialized_gaussians"]
    source_background_count = int(source_counts["Background"])
    reproduced_background_count = int(reproduced_counts["Background"])
    source_rigid_count = int(source_counts["RigidNodes"])
    reproduced_rigid_count = int(reproduced_counts["RigidNodes"])
    return {
        "background_lidar_tensor_exact": bool(background_lidar_exact),
        "actor_input_tensors_exact": bool(actor_inputs_exact),
        "recorded_lidar_actor_inputs_exact": bool(
            background_lidar_exact and actor_inputs_exact
        ),
        "rigid_initial_gaussian_count_exact": source_rigid_count
        == reproduced_rigid_count,
        "source_background_initial_gaussian_count": source_background_count,
        "reproduced_background_initial_gaussian_count": reproduced_background_count,
        "background_initial_gaussian_count_delta": reproduced_background_count
        - source_background_count,
        "source_rigid_initial_gaussian_count": source_rigid_count,
        "reproduced_rigid_initial_gaussian_count": reproduced_rigid_count,
        "exact_recorded_provenance": source == reproduced,
    }


def sparse_depth_residuals(
    *,
    rendered_depth: np.ndarray,
    rendered_opacity: np.ndarray,
    lidar_depth: np.ndarray,
    lidar_valid_minimum_meters: float,
    rendered_valid_minimum_meters: float,
    minimum_rendered_opacity: float,
) -> dict[str, Any]:
    prediction = np.asarray(rendered_depth, dtype=np.float64).squeeze()
    opacity = np.asarray(rendered_opacity, dtype=np.float64).squeeze()
    target = np.asarray(lidar_depth, dtype=np.float64).squeeze()
    if prediction.shape != target.shape or opacity.shape != target.shape:
        raise ValueError("depth, opacity, and LiDAR map shapes must match")
    candidate = np.isfinite(target) & (target > lidar_valid_minimum_meters)
    valid = (
        candidate
        & np.isfinite(prediction)
        & (prediction > rendered_valid_minimum_meters)
        & np.isfinite(opacity)
        & (opacity >= minimum_rendered_opacity)
    )
    absolute = np.abs(prediction[valid] - target[valid])
    relative = absolute / np.maximum(target[valid], lidar_valid_minimum_meters)
    return {
        "candidate_mask": candidate,
        "valid_mask": valid,
        "candidate_count": int(candidate.sum()),
        "valid_count": int(valid.sum()),
        "absolute_residual_meters": absolute,
        "relative_residual": relative,
    }


def summarize_depth_residual(
    *,
    absolute_values: list[float] | np.ndarray,
    relative_values: list[float] | np.ndarray,
    valid_count: int,
    candidate_count: int,
    minimum_valid_points: int,
    minimum_coverage: float,
) -> dict[str, Any]:
    status = coverage_status(
        valid_count=valid_count,
        candidate_count=candidate_count,
        minimum_valid_count=minimum_valid_points,
        minimum_coverage=minimum_coverage,
        zero_reason="ZERO_VALID_INITIAL_DEPTH_SUPPORT",
    )
    return {
        **status,
        "absolute_residual_meters": summarize_distribution(absolute_values),
        "relative_residual": summarize_distribution(relative_values),
    }
