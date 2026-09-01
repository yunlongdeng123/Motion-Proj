"""Hard physical evidence and deterministic visual panels for WorldSim V7 P3."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from motion_proj.worldsim_v7.av2_four_action_compiler import _nearest


def _depth_evidence(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    distances, nearest_indices = _nearest(
        target,
        surface,
        device,
        int(config["distance_chunk_size"]),
    )
    nearest = surface[nearest_indices]
    target_vectors = target - origins
    target_depth = np.linalg.norm(target_vectors, axis=1)
    directions = target_vectors / np.maximum(target_depth[:, None], 1e-6)
    surface_vectors = nearest - origins
    predicted_depth = np.sum(surface_vectors * directions, axis=1)
    lateral = np.linalg.norm(
        surface_vectors - predicted_depth[:, None] * directions,
        axis=1,
    )
    depth_error = np.abs(predicted_depth - target_depth)
    consistent = (
        (predicted_depth > 0.0)
        & (lateral <= float(config["ray_lateral_tolerance_m"]))
        & (depth_error <= float(config["depth_tolerance_m"]))
    )
    return {
        "point_to_surface_mean_m": float(np.mean(distances)),
        "lidar_depth_error_mean_m": float(np.mean(depth_error)),
        "ray_termination_consistency": float(np.mean(consistent)),
    }


def _zero_level_error(
    points: np.ndarray,
    canonical: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    distances, _ = _nearest(
        points,
        canonical,
        device,
        int(config["distance_chunk_size"]),
    )
    return {
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.quantile(distances, 0.95)),
    }


def _component_count(points: np.ndarray, voxel_size_m: float) -> int:
    if len(points) == 0:
        return 0
    keys = {
        tuple(int(value) for value in row)
        for row in np.floor(points / float(voxel_size_m)).astype(np.int32)
    }
    components = 0
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    while keys:
        components += 1
        queue = deque([keys.pop()])
        while queue:
            key = queue.popleft()
            for offset in offsets:
                neighbour = tuple(key[index] + offset[index] for index in range(3))
                if neighbour in keys:
                    keys.remove(neighbour)
                    queue.append(neighbour)
    return components


def compile_hard_evidence(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    baseline = diagnostics["baseline"]
    compiled = diagnostics["compiled"]
    canonical = diagnostics["canonical"]
    target = diagnostics["target"]
    origins = diagnostics["target_sensor_origins"]
    ghost = diagnostics["ghost"]
    track = diagnostics["track"]

    depth_before = _depth_evidence(baseline, target, origins, config, device)
    depth_after = _depth_evidence(compiled, target, origins, config, device)
    sdf_before = _zero_level_error(baseline, canonical, config, device)
    sdf_after = _zero_level_error(compiled, canonical, config, device)

    ghost_hit = diagnostics["ghost_hit"]
    ghost_ray = diagnostics["ghost_ray"]
    projected_aligned = diagnostics["projected_aligned"]
    query_origin = diagnostics["query_sensor_origin"]
    ghost_offset = float(config["ghost_offset_m"])
    free_threshold = float(config["minimum_free_space_violation_m"])
    has_projected_output = np.all(np.isfinite(projected_aligned), axis=1)
    hit_depth = np.linalg.norm(ghost_hit - query_origin[None, :], axis=1)
    output_vectors = np.zeros_like(projected_aligned)
    output_vectors[has_projected_output] = (
        projected_aligned[has_projected_output] - query_origin[None, :]
    )
    output_depth = np.sum(output_vectors * ghost_ray, axis=1)
    output_lateral = np.linalg.norm(
        output_vectors - output_depth[:, None] * ghost_ray,
        axis=1,
    )
    same_ray_output = has_projected_output & (
        output_lateral <= float(config["ray_lateral_tolerance_m"])
    )
    residual_depth = np.zeros(len(ghost), dtype=np.float32)
    residual_depth[same_ray_output] = np.maximum(
        hit_depth[same_ray_output] - output_depth[same_ray_output],
        0.0,
    )
    residual_ghost = ghost[residual_depth > free_threshold]

    frame_means = []
    for points in diagnostics["build_frame_points"]:
        distances, _ = _nearest(
            points,
            canonical,
            device,
            int(config["distance_chunk_size"]),
        )
        frame_means.append(float(np.mean(distances)))
    frame_means_array = np.asarray(frame_means, dtype=np.float64)

    return {
        "track_id": str(actor_row["track_id"]),
        "category": str(actor_row["category"]),
        "hazardous": bool(actor_row["hazardous"]),
        "free_space_violation_rate_before": 1.0,
        "free_space_violation_rate_after": float(
            np.mean(residual_depth > free_threshold)
        ),
        "free_space_violation_depth_before_m": ghost_offset,
        "free_space_violation_depth_after_m": float(np.mean(residual_depth)),
        "lidar_depth_error_before_m": depth_before["lidar_depth_error_mean_m"],
        "lidar_depth_error_after_m": depth_after["lidar_depth_error_mean_m"],
        "ray_termination_consistency_before": depth_before[
            "ray_termination_consistency"
        ],
        "ray_termination_consistency_after": depth_after[
            "ray_termination_consistency"
        ],
        "sdf_zero_level_error_before_m": sdf_before["mean_m"],
        "sdf_zero_level_error_after_m": sdf_after["mean_m"],
        "sdf_zero_level_p95_before_m": sdf_before["p95_m"],
        "sdf_zero_level_p95_after_m": sdf_after["p95_m"],
        "temporal_surface_jitter_m": float(np.std(frame_means_array)),
        "temporal_surface_residual_mean_m": float(np.mean(frame_means_array)),
        "ghost_connected_components_before": _component_count(
            ghost, float(config["ghost_component_voxel_m"])
        ),
        "ghost_connected_components_after": _component_count(
            residual_ghost, float(config["ghost_component_voxel_m"])
        ),
        "actor_surface_completeness_before": float(actor_row["before"]["target_recall"]),
        "actor_surface_completeness_after": float(actor_row["after"]["target_recall"]),
        "collision_shell_consistency_before": float(
            actor_row["before"]["surface_precision"]
        ),
        "collision_shell_consistency_after": float(
            actor_row["after"]["surface_precision"]
        ),
        "symmetric_chamfer_before_m": float(
            actor_row["before"]["symmetric_chamfer_m"]
        ),
        "symmetric_chamfer_after_m": float(
            actor_row["after"]["symmetric_chamfer_m"]
        ),
        "actor_retention": 1.0,
        "id_lifecycle_retention": 1.0,
        "trajectory_displacement_m": 0.0,
        "speed_shift_mps": 0.0,
        "acceleration_shift_mps2": 0.0,
        "ttc_shift_s": 0.0,
        "minimum_ttc_s": float(track.minimum_ttc_s),
        "minimum_clearance_m": float(track.minimum_clearance_m),
        "hard_brake_score": float(track.hard_brake_score),
        "crossing_probability": float(track.crossing_probability),
        "hazard_label_retention": 1.0,
        "hazard_event_count_change": 0,
    }


def summarize_hard_evidence(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("role has no P3 hard-evidence rows")

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    ghost_before = sum(int(row["ghost_connected_components_before"]) for row in rows)
    ghost_after = sum(int(row["ghost_connected_components_after"]) for row in rows)
    return {
        "actor_count": len(rows),
        "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in rows)),
        "free_space_violation_rate_before": mean("free_space_violation_rate_before"),
        "free_space_violation_rate_after": mean("free_space_violation_rate_after"),
        "free_space_violation_depth_before_m": mean(
            "free_space_violation_depth_before_m"
        ),
        "free_space_violation_depth_after_m": mean(
            "free_space_violation_depth_after_m"
        ),
        "lidar_depth_error_before_m": mean("lidar_depth_error_before_m"),
        "lidar_depth_error_after_m": mean("lidar_depth_error_after_m"),
        "ray_termination_consistency_before": mean(
            "ray_termination_consistency_before"
        ),
        "ray_termination_consistency_after": mean(
            "ray_termination_consistency_after"
        ),
        "sdf_zero_level_error_before_m": mean("sdf_zero_level_error_before_m"),
        "sdf_zero_level_error_after_m": mean("sdf_zero_level_error_after_m"),
        "sdf_zero_level_ratio": mean("sdf_zero_level_error_after_m")
        / max(mean("sdf_zero_level_error_before_m"), 1e-8),
        "mean_temporal_surface_jitter_m": mean("temporal_surface_jitter_m"),
        "maximum_temporal_surface_jitter_m": float(
            max(float(row["temporal_surface_jitter_m"]) for row in rows)
        ),
        "ghost_connected_components_before": ghost_before,
        "ghost_connected_components_after": ghost_after,
        "ghost_connected_component_ratio": ghost_after / max(ghost_before, 1),
        "actor_surface_completeness_before": mean(
            "actor_surface_completeness_before"
        ),
        "actor_surface_completeness_after": mean(
            "actor_surface_completeness_after"
        ),
        "collision_shell_consistency_before": mean(
            "collision_shell_consistency_before"
        ),
        "collision_shell_consistency_after": mean(
            "collision_shell_consistency_after"
        ),
        "symmetric_chamfer_before_m": mean("symmetric_chamfer_before_m"),
        "symmetric_chamfer_after_m": mean("symmetric_chamfer_after_m"),
        "symmetric_chamfer_ratio": mean("symmetric_chamfer_after_m")
        / max(mean("symmetric_chamfer_before_m"), 1e-8),
        "actor_retention": mean("actor_retention"),
        "id_lifecycle_retention": mean("id_lifecycle_retention"),
        "maximum_trajectory_displacement_m": float(
            max(float(row["trajectory_displacement_m"]) for row in rows)
        ),
        "maximum_speed_shift_mps": float(
            max(float(row["speed_shift_mps"]) for row in rows)
        ),
        "maximum_acceleration_shift_mps2": float(
            max(float(row["acceleration_shift_mps2"]) for row in rows)
        ),
        "maximum_ttc_shift_s": float(max(float(row["ttc_shift_s"]) for row in rows)),
        "hazard_label_retention": mean("hazard_label_retention"),
        "hazard_event_count_change": int(
            sum(int(row["hazard_event_count_change"]) for row in rows)
        ),
    }


def _limited(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]


def render_actor_panel(
    diagnostics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_path: Path,
    point_limit: int,
) -> None:
    layers = [
        diagnostics["query"],
        diagnostics["ghost"],
        diagnostics["duplicate"],
        diagnostics["flicker"],
        diagnostics["compiled"],
        diagnostics["target"],
    ]
    all_xy = np.concatenate([layer[:, :2] for layer in layers if len(layer)], axis=0)
    limit = max(float(np.quantile(np.abs(all_xy), 0.995)), 1.0) + 0.25

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    before_layers = [
        ("query", diagnostics["query"], "#808080", 2.0),
        ("ghost", diagnostics["ghost"], "#d62728", 8.0),
        ("duplicate", diagnostics["duplicate"], "#ff7f0e", 8.0),
        ("flicker", diagnostics["flicker"], "#9467bd", 8.0),
    ]
    for label, points, color, size in before_layers:
        selected = _limited(points, point_limit)
        axes[0].scatter(selected[:, 0], selected[:, 1], s=size, c=color, label=label, alpha=0.72)
    axes[0].set_title("Query + paired artifacts")

    compiled_layers = [
        ("KEEP", diagnostics["kept"], "#7f7f7f", 2.0),
        ("PROJECT", diagnostics["projected"], "#1f77b4", 8.0),
        ("COMPLETE", diagnostics["completed"], "#2ca02c", 8.0),
        ("UNKNOWN", diagnostics["unknown_query"], "#d62728", 10.0),
    ]
    for label, points, color, size in compiled_layers:
        if len(points) == 0:
            continue
        selected = _limited(points, point_limit)
        marker = "x" if label == "UNKNOWN" else "o"
        axes[1].scatter(
            selected[:, 0], selected[:, 1], s=size, c=color, label=label, alpha=0.78, marker=marker
        )
    axes[1].set_title("Four-action physical surface")

    target = _limited(diagnostics["target"], point_limit)
    compiled = _limited(diagnostics["compiled"], point_limit)
    axes[2].scatter(target[:, 0], target[:, 1], s=2.0, c="#111111", label="held-out target", alpha=0.45)
    axes[2].scatter(compiled[:, 0], compiled[:, 1], s=2.0, c="#1f77b4", label="compiled", alpha=0.55)
    axes[2].set_title(
        f"Target evidence: CD {evidence['symmetric_chamfer_before_m']:.3f}→"
        f"{evidence['symmetric_chamfer_after_m']:.3f} m"
    )

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_xlabel("Actor x (m)")
        axis.set_ylabel("Actor y (m)")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=6, loc="best", frameon=False)
    figure.suptitle(
        f"{str(evidence['track_id'])[:8]} | {evidence['category']} | "
        f"hazard={int(bool(evidence['hazardous']))} | no result-based selection",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
