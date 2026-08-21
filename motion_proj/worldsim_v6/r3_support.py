"""R3 support-deviation 的观测投影、分区误差与排序裁决。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr


class R3AnalysisError(RuntimeError):
    """R3 输入、denominator 或排序合同失败。"""


def _load_map(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _render_index(root: Path) -> dict[tuple[str, int, str, float], Path]:
    result = {}
    for row in _load_map(root / "RENDER_MAP.jsonl"):
        key = (
            row["frontend"],
            int(row["frame_index"]),
            row["variant"],
            float(row["lateral_offset_m"]),
        )
        result[key] = root / row["path"]
    return result


def _camera_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    left = first[:3, 2] / np.linalg.norm(first[:3, 2])
    right = second[:3, 2] / np.linalg.norm(second[:3, 2])
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0))))


def _project_observations(
    support: Mapping[str, np.ndarray], lateral_offset: float, forward_offset: float = 0.0
) -> dict[str, np.ndarray]:
    target_c2w = support["cam0_camera_to_world"].astype(np.float64).copy()
    target_c2w[:3, 3] += target_c2w[:3, 0] * lateral_offset
    target_c2w[:3, 3] += target_c2w[:3, 2] * forward_offset
    target_w2c = np.linalg.inv(target_c2w)
    target_k = support["cam0_intrinsics"].astype(np.float64)
    height, width = support["cam0_lidar_depth"].shape
    projected = []
    for camera_id in (0, 1, 2):
        prefix = f"cam{camera_id}"
        depth = support[f"{prefix}_lidar_depth"].astype(np.float64)
        valid_y, valid_x = np.nonzero(np.isfinite(depth) & (depth > 0))
        if valid_x.size == 0:
            continue
        z = depth[valid_y, valid_x]
        pixels = np.stack((valid_x, valid_y, np.ones_like(valid_x)), axis=0)
        camera_points = (np.linalg.inv(support[f"{prefix}_intrinsics"].astype(np.float64)) @ pixels) * z
        camera_to_world = support[f"{prefix}_camera_to_world"].astype(np.float64)
        world_points = camera_to_world[:3, :3] @ camera_points + camera_to_world[:3, 3:4]
        target_points = target_w2c[:3, :3] @ world_points + target_w2c[:3, 3:4]
        target_z = target_points[2]
        uvw = target_k @ target_points
        x = np.rint(uvw[0] / np.maximum(uvw[2], 1e-8)).astype(np.int64)
        y = np.rint(uvw[1] / np.maximum(uvw[2], 1e-8)).astype(np.int64)
        keep = (target_z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        rgb = support[f"{prefix}_rgb"][valid_y, valid_x].astype(np.float32) / 255.0
        dynamic = support[f"{prefix}_dynamic_mask"][valid_y, valid_x].astype(bool)
        projected.append(
            np.rec.fromarrays(
                [x[keep], y[keep], target_z[keep], rgb[keep, 0], rgb[keep, 1], rgb[keep, 2], dynamic[keep]],
                names="x,y,z,r,g,b,dynamic",
            )
        )
    if not projected:
        raise R3AnalysisError("没有可投影 LiDAR observation")
    values = np.concatenate(projected)
    linear = values.y.astype(np.int64) * width + values.x.astype(np.int64)
    order = np.lexsort((values.z, linear))
    ordered_linear = linear[order]
    first = np.r_[True, ordered_linear[1:] != ordered_linear[:-1]]
    selected = values[order[first]]
    return {
        "x": selected.x.astype(np.int64),
        "y": selected.y.astype(np.int64),
        "z": selected.z.astype(np.float32),
        "rgb": np.stack((selected.r, selected.g, selected.b), axis=-1).astype(np.float32),
        "dynamic": selected.dynamic.astype(bool),
        "target_camera_to_world": target_c2w.astype(np.float32),
        "height": np.asarray(height),
        "width": np.asarray(width),
    }


def _as_rgb(data: Mapping[str, np.ndarray]) -> np.ndarray:
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return np.clip(rgb, 0.0, 1.0)


def _as_map(value: np.ndarray) -> np.ndarray:
    result = np.squeeze(np.asarray(value, dtype=np.float32))
    if result.ndim != 2:
        raise R3AnalysisError(f"预期二维 raster，得到 {result.shape}")
    return result


def _safe_mean(value: np.ndarray) -> float | None:
    return float(np.mean(value)) if value.size else None


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def analyze_support_deviation(
    run_dir: Path,
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    """生成每 case 指标、排序 aggregate 和 actor 编辑 effect。"""
    thresholds = config["metrics"]
    support_cfg = config["support_signal"]
    weights = support_cfg["weights"]
    all_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    temporal_lookup: dict[tuple[str, str, int, float], np.ndarray] = {}
    for scene_row in config["cohort"]["scenes"]:
        scene = scene_row["scene"]
        frames = [int(value) for value in scene_row["source_frame_indices"]]
        street_root = run_dir / "renders" / scene / "streetgs"
        adgs_root = run_dir / "renders" / scene / "ad_gs"
        indexes = {
            "streetgs": _render_index(street_root),
            "ad_gs": _render_index(adgs_root),
        }
        training_poses = [
            np.asarray(row["camera_to_world"], dtype=np.float64)
            for row in json.loads((street_root / "TRAINING_CAMERA_SUPPORT.json").read_text(encoding="utf-8"))
        ]
        for frame_index in frames:
            support = np.load(street_root / f"support_frame_{frame_index:03d}.npz", allow_pickle=False)
            projected_by_offset = {
                float(offset): _project_observations(support, float(offset))
                for offset in config["camera_deviation_profile"]["lateral_offsets_m"]
            }
            logged_count = int(projected_by_offset[0.0]["x"].size)
            if logged_count < int(thresholds["minimum_global_projected_lidar_pixels"]):
                raise R3AnalysisError(f"{scene}/{frame_index} logged LiDAR denominator 过小：{logged_count}")
            for offset in config["camera_deviation_profile"]["lateral_offsets_m"]:
                offset = float(offset)
                observation = projected_by_offset[offset]
                x, y = observation["x"], observation["y"]
                support_count = int(x.size)
                target_pose = observation["target_camera_to_world"].astype(np.float64)
                distances = [np.linalg.norm(target_pose[:3, 3] - pose[:3, 3]) for pose in training_poses]
                nearest_index = int(np.argmin(distances))
                nearest_distance = float(distances[nearest_index])
                nearest_angle = _camera_angle_deg(target_pose, training_poses[nearest_index])
                q_position = math.exp(-nearest_distance / float(support_cfg["components"]["nearest_training_camera_position_scale_m"]))
                q_angle = math.exp(-nearest_angle / float(support_cfg["components"]["nearest_training_camera_angle_scale_deg"]))
                q_observed = min(1.0, support_count / logged_count)
                support_score = (
                    float(weights["camera_position"]) * q_position
                    + float(weights["camera_angle"]) * q_angle
                    + float(weights["observed_projection"]) * q_observed
                )
                renders = {
                    frontend: np.load(indexes[frontend][(frontend, frame_index, "camera_lateral", offset)], allow_pickle=False)
                    for frontend in ("streetgs", "ad_gs")
                }
                rgbs = {frontend: _as_rgb(data) for frontend, data in renders.items()}
                dynamic_maps = {
                    frontend: _as_map(data["dynamic_opacity"]) > float(thresholds["actor_opacity_threshold"])
                    for frontend, data in renders.items()
                }
                actor_union = dynamic_maps["streetgs"] | dynamic_maps["ad_gs"]
                boundary = binary_dilation(actor_union, iterations=int(thresholds["boundary_radius_px"])) ^ binary_erosion(
                    actor_union, iterations=int(thresholds["boundary_radius_px"]), border_value=0
                )
                cross_pixel = np.mean(np.abs(rgbs["streetgs"] - rgbs["ad_gs"]), axis=-1)
                cross_metrics = {
                    "global_rgb_disagreement": float(cross_pixel.mean()),
                    "static_rgb_disagreement": _safe_mean(cross_pixel[~actor_union]),
                    "actor_rgb_disagreement": _safe_mean(cross_pixel[actor_union]),
                    "actor_boundary_rgb_disagreement": _safe_mean(cross_pixel[boundary]),
                    "actor_pixel_count": int(actor_union.sum()),
                    "boundary_pixel_count": int(boundary.sum()),
                }
                for frontend in ("streetgs", "ad_gs"):
                    rgb = rgbs[frontend]
                    depth = _as_map(renders[frontend]["depth"])
                    observed_error = np.mean(np.abs(rgb[y, x] - observation["rgb"]), axis=-1)
                    dynamic = observation["dynamic"]
                    depth_relative = np.abs(depth[y, x] - observation["z"]) / np.maximum(observation["z"], 1.0)
                    projected_actor_count = int(dynamic.sum())
                    row = {
                        "scene": scene,
                        "frontend": frontend,
                        "frame_index": frame_index,
                        "lateral_offset_m": offset,
                        "support_score": support_score,
                        "support_invalidity": 1.0 - support_score,
                        "q_camera_position": q_position,
                        "q_camera_angle": q_angle,
                        "q_observed_projection": q_observed,
                        "nearest_training_camera_distance_m": nearest_distance,
                        "nearest_training_camera_angle_deg": nearest_angle,
                        "projected_lidar_pixel_count": support_count,
                        "projected_actor_lidar_pixel_count": projected_actor_count,
                        "projected_observation_rgb_error": float(observed_error.mean()),
                        "projected_static_rgb_error": _safe_mean(observed_error[~dynamic]),
                        "projected_actor_rgb_error": _safe_mean(observed_error[dynamic]),
                        "lidar_relative_depth_error": float(np.median(np.clip(depth_relative, 0.0, 10.0))),
                        **cross_metrics,
                    }
                    row["actor_lidar_denominator_passed"] = projected_actor_count >= int(
                        thresholds["minimum_actor_projected_lidar_pixels"]
                    )
                    row["downstream_error"] = (
                        0.5 * row["projected_observation_rgb_error"]
                        + 0.25 * min(1.0, row["lidar_relative_depth_error"])
                        + 0.25 * row["global_rgb_disagreement"]
                    )
                    all_rows.append(row)
                    temporal_lookup[(scene, frontend, frame_index, offset)] = rgb

            for frontend in ("streetgs", "ad_gs"):
                logged_path = indexes[frontend][(frontend, frame_index, "camera_lateral", 0.0)]
                logged_rgb = _as_rgb(np.load(logged_path, allow_pickle=False))
                for operation in config["actor_edit_profile"]["operations"]:
                    edited = _as_rgb(
                        np.load(indexes[frontend][(frontend, frame_index, operation, 0.0)], allow_pickle=False)
                    )
                    edit_rows.append(
                        {
                            "scene": scene,
                            "frontend": frontend,
                            "frame_index": frame_index,
                            "operation": operation,
                            "global_rgb_effect": float(np.mean(np.abs(edited - logged_rgb))),
                            "nonzero_pixel_fraction": float(np.mean(np.max(np.abs(edited - logged_rgb), axis=-1) > (1.0 / 255.0))),
                        }
                    )

            forward_offset = float(config["camera_deviation_profile"]["forward_extension_m"])
            forward_observation = _project_observations(support, 0.0, forward_offset)
            forward_renders = {
                frontend: np.load(
                    indexes[frontend][(frontend, frame_index, "camera_forward_extension", 0.0)],
                    allow_pickle=False,
                )
                for frontend in ("streetgs", "ad_gs")
            }
            forward_rgbs = {frontend: _as_rgb(data) for frontend, data in forward_renders.items()}
            forward_cross = float(np.mean(np.abs(forward_rgbs["streetgs"] - forward_rgbs["ad_gs"])))
            x, y = forward_observation["x"], forward_observation["y"]
            for frontend in ("streetgs", "ad_gs"):
                forward_depth = _as_map(forward_renders[frontend]["depth"])
                observation_rgb_error = float(
                    np.mean(np.abs(forward_rgbs[frontend][y, x] - forward_observation["rgb"]))
                )
                depth_relative = np.abs(forward_depth[y, x] - forward_observation["z"]) / np.maximum(
                    forward_observation["z"], 1.0
                )
                forward_rows.append(
                    {
                        "scene": scene,
                        "frontend": frontend,
                        "frame_index": frame_index,
                        "forward_offset_m": forward_offset,
                        "projected_lidar_pixel_count": int(x.size),
                        "projected_observation_rgb_error": observation_rgb_error,
                        "lidar_relative_depth_error": float(np.median(np.clip(depth_relative, 0.0, 10.0))),
                        "global_rgb_disagreement": forward_cross,
                    }
                )

        first_frame, second_frame = frames
        for frontend in ("streetgs", "ad_gs"):
            logged_change = float(
                np.mean(np.abs(temporal_lookup[(scene, frontend, second_frame, 0.0)] - temporal_lookup[(scene, frontend, first_frame, 0.0)]))
            )
            for offset in config["camera_deviation_profile"]["lateral_offsets_m"]:
                offset = float(offset)
                change = float(
                    np.mean(
                        np.abs(
                            temporal_lookup[(scene, frontend, second_frame, offset)]
                            - temporal_lookup[(scene, frontend, first_frame, offset)]
                        )
                    )
                )
                for row in all_rows:
                    if row["scene"] == scene and row["frontend"] == frontend and row["lateral_offset_m"] == offset:
                        row["temporal_change_error"] = abs(change - logged_change)

    invalidity = [row["support_invalidity"] for row in all_rows]
    distance = [row["lateral_offset_m"] for row in all_rows]
    errors = [row["downstream_error"] for row in all_rows]
    support_spearman = _spearman(invalidity, errors)
    distance_spearman = _spearman(distance, errors)
    offsets = sorted(set(distance))
    invalidity_residual = []
    error_residual = []
    for offset in offsets:
        indices = [index for index, value in enumerate(distance) if value == offset]
        invalidity_mean = float(np.mean([invalidity[index] for index in indices]))
        error_mean = float(np.mean([errors[index] for index in indices]))
        invalidity_residual.extend(invalidity[index] - invalidity_mean for index in indices)
        error_residual.extend(errors[index] - error_mean for index in indices)
    residual_spearman = _spearman(invalidity_residual, error_residual)
    group_rows = []
    for scene in sorted({row["scene"] for row in all_rows}):
        for frontend in ("streetgs", "ad_gs"):
            subset = [row for row in all_rows if row["scene"] == scene and row["frontend"] == frontend]
            group_rows.append(
                {
                    "scene": scene,
                    "frontend": frontend,
                    "support_spearman": _spearman(
                        [row["support_invalidity"] for row in subset],
                        [row["downstream_error"] for row in subset],
                    ),
                    "distance_spearman": _spearman(
                        [row["lateral_offset_m"] for row in subset],
                        [row["downstream_error"] for row in subset],
                    ),
                }
            )
    gate_cfg = config["ranking_gate"]
    gain = support_spearman - distance_spearman
    stable_positive_groups = sum(row["support_spearman"] > 0 for row in group_rows)
    gate = {
        "support_spearman_passed": support_spearman >= float(gate_cfg["minimum_support_spearman"]),
        "distance_gain_passed": gain >= float(gate_cfg["minimum_gain_over_distance_spearman"]),
        "distance_residual_passed": residual_spearman >= float(gate_cfg["minimum_distance_residual_spearman"]),
        "stable_group_ordering_passed": stable_positive_groups >= 3,
    }
    aggregate = {
        "schema_version": "worldsim_v6.r3_support_deviation_aggregate.v1",
        "row_count": len(all_rows),
        "scene_count": len({row["scene"] for row in all_rows}),
        "frontend_count": len({row["frontend"] for row in all_rows}),
        "support_invalidity_vs_downstream_error_spearman": support_spearman,
        "distance_vs_downstream_error_spearman": distance_spearman,
        "support_gain_over_distance_spearman": gain,
        "support_distance_residual_spearman": residual_spearman,
        "group_ordering": group_rows,
        "gate": {**gate, "passed": all(gate.values())},
        "decision": (
            "support_certificate_promising"
            if all(gate.values())
            else "reject_or_revise_analytic_support_before_any_learned_model"
        ),
    }
    return all_rows, aggregate, edit_rows, forward_rows
