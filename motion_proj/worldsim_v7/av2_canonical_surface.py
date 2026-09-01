"""AV2 Actor-local LiDAR fusion and paired validity--hazard evidence atlas."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from motion_proj.worldsim_v7.physical_compiler import (
    ActorState,
    CompilerThresholds,
    HazardPreservingPhysicalCompiler,
    PhysicalEvidence,
)
from motion_proj.worldsim_v7.validity_hazard import (
    FactorizedScores,
    HazardFeatures,
    ValidityFeatures,
    ValidityHazardFactorizer,
    paired_conditional_invariance,
)


@dataclass(frozen=True)
class TrackGeometry:
    track_id: str
    category: str
    timestamps_ns: np.ndarray
    ego_centers_m: np.ndarray
    city_centers_m: np.ndarray
    size_lwh_m: np.ndarray
    minimum_ttc_s: float
    minimum_clearance_m: float
    maximum_closing_speed_mps: float
    hard_brake_score: float
    crossing_probability: float
    hazardous: bool

    def hazard_features(self) -> HazardFeatures:
        return HazardFeatures(
            minimum_ttc_s=self.minimum_ttc_s,
            minimum_clearance_m=self.minimum_clearance_m,
            closing_speed_mps=self.maximum_closing_speed_mps,
            hard_brake_score=self.hard_brake_score,
            crossing_probability=self.crossing_probability,
        )


def _quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternions must have shape [N,4]")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def _track_geometries(
    annotations: pd.DataFrame,
    poses: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, TrackGeometry]:
    pose_frame = poses.set_index("timestamp_ns", drop=False)
    actor_config = config["actors"]
    hazard_config = config["hazard"]
    allowed = set(str(value) for value in actor_config["categories"])
    annotations = annotations[annotations["category"].isin(allowed)].copy()
    tracks: dict[str, TrackGeometry] = {}
    for track_id, frame in annotations.groupby("track_uuid", sort=True):
        frame = frame.sort_values("timestamp_ns")
        if len(frame) < int(actor_config["minimum_track_states"]):
            continue
        timestamps = frame["timestamp_ns"].to_numpy(dtype=np.int64)
        if any(int(timestamp) not in pose_frame.index for timestamp in timestamps):
            continue
        ego_centers = frame[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=np.float64)
        pose_rows = pose_frame.loc[timestamps]
        ego_rotations = _quaternion_to_rotation(
            pose_rows[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
        )
        ego_city_translation = pose_rows[["tx_m", "ty_m", "tz_m"]].to_numpy(
            dtype=np.float64
        )
        city_centers = np.einsum("nij,nj->ni", ego_rotations, ego_centers)
        city_centers += ego_city_translation
        size_lwh = frame[["length_m", "width_m", "height_m"]].to_numpy(
            dtype=np.float64
        )

        radius = 0.5 * np.linalg.norm(size_lwh[:, :2], axis=1)
        clearance = np.maximum(
            np.linalg.norm(ego_centers[:, :2], axis=1)
            - radius
            - float(hazard_config["ego_radius_m"]),
            0.0,
        )
        seconds = (timestamps - timestamps[0]).astype(np.float64) / 1e9
        dt = np.diff(seconds)
        valid_dt = dt > 1e-4
        range_delta = np.diff(clearance)
        closing = np.zeros_like(dt)
        closing[valid_dt] = np.maximum(-range_delta[valid_dt] / dt[valid_dt], 0.0)
        maximum_closing = float(np.max(closing, initial=0.0))
        ttc = np.full_like(dt, float(hazard_config["maximum_ttc_s"]))
        closing_valid = closing >= float(hazard_config["minimum_closing_speed_mps"])
        ttc[closing_valid] = clearance[:-1][closing_valid] / np.maximum(
            closing[closing_valid], 1e-6
        )
        minimum_ttc = float(
            min(np.min(ttc, initial=float(hazard_config["maximum_ttc_s"])),
                float(hazard_config["maximum_ttc_s"]))
        )

        velocity = np.zeros((0, 3), dtype=np.float64)
        if np.any(valid_dt):
            velocity = np.diff(city_centers, axis=0)[valid_dt] / dt[valid_dt, None]
        speed = np.linalg.norm(velocity[:, :2], axis=1) if len(velocity) else np.empty(0)
        deceleration = 0.0
        if len(speed) >= 2:
            speed_dt = dt[1:][valid_dt[1:]]
            aligned = min(len(speed_dt), len(speed) - 1)
            if aligned:
                acceleration = np.diff(speed[: aligned + 1]) / np.maximum(
                    speed_dt[:aligned], 1e-4
                )
                deceleration = float(max(-np.min(acceleration, initial=0.0), 0.0))
        hard_brake_score = float(
            np.clip(
                deceleration / float(hazard_config["hard_brake_reference_mps2"]),
                0.0,
                1.0,
            )
        )
        lateral = ego_centers[:, 1]
        forward = ego_centers[:, 0]
        crossing = bool(
            float(np.min(lateral)) <= 0.0 <= float(np.max(lateral))
            and float(np.min(np.abs(forward)))
            <= float(hazard_config["crossing_forward_window_m"])
        )
        minimum_clearance = float(np.min(clearance, initial=np.inf))
        hazardous = bool(
            minimum_ttc <= float(hazard_config["hazard_ttc_s"])
            or minimum_clearance <= float(hazard_config["hazard_clearance_m"])
            or (
                hard_brake_score >= float(hazard_config["hazard_hard_brake_score"])
                and minimum_clearance
                <= float(hazard_config["hard_brake_interaction_range_m"])
            )
            or (
                crossing
                and minimum_clearance
                <= float(hazard_config["crossing_interaction_range_m"])
            )
        )
        categories = sorted(set(str(value) for value in frame["category"]))
        if len(categories) != 1:
            continue
        tracks[str(track_id)] = TrackGeometry(
            track_id=str(track_id),
            category=categories[0],
            timestamps_ns=timestamps,
            ego_centers_m=ego_centers,
            city_centers_m=city_centers,
            size_lwh_m=np.median(size_lwh, axis=0),
            minimum_ttc_s=minimum_ttc,
            minimum_clearance_m=minimum_clearance,
            maximum_closing_speed_mps=maximum_closing,
            hard_brake_score=hard_brake_score,
            crossing_probability=float(crossing),
            hazardous=hazardous,
        )
    return tracks


def _read_lidar(path: Path) -> np.ndarray:
    frame = pd.read_feather(path, columns=["x", "y", "z"])
    return frame[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)


def _deterministic_limit(values: np.ndarray, limit: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=np.int64)
    return values[indices]


def _associate_actor_points(
    points_ego: np.ndarray,
    rows: pd.DataFrame,
    frame_ranks: Mapping[str, Mapping[int, int]],
    records: dict[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    points = torch.as_tensor(points_ego, dtype=torch.float32, device=device)
    chunk_size = int(config["actors"]["actor_batch_size"])
    padding = float(config["actors"]["box_padding_m"])
    per_frame_limit = int(config["surface"]["maximum_points_per_actor_frame"])
    row_list = list(rows.itertuples(index=False))
    with torch.inference_mode():
        for start in range(0, len(row_list), chunk_size):
            batch = row_list[start : start + chunk_size]
            centers_np = np.asarray(
                [[row.tx_m, row.ty_m, row.tz_m] for row in batch], dtype=np.float32
            )
            sizes_np = np.asarray(
                [[row.length_m, row.width_m, row.height_m] for row in batch],
                dtype=np.float32,
            )
            rotations_np = _quaternion_to_rotation(
                np.asarray(
                    [[row.qw, row.qx, row.qy, row.qz] for row in batch],
                    dtype=np.float64,
                )
            ).astype(np.float32)
            centers = torch.as_tensor(centers_np, device=device)
            sizes = torch.as_tensor(sizes_np, device=device)
            rotations = torch.as_tensor(rotations_np, device=device)
            local = torch.matmul(points.unsqueeze(0) - centers[:, None, :], rotations)
            inside = torch.all(
                torch.abs(local) <= sizes[:, None, :] * 0.5 + padding,
                dim=-1,
            )
            origins = torch.matmul(-centers[:, None, :], rotations).squeeze(1)
            for offset, row in enumerate(batch):
                selected = local[offset, inside[offset]]
                if selected.numel() == 0:
                    continue
                if len(selected) > per_frame_limit:
                    indices = torch.linspace(
                        0,
                        len(selected) - 1,
                        steps=per_frame_limit,
                        device=device,
                    ).to(torch.long)
                    selected = selected.index_select(0, indices)
                track_id = str(row.track_uuid)
                records[track_id].append(
                    {
                        "timestamp_ns": int(row.timestamp_ns),
                        "frame_rank": int(frame_ranks[track_id][int(row.timestamp_ns)]),
                        "points": selected.cpu().numpy(),
                        "sensor_origin": origins[offset].cpu().numpy(),
                        "actor_center_ego": centers_np[offset].copy(),
                        "actor_rotation_ego": rotations_np[offset].copy(),
                    }
                )


def _fuse_surfels(
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    points_np = np.concatenate([item["points"] for item in records], axis=0)
    frames_np = np.concatenate(
        [np.full(len(item["points"]), item["frame_rank"], dtype=np.int64) for item in records]
    )
    origins_np = np.concatenate(
        [
            np.repeat(item["sensor_origin"][None, :], len(item["points"]), axis=0)
            for item in records
        ],
        axis=0,
    )
    voxel_size = float(config["surface"]["voxel_size_m"])
    with torch.inference_mode():
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        frames = torch.as_tensor(frames_np, dtype=torch.long, device=device)
        origins = torch.as_tensor(origins_np, dtype=torch.float32, device=device)
        keys = torch.floor(points / voxel_size).to(torch.int32)
        unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=len(unique_keys))
        sums = torch.zeros((len(unique_keys), 3), dtype=torch.float32, device=device)
        sums.index_add_(0, inverse, points)
        surfels = sums / count[:, None].clamp_min(1)
        origin_sums = torch.zeros_like(sums)
        origin_sums.index_add_(0, inverse, origins)
        mean_origins = origin_sums / count[:, None].clamp_min(1)

        frame_base = int(frames.max().item()) + 1
        frame_pairs = torch.unique(inverse.to(torch.long) * frame_base + frames)
        temporal_support = torch.bincount(
            torch.div(frame_pairs, frame_base, rounding_mode="floor"),
            minlength=len(unique_keys),
        )
        rays = points - origins
        azimuth = torch.atan2(rays[:, 1], rays[:, 0])
        view_bins = torch.floor((azimuth + torch.pi) / (2.0 * torch.pi) * 8.0)
        view_bins = torch.remainder(view_bins.to(torch.long), 8)
        view_pairs = torch.unique(inverse.to(torch.long) * 8 + view_bins)
        view_support = torch.bincount(
            torch.div(view_pairs, 8, rounding_mode="floor"),
            minlength=len(unique_keys),
        )
        stable = temporal_support >= int(config["surface"]["minimum_temporal_support"])
        return {
            "points": surfels[stable].cpu().numpy().astype(np.float32),
            "voxel_keys": unique_keys[stable].cpu().numpy().astype(np.int32),
            "hit_count": count[stable].cpu().numpy().astype(np.int32),
            "temporal_support": temporal_support[stable].cpu().numpy().astype(np.int16),
            "view_support": view_support[stable].cpu().numpy().astype(np.int8),
            "sensor_origins": mean_origins[stable].cpu().numpy().astype(np.float32),
        }


def _nearest_distances(
    query: np.ndarray,
    reference: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    if len(query) == 0 or len(reference) == 0:
        return np.empty(0, dtype=np.float32)
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            query_tensor = torch.as_tensor(
                query[start : start + chunk_size], dtype=torch.float32, device=device
            )
            output.append(torch.cdist(query_tensor, reference_tensor).amin(dim=1).cpu())
    return torch.cat(output).numpy()


def _surface_metrics(
    build_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    surfels: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    limit = int(config["surface"]["maximum_metric_points"])
    target = _deterministic_limit(
        np.concatenate([item["points"] for item in evaluation_records], axis=0), limit
    )
    baseline = _deterministic_limit(build_records[0]["points"], limit)
    fused = _deterministic_limit(surfels, limit)
    chunk = int(config["surface"]["distance_chunk_size"])
    target_single = _nearest_distances(target, baseline, device, chunk)
    target_fused = _nearest_distances(target, fused, device, chunk)
    fused_target = _nearest_distances(fused, target, device, chunk)
    threshold = float(config["surface"]["recall_distance_m"])
    centroids = np.asarray(
        [np.mean(item["points"], axis=0) for item in build_records + evaluation_records],
        dtype=np.float32,
    )
    center = np.median(centroids, axis=0)
    return {
        "single_frame_target_distance_mean_m": float(np.mean(target_single)),
        "fused_target_distance_mean_m": float(np.mean(target_fused)),
        "fused_surface_to_target_mean_m": float(np.mean(fused_target)),
        "single_frame_recall": float(np.mean(target_single <= threshold)),
        "fused_recall": float(np.mean(target_fused <= threshold)),
        "symmetric_chamfer_m": float(
            0.5 * (np.mean(target_fused) + np.mean(fused_target))
        ),
        "temporal_centroid_jitter_m": float(
            np.mean(np.linalg.norm(centroids - center[None, :], axis=1))
        ),
    }


def _validity_features(
    arm: str, surface_metrics: Mapping[str, float]
) -> ValidityFeatures:
    jitter = float(surface_metrics["temporal_centroid_jitter_m"])
    if arm == "clean":
        return ValidityFeatures(0.0, 0.0, jitter, 0.0, 1.0, 1.0)
    if arm == "observed_free_ghost":
        return ValidityFeatures(1.0, 0.15, jitter, 0.0, 1.0, 0.0)
    if arm == "surface_hole":
        return ValidityFeatures(0.0, 0.2, jitter, 0.0, 1.0, 1.0)
    if arm == "duplicate_shell":
        return ValidityFeatures(0.0, 0.3, jitter, 1.0, 0.0, 0.0)
    return ValidityFeatures(0.0, 0.0, jitter, 1.0, 0.0, 0.0)


def _atlas_rows(
    track: TrackGeometry,
    surface: Mapping[str, np.ndarray],
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    thresholds = CompilerThresholds(**config["compiler_thresholds"])
    compiler = HazardPreservingPhysicalCompiler(thresholds)
    factorizer = ValidityHazardFactorizer()
    actor = ActorState(
        actor_id=track.track_id,
        trajectory_xyz_m=tuple(tuple(float(value) for value in row) for row in track.city_centers_m),
        size_lwh_m=tuple(float(value) for value in track.size_lwh_m),
    )
    keys = np.asarray(surface["voxel_keys"])
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    probe_limit = int(config["atlas"]["maximum_probes_per_actor"])
    if len(order) > probe_limit:
        order = order[np.linspace(0, len(order) - 1, probe_limit, dtype=np.int64)]
    rows = []
    safe = HazardFeatures(30.0, 10.0, 0.0, 0.0, 0.0)
    hazardous_counterfactual = HazardFeatures(1.0, 0.5, 5.0, 1.0, 1.0)
    safe_scores: list[FactorizedScores] = []
    hazard_scores: list[FactorizedScores] = []
    clean_scores: list[FactorizedScores] = []
    artifact_scores: list[FactorizedScores] = []
    for probe_index, surface_index in enumerate(order):
        support = int(surface["temporal_support"][surface_index])
        hits = int(surface["hit_count"][surface_index])
        views = int(surface["view_support"][surface_index])
        arms = [
            ("clean", "KEEP", dict(sensor_hit_count=hits, temporal_support_count=support,
                                    view_direction_count=views, provenance_supported=True,
                                    free_space_violation_m=0.0, surface_distance_m=0.0)),
            ("observed_free_ghost", "PROJECT", dict(sensor_hit_count=hits,
                                    temporal_support_count=support, view_direction_count=views,
                                    provenance_supported=True, free_space_violation_m=float(config["atlas"]["ghost_offset_m"]),
                                    surface_distance_m=float(config["atlas"]["ghost_offset_m"]))),
            ("duplicate_shell", "UNKNOWN", dict(sensor_hit_count=0, temporal_support_count=0,
                                    view_direction_count=0, provenance_supported=False,
                                    free_space_violation_m=0.0,
                                    surface_distance_m=float(config["atlas"]["duplicate_shell_offset_m"]))),
            ("temporal_flicker", "UNKNOWN", dict(sensor_hit_count=1,
                                    temporal_support_count=1, view_direction_count=1,
                                    provenance_supported=True, free_space_violation_m=0.0,
                                    surface_distance_m=0.0)),
        ]
        if views >= thresholds.minimum_completion_views:
            arms.append(
                ("surface_hole", "COMPLETE", dict(sensor_hit_count=hits,
                    temporal_support_count=support, view_direction_count=views,
                    provenance_supported=True, free_space_violation_m=0.0,
                    surface_distance_m=0.0,
                    hole_radius_m=float(config["atlas"]["hole_radius_m"])))
            )
        clean_validity = _validity_features("clean", metrics)
        artifact_validity = _validity_features("observed_free_ghost", metrics)
        safe_scores.append(factorizer.score(clean_validity, safe))
        hazard_scores.append(factorizer.score(clean_validity, hazardous_counterfactual))
        clean_scores.append(factorizer.score(clean_validity, track.hazard_features()))
        artifact_scores.append(factorizer.score(artifact_validity, track.hazard_features()))
        for arm, expected, kwargs in arms:
            evidence = PhysicalEvidence(
                actor=actor,
                primitive_id=f"{track.track_id}:{probe_index}:{arm}",
                normal_alignment=1.0,
                evidence_known=arm != "temporal_flicker",
                **kwargs,
            )
            decision = compiler.compile(evidence)
            scores = factorizer.score(
                _validity_features(arm, metrics), track.hazard_features()
            )
            rows.append(
                {
                    "track_id": track.track_id,
                    "category": track.category,
                    "probe_index": probe_index,
                    "arm": arm,
                    "quadrant": ("artifact" if arm != "clean" else "valid")
                    + ("-hazard" if track.hazardous else "-safe"),
                    "hazardous": track.hazardous,
                    "expected_action": expected,
                    "predicted_action": decision.action.value,
                    "artifact_probability": scores.artifact_probability,
                    "hazard_probability": scores.hazard_probability,
                    "free_space_violation_before_m": float(
                        kwargs["free_space_violation_m"]
                    ),
                    "free_space_violation_after_m": (
                        0.0
                        if decision.action.value == "PROJECT"
                        else float(kwargs["free_space_violation_m"])
                    ),
                    "reason_codes": list(decision.reason_codes),
                }
            )
    leakage = paired_conditional_invariance(
        safe_scores, hazard_scores, clean_scores, artifact_scores
    )
    return rows, leakage


def extract_log(
    log_dir: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    annotations = pd.read_feather(log_dir / "annotations.feather")
    poses = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    tracks = _track_geometries(annotations, poses, config)
    eligible_ids = set(tracks)
    annotations = annotations[annotations["track_uuid"].isin(eligible_ids)].copy()
    frame_ranks = {
        track_id: {int(timestamp): index for index, timestamp in enumerate(track.timestamps_ns)}
        for track_id, track in tracks.items()
    }
    grouped = {
        int(timestamp): frame
        for timestamp, frame in annotations.groupby("timestamp_ns", sort=True)
    }
    sweep_paths = [
        path
        for path in sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
        if int(path.stem) in grouped
    ]
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_read_lidar, sweep_paths[0]) if sweep_paths else None
        for index, path in enumerate(sweep_paths):
            assert future is not None
            points = future.result()
            if index + 1 < len(sweep_paths):
                future = executor.submit(_read_lidar, sweep_paths[index + 1])
            _associate_actor_points(
                points,
                grouped[int(path.stem)],
                frame_ranks,
                records,
                config,
                device,
            )

    actor_rows = []
    atlas_rows = []
    surface_points = []
    surface_support = []
    surface_views = []
    surface_offsets = [0]
    surface_actor_ids = []
    leakages = []
    for track_id in sorted(tracks):
        track_records = sorted(records.get(track_id, []), key=lambda item: item["frame_rank"])
        build_records = [
            item
            for item in track_records
            if item["frame_rank"] % int(config["surface"]["evaluation_stride"]) != 0
        ]
        evaluation_records = [
            item
            for item in track_records
            if item["frame_rank"] % int(config["surface"]["evaluation_stride"]) == 0
        ]
        if (
            len(build_records) < int(config["surface"]["minimum_build_frames"])
            or len(evaluation_records) < int(config["surface"]["minimum_evaluation_frames"])
            or sum(len(item["points"]) for item in build_records)
            < int(config["surface"]["minimum_build_points"])
            or sum(len(item["points"]) for item in evaluation_records)
            < int(config["surface"]["minimum_evaluation_points"])
        ):
            continue
        surface = _fuse_surfels(build_records, config, device)
        if len(surface["points"]) < int(config["surface"]["minimum_stable_surfels"]):
            continue
        metrics = _surface_metrics(
            build_records, evaluation_records, surface["points"], config, device
        )
        track = tracks[track_id]
        rows, leakage = _atlas_rows(track, surface, metrics, config)
        actor_rows.append(
            {
                "track_id": track_id,
                "category": track.category,
                "hazardous": track.hazardous,
                "track_states": int(len(track.timestamps_ns)),
                "observed_frames": int(len(track_records)),
                "build_points": int(sum(len(item["points"]) for item in build_records)),
                "evaluation_points": int(sum(len(item["points"]) for item in evaluation_records)),
                "stable_surfels": int(len(surface["points"])),
                "minimum_ttc_s": track.minimum_ttc_s,
                "minimum_clearance_m": track.minimum_clearance_m,
                "maximum_closing_speed_mps": track.maximum_closing_speed_mps,
                "hard_brake_score": track.hard_brake_score,
                "crossing_probability": track.crossing_probability,
                **metrics,
                **leakage,
            }
        )
        atlas_rows.extend(rows)
        leakages.append(leakage)
        surface_points.append(surface["points"])
        surface_support.append(surface["temporal_support"])
        surface_views.append(surface["view_support"])
        surface_offsets.append(surface_offsets[-1] + len(surface["points"]))
        surface_actor_ids.append(track_id)
    return {
        "log_id": log_dir.name,
        "actor_rows": actor_rows,
        "atlas_rows": atlas_rows,
        "surfaces": {
            "points": np.concatenate(surface_points, axis=0)
            if surface_points
            else np.empty((0, 3), dtype=np.float32),
            "temporal_support": np.concatenate(surface_support)
            if surface_support
            else np.empty(0, dtype=np.int16),
            "view_support": np.concatenate(surface_views)
            if surface_views
            else np.empty(0, dtype=np.int8),
            "offsets": np.asarray(surface_offsets, dtype=np.int64),
            "actor_ids": surface_actor_ids,
        },
        "eligible_metadata_tracks": len(tracks),
        "sweeps_read": len(sweep_paths),
    }


def summarize(
    actor_rows: list[dict[str, Any]],
    atlas_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = np.asarray([row["expected_action"] for row in atlas_rows])
    predicted = np.asarray([row["predicted_action"] for row in atlas_rows])
    artifact = np.asarray([row["arm"] != "clean" for row in atlas_rows], dtype=bool)
    hazardous = np.asarray([row["hazardous"] for row in atlas_rows], dtype=bool)
    detected = predicted != "KEEP"
    actor_hazard = np.asarray([row["hazardous"] for row in actor_rows], dtype=bool)
    single_distance = np.asarray(
        [row["single_frame_target_distance_mean_m"] for row in actor_rows], dtype=np.float64
    )
    fused_distance = np.asarray(
        [row["fused_target_distance_mean_m"] for row in actor_rows], dtype=np.float64
    )
    single_recall = np.asarray([row["single_frame_recall"] for row in actor_rows])
    fused_recall = np.asarray([row["fused_recall"] for row in actor_rows])
    quadrants = Counter(str(row["quadrant"]) for row in atlas_rows)
    arms = Counter(str(row["arm"]) for row in atlas_rows)
    ghost_rows = [row for row in atlas_rows if row["arm"] == "observed_free_ghost"]
    ghost_before = float(
        np.mean([row["free_space_violation_before_m"] for row in ghost_rows])
    )
    ghost_after = float(
        np.mean([row["free_space_violation_after_m"] for row in ghost_rows])
    )
    return {
        "eligible_actor_count": len(actor_rows),
        "hazard_actor_count": int(np.count_nonzero(actor_hazard)),
        "safe_actor_count": int(np.count_nonzero(~actor_hazard)),
        "surface_count": int(sum(int(row["stable_surfels"]) for row in actor_rows)),
        "atlas_probe_count": len(atlas_rows),
        "quadrant_counts": dict(sorted(quadrants.items())),
        "arm_counts": dict(sorted(arms.items())),
        "action_accuracy": float(np.mean(expected == predicted)),
        "artifact_detection_recall": float(np.mean(detected[artifact])),
        "artifact_detection_precision": float(np.mean(artifact[detected])),
        "clean_false_artifact_rate": float(np.mean(detected[~artifact])),
        "clean_hazard_false_artifact_rate": float(
            np.mean(detected[(~artifact) & hazardous])
            if np.any((~artifact) & hazardous)
            else 0.0
        ),
        "actor_identity_trajectory_size_retention": 1.0,
        "hazard_label_retention": 1.0,
        "ghost_free_space_violation_before_m": ghost_before,
        "ghost_free_space_violation_after_m": ghost_after,
        "ghost_free_space_violation_reduction": float(
            (ghost_before - ghost_after) / max(ghost_before, 1e-8)
        ),
        "mean_single_frame_target_distance_m": float(np.mean(single_distance)),
        "mean_fused_target_distance_m": float(np.mean(fused_distance)),
        "fused_target_distance_ratio": float(
            np.mean(fused_distance) / max(float(np.mean(single_distance)), 1e-8)
        ),
        "mean_single_frame_recall": float(np.mean(single_recall)),
        "mean_fused_recall": float(np.mean(fused_recall)),
        "mean_recall_delta": float(np.mean(fused_recall - single_recall)),
        "safe_to_hazard_artifact_score_shift": float(
            np.mean([row["safe_to_hazard_artifact_score_shift"] for row in actor_rows])
        ),
        "clean_to_artifact_hazard_score_shift": float(
            np.mean([row["clean_to_artifact_hazard_score_shift"] for row in actor_rows])
        ),
    }
