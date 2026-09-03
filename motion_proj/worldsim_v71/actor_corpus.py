"""紧凑 Actor corpus 的物化与读取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries


def _deterministic_limit(points: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points)
    if len(points) <= maximum:
        return points, np.arange(len(points), dtype=np.int64)
    indices = np.linspace(0, len(points) - 1, num=int(maximum), dtype=np.int64)
    return points[indices], indices


def materialize_actor_cache(
    bundle: Mapping[str, Any],
    output_path: Path,
    config: Mapping[str, Any],
    *,
    oracle_by_track: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    diagnostics = bundle["diagnostics"]
    row = bundle["row"]
    track_id = str(row["track_id"])
    build_points_parts = [
        np.asarray(points, dtype=np.float32).reshape(-1, 3)
        for points in diagnostics["build_frame_points"]
    ]
    build_origins_parts = [
        np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(points), axis=0)
        for points, origin in zip(build_points_parts, diagnostics["build_sensor_origins"])
    ]
    build_points = np.concatenate(build_points_parts, axis=0)
    build_origins = np.concatenate(build_origins_parts, axis=0)
    build_points, selected = _deterministic_limit(
        build_points, int(config["maximum_build_evidence_points"])
    )
    build_origins = build_origins[selected]
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32).reshape(-1, 3)
    evidence = build_evidential_queries(
        candidates,
        build_origins,
        build_points,
        beam_radius_m=float(config["beam_radius_m"]),
        endpoint_radius_m=float(config["endpoint_radius_m"]),
        device=device,
        query_chunk_size=int(config["query_chunk_size"]),
    )
    anchors = np.concatenate(
        [
            np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
            np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
        ],
        axis=0,
    )
    canonical, _ = _deterministic_limit(
        np.asarray(diagnostics["canonical"], dtype=np.float32).reshape(-1, 3),
        int(config["maximum_input_surfels"]),
    )
    target, selected_target = _deterministic_limit(
        np.asarray(diagnostics["target"], dtype=np.float32).reshape(-1, 3),
        int(config["maximum_target_rays"]),
    )
    target_origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32).reshape(-1, 3)[selected_target]
    oracle = oracle_by_track.get(track_id)
    oracle_displacement = np.empty((0, 3), dtype=np.float32)
    if oracle is not None:
        candidate_displacement = np.asarray(oracle["displacement"], dtype=np.float32).reshape(-1, 3)
        if len(candidate_displacement) == len(candidates):
            oracle_displacement = candidate_displacement
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("worldsim_v71.actor_corpus.v1"),
        track_id=np.asarray(track_id),
        scene_name=np.asarray(bundle["scene_name"]),
        category=np.asarray(str(row["category"])),
        hazardous=np.asarray(bool(row["hazardous"])),
        size_lwh_m=np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32),
        trajectory_xyz_m=np.asarray(diagnostics["track"].city_centers_m, dtype=np.float32),
        canonical=canonical.astype(np.float32),
        anchors=anchors.astype(np.float32),
        candidates=candidates.astype(np.float32),
        base_features=np.asarray(diagnostics["completion_features"], dtype=np.float32),
        evidence_masses=evidence.masses.astype(np.float32),
        evidence_opportunities=evidence.opportunity_count.astype(np.int32),
        query_sensor_origin=np.asarray(diagnostics["query_sensor_origin"], dtype=np.float32),
        target=target.astype(np.float32),
        target_sensor_origins=target_origins.astype(np.float32),
        oracle_displacement=oracle_displacement,
    )
    temporary.replace(output_path)
    return {
        "track_id": track_id,
        "hazardous": bool(row["hazardous"]),
        "candidate_count": len(candidates),
        "oracle_target": bool(len(oracle_displacement)),
        "bytes": output_path.stat().st_size,
    }


def load_actor_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
