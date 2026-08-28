"""将冻结两级certificate烘焙为Actor-preserving runtime package。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from motion_proj.worldsim_v61.occupancy import UNKNOWN


PACKAGE_FILES = (
    "STATIC_STATE.npz",
    "ACTORS.jsonl",
    "ACTOR_PRIMITIVES.npz",
    "ARTIFACT_FACTORS.jsonl",
    "REPAIR_LOG.jsonl",
    "HAZARD_ATTRIBUTES.jsonl",
    "PROVENANCE.jsonl",
    "RUNTIME_MANIFEST.json",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rle_encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = np.asarray(values, dtype=np.uint8).reshape(-1)
    starts = np.r_[0, np.flatnonzero(flattened[1:] != flattened[:-1]) + 1]
    lengths = np.diff(np.r_[starts, flattened.size]).astype(np.int32)
    return flattened[starts], lengths


def _counts(values: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {int(actor_id): int(count) for actor_id, count in zip(ids, counts)}


def _actor_id(base_id: str) -> int:
    return int(str(base_id).rsplit("actor-", maxsplit=1)[1])


def _trajectory_row(info: Mapping[str, Any], frame: int) -> dict[str, Any]:
    annotations = info["frame_annotations"]
    frames = [int(value) for value in annotations["frame_idx"]]
    position = frames.index(int(frame))
    transform = np.asarray(annotations["obj_to_world"][position], dtype=np.float64)
    return {
        "frame": int(frame),
        "center_global_m": [float(value) for value in transform[:3, 3]],
        "yaw_global_rad": float(math.atan2(transform[1, 0], transform[0, 0])),
        "box_size_lwh_m": [
            float(value) for value in annotations["box_size"][position]
        ],
    }


def _hazard_attributes(
    scene_root: Path,
    info: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    trajectory = [_trajectory_row(info, frame) for frame in sorted(set(target_frames))]
    distances = []
    for row in trajectory:
        ego = np.loadtxt(scene_root / f"lidar_pose/{int(row['frame']):03d}.txt")[:3, 3]
        actor = np.asarray(row["center_global_m"], dtype=np.float64)
        distances.append(float(np.linalg.norm(actor[:2] - ego[:2])))
    closing = [
        max(0.0, (left - right) / ((b["frame"] - a["frame"]) * 0.1))
        for left, right, a, b in zip(distances[:-1], distances[1:], trajectory[:-1], trajectory[1:])
        if b["frame"] > a["frame"]
    ]
    actor_speed = [
        float(
            np.linalg.norm(
                np.asarray(b["center_global_m"])[:2]
                - np.asarray(a["center_global_m"])[:2]
            )
            / ((b["frame"] - a["frame"]) * 0.1)
        )
        for a, b in zip(trajectory[:-1], trajectory[1:])
        if b["frame"] > a["frame"]
    ]
    return {
        "sampled_frame_count": len(trajectory),
        "minimum_ego_center_distance_m": min(distances) if distances else None,
        "maximum_closing_speed_mps": max(closing, default=0.0),
        "maximum_actor_speed_mps": max(actor_speed, default=0.0),
        "controls_actor_existence": False,
        "use": "P7_distribution_audit_candidate",
    }


def bake_package(config: Mapping[str, Any], runs_root: Path, package_dir: Path) -> dict[str, Any]:
    sources = config["sources"]
    evidence_root = runs_root / str(sources["evidence_run"])
    score_path = (
        runs_root
        / str(sources["local_geometry_run"])
        / str(sources["local_geometry_scores"])
    )
    score_rows = sorted(_jsonl(score_path), key=lambda row: str(row["base_id"]))
    sanitized_scores = {
        str(row["base_id"]): {
            "base_id": str(row["base_id"]),
            "scene": str(row["scene"]),
            "q0_mean": float(row["q0_mean"]),
            "p_local_conflict": float(row["p_local_conflict"]),
        }
        for row in score_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sanitized_scores.values():
        _, unit, _ = row["base_id"].split("/")
        grouped[(row["scene"], unit)].append(row)

    static_values: list[np.ndarray] = []
    static_lengths: list[np.ndarray] = []
    static_offsets = [0]
    unit_names = []
    primitive_indices: list[np.ndarray] = []
    primitive_support: list[np.ndarray] = []
    primitive_offsets = [0]
    state_rows = []
    factor_rows = []
    repair_rows = []
    provenance_rows = []
    actor_state_indices: dict[str, list[int]] = defaultdict(list)
    actor_target_frames: dict[str, list[int]] = defaultdict(list)
    unit_grid_origin = None
    unit_voxel_size = None
    unit_grid_shape = None

    for scene, unit in sorted(grouped):
        path = evidence_root / "units" / scene / unit / "METHOD_EVIDENCE.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
        semantics = np.asarray(arrays["semantics"], dtype=np.uint8).copy()
        envelope = np.asarray(arrays["actor_envelope_indices"], dtype=np.int32)
        if envelope.size:
            semantics[envelope[:, 0], envelope[:, 1], envelope[:, 2]] = UNKNOWN
        values, lengths = _rle_encode(semantics)
        static_values.append(values)
        static_lengths.append(lengths)
        static_offsets.append(static_offsets[-1] + len(values))
        unit_names.append(f"{scene}/{unit}")
        unit_grid_origin = np.asarray(arrays["grid_origin_m"], dtype=np.float64)
        unit_voxel_size = float(arrays["voxel_size_m"])
        unit_grid_shape = np.asarray(arrays["grid_shape"], dtype=np.int64)

        current_indices = np.asarray(arrays["actor_current_envelope_indices"], dtype=np.int32)
        current_ids = np.asarray(arrays["actor_current_envelope_ids"], dtype=np.int32)
        hit_indices = np.asarray(arrays["actor_hit_indices"], dtype=np.int32)
        hit_ids = np.asarray(arrays["actor_hit_ids"], dtype=np.int32)
        hit_counts = _counts(hit_ids)
        current_counts = _counts(current_ids)
        swept_counts = _counts(arrays["actor_swept_envelope_ids"])
        shape = tuple(int(value) for value in unit_grid_shape)
        target_frame = int(unit.removeprefix("f"))

        for score in sorted(grouped[(scene, unit)], key=lambda row: str(row["base_id"])):
            actor_id = _actor_id(score["base_id"])
            owned = current_indices[current_ids == actor_id]
            actor_hits = hit_indices[hit_ids == actor_id]
            owned_linear = np.ravel_multi_index(owned.T, shape)
            hit_linear = (
                np.ravel_multi_index(actor_hits.T, shape)
                if actor_hits.size
                else np.empty(0, dtype=np.int64)
            )
            support = np.isin(owned_linear, hit_linear).astype(np.uint8)
            state_index = len(state_rows)
            actor_key = f"{scene}/actor-{actor_id}"
            actor_state_indices[actor_key].append(state_index)
            actor_target_frames[actor_key].append(target_frame)
            primitive_indices.append(owned)
            primitive_support.append(support)
            primitive_offsets.append(primitive_offsets[-1] + len(owned))
            state_rows.append(
                {
                    "state_index": state_index,
                    "base_id": score["base_id"],
                    "actor_key": actor_key,
                    "scene": scene,
                    "unit": unit,
                    "actor_id": actor_id,
                    "existence_state": "SUPPORTED_ACTOR",
                    "local_geometry_state": "RANKED_UNCERTAINTY",
                    "local_geometry_score": score["p_local_conflict"],
                    "repair_action": "RANK_REPAIR_OR_ABSTAIN",
                    "owned_primitive_offset": primitive_offsets[-2],
                    "owned_primitive_count": len(owned),
                }
            )
            factor_rows.append(
                {
                    "base_id": score["base_id"],
                    "actor_key": actor_key,
                    "sensor_hit_count": hit_counts.get(actor_id, 0),
                    "current_envelope_count": current_counts.get(actor_id, 0),
                    "swept_envelope_count": swept_counts.get(actor_id, 0),
                    "q0_mean": score["q0_mean"],
                    "local_geometry_ranking_score": score["p_local_conflict"],
                    "deterministic_existence_reasons": [],
                }
            )
            repair_rows.append(
                {
                    "base_id": score["base_id"],
                    "actor_key": actor_key,
                    "action": "RANK_REPAIR_OR_ABSTAIN",
                    "actor_removed": False,
                    "physical_geometry_mutated": False,
                    "threshold": None,
                }
            )
            provenance_rows.append(
                {
                    "base_id": score["base_id"],
                    "actor_key": actor_key,
                    "evidence_unit": str(path.relative_to(evidence_root)),
                    "sensor_hit_count": hit_counts.get(actor_id, 0),
                    "owned_primitive_count": len(owned),
                    "source_kind": "observable_method_evidence_and_frozen_offline_score",
                }
            )

    package_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        package_dir / "STATIC_STATE.npz",
        units=np.asarray(unit_names),
        rle_offsets=np.asarray(static_offsets, dtype=np.int64),
        rle_values=np.concatenate(static_values).astype(np.uint8),
        rle_lengths=np.concatenate(static_lengths).astype(np.int32),
        grid_origin_m=unit_grid_origin,
        voxel_size_m=np.asarray(unit_voxel_size, dtype=np.float64),
        grid_shape=unit_grid_shape,
    )
    np.savez_compressed(
        package_dir / "ACTOR_PRIMITIVES.npz",
        state_base_ids=np.asarray([row["base_id"] for row in state_rows]),
        primitive_offsets=np.asarray(primitive_offsets, dtype=np.int64),
        primitive_indices=np.concatenate(primitive_indices).astype(np.int32),
        primitive_sensor_supported=np.concatenate(primitive_support).astype(np.uint8),
        grid_origin_m=unit_grid_origin,
        voxel_size_m=np.asarray(unit_voxel_size, dtype=np.float64),
        grid_shape=unit_grid_shape,
    )

    processed_root = Path(str(sources["processed_root"]))
    scene_indices = {str(row["name"]): int(row["processed_index"]) for row in config["scenes"]}
    actor_rows = []
    hazard_rows = []
    for actor_key in sorted(actor_state_indices):
        scene, actor_name = actor_key.split("/")
        actor_id = int(actor_name.removeprefix("actor-"))
        scene_root = processed_root / f"{scene_indices[scene]:03d}"
        instances = json.loads(
            (scene_root / "instances/instances_info.json").read_text(encoding="utf-8")
        )
        info = instances[str(actor_id)]
        frames = sorted(set(actor_target_frames[actor_key]))
        trajectory = [_trajectory_row(info, frame) for frame in frames]
        actor_rows.append(
            {
                "actor_key": actor_key,
                "scene": scene,
                "actor_id": actor_id,
                "class": str(info["class_name"]),
                "track_id": actor_key,
                "lifecycle": {
                    "first_frame": int(min(info["frame_annotations"]["frame_idx"])),
                    "last_frame": int(max(info["frame_annotations"]["frame_idx"])),
                    "annotation_count": len(info["frame_annotations"]["frame_idx"]),
                },
                "trajectory": trajectory,
                "state_indices": actor_state_indices[actor_key],
                "existence_authority": "DETERMINISTIC_SUPPORT_ONLY",
            }
        )
        hazard_rows.append(
            {
                "actor_key": actor_key,
                **_hazard_attributes(scene_root, info, frames),
            }
        )

    _write_jsonl(package_dir / "ACTORS.jsonl", actor_rows)
    _write_jsonl(package_dir / "ARTIFACT_FACTORS.jsonl", factor_rows)
    _write_jsonl(package_dir / "REPAIR_LOG.jsonl", repair_rows)
    _write_jsonl(package_dir / "HAZARD_ATTRIBUTES.jsonl", hazard_rows)
    _write_jsonl(package_dir / "PROVENANCE.jsonl", provenance_rows)
    manifest = {
        "schema_version": "worldsim_v66.harp_runtime_manifest.v1",
        "task_id": config["task_id"],
        "package_files": list(PACKAGE_FILES[:-1]),
        "counts": {
            "unit_count": len(unit_names),
            "actor_count": len(actor_rows),
            "actor_state_count": len(state_rows),
            "actor_primitive_count": primitive_offsets[-1],
        },
        "runtime_contract": {
            "learned_model_loaded": False,
            "hidden_target_loaded": False,
            "hazard_controls_actor_existence": False,
            "actor_static_layers_separated": True,
            "physical_appearance_layers_separated": True,
            "local_geometry_threshold": None,
            "local_geometry_action": "RANK_REPAIR_OR_ABSTAIN",
            "actor_existence_authority": "DETERMINISTIC_SUPPORT_ONLY",
        },
        "claim_boundary": config["claim_boundary"],
    }
    (package_dir / "RUNTIME_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "actor_rows": actor_rows,
        "state_rows": state_rows,
        "factor_rows": factor_rows,
        "repair_rows": repair_rows,
        "hazard_rows": hazard_rows,
        "provenance_rows": provenance_rows,
    }
