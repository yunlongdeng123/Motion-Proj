"""在独立 legacy cohort 上诊断 Actor-owned observed-FREE 局部几何冲突。"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v61.occupancy import FREE
from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.native_voxel_uq import (
    _evidence_on_native_grid,
    _native_unit_dir,
    _unit_dirs,
)
from motion_proj.worldsim_v66.actor_factorial import (
    _count_ids,
    _ground_actor_ids,
    _q0_scores,
    certificate,
)


def _load_unit(
    descriptor: tuple[int, str, Path, Path],
    *,
    origin: np.ndarray,
    voxel_size: float,
    point_limit: int,
    seed: int,
) -> dict[str, Any]:
    scene_index, scene, evidence_unit, native_unit = descriptor
    indices, centers, features = _target_free_boundary(
        evidence_unit,
        native_unit,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    with np.load(evidence_unit / "METHOD_EVIDENCE.npz", allow_pickle=False) as source:
        method = {name: np.asarray(source[name]) for name in source.files}
    with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
        target = {name: np.asarray(source[name]) for name in source.files}
    native_shape = tuple(
        int(value) for value in np.load(native_unit / "ARGMAX.npy", mmap_mode="r").shape
    )
    target_state, target_valid = _evidence_on_native_grid(
        target,
        native_shape=native_shape,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    x, y, z = indices.T
    valid = target_valid[x, y, z]
    indices, centers, features = indices[valid], centers[valid], features[valid]
    labels = np.asarray(target_state[x, y, z][valid] == FREE, dtype=bool)
    if features.shape[0] > int(point_limit):
        unit_seed = int(seed) + scene_index * 1009 + int(evidence_unit.name.removeprefix("f"))
        rng = np.random.default_rng(unit_seed)
        chosen = np.sort(rng.choice(features.shape[0], size=int(point_limit), replace=False))
        indices, centers, features, labels = (
            indices[chosen],
            centers[chosen],
            features[chosen],
            labels[chosen],
        )
    actor_ids, ambiguous_query_count = _ground_actor_ids(centers, method)
    return {
        "scene_index": int(scene_index),
        "scene": scene,
        "unit": evidence_unit.name,
        "features": np.asarray(features, dtype=np.float32),
        "labels": labels,
        "actor_ids": actor_ids,
        "hit_counts": _count_ids(method["actor_hit_ids"]),
        "current_counts": _count_ids(method["actor_current_envelope_ids"]),
        "swept_counts": _count_ids(method["actor_swept_envelope_ids"]),
        "ambiguous_query_count": ambiguous_query_count,
    }


def _actor_rows(
    loaded: Mapping[str, Any],
    q0_scores: np.ndarray,
    minimum_boundary_points: int,
    certificate_config: Mapping[str, float],
) -> list[dict[str, Any]]:
    actor_ids = np.asarray(loaded["actor_ids"], dtype=np.int32)
    labels = np.asarray(loaded["labels"], dtype=bool)
    rows = []
    for actor_id in np.unique(actor_ids[actor_ids >= 0]):
        members = actor_ids == int(actor_id)
        boundary_count = int(np.count_nonzero(members))
        hit_count = int(loaded["hit_counts"].get(int(actor_id), 0))
        current_count = int(loaded["current_counts"].get(int(actor_id), 0))
        swept_count = int(loaded["swept_counts"].get(int(actor_id), 0))
        if (
            boundary_count < int(minimum_boundary_points)
            or hit_count <= 0
            or current_count <= 0
            or swept_count <= 0
        ):
            continue
        hidden_free_count = int(np.count_nonzero(labels[members]))
        row = {
            "base_id": f"{loaded['scene']}/{loaded['unit']}/actor-{int(actor_id)}",
            "scene_index": int(loaded["scene_index"]),
            "scene": str(loaded["scene"]),
            "unit": str(loaded["unit"]),
            "actor_id": int(actor_id),
            "boundary_count": boundary_count,
            "hidden_free_count": hidden_free_count,
            "hidden_free_rate": float(hidden_free_count / boundary_count),
            "local_geometry_conflict": bool(hidden_free_count > 0),
            "q0_mean": float(np.mean(q0_scores[members])),
            "q0_p90": float(np.quantile(q0_scores[members], 0.9)),
            "sensor_hit_count": hit_count,
            "current_envelope_count": current_count,
            "swept_envelope_count": swept_count,
            "provenance_supported": True,
            "duplicate_overlap": 0.0,
            "lifecycle_gap_count": 0,
            "kinematic_jump_m": 0.0,
            "identity_discontinuity": False,
            "shape_ratio_jump": 1.0,
        }
        score, state, reasons = certificate(
            row,
            maximum_kinematic_jump_m=float(
                certificate_config["maximum_supported_kinematic_jump_m"]
            ),
            maximum_shape_ratio_jump=float(
                certificate_config["maximum_shape_ratio_jump"]
            ),
        )
        row.update(
            certificate_score=score,
            certificate_state=state,
            certificate_reason_codes=reasons,
        )
        rows.append(row)
    return rows


def materialize_rows(config: Mapping[str, Any], runs_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    inputs = config["inputs"]
    evidence_root = runs_root / str(inputs["evidence_run"])
    native_root = runs_root / str(inputs["native_run"])
    partition = str(inputs["native_partition"])
    model = joblib.load(runs_root / str(inputs["risk_run"]) / str(inputs["risk_model_relative_path"]))
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    descriptors = []
    for scene_index, scene in enumerate(config["scenes"]):
        for evidence_unit in _unit_dirs(evidence_root, str(scene)):
            descriptors.append(
                (
                    scene_index,
                    str(scene),
                    evidence_unit,
                    _native_unit_dir(
                        native_root,
                        str(scene),
                        evidence_unit.name,
                        {str(scene): partition},
                    ),
                )
            )
    rows = []
    ambiguous_query_count = 0
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["sampling"]["io_prefetch_workers"])
    )

    def submit(descriptor: tuple[int, str, Path, Path]):
        return executor.submit(
            _load_unit,
            descriptor,
            origin=origin,
            voxel_size=voxel_size,
            point_limit=int(config["sampling"]["maximum_boundary_points_per_unit"]),
            seed=int(config["seed"]),
        )

    future = submit(descriptors[0])
    try:
        for position, descriptor in enumerate(descriptors):
            loaded = future.result()
            if position + 1 < len(descriptors):
                future = submit(descriptors[position + 1])
            q0 = _q0_scores(model, loaded["features"])
            unit_rows = _actor_rows(
                loaded,
                q0,
                int(config["sampling"]["minimum_actor_boundary_points"]),
                config["certificate"],
            )
            rows.extend(unit_rows)
            ambiguous_query_count += int(loaded["ambiguous_query_count"])
            print(
                f"P2N {position + 1}/{len(descriptors)} scene={descriptor[1]} "
                f"unit={descriptor[2].name} actors={len(unit_rows)}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
    return rows, {
        "source_unit_count": len(descriptors),
        "eligible_actor_unit_count": len(rows),
        "ambiguous_grounded_query_count": ambiguous_query_count,
    }


def evaluate_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([bool(row["local_geometry_conflict"]) for row in rows])
    target = np.asarray([float(row["hidden_free_rate"]) for row in rows])
    q0 = np.asarray([float(row["q0_mean"]) for row in rows])
    certificate_scores = np.asarray([float(row["certificate_score"]) for row in rows])
    if np.unique(labels).size != 2:
        raise RuntimeError("P2N requires both conflict classes")
    return {
        "row_count": len(rows),
        "conflict_actor_unit_count": int(np.count_nonzero(labels)),
        "clean_actor_unit_count": int(np.count_nonzero(~labels)),
        "conflict_prevalence": float(np.mean(labels)),
        "q0_conflict_auroc": float(roc_auc_score(labels, q0)),
        "q0_conflict_auprc": float(average_precision_score(labels, q0)),
        "q0_hidden_free_rate_spearman": float(spearmanr(target, q0).statistic),
        "certificate_conflict_recall": float(np.mean(certificate_scores[labels] >= 0.5)),
        "certificate_clean_false_conflict": float(np.mean(certificate_scores[~labels] >= 0.5)),
        "certificate_conflict_auroc": float(roc_auc_score(labels, certificate_scores)),
        "certificate_conflict_auprc": float(average_precision_score(labels, certificate_scores)),
    }
