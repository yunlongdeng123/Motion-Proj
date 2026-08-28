"""构建 Actor-grounded validity×hazard 配对 atlas。"""

from __future__ import annotations

import concurrent.futures
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.native_voxel_uq import _native_unit_dir, _unit_dirs


ARTIFACT_FAMILIES = (
    "unsupported_ghost",
    "duplicate_shell",
    "lifecycle_flicker",
    "teleport",
    "shape_jump",
)


def _count_ids(values: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {int(actor_id): int(count) for actor_id, count in zip(ids, counts)}


def _ground_actor_ids(
    centers: np.ndarray,
    method: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    origin = np.asarray(method["grid_origin_m"], dtype=np.float64)
    voxel_size = float(method["voxel_size_m"])
    shape = tuple(int(value) for value in np.asarray(method["grid_shape"]))
    query_indices = np.floor(
        (np.asarray(centers, dtype=np.float64) - origin[None, :]) / voxel_size
    ).astype(np.int64)
    valid = np.all(
        (query_indices >= 0) & (query_indices < np.asarray(shape)[None, :]), axis=1
    )
    query_linear = np.full(query_indices.shape[0], -1, dtype=np.int64)
    query_linear[valid] = np.ravel_multi_index(query_indices[valid].T, shape)

    envelope_indices = np.asarray(method["actor_envelope_indices"], dtype=np.int64)
    envelope_ids = np.asarray(method["actor_envelope_ids"], dtype=np.int64)
    result = np.full(query_indices.shape[0], -1, dtype=np.int32)
    if envelope_indices.shape[0] == 0:
        return result, 0
    envelope_linear = np.ravel_multi_index(envelope_indices.T, shape)
    order = np.argsort(envelope_linear, kind="stable")
    sorted_linear = envelope_linear[order]
    sorted_ids = envelope_ids[order]
    positions = np.searchsorted(sorted_linear, query_linear)
    matched = valid & (positions < sorted_linear.size)
    matched_indices = np.flatnonzero(matched)
    matched[matched_indices] = (
        sorted_linear[positions[matched_indices]] == query_linear[matched_indices]
    )
    result[matched] = sorted_ids[positions[matched]].astype(np.int32)

    unique_linear, counts = np.unique(envelope_linear, return_counts=True)
    ambiguous = set(unique_linear[counts > 1].tolist())
    ambiguous_query_count = sum(int(value) in ambiguous for value in query_linear[matched])
    return result, int(ambiguous_query_count)


def _load_actor_unit(
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
    if features.shape[0] > int(point_limit):
        unit_seed = int(seed) + scene_index * 1009 + int(evidence_unit.name.removeprefix("f"))
        rng = np.random.default_rng(unit_seed)
        chosen = np.sort(rng.choice(features.shape[0], size=int(point_limit), replace=False))
        indices, centers, features = indices[chosen], centers[chosen], features[chosen]
    with np.load(evidence_unit / "METHOD_EVIDENCE.npz", allow_pickle=False) as source:
        method = {name: np.asarray(source[name]) for name in source.files}
    actor_ids, ambiguous_query_count = _ground_actor_ids(centers, method)
    x, y, z = indices.T
    entropy = np.asarray(np.load(native_unit / "ENTROPY.npy", mmap_mode="r")[x, y, z])
    margin = np.asarray(np.load(native_unit / "MARGIN.npy", mmap_mode="r")[x, y, z])
    return {
        "scene_index": int(scene_index),
        "scene": scene,
        "unit": evidence_unit.name,
        "features": np.asarray(features, dtype=np.float32),
        "actor_ids": actor_ids,
        "entropy": entropy.astype(np.float32),
        "margin": margin.astype(np.float32),
        "hit_counts": _count_ids(method["actor_hit_ids"]),
        "current_counts": _count_ids(method["actor_current_envelope_ids"]),
        "swept_counts": _count_ids(method["actor_swept_envelope_ids"]),
        "ambiguous_query_count": ambiguous_query_count,
    }


def _q0_scores(model: object, features: np.ndarray) -> np.ndarray:
    values = (np.asarray(features, dtype=np.float32) - model.mean) / model.scale
    network = model.model.cuda().eval()
    parts = []
    with torch.inference_mode():
        for offset in range(0, values.shape[0], 131072):
            batch = torch.from_numpy(values[offset : offset + 131072]).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = network(batch).reshape(-1)
            parts.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _base_actor_rows(
    loaded: Mapping[str, Any],
    scores: np.ndarray,
    minimum_boundary_points: int,
) -> list[dict[str, Any]]:
    actor_ids = np.asarray(loaded["actor_ids"], dtype=np.int32)
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
        actor_scores = scores[members]
        rows.append(
            {
                "base_id": f"{loaded['scene']}/{loaded['unit']}/actor-{int(actor_id)}",
                "scene_index": int(loaded["scene_index"]),
                "scene": str(loaded["scene"]),
                "unit": str(loaded["unit"]),
                "actor_id": int(actor_id),
                "boundary_count": boundary_count,
                "sensor_hit_count": hit_count,
                "current_envelope_count": current_count,
                "swept_envelope_count": swept_count,
                "q0_mean": float(np.mean(actor_scores)),
                "q0_p90": float(np.quantile(actor_scores, 0.9)),
                "entropy_mean": float(np.mean(np.asarray(loaded["entropy"])[members])),
                "margin_mean": float(np.mean(np.asarray(loaded["margin"])[members])),
            }
        )
    return rows


def _artifact_factors(base: Mapping[str, Any], family: str, artifact: bool) -> dict[str, Any]:
    factors = {
        "sensor_hit_count": int(base["sensor_hit_count"]),
        "provenance_supported": True,
        "duplicate_overlap": 0.0,
        "lifecycle_gap_count": 0,
        "kinematic_jump_m": 0.0,
        "identity_discontinuity": False,
        "shape_ratio_jump": 1.0,
    }
    if not artifact:
        return factors
    if family == "unsupported_ghost":
        factors.update(sensor_hit_count=0, provenance_supported=False)
    elif family == "duplicate_shell":
        factors.update(sensor_hit_count=0, provenance_supported=False, duplicate_overlap=1.0)
    elif family == "lifecycle_flicker":
        factors.update(lifecycle_gap_count=1)
    elif family == "teleport":
        factors.update(kinematic_jump_m=8.0, identity_discontinuity=True)
    elif family == "shape_jump":
        factors.update(shape_ratio_jump=2.0)
    else:
        raise ValueError(f"unknown artifact family: {family}")
    return factors


def certificate(
    row: Mapping[str, Any],
    *,
    maximum_kinematic_jump_m: float,
    maximum_shape_ratio_jump: float,
) -> tuple[float, str, list[str]]:
    reasons = []
    if int(row["sensor_hit_count"]) == 0 and not bool(row["provenance_supported"]):
        reasons.append("sensor_and_provenance_missing")
    if float(row["duplicate_overlap"]) >= 0.5:
        reasons.append("duplicate_overlap")
    if int(row["lifecycle_gap_count"]) > 0:
        reasons.append("lifecycle_gap")
    if (
        float(row["kinematic_jump_m"]) > float(maximum_kinematic_jump_m)
        or bool(row["identity_discontinuity"])
    ):
        reasons.append("kinematic_or_identity_discontinuity")
    if float(row["shape_ratio_jump"]) > float(maximum_shape_ratio_jump):
        reasons.append("shape_ratio_jump")
    score = float(bool(reasons))
    return score, "ARTIFACT" if reasons else "LEGITIMATE", reasons


def build_factorial_rows(
    base_rows: Iterable[Mapping[str, Any]],
    artifact_families: Iterable[str],
    certificate_config: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for base in base_rows:
        for family in artifact_families:
            if family not in ARTIFACT_FAMILIES:
                raise ValueError(f"unsupported artifact family: {family}")
            for artifact in (False, True):
                factors = _artifact_factors(base, family, artifact)
                for hazard in (False, True):
                    row = {
                        **base,
                        **factors,
                        "cluster_id": f"{base['base_id']}/{family}",
                        "variant_id": f"V{int(artifact)}-H{int(hazard)}",
                        "artifact_family": family,
                        "artifact_label": bool(artifact),
                        "hazard_label": bool(hazard),
                        "hazard_score": float(hazard),
                    }
                    cert_score, decision, reasons = certificate(
                        row,
                        maximum_kinematic_jump_m=float(
                            certificate_config["maximum_supported_kinematic_jump_m"]
                        ),
                        maximum_shape_ratio_jump=float(
                            certificate_config["maximum_shape_ratio_jump"]
                        ),
                    )
                    row.update(
                        certificate_score=cert_score,
                        certificate_decision=decision,
                        certificate_reason_codes=reasons,
                    )
                    rows.append(row)
    return rows


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def evaluate_atlas(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([bool(row["artifact_label"]) for row in rows])
    q0 = np.asarray([float(row["q0_mean"]) for row in rows])
    certificate_scores = np.asarray([float(row["certificate_score"]) for row in rows])
    hazard = np.asarray([bool(row["hazard_label"]) for row in rows])
    clean_hazard = (~labels) & hazard
    artifact_hazard = labels & hazard
    family_recall = {}
    for family in sorted({str(row["artifact_family"]) for row in rows}):
        members = np.asarray(
            [str(row["artifact_family"]) == family and bool(row["artifact_label"]) for row in rows]
        )
        family_recall[family] = float(np.mean(certificate_scores[members] >= 0.5))

    paired_q0_delta = []
    paired_certificate_delta = []
    by_key: dict[tuple[str, bool], dict[bool, Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(
            (str(row["cluster_id"]), bool(row["artifact_label"])), {}
        )[bool(row["hazard_label"])] = row
    for pair in by_key.values():
        if set(pair) != {False, True}:
            continue
        paired_q0_delta.append(abs(float(pair[True]["q0_mean"]) - float(pair[False]["q0_mean"])))
        paired_certificate_delta.append(
            abs(float(pair[True]["certificate_score"]) - float(pair[False]["certificate_score"]))
        )
    predicted_artifact = certificate_scores >= 0.5
    return {
        "row_count": len(rows),
        "base_actor_unit_count": len({str(row["base_id"]) for row in rows}),
        "cluster_count": len({str(row["cluster_id"]) for row in rows}),
        "quadrant_counts": dict(Counter(str(row["variant_id"]) for row in rows)),
        "q0_artifact_ranking": _binary_metrics(labels, q0),
        "q0_hazard_ranking": _binary_metrics(hazard, q0),
        "certificate_artifact_ranking": _binary_metrics(labels, certificate_scores),
        "certificate_family_recall": family_recall,
        "clean_hazard_false_artifact_rate": float(np.mean(predicted_artifact[clean_hazard])),
        "legitimate_hazardous_actor_retention": float(np.mean(~predicted_artifact[clean_hazard])),
        "artifact_hazard_detection_rate": float(np.mean(predicted_artifact[artifact_hazard])),
        "mean_absolute_hazard_pair_q0_delta": float(np.mean(paired_q0_delta)),
        "mean_absolute_hazard_pair_certificate_delta": float(
            np.mean(paired_certificate_delta)
        ),
    }


def materialize_base_rows(config: Mapping[str, Any], runs_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    inputs = config["inputs"]
    evidence_root = runs_root / str(inputs["evidence_run"])
    native_root = runs_root / str(inputs["native_run"])
    partition = str(inputs["native_partition"])
    model = joblib.load(runs_root / str(inputs["risk_run"]) / str(inputs["risk_model_relative_path"]))
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    point_limit = int(config["sampling"]["maximum_boundary_points_per_unit"])
    minimum_boundary = int(config["sampling"]["minimum_actor_boundary_points"])
    seed = int(config["seed"])

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
    if not descriptors:
        raise RuntimeError("P1-D found no evidence units")

    rows = []
    ambiguous_query_count = 0
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["sampling"]["io_prefetch_workers"])
    )

    def submit(descriptor: tuple[int, str, Path, Path]):
        return executor.submit(
            _load_actor_unit,
            descriptor,
            origin=origin,
            voxel_size=voxel_size,
            point_limit=point_limit,
            seed=seed,
        )

    future = submit(descriptors[0])
    try:
        for position, descriptor in enumerate(descriptors):
            loaded = future.result()
            if position + 1 < len(descriptors):
                future = submit(descriptors[position + 1])
            scores = _q0_scores(model, loaded["features"])
            unit_rows = _base_actor_rows(loaded, scores, minimum_boundary)
            rows.extend(unit_rows)
            ambiguous_query_count += int(loaded["ambiguous_query_count"])
            print(
                f"P1-D {position + 1}/{len(descriptors)} scene={descriptor[1]} "
                f"unit={descriptor[2].name} actors={len(unit_rows)}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
    return rows, {
        "source_unit_count": len(descriptors),
        "eligible_base_actor_unit_count": len(rows),
        "ambiguous_grounded_query_count": ambiguous_query_count,
    }
