"""直接在 IR-WM 原生体素边界上评估 feature-density uncertainty。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN
from motion_proj.worldsim_v64.retrospective_uq import (
    PointChunk,
    _binary_metrics,
    _risk_coverage,
    _softmax_uncertainty,
)


def _unit_dirs(root: Path, scene: str) -> list[Path]:
    scene_root = root / "units" / scene
    units = sorted(path for path in scene_root.iterdir() if path.is_dir())
    if not units:
        raise RuntimeError(f"scene has no evidence units: {scene}")
    return units


def _native_unit_dir(
    native_root: Path,
    scene: str,
    unit_name: str,
    partition_by_scene: Mapping[str, str],
) -> Path:
    partition = str(partition_by_scene[scene])
    return native_root / "units" / partition / scene / unit_name


def _evidence_on_native_grid(
    evidence: Mapping[str, np.ndarray],
    *,
    native_shape: tuple[int, int, int],
    native_origin_m: np.ndarray,
    native_voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    evidence_origin = np.asarray(evidence["grid_origin_m"], dtype=np.float64)
    evidence_voxel_size = float(evidence["voxel_size_m"])
    evidence_shape = np.asarray(evidence["grid_shape"], dtype=np.int64)
    axes = []
    valids = []
    for axis, size in enumerate(native_shape):
        centers = native_origin_m[axis] + (
            np.arange(size, dtype=np.float64) + 0.5
        ) * float(native_voxel_size_m)
        indices = np.floor(
            (centers - evidence_origin[axis]) / evidence_voxel_size
        ).astype(np.int64)
        valids.append((indices >= 0) & (indices < evidence_shape[axis]))
        axes.append(np.clip(indices, 0, evidence_shape[axis] - 1))
    valid = (
        valids[0][:, None, None]
        & valids[1][None, :, None]
        & valids[2][None, None, :]
    )
    sampled = np.asarray(evidence["semantics"])[
        axes[0][:, None, None],
        axes[1][None, :, None],
        axes[2][None, None, :],
    ]
    return sampled, valid


def _native_boundary_chunk(
    evidence_unit: Path,
    native_unit: Path,
    *,
    native_origin_m: np.ndarray,
    native_voxel_size_m: float,
) -> PointChunk:
    with np.load(
        evidence_unit / "METHOD_EVIDENCE.npz", allow_pickle=False
    ) as source:
        method = {name: np.asarray(source[name]) for name in source.files}
    with np.load(
        evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False
    ) as source:
        target = {name: np.asarray(source[name]) for name in source.files}

    argmax = np.load(native_unit / "ARGMAX.npy", mmap_mode="r")
    native_shape = tuple(int(value) for value in argmax.shape)
    method_state, native_valid = _evidence_on_native_grid(
        method,
        native_shape=native_shape,
        native_origin_m=native_origin_m,
        native_voxel_size_m=native_voxel_size_m,
    )
    target_state, target_valid = _evidence_on_native_grid(
        target,
        native_shape=native_shape,
        native_origin_m=native_origin_m,
        native_voxel_size_m=native_voxel_size_m,
    )
    method_contradiction, contradiction_valid = _evidence_on_native_grid(
        {**method, "semantics": method["contradiction"]},
        native_shape=native_shape,
        native_origin_m=native_origin_m,
        native_voxel_size_m=native_voxel_size_m,
    )
    proposal_occupied = (np.asarray(argmax) != 0) | (method_state == OCCUPIED)
    structure = ndimage.generate_binary_structure(3, 1)
    boundary = proposal_occupied & ~ndimage.binary_erosion(
        proposal_occupied, structure=structure, border_value=0
    )
    eligible = (
        boundary
        & native_valid
        & target_valid
        & contradiction_valid
        & (method_state == UNKNOWN)
        & ~method_contradiction.astype(bool)
    )
    indices = np.argwhere(eligible)
    if indices.shape[0] == 0:
        raise RuntimeError(f"native boundary denominator is empty: {evidence_unit}")

    logits_grid = np.load(native_unit / "NATIVE_LOGITS.npy", mmap_mode="r")
    bev_grid = np.load(native_unit / "BEV_LATENT.npy", mmap_mode="r")
    x, y, z = indices.T
    logits = np.asarray(logits_grid[x, y, z], dtype=np.float32)
    bev = np.asarray(bev_grid[x, y], dtype=np.float32)
    hidden_free = np.asarray(target_state[x, y, z] == FREE, dtype=bool)
    return PointChunk(
        features=np.concatenate((logits, bev), axis=1),
        logits=logits,
        hidden_free=hidden_free,
    )


def sample_training_points_native(
    evidence_root: Path,
    native_root: Path,
    scenes: Sequence[str],
    *,
    partition_by_scene: Mapping[str, str],
    maximum_points_per_scene: int,
    seed: int,
    native_origin_m: Sequence[float],
    native_voxel_size_m: float,
) -> PointChunk:
    rng = np.random.default_rng(int(seed))
    sampled = []
    origin = np.asarray(native_origin_m, dtype=np.float64)
    for scene in scenes:
        units = _unit_dirs(evidence_root, scene)
        per_unit = max(1, int(np.ceil(maximum_points_per_scene / len(units))))
        parts = []
        for evidence_unit in units:
            native_unit = _native_unit_dir(
                native_root, scene, evidence_unit.name, partition_by_scene
            )
            part = _native_boundary_chunk(
                evidence_unit,
                native_unit,
                native_origin_m=origin,
                native_voxel_size_m=native_voxel_size_m,
            )
            if part.features.shape[0] > per_unit:
                keep = rng.choice(part.features.shape[0], size=per_unit, replace=False)
                part = PointChunk(
                    features=part.features[keep],
                    logits=part.logits[keep],
                    hidden_free=part.hidden_free[keep],
                )
            parts.append(part)
        features = np.concatenate([part.features for part in parts])
        logits = np.concatenate([part.logits for part in parts])
        hidden_free = np.concatenate([part.hidden_free for part in parts])
        if features.shape[0] > maximum_points_per_scene:
            keep = rng.choice(
                features.shape[0], size=maximum_points_per_scene, replace=False
            )
            features, logits, hidden_free = (
                features[keep],
                logits[keep],
                hidden_free[keep],
            )
        sampled.append(PointChunk(features, logits, hidden_free))
    return PointChunk(
        features=np.concatenate([part.features for part in sampled]),
        logits=np.concatenate([part.logits for part in sampled]),
        hidden_free=np.concatenate([part.hidden_free for part in sampled]),
    )


def evaluate_scene_native(
    model: object,
    evidence_root: Path,
    native_root: Path,
    scene: str,
    *,
    partition_by_scene: Mapping[str, str],
    native_origin_m: Sequence[float],
    native_voxel_size_m: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    labels_parts = []
    score_parts: dict[str, list[np.ndarray]] = {
        "u0_max_probability": [],
        "u0_entropy": [],
        "u0_inverse_margin": [],
        "u2_feature_density": [],
    }
    origin = np.asarray(native_origin_m, dtype=np.float64)
    for evidence_unit in _unit_dirs(evidence_root, scene):
        native_unit = _native_unit_dir(
            native_root, scene, evidence_unit.name, partition_by_scene
        )
        chunk = _native_boundary_chunk(
            evidence_unit,
            native_unit,
            native_origin_m=origin,
            native_voxel_size_m=native_voxel_size_m,
        )
        labels_parts.append(chunk.hidden_free)
        for name, values in _softmax_uncertainty(chunk.logits).items():
            score_parts[name].append(np.asarray(values, dtype=np.float32))
        score_parts["u2_feature_density"].append(
            model.score(chunk.features, chunk.logits)
        )
    labels = np.concatenate(labels_parts)
    scores = {name: np.concatenate(parts) for name, parts in score_parts.items()}
    metrics = {
        "scene": scene,
        "point_count": int(labels.size),
        "hidden_free_count": int(labels.sum()),
        "hidden_free_prevalence": float(labels.mean()),
        "scores": {
            name: {
                **_binary_metrics(labels, values),
                "risk_coverage": _risk_coverage(labels, values),
            }
            for name, values in scores.items()
        },
    }
    return metrics, {"labels": labels, **scores}
