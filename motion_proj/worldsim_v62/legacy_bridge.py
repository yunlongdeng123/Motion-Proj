"""Lossy class-prototype bridge from V6.1 argmax artifacts to frozen CPSC-Lite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN
from motion_proj.worldsim_v62.cpsc_lite import CPSCLite, UnitArrays, load_unit_arrays
from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
)


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def fit_query_weighted_class_prototypes(
    p2_run: Path,
    p4_run: Path,
    scenes: list[str],
    target_frames: list[int],
) -> dict[str, np.ndarray]:
    """Fit 17 deterministic feature means using only the frozen P5 training split."""
    counts = np.zeros(17, dtype=np.int64)
    logit_sums = np.zeros((17, 17), dtype=np.float64)
    bev_sums = np.zeros((17, 256), dtype=np.float64)
    for scene in scenes:
        for target_frame in target_frames:
            unit = load_unit_arrays(p2_run, p4_run, scene, int(target_frame))
            logits = unit.prior_features[:, :17]
            bev = unit.prior_features[:, 22:]
            classes = logits.argmax(axis=1)
            for class_index in range(17):
                selected = unit.prior_valid & (classes == class_index)
                count = int(np.count_nonzero(selected))
                counts[class_index] += count
                if count:
                    logit_sums[class_index] += logits[selected].sum(
                        axis=0, dtype=np.float64
                    )
                    bev_sums[class_index] += bev[selected].sum(
                        axis=0, dtype=np.float64
                    )
    if np.any(counts == 0):
        raise ValueError(
            f"P5 training split lacks prototype classes: {np.flatnonzero(counts == 0)}"
        )
    return {
        "counts": counts,
        "logits": (logit_sums / counts[:, None]).astype(np.float32),
        "bev": (bev_sums / counts[:, None]).astype(np.float32),
    }


def _prototype_features(
    classes: np.ndarray,
    source_valid: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    raw_logits = prototypes["logits"][classes].astype(np.float32, copy=True)
    bev_features = prototypes["bev"][classes].astype(np.float32, copy=True)
    raw_logits[~source_valid] = 0.0
    bev_features[~source_valid] = 0.0
    semantic_probabilities = _stable_softmax(raw_logits)
    entropy = -np.sum(
        semantic_probabilities
        * np.log(np.clip(semantic_probabilities, 1e-8, None)),
        axis=1,
    ) / np.log(17.0)
    prior_tristate = np.stack(
        (
            semantic_probabilities[:, 0],
            semantic_probabilities[:, 1:].sum(axis=1),
            np.zeros(classes.shape[0], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    prior_tristate[~source_valid] = np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float32
    )
    entropy[~source_valid] = 1.0
    prior_features = np.concatenate(
        (
            raw_logits,
            entropy[:, None].astype(np.float32),
            prior_tristate,
            source_valid[:, None].astype(np.float32),
            bev_features,
        ),
        axis=1,
    ).astype(np.float32)
    return prior_features, prior_tristate


def _method_class_index(method_semantics: np.ndarray) -> np.ndarray:
    method_class = np.full(method_semantics.shape, UNKNOWN_INDEX, dtype=np.int64)
    method_class[method_semantics == FREE] = FREE_INDEX
    method_class[method_semantics == OCCUPIED] = OCCUPIED_INDEX
    return method_class


def _query_features(
    indices: np.ndarray,
    shape: tuple[int, int, int],
    method_class: np.ndarray,
    actor_bound: np.ndarray,
    prior_tristate: np.ndarray,
) -> np.ndarray:
    normalized_coordinates = 2.0 * (
        (indices.astype(np.float32) + 0.5)
        / np.asarray(shape, dtype=np.float32)[None]
    ) - 1.0
    method_one_hot = np.eye(3, dtype=np.float32)[method_class]
    actor_features = np.stack(
        (
            actor_bound,
            actor_bound,
            np.zeros(actor_bound.shape[0], dtype=bool),
        ),
        axis=1,
    ).astype(np.float32)


def bridge_unit_features(
    unit: UnitArrays, prototypes: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Build the pure prototype view corresponding to one frozen P2/P4 unit."""
    classes = unit.prior_features[:, :17].argmax(axis=1).astype(np.int64)
    prior_features, prior_tristate = _prototype_features(
        classes, unit.prior_valid, prototypes
    )
    query_features = unit.query_features.copy()
    method_one_hot = np.eye(3, dtype=np.float32)[unit.method_class]
    query_features[:, -3:] = prior_tristate - method_one_hot
    return prior_features, query_features
    return np.concatenate(
        (
            normalized_coordinates,
            method_one_hot,
            np.zeros((indices.shape[0], 1), dtype=np.float32),
            actor_features,
            prior_tristate - method_one_hot,
        ),
        axis=1,
    ).astype(np.float32)


def _tristate_to_semantics(class_index: np.ndarray) -> np.ndarray:
    semantics = np.full(class_index.shape, UNKNOWN, dtype=np.uint8)
    semantics[class_index == FREE_INDEX] = FREE
    semantics[class_index == OCCUPIED_INDEX] = OCCUPIED
    return semantics


@torch.no_grad()
def infer_legacy_tristate_grids(
    model: CPSCLite,
    *,
    predicted_class: np.ndarray,
    source_valid: np.ndarray,
    method_semantics: np.ndarray,
    actor_grid: np.ndarray,
    prototypes: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run B1/B3/B5 over a V6.1 0.2m grid without model-backbone replay."""
    shape = tuple(int(value) for value in method_semantics.shape)
    if predicted_class.shape != shape or source_valid.shape != shape:
        raise ValueError("legacy class and method grids are not aligned")
    total = int(np.prod(shape))
    flat_class = predicted_class.reshape(-1)
    flat_valid = source_valid.reshape(-1)
    flat_method = method_semantics.reshape(-1)
    flat_actor = actor_grid.reshape(-1) >= 0
    outputs = {
        "B1-HARD-CLIP": np.full(total, UNKNOWN, dtype=np.uint8),
        "B3-EVIDENTIAL-NO-PROJECTION": np.full(total, UNKNOWN, dtype=np.uint8),
        "B5-CPSC-LITE-PRECONFORMAL": np.full(total, UNKNOWN, dtype=np.uint8),
    }
    hard_constraints = 0
    b3_hard_violations = 0
    b5_hard_violations = 0
    source_valid_count = int(np.count_nonzero(flat_valid))
    source_valid_unknown = {name: 0 for name in outputs}
    model.eval()
    for start in range(0, total, int(batch_size)):
        stop = min(total, start + int(batch_size))
        linear = np.arange(start, stop, dtype=np.int64)
        indices = np.stack(np.unravel_index(linear, shape), axis=1).astype(np.int32)
        valid = flat_valid[start:stop]
        classes = np.clip(flat_class[start:stop], 0, 16).astype(np.int64)
        method_class = _method_class_index(flat_method[start:stop])
        prior_features, prior_tristate = _prototype_features(
            classes, valid, prototypes
        )
        query_features = _query_features(
            indices,
            shape,
            method_class,
            flat_actor[start:stop],
            prior_tristate,
        )
        observed_free = method_class == FREE_INDEX
        observed_occupied = method_class == OCCUPIED_INDEX
        prior_class = np.full(classes.shape, UNKNOWN_INDEX, dtype=np.int64)
        prior_class[valid & (classes == 0)] = FREE_INDEX
        prior_class[valid & (classes > 0)] = OCCUPIED_INDEX
        b1 = prior_class.copy()
        b1[observed_free] = FREE_INDEX
        b1[observed_occupied] = OCCUPIED_INDEX
        result = model(
            torch.from_numpy(prior_features).to(device),
            torch.from_numpy(query_features).to(device),
            observed_free=torch.from_numpy(observed_free).to(device),
            observed_occupied=torch.from_numpy(observed_occupied).to(device),
            contradiction=torch.zeros(stop - start, dtype=torch.bool, device=device),
        )
        b3 = result["base_probabilities"].argmax(dim=1).cpu().numpy()
        b5 = result["probabilities"].argmax(dim=1).cpu().numpy()
        expected = np.full(method_class.shape, UNKNOWN_INDEX, dtype=np.int64)
        expected[observed_free] = FREE_INDEX
        expected[observed_occupied] = OCCUPIED_INDEX
        constrained = observed_free | observed_occupied
        hard_constraints += int(np.count_nonzero(constrained))
        b3_hard_violations += int(np.count_nonzero(constrained & (b3 != expected)))
        b5_hard_violations += int(np.count_nonzero(constrained & (b5 != expected)))
        classes_by_arm = {
            "B1-HARD-CLIP": b1,
            "B3-EVIDENTIAL-NO-PROJECTION": b3,
            "B5-CPSC-LITE-PRECONFORMAL": b5,
        }
        for name, class_index in classes_by_arm.items():
            semantics = _tristate_to_semantics(class_index)
            outputs[name][start:stop] = semantics
            source_valid_unknown[name] += int(
                np.count_nonzero(valid & (class_index == UNKNOWN_INDEX))
            )
    reshaped = {name: values.reshape(shape) for name, values in outputs.items()}
    diagnostics = {
        "voxel_count": total,
        "source_valid_count": source_valid_count,
        "hard_constraint_count": hard_constraints,
        "b3_hard_violation_count": b3_hard_violations,
        "b5_hard_violation_count": b5_hard_violations,
        "source_valid_unknown_fraction": {
            name: count / max(1, source_valid_count)
            for name, count in source_valid_unknown.items()
        },
    }
    return reshaped, diagnostics
