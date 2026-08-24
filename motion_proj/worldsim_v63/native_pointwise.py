"""Run the frozen V6.2 pointwise model with true V6.3 native sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN
from motion_proj.worldsim_v62.cpsc_lite import CPSCLite
from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
)

from .native_features import target_points_to_native_indices


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


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
        (actor_bound, actor_bound, np.zeros(actor_bound.shape[0], dtype=bool)),
        axis=1,
    ).astype(np.float32)
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
def infer_native_pointwise_grids(
    model: CPSCLite,
    *,
    native_unit_dir: Path,
    method_semantics: np.ndarray,
    actor_grid: np.ndarray,
    target_origin_m: np.ndarray,
    target_voxel_size_m: float,
    source_origin_m: np.ndarray,
    source_voxel_size_m: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate native B1/B3/B2 on an entire target grid."""
    native_logits = np.load(native_unit_dir / "NATIVE_LOGITS.npy", mmap_mode="r")
    native_bev = np.load(native_unit_dir / "BEV_LATENT.npy", mmap_mode="r")
    source_shape = tuple(int(value) for value in native_logits.shape[:3])
    if native_logits.shape != (*source_shape, 17) or native_bev.shape != (
        source_shape[0],
        source_shape[1],
        256,
    ):
        raise ValueError("native sidecar shape drift")
    shape = tuple(int(value) for value in method_semantics.shape)
    if actor_grid.shape != shape:
        raise ValueError("method and actor grids are not aligned")
    total = int(np.prod(shape))
    flat_method = method_semantics.reshape(-1)
    flat_actor = actor_grid.reshape(-1) >= 0
    outputs = {
        "B1-HARD-CLIP": np.full(total, UNKNOWN, dtype=np.uint8),
        "B3-NATIVE-NO-PROJECTION": np.full(total, UNKNOWN, dtype=np.uint8),
        "B2-NATIVE-CPSC-LITE": np.full(total, UNKNOWN, dtype=np.uint8),
    }
    hard_constraints = 0
    b3_hard_violations = 0
    b2_hard_violations = 0
    source_valid_count = 0
    source_valid_unknown = {name: 0 for name in outputs}
    model.eval()
    target_origin = np.asarray(target_origin_m, dtype=np.float64)
    for start in range(0, total, int(batch_size)):
        stop = min(total, start + int(batch_size))
        linear = np.arange(start, stop, dtype=np.int64)
        indices = np.stack(np.unravel_index(linear, shape), axis=1).astype(np.int32)
        points_m = target_origin[None] + (
            indices.astype(np.float64) + 0.5
        ) * float(target_voxel_size_m)
        native_indices, valid = target_points_to_native_indices(
            points_m,
            source_origin_m=np.asarray(source_origin_m, dtype=np.float64),
            source_voxel_size_m=float(source_voxel_size_m),
            source_shape=source_shape,
        )
        raw_logits = np.zeros((stop - start, 17), dtype=np.float32)
        bev_features = np.zeros((stop - start, 256), dtype=np.float32)
        if np.any(valid):
            selected = native_indices[valid]
            raw_logits[valid] = native_logits[
                selected[:, 0], selected[:, 1], selected[:, 2]
            ].astype(np.float32)
            bev_features[valid] = native_bev[
                selected[:, 0], selected[:, 1]
            ].astype(np.float32)
        semantic_probabilities = _softmax(raw_logits)
        entropy = -np.sum(
            semantic_probabilities
            * np.log(np.clip(semantic_probabilities, 1e-8, None)),
            axis=1,
        ) / np.log(17.0)
        prior_tristate = np.stack(
            (
                semantic_probabilities[:, 0],
                semantic_probabilities[:, 1:].sum(axis=1),
                np.zeros(stop - start, dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float32)
        prior_tristate[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        entropy[~valid] = 1.0
        prior_features = np.concatenate(
            (
                raw_logits,
                entropy[:, None].astype(np.float32),
                prior_tristate,
                valid[:, None].astype(np.float32),
                bev_features,
            ),
            axis=1,
        ).astype(np.float32)

        method_class = _method_class_index(flat_method[start:stop])
        query_features = _query_features(
            indices,
            shape,
            method_class,
            flat_actor[start:stop],
            prior_tristate,
        )
        observed_free = method_class == FREE_INDEX
        observed_occupied = method_class == OCCUPIED_INDEX
        prior_class = np.full(stop - start, UNKNOWN_INDEX, dtype=np.int64)
        argmax = raw_logits.argmax(axis=1)
        prior_class[valid & (argmax == 0)] = FREE_INDEX
        prior_class[valid & (argmax > 0)] = OCCUPIED_INDEX
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
        b2 = result["probabilities"].argmax(dim=1).cpu().numpy()
        expected = np.full(method_class.shape, UNKNOWN_INDEX, dtype=np.int64)
        expected[observed_free] = FREE_INDEX
        expected[observed_occupied] = OCCUPIED_INDEX
        constrained = observed_free | observed_occupied
        hard_constraints += int(np.count_nonzero(constrained))
        b3_hard_violations += int(np.count_nonzero(constrained & (b3 != expected)))
        b2_hard_violations += int(np.count_nonzero(constrained & (b2 != expected)))
        classes_by_arm = {
            "B1-HARD-CLIP": b1,
            "B3-NATIVE-NO-PROJECTION": b3,
            "B2-NATIVE-CPSC-LITE": b2,
        }
        source_valid_count += int(np.count_nonzero(valid))
        for name, class_index in classes_by_arm.items():
            outputs[name][start:stop] = _tristate_to_semantics(class_index)
            source_valid_unknown[name] += int(
                np.count_nonzero(valid & (class_index == UNKNOWN_INDEX))
            )
    diagnostics = {
        "voxel_count": total,
        "source_valid_count": source_valid_count,
        "hard_constraint_count": hard_constraints,
        "b3_hard_violation_count": b3_hard_violations,
        "b2_hard_violation_count": b2_hard_violations,
        "source_valid_unknown_fraction": {
            name: count / max(1, source_valid_count)
            for name, count in source_valid_unknown.items()
        },
        "prototype_used": False,
    }
    return {name: values.reshape(shape) for name, values in outputs.items()}, diagnostics
