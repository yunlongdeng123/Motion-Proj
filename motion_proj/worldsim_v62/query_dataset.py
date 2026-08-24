"""Sparse query sampling for the V6.2 evidence dataset."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN
from motion_proj.worldsim_v62.evidence import EvidenceGrid
from motion_proj.worldsim_v62.projection import FREE_INDEX, OCCUPIED_INDEX, UNKNOWN_INDEX

QUERY_TYPES = {
    "hard_free": 0,
    "hard_occupied": 1,
    "behind_hit_unknown": 2,
    "evidence_boundary": 3,
    "actor_envelope": 4,
    "contradiction": 5,
}


def _boundary_pool(semantics: np.ndarray) -> np.ndarray:
    observed = semantics != UNKNOWN
    boundary = np.zeros_like(observed)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        different = semantics[tuple(left)] != semantics[tuple(right)]
        active = different & (observed[tuple(left)] | observed[tuple(right)])
        boundary[tuple(left)] |= active
        boundary[tuple(right)] |= active
    return np.flatnonzero(boundary)


def _linear(indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if indices.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    return np.ravel_multi_index(np.asarray(indices, dtype=np.int64).T, shape)


def _draw(pool: np.ndarray, count: int, rng: np.random.Generator, name: str) -> np.ndarray:
    values = np.unique(np.asarray(pool, dtype=np.int64))
    if values.size == 0:
        raise RuntimeError(f"query pool is empty: {name}")
    return rng.choice(values, size=int(count), replace=values.size < int(count))


def _actor_ids_for_queries(
    query_linear: np.ndarray,
    envelope_indices: np.ndarray,
    envelope_ids: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    result = np.full(query_linear.shape[0], -1, dtype=np.int32)
    envelope_linear = _linear(envelope_indices, shape)
    if envelope_linear.size == 0:
        return result
    order = np.argsort(envelope_linear)
    sorted_linear = envelope_linear[order]
    sorted_ids = np.asarray(envelope_ids, dtype=np.int32)[order]
    positions = np.searchsorted(sorted_linear, query_linear)
    valid = positions < sorted_linear.size
    matched = np.zeros_like(valid)
    matched[valid] = sorted_linear[positions[valid]] == query_linear[valid]
    result[matched] = sorted_ids[positions[matched]]
    return result


def _model_class_index(evidence_state: np.ndarray) -> np.ndarray:
    result = np.full(evidence_state.shape, UNKNOWN_INDEX, dtype=np.uint8)
    result[evidence_state == FREE] = FREE_INDEX
    result[evidence_state == OCCUPIED] = OCCUPIED_INDEX
    return result


def build_query_arrays(
    method: EvidenceGrid,
    dropout: EvidenceGrid,
    target: EvidenceGrid,
    query_config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    method_semantics = np.asarray(method.arrays["semantics"], dtype=np.uint8)
    dropout_semantics = np.asarray(dropout.arrays["semantics"], dtype=np.uint8)
    target_semantics = np.asarray(target.arrays["semantics"], dtype=np.uint8)
    shape = tuple(int(value) for value in method_semantics.shape)
    rng = np.random.default_rng(int(seed))
    flat_method = method_semantics.reshape(-1)
    flat_dropout = dropout_semantics.reshape(-1)
    flat_target = target_semantics.reshape(-1)

    opposite_dropout = (
        ((flat_method == FREE) & (flat_dropout == OCCUPIED))
        | ((flat_method == OCCUPIED) & (flat_dropout == FREE))
    )
    contradiction_pool = np.flatnonzero(
        np.asarray(method.arrays["contradiction"], dtype=bool).reshape(-1)
        | opposite_dropout
    )
    pools = {
        "hard_free": np.flatnonzero(flat_method == FREE),
        "hard_occupied": np.flatnonzero(flat_method == OCCUPIED),
        "behind_hit_unknown": np.flatnonzero(
            np.asarray(method.arrays["behind_hit"], dtype=bool).reshape(-1)
            & (flat_method == UNKNOWN)
        ),
        "evidence_boundary": _boundary_pool(method_semantics),
        "actor_envelope": _linear(
            np.asarray(method.arrays["actor_envelope_indices"], dtype=np.int32), shape
        ),
        "contradiction": contradiction_pool,
    }

    query_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    pool_counts: dict[str, int] = {}
    for name, count in query_config["quotas"].items():
        pool = pools[name]
        pool_counts[name] = int(np.unique(pool).size)
        selected = _draw(pool, int(count), rng, name)
        query_parts.append(selected)
        type_parts.append(np.full(selected.shape[0], QUERY_TYPES[name], dtype=np.uint8))

    query_linear = np.concatenate(query_parts)
    query_type = np.concatenate(type_parts)
    order = rng.permutation(query_linear.shape[0])
    query_linear = query_linear[order]
    query_type = query_type[order]
    query_indices = np.stack(np.unravel_index(query_linear, shape), axis=1).astype(np.int32)
    actor_ids = _actor_ids_for_queries(
        query_linear,
        np.asarray(method.arrays["actor_envelope_indices"], dtype=np.int32),
        np.asarray(method.arrays["actor_envelope_ids"], dtype=np.int32),
        shape,
    )

    method_state = flat_method[query_linear].astype(np.uint8)
    dropout_state = flat_dropout[query_linear].astype(np.uint8)
    target_state = flat_target[query_linear].astype(np.uint8)
    arrays = {
        "query_indices": query_indices,
        "query_type": query_type,
        "method_evidence_state": method_state,
        "dropout_evidence_state": dropout_state,
        "target_evidence_state": target_state,
        "method_class_index": _model_class_index(method_state),
        "dropout_class_index": _model_class_index(dropout_state),
        "target_class_index": _model_class_index(target_state),
        "method_contradiction": np.asarray(
            method.arrays["contradiction"], dtype=bool
        ).reshape(-1)[query_linear],
        "target_contradiction": np.asarray(
            target.arrays["contradiction"], dtype=bool
        ).reshape(-1)[query_linear],
        "actor_id": actor_ids,
        "grid_origin_m": np.asarray(method.arrays["grid_origin_m"], dtype=np.float64),
        "voxel_size_m": np.asarray(method.arrays["voxel_size_m"], dtype=np.float64),
        "grid_shape": np.asarray(shape, dtype=np.int64),
    }
    summary = {
        "query_count": int(query_linear.shape[0]),
        "query_type_counts": {
            name: int(np.count_nonzero(query_type == value))
            for name, value in QUERY_TYPES.items()
        },
        "candidate_pool_counts": pool_counts,
        "method_state_counts": {
            "unknown": int(np.count_nonzero(method_state == UNKNOWN)),
            "free": int(np.count_nonzero(method_state == FREE)),
            "occupied": int(np.count_nonzero(method_state == OCCUPIED)),
        },
        "target_supervised_count": int(np.count_nonzero(target_state != UNKNOWN)),
        "actor_query_count": int(np.count_nonzero(arrays["actor_id"] >= 0)),
        "evidence_state_encoding": {"unknown": 0, "free": 1, "occupied": 2},
        "model_class_index_encoding": {"free": 0, "occupied": 1, "unknown": 2},
    }
    return arrays, summary
