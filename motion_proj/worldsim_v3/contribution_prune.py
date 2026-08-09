"""A4-P1 contribution pruning 的确定性排序、裁剪与裁决工具。"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


GAUSSIAN_FIELDS = (
    "_means",
    "_scales",
    "_quats",
    "_features_dc",
    "_features_rest",
    "_opacities",
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def index_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(index) for index in indices], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """对 NumPy 数组的 dtype、shape 与连续字节做稳定指纹。"""
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stable_remove_indices(
    *,
    train_alpha_weight_sum: np.ndarray,
    train_visible_view_count: np.ndarray,
    learned_opacity: np.ndarray,
    gaussian_ids: np.ndarray,
    asset_indices: np.ndarray,
    prune_fraction: float,
    decimal_places: int = 12,
) -> np.ndarray:
    """按冻结的五级排序键返回一个资产要删除的 model-flat indices。"""
    arrays = [
        np.asarray(train_alpha_weight_sum),
        np.asarray(train_visible_view_count),
        np.asarray(learned_opacity),
        np.asarray(gaussian_ids),
    ]
    total = len(arrays[0])
    if any(array.ndim != 1 or len(array) != total for array in arrays):
        raise ValueError("A4-P1 ranking arrays must be aligned one-dimensional arrays")
    indices = np.asarray(asset_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= total):
        raise ValueError("A4-P1 asset indices are invalid")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("A4-P1 asset indices must be unique")
    if not 0.0 <= float(prune_fraction) < 1.0:
        raise ValueError("A4-P1 prune fraction must be in [0,1)")
    remove_count = math.floor(len(indices) * float(prune_fraction))
    if remove_count == 0:
        return np.empty(0, dtype=np.int64)
    score = np.round(arrays[0][indices].astype(np.float64), decimals=decimal_places)
    visibility = arrays[1][indices].astype(np.int64)
    opacity = arrays[2][indices].astype(np.float64)
    ids = arrays[3][indices].astype(np.int64)
    order = np.lexsort((indices, ids, opacity, visibility, score))
    return np.sort(indices[order[:remove_count]])


def build_candidate_masks(
    *,
    background_scores: Mapping[str, np.ndarray],
    rigid_scores: Mapping[str, np.ndarray],
    rigid_point_ids: np.ndarray,
    prune_fraction: float,
    decimal_places: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """为 Background 与每个非空 actor 独立生成同一预算的 keep mask。"""
    background_count = len(background_scores["train_alpha_weight_sum"])
    rigid_count = len(rigid_scores["train_alpha_weight_sum"])
    point_ids = np.asarray(rigid_point_ids, dtype=np.int64).reshape(-1)
    if len(point_ids) != rigid_count:
        raise ValueError("A4-P1 Rigid point ids do not align with scores")
    background_indices = np.arange(background_count, dtype=np.int64)
    background_removed = stable_remove_indices(
        train_alpha_weight_sum=background_scores["train_alpha_weight_sum"],
        train_visible_view_count=background_scores["train_visible_view_count"],
        learned_opacity=background_scores["learned_opacity"],
        gaussian_ids=background_scores["gaussian_ids"],
        asset_indices=background_indices,
        prune_fraction=prune_fraction,
        decimal_places=decimal_places,
    )
    background_keep = np.ones(background_count, dtype=bool)
    background_keep[background_removed] = False
    manifest = [
        {
            "asset": "Background",
            "rigid_model_index": None,
            "source_count": background_count,
            "removed_count": len(background_removed),
            "remaining_count": int(background_keep.sum()),
            "removed_flat_indices_sha256": index_sha256(background_removed.tolist()),
            "removed_gaussian_ids_sha256": index_sha256(
                background_scores["gaussian_ids"][background_removed].tolist()
            ),
        }
    ]
    rigid_keep = np.ones(rigid_count, dtype=bool)
    for model_index in sorted(np.unique(point_ids).tolist()):
        asset_indices = np.flatnonzero(point_ids == model_index)
        removed = stable_remove_indices(
            train_alpha_weight_sum=rigid_scores["train_alpha_weight_sum"],
            train_visible_view_count=rigid_scores["train_visible_view_count"],
            learned_opacity=rigid_scores["learned_opacity"],
            gaussian_ids=rigid_scores["gaussian_ids"],
            asset_indices=asset_indices,
            prune_fraction=prune_fraction,
            decimal_places=decimal_places,
        )
        rigid_keep[removed] = False
        manifest.append(
            {
                "asset": "RigidNodes",
                "rigid_model_index": int(model_index),
                "source_count": len(asset_indices),
                "removed_count": len(removed),
                "remaining_count": int(len(asset_indices) - len(removed)),
                "removed_flat_indices_sha256": index_sha256(removed.tolist()),
                "removed_gaussian_ids_sha256": index_sha256(
                    rigid_scores["gaussian_ids"][removed].tolist()
                ),
            }
        )
    return background_keep, rigid_keep, manifest


def prune_model_state(
    model_state: Mapping[str, Any], keep_mask: torch.Tensor, *, rigid: bool
) -> OrderedDict[str, Any]:
    """用同一个 keep mask 裁剪所有 Gaussian 参数、point_ids 与 ancestry rows。"""
    count = int(keep_mask.numel())
    if keep_mask.dtype != torch.bool or keep_mask.ndim != 1:
        raise ValueError("A4-P1 keep mask must be one-dimensional bool")
    output: OrderedDict[str, Any] = OrderedDict()
    for name, value in model_state.items():
        if name in GAUSSIAN_FIELDS or (rigid and name == "points_ids"):
            if not isinstance(value, torch.Tensor) or value.shape[0] != count:
                raise ValueError(f"A4-P1 row-aligned field drift: {name}")
            output[name] = value[keep_mask].contiguous()
        elif name == "worldsim_a2_ancestry":
            ancestry = dict(value)
            fields = {}
            for field_name, field_value in ancestry["fields"].items():
                if not isinstance(field_value, torch.Tensor) or field_value.shape[0] != count:
                    raise ValueError(f"A4-P1 ancestry field drift: {field_name}")
                fields[field_name] = field_value[keep_mask].contiguous()
            ancestry["fields"] = fields
            output[name] = ancestry
        else:
            output[name] = value
    return output


def prune_checkpoint_state(
    checkpoint: Mapping[str, Any], background_keep: torch.Tensor, rigid_keep: torch.Tensor
) -> OrderedDict[str, Any]:
    """保持 checkpoint schema 不变并只替换两个 Gaussian model state。"""
    output: OrderedDict[str, Any] = OrderedDict()
    for name, value in checkpoint.items():
        if name != "models":
            output[name] = value
            continue
        models: OrderedDict[str, Any] = OrderedDict()
        for model_name, model_state in value.items():
            if model_name == "Background":
                models[model_name] = prune_model_state(
                    model_state, background_keep, rigid=False
                )
            elif model_name == "RigidNodes":
                models[model_name] = prune_model_state(
                    model_state, rigid_keep, rigid=True
                )
            else:
                models[model_name] = model_state
        output[name] = models
    return output


def half_open_ranges(indices: Sequence[int]) -> list[list[int]]:
    values = sorted(int(index) for index in indices)
    if not values:
        return []
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append([start, previous + 1])
            start = value
        previous = value
    ranges.append([start, previous + 1])
    return ranges


def build_candidate_registry(
    source_registry: Mapping[str, Any], point_ids: Sequence[int], checkpoint_sha256: str
) -> dict[str, Any]:
    """从裁剪后的 runtime point_ids 重建 v2 actor registry 索引层。"""
    result = deepcopy(dict(source_registry))
    ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    for actor in result["actors"]:
        model_index = int(actor["rigid_model_index"])
        indices = np.flatnonzero(ids == model_index).tolist()
        actor["availability"] = (
            "available" if indices else "unavailable_empty_checkpoint_slice"
        )
        actor["checkpoint_tensor_slice"] = {
            "selector": f"models.RigidNodes.points_ids[:,0] == {model_index}",
            "gaussian_count": len(indices),
            "flat_indices_sha256": index_sha256(indices),
            "flat_index_ranges_half_open": half_open_ranges(indices),
        }
    result["checkpoint_sha256"] = checkpoint_sha256
    result["available_actor_count"] = sum(
        actor["availability"] == "available" for actor in result["actors"]
    )
    result["empty_checkpoint_actor_count"] = sum(
        actor["availability"] != "available" for actor in result["actors"]
    )
    smoke_index = int(result["selected_smoke_actor"]["rigid_model_index"])
    result["selected_smoke_actor"] = deepcopy(
        next(
            actor
            for actor in result["actors"]
            if int(actor["rigid_model_index"]) == smoke_index
        )
    )
    result["source"] = {
        **result["source"],
        "checkpoint": "candidate_checkpoint_recorded_in_run_manifest",
    }
    result.pop("actor_registry_sha256", None)
    result["actor_registry_sha256"] = canonical_sha256(result)
    return result


def metric_passes(
    baseline: float, candidate: float, *, direction: str, maximum_regression: float
) -> bool:
    if not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
        return False
    if direction == "higher":
        return float(candidate) >= float(baseline) - float(maximum_regression)
    if direction == "lower":
        return float(candidate) <= float(baseline) + float(maximum_regression)
    raise ValueError(f"unknown A4-P1 metric direction: {direction}")


def compare_metric_group(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for contract in contracts:
        name = str(contract["name"])
        before = baseline.get(name)
        after = candidate.get(name)
        passed = (
            before is not None
            and after is not None
            and metric_passes(
                float(before),
                float(after),
                direction=str(contract["direction"]),
                maximum_regression=float(contract["maximum_regression"]),
            )
        )
        rows.append(
            {
                "name": name,
                "direction": contract["direction"],
                "maximum_regression": float(contract["maximum_regression"]),
                "baseline": before,
                "candidate": after,
                "delta": (
                    float(after) - float(before)
                    if before is not None and after is not None
                    else None
                ),
                "passed": passed,
            }
        )
    return rows


def select_largest_eligible_arm(
    arms: Sequence[Mapping[str, Any]], *, source_arm: str = "p1-source"
) -> dict[str, Any]:
    candidates = [
        arm
        for arm in arms
        if arm["id"] != source_arm
        and bool(arm.get("candidate_checkpoint_reload_exact"))
        and bool(arm.get("expected_counts_exact"))
        and bool(arm.get("all_quality_safeguards_pass"))
        and bool(arm.get("checkpoint_bytes_strictly_less_than_source"))
        and bool(arm.get("source_inputs_unchanged"))
        and bool(arm.get("resources_within_frozen_ceilings"))
    ]
    if not candidates:
        return {
            "selected_arm": source_arm,
            "selected_prune_fraction": 0.0,
            "method_state": "rejected_quality_or_integrity_gate",
            "fallback_exact_alias": True,
        }
    selected = max(candidates, key=lambda arm: (float(arm["prune_fraction"]), str(arm["id"])))
    return {
        "selected_arm": selected["id"],
        "selected_prune_fraction": float(selected["prune_fraction"]),
        "method_state": "selected_bounded_quality_loss_candidate",
        "fallback_exact_alias": False,
    }
