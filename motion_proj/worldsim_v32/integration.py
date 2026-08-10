"""WorldSim V3.2 最终资产集成、语义扩展和分包协议工具。"""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch


NEGATIVE_LABEL = 0


def atomic_savez_compressed(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """原子写入 NPZ，禁止覆盖既有正式资产。"""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def extend_semantic_sidecar(
    source_path: Path,
    output_path: Path,
    *,
    old_background_count: int,
    generated_background_count: int,
    rigid_count: int,
) -> dict[str, Any]:
    """在 Background 与 RigidNodes 之间插入 generated-background 的负语义行。"""
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    old_total = old_background_count + rigid_count
    new_total = old_total + generated_background_count
    output: MutableMapping[str, np.ndarray] = OrderedDict()
    extended_fields: list[str] = []
    for name, value in arrays.items():
        if name == "background_count":
            output[name] = np.asarray(old_background_count + generated_background_count, dtype=value.dtype)
            continue
        if name == "rigid_point_ids":
            if value.shape != (rigid_count,):
                raise ValueError(f"rigid_point_ids shape 漂移：{value.shape}")
            output[name] = value
            continue
        if value.ndim == 0 or value.shape[0] != old_total:
            raise ValueError(f"未知 sidecar 字段合同：{name} {value.shape}")
        fill_shape = (generated_background_count, *value.shape[1:])
        fill = np.zeros(fill_shape, dtype=value.dtype)
        if name == "labels":
            fill.fill(NEGATIVE_LABEL)
        output[name] = np.concatenate(
            [value[:old_background_count], fill, value[old_background_count:]],
            axis=0,
        )
        if output[name].shape[0] != new_total:
            raise RuntimeError(f"sidecar 扩展失败：{name}")
        extended_fields.append(name)
    atomic_savez_compressed(output_path, output)
    return {
        "old_background_count": old_background_count,
        "generated_background_count": generated_background_count,
        "new_background_count": old_background_count + generated_background_count,
        "rigid_count": rigid_count,
        "old_total": old_total,
        "new_total": new_total,
        "extended_fields": extended_fields,
    }


def validate_extended_semantic_sidecar(
    source_path: Path,
    candidate_path: Path,
    *,
    old_background_count: int,
    generated_background_count: int,
    rigid_count: int,
) -> dict[str, Any]:
    """核对旧 Background、RigidNodes 精确不变，新增行全为 actor-negative/zero evidence。"""
    with np.load(source_path, allow_pickle=False) as source_npz:
        source = {name: np.array(source_npz[name], copy=True) for name in source_npz.files}
    with np.load(candidate_path, allow_pickle=False) as candidate_npz:
        candidate = {
            name: np.array(candidate_npz[name], copy=True) for name in candidate_npz.files
        }
    if set(source) != set(candidate):
        raise RuntimeError("semantic sidecar 字段集合漂移")
    old_total = old_background_count + rigid_count
    new_background_count = old_background_count + generated_background_count
    prefix_exact = True
    rigid_exact = True
    generated_zero = True
    for name, source_value in source.items():
        candidate_value = candidate[name]
        if name == "background_count":
            generated_zero = generated_zero and int(candidate_value) == new_background_count
            continue
        if name == "rigid_point_ids":
            rigid_exact = rigid_exact and np.array_equal(source_value, candidate_value)
            continue
        if source_value.shape[0] != old_total:
            raise RuntimeError(f"未知 semantic 字段：{name}")
        prefix_exact = prefix_exact and np.array_equal(
            source_value[:old_background_count],
            candidate_value[:old_background_count],
        )
        rigid_exact = rigid_exact and np.array_equal(
            source_value[old_background_count:],
            candidate_value[new_background_count:],
        )
        generated = candidate_value[old_background_count:new_background_count]
        if name == "labels":
            generated_zero = generated_zero and bool(np.all(generated == NEGATIVE_LABEL))
        else:
            generated_zero = generated_zero and bool(np.all(generated == 0))
    return {
        "field_set_exact": set(source) == set(candidate),
        "old_background_prefix_exact": bool(prefix_exact),
        "rigid_suffix_exact": bool(rigid_exact),
        "generated_rows_actor_negative_zero_evidence": bool(generated_zero),
        "all_exact": bool(prefix_exact and rigid_exact and generated_zero),
    }


def _discover_row_tensors(value: Any, row_count: int, prefix: str = "") -> OrderedDict[str, dict[str, Any]]:
    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and int(value.shape[0]) == row_count:
            rows[prefix] = {
                "path": prefix,
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape_tail": [int(item) for item in value.shape[1:]],
            }
        return rows
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            rows.update(_discover_row_tensors(child, row_count, child_prefix))
    return rows


def discover_model_row_schema(model: Mapping[str, Any]) -> OrderedDict[str, dict[str, Any]]:
    """按 `_means` 第一维发现 Gaussian row tensor，保留 checkpoint 顺序。"""
    means = model.get("_means")
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("模型缺少合法 _means")
    rows = _discover_row_tensors(model, int(means.shape[0]))
    if "_means" not in rows:
        raise RuntimeError("row schema 未发现 _means")
    return rows


def build_chunk_protocol(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_record: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    cell_size_m: float = 50.0,
) -> dict[str, Any]:
    """从 V3.2 mixed checkpoint 生成 exact static/actor 分包协议。"""
    models = checkpoint["models"]
    background = models["Background"]
    rigid = models["RigidNodes"]
    background_count = int(background["_means"].shape[0])
    rigid_count = int(rigid["_means"].shape[0])
    background_rows = discover_model_row_schema(background)
    rigid_rows = discover_model_row_schema(rigid)
    common_paths = [
        path
        for path, schema in background_rows.items()
        if path in rigid_rows and schema == rigid_rows[path]
    ]
    background_additional = [
        schema for path, schema in background_rows.items() if path not in common_paths
    ]
    rigid_additional = [
        schema for path, schema in rigid_rows.items() if path not in common_paths
    ]
    point_ids = rigid.get("points_ids")
    if not isinstance(point_ids, torch.Tensor) or point_ids.numel() != rigid_count:
        raise ValueError("RigidNodes points_ids 合同失败")
    actor_min = int(point_ids.min().item())
    actor_max = int(point_ids.max().item())
    return {
        "schema_version": 1,
        "selected_asset": {
            "checkpoint": dict(checkpoint_record),
            "actor_registry": dict(registry_record),
        },
        "row_tensor_schema": {
            "common_gaussian_row_tensors": [background_rows[path] for path in common_paths],
            "models": {
                "Background": {
                    "row_count": background_count,
                    "additional_row_tensors": background_additional,
                },
                "RigidNodes": {
                    "row_count": rigid_count,
                    "additional_row_tensors": rigid_additional,
                },
            },
        },
        "static_chunk_contract": {
            "model": "Background",
            "coordinate_tensor_path": "_means",
            "axes": ["x", "y"],
            "origin_xy_m": [0.0, 0.0],
            "cell_size_m": float(cell_size_m),
        },
        "actor_chunk_contract": {
            "model": "RigidNodes",
            "assignment_tensor_path": "points_ids",
            "actor_index_domain_inclusive": [actor_min, actor_max],
        },
        "package_contract": {
            "package_format": "worldsim_v32_chunk_package_v1",
        },
    }
