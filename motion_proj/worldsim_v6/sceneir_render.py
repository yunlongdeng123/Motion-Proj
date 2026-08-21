"""SceneIR v0 到前端中立 Gaussian renderer view 的适配层。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from motion_proj.worldsim_v6.sceneir import SceneIRError, load_sceneir
from motion_proj.worldsim_v6.sceneir_adapters import (
    normalize_quaternions,
    quaternion_multiply,
    quaternion_to_matrix,
)


VIEW_ARRAYS = (
    "means_m",
    "scales_m",
    "quaternions_wxyz",
    "opacities",
    "features_dc",
    "features_rest",
    "velocities_mps",
    "source_indices",
)


def _chunk_values(chunk: dict[str, Any], blobs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: blobs[ref["sha256"]] for name, ref in chunk["arrays"].items()}


def _transform_at(document: dict[str, Any], frame_id: str, timestamp_us: int) -> dict[str, Any]:
    name = f"T_world_{frame_id}"
    matches = [
        row
        for row in document["transforms"]
        if row["name"] == name and row["timestamp_us"] == timestamp_us
    ]
    if len(matches) != 1:
        raise SceneIRError(f"{name}@{timestamp_us} 不唯一或不存在")
    return matches[0]


def render_view(package: Path, timestamp_us: int) -> dict[str, dict[str, np.ndarray]]:
    """加载静态与 actor Gaussian，并在给定时间组装到 world frame。"""
    document, blobs = load_sceneir(package)
    if not document["episode"]["start_timestamp_us"] <= timestamp_us <= document["episode"]["end_timestamp_us"]:
        raise SceneIRError("render timestamp 越出 episode")
    groups: dict[str, list[dict[str, np.ndarray]]] = {"static": [], "dynamic": []}
    for chunk in document["chunks"]:
        values = _chunk_values(chunk, blobs)
        if chunk["role"] == "static":
            groups["static"].append(values)
            continue
        transform = _transform_at(document, chunk["frame_id"], timestamp_us)
        pose_q = normalize_quaternions(np.asarray(transform["rotation_wxyz"], dtype=np.float32))
        pose_t = np.asarray(transform["translation_m"], dtype=np.float32)
        rotation = quaternion_to_matrix(pose_q)
        transformed = dict(values)
        transformed["means_m"] = values["means_m"] @ rotation.T + pose_t
        transformed["quaternions_wxyz"] = normalize_quaternions(
            quaternion_multiply(pose_q, values["quaternions_wxyz"])
        )
        if "velocities_mps" in values:
            transformed["velocities_mps"] = values["velocities_mps"] @ rotation.T
        groups["dynamic"].append(transformed)

    result: dict[str, dict[str, np.ndarray]] = {}
    for group_name, chunks in groups.items():
        if not chunks:
            result[group_name] = {}
            continue
        common_names = set.intersection(*(set(chunk) for chunk in chunks))
        merged = {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in VIEW_ARRAYS
            if name in common_names
        }
        order = np.argsort(merged["source_indices"], kind="stable")
        result[group_name] = {name: value[order] for name, value in merged.items()}
    return result


def compare_views(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """逐字段比较 Gaussian view，quaternion 同时接受 q 与 -q。"""
    if set(expected) != set(actual):
        raise SceneIRError(f"runtime view 字段漂移：{sorted(expected)} != {sorted(actual)}")
    fields: dict[str, Any] = {}
    passed = True
    for name in sorted(expected):
        left = np.asarray(expected[name])
        right = np.asarray(actual[name])
        shape_equal = left.shape == right.shape
        if not shape_equal:
            fields[name] = {"shape_equal": False, "passed": False}
            passed = False
            continue
        if name == "quaternions_wxyz":
            direct = np.max(np.abs(left - right), axis=-1)
            antipodal = np.max(np.abs(left + right), axis=-1)
            error = np.minimum(direct, antipodal)
            max_abs = float(error.max(initial=0.0))
            field_passed = bool(np.all(error <= atol + rtol))
        elif np.issubdtype(left.dtype, np.integer) or left.dtype == bool:
            max_abs = 0.0 if np.array_equal(left, right) else float("inf")
            field_passed = bool(np.array_equal(left, right))
        else:
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            max_abs = float(difference.max(initial=0.0))
            field_passed = bool(np.allclose(left, right, atol=atol, rtol=rtol))
        fields[name] = {
            "shape_equal": True,
            "shape": list(left.shape),
            "max_abs_error": max_abs,
            "passed": field_passed,
        }
        passed = passed and field_passed
    return {"passed": passed, "fields": fields, "atol": atol, "rtol": rtol}
