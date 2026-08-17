"""V5.1 Stage B 的确定性 H feature/PCA sidecar 基础算子。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def select_h_uplift_records(
    manifest: Mapping[str, Any],
    *,
    scenes: Sequence[str],
    frames: Sequence[int],
    cameras: Sequence[int],
) -> list[dict[str, Any]]:
    """按冻结 scene/frame/camera 顺序选择且完整核对 H uplift 图像。"""
    scene_order = {name: index for index, name in enumerate(scenes)}
    frame_set = {int(value) for value in frames}
    camera_set = {int(value) for value in cameras}
    selected = [
        dict(record)
        for record in manifest.get("records", [])
        if record.get("role") == "historical_diagnostic"
        and record.get("scene") in scene_order
        and int(record.get("frame", -1)) in frame_set
        and int(record.get("camera", -1)) in camera_set
    ]
    selected.sort(
        key=lambda row: (
            scene_order[row["scene"]],
            int(row["frame"]),
            int(row["camera"]),
        )
    )
    expected = {
        (scene, int(frame), int(camera))
        for scene in scenes
        for frame in frames
        for camera in cameras
    }
    observed = {
        (row["scene"], int(row["frame"]), int(row["camera"]))
        for row in selected
    }
    if observed != expected or len(selected) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"H uplift image grid 不完整: missing={missing}, extra={extra}")
    for row in selected:
        if [int(row.get("width", -1)), int(row.get("height", -1))] != [1600, 900]:
            raise ValueError(f"H uplift image size 漂移: {row.get('path')}")
    return selected


def record_chain_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """对有序 record 形成无空白 canonical JSONL chain SHA。"""
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """哈希 C-contiguous 数组的 dtype/shape/content，避免容器元数据歧义。"""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def feature_mean_std_correction1(
    features: np.ndarray, *, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """固定 row order 的两遍 float64 mean/std（sample correction=1）。"""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("feature statistics 输入必须为至少两行的二维矩阵")
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows 必须为正")
    count, dimension = matrix.shape
    total = np.zeros(dimension, dtype=np.float64)
    for start in range(0, count, int(chunk_rows)):
        block = np.asarray(matrix[start : start + int(chunk_rows)], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("raw feature 含非 finite")
        total += block.sum(axis=0, dtype=np.float64)
    mean = total / float(count)
    squared = np.zeros(dimension, dtype=np.float64)
    for start in range(0, count, int(chunk_rows)):
        block = np.asarray(matrix[start : start + int(chunk_rows)], dtype=np.float64)
        delta = block - mean
        squared += np.einsum("ij,ij->j", delta, delta, dtype=np.float64)
    std = np.sqrt(squared / float(count - 1))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("feature mean/std 非 finite")
    if np.any(std <= 1e-12):
        raise ValueError("feature std 出现零或近零维度")
    return mean, std


def standardize_in_place(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    chunk_rows: int,
) -> None:
    """按固定 chunk 将 float32 memmap 原位标准化。"""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.dtype != np.float32:
        raise ValueError("in-place standardization 要求二维 float32")
    mean64 = np.asarray(mean, dtype=np.float64)
    std64 = np.asarray(std, dtype=np.float64)
    if mean64.shape != (matrix.shape[1],) or std64.shape != mean64.shape:
        raise ValueError("mean/std shape 漂移")
    for start in range(0, matrix.shape[0], int(chunk_rows)):
        stop = min(start + int(chunk_rows), matrix.shape[0])
        block = np.asarray(matrix[start:stop], dtype=np.float64)
        matrix[start:stop] = ((block - mean64) / std64).astype(np.float32)
    if hasattr(features, "flush"):
        features.flush()


def pca_patch_grid(
    standardized_rows: np.ndarray,
    *,
    pca_mean: np.ndarray,
    components: np.ndarray,
    grid_hw: Sequence[int],
) -> np.ndarray:
    """把一个 view 的标准化 patches 变换为 `[D,H,W]` float32。"""
    rows = np.asarray(standardized_rows, dtype=np.float32)
    mean = np.asarray(pca_mean, dtype=np.float32)
    basis = np.asarray(components, dtype=np.float32)
    height, width = [int(value) for value in grid_hw]
    if rows.shape != (height * width, basis.shape[1]):
        raise ValueError("PCA view row shape 漂移")
    if mean.shape != (basis.shape[1],):
        raise ValueError("PCA mean shape 漂移")
    transformed = (rows - mean) @ basis.T
    grid = transformed.reshape(height, width, basis.shape[0]).transpose(2, 0, 1)
    result = np.ascontiguousarray(grid, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("PCA sidecar 含非 finite")
    return result


def validate_sidecar_identity(record: Mapping[str, Any], feature: np.ndarray) -> None:
    """验证 sidecar metadata 与数组内容一致。"""
    value = np.asarray(feature)
    if list(value.shape) != list(record["shape"]):
        raise ValueError("sidecar shape identity 漂移")
    if str(value.dtype) != record["dtype"]:
        raise ValueError("sidecar dtype identity 漂移")
    if array_sha256(value) != record["content_sha256"]:
        raise ValueError("sidecar content identity 漂移")


def sidecar_relative_path(record: Mapping[str, Any]) -> Path:
    """构造固定且不依赖绝对数据根的 sidecar 路径。"""
    return (
        Path("artifacts/features")
        / str(record["scene"])
        / f"{int(record['frame']):03d}_{int(record['camera'])}.npz"
    )
