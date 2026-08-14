"""V5 结构化 ownership evidence 的确定性、run-local 持久化合同。"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Mapping
import zipfile

import numpy as np


GAUSSIAN_ROW_FIELDS = {
    "gaussian_id",
    "base_model",
    "base_index",
    "center",
    "covariance",
    "normal_proxy",
    "normal_available",
    "prior",
    "unary_posterior",
    "unary_uncertainty",
    "effective_evidence_count",
    "multi_view_disagreement",
    "boundary_ambiguity",
    "depth_support",
    "lidar_support",
    "lidar_support_available",
    "motion_consistency",
    "motion_consistency_available",
}
GAUSSIAN_SCALARS = {"scene", "role"}
OBSERVATION_FIELDS = {
    "gaussian_id",
    "view_id",
    "frame_id",
    "camera_id",
    "projected_pixel",
    "visibility",
    "sam_probability",
    "sam_logit",
    "sam_probability_available",
    "mask_quality_accepted",
    "mask_boundary_distance",
    "depth_residual",
    "depth_consistent",
    "lidar_support",
    "lidar_support_available",
    "view_angle_cosine",
    "positive_observation",
    "negative_observation",
    "reliability",
    "contribution_weight",
}
OBSERVATION_SCALARS = {"scene", "role", "sam_probability_source"}
EDGE_FIELDS = {
    "source_gaussian_id",
    "target_gaussian_id",
    "mahalanobis_distance",
    "normal_distance",
    "motion_distance",
    "boundary_barrier",
    "edge_affinity",
}
EDGE_SCALARS = {"scene", "role"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(payload: Mapping[str, np.ndarray], required: set[str], label: str) -> None:
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{label} 缺少字段: {sorted(missing)}")


def _validate_probability(name: str, value: np.ndarray) -> None:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} 必须为有限 [0,1]")


def _validate_scalar_text(payload: Mapping[str, np.ndarray], names: set[str]) -> None:
    for name in names:
        if np.asarray(payload[name]).shape != () or not str(np.asarray(payload[name]).item()):
            raise ValueError(f"{name} 必须是非空 scalar text")


def validate_gaussian_table(table: Mapping[str, np.ndarray]) -> None:
    _require_fields(table, GAUSSIAN_ROW_FIELDS | GAUSSIAN_SCALARS, "per-Gaussian table")
    _validate_scalar_text(table, GAUSSIAN_SCALARS)
    gaussian_id = np.asarray(table["gaussian_id"], dtype=np.int64)
    count = gaussian_id.size
    if gaussian_id.ndim != 1 or not np.array_equal(gaussian_id, np.arange(count)):
        raise ValueError("gaussian_id 必须是连续全局索引")
    vector_shapes = {
        "base_model": (count,),
        "base_index": (count,),
        "center": (count, 3),
        "covariance": (count, 3, 3),
        "normal_proxy": (count, 3),
    }
    for name, shape in vector_shapes.items():
        if np.asarray(table[name]).shape != shape:
            raise ValueError(f"{name} shape 必须为 {shape}")
    scalar_rows = GAUSSIAN_ROW_FIELDS - set(vector_shapes) - {"gaussian_id"}
    for name in scalar_rows:
        if np.asarray(table[name]).shape != (count,):
            raise ValueError(f"{name} shape 必须为 {(count,)}")
    for name in (
        "prior",
        "unary_posterior",
        "unary_uncertainty",
        "multi_view_disagreement",
        "boundary_ambiguity",
        "depth_support",
        "lidar_support",
        "motion_consistency",
        "normal_available",
        "lidar_support_available",
        "motion_consistency_available",
    ):
        _validate_probability(name, np.asarray(table[name]))
    effective = np.asarray(table["effective_evidence_count"], dtype=np.float64)
    if not np.isfinite(effective).all() or np.any(effective < 0.0):
        raise ValueError("effective_evidence_count 必须有限非负")
    for name in ("center", "covariance", "normal_proxy"):
        if not np.isfinite(np.asarray(table[name], dtype=np.float64)).all():
            raise ValueError(f"{name} 必须有限")
    covariance = np.asarray(table["covariance"], dtype=np.float64)
    if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-6):
        raise ValueError("covariance 必须对称")
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise ValueError("covariance 必须正定")
    normal = np.asarray(table["normal_proxy"], dtype=np.float64)
    normal_available = np.asarray(table["normal_available"], dtype=bool)
    if np.any(
        np.abs(np.linalg.norm(normal[normal_available], axis=1) - 1.0) > 1e-4
    ):
        raise ValueError("available normal_proxy 必须为单位向量")
    for value_name, availability_name in (
        ("lidar_support", "lidar_support_available"),
        ("motion_consistency", "motion_consistency_available"),
    ):
        values = np.asarray(table[value_name], dtype=np.float64)
        available = np.asarray(table[availability_name], dtype=bool)
        if np.any(values[~available] != 0.0):
            raise ValueError(f"unavailable {value_name} 必须显式为 0")


def validate_observation_chunk(
    chunk: Mapping[str, np.ndarray], *, gaussian_count: int | None = None
) -> None:
    _require_fields(chunk, OBSERVATION_FIELDS | OBSERVATION_SCALARS, "per-view observation")
    _validate_scalar_text(chunk, OBSERVATION_SCALARS)
    gaussian_id = np.asarray(chunk["gaussian_id"], dtype=np.int64)
    count = gaussian_id.size
    if gaussian_id.ndim != 1 or np.any(gaussian_id < 0):
        raise ValueError("observation gaussian_id 必须为一维非负")
    if gaussian_count is not None and np.any(gaussian_id >= gaussian_count):
        raise ValueError("observation gaussian_id 超出 per-Gaussian table")
    for name in OBSERVATION_FIELDS - {"gaussian_id", "projected_pixel"}:
        if np.asarray(chunk[name]).shape != (count,):
            raise ValueError(f"observation {name} shape 必须为 {(count,)}")
    if np.asarray(chunk["projected_pixel"]).shape != (count, 2):
        raise ValueError("projected_pixel shape 必须为 (N,2)")
    for name in (
        "visibility",
        "sam_probability",
        "sam_probability_available",
        "mask_quality_accepted",
        "depth_consistent",
        "lidar_support",
        "lidar_support_available",
        "view_angle_cosine",
        "positive_observation",
        "negative_observation",
        "reliability",
    ):
        _validate_probability(name, np.asarray(chunk[name]))
    for name in (
        "projected_pixel",
        "sam_logit",
        "mask_boundary_distance",
        "depth_residual",
        "contribution_weight",
    ):
        value = np.asarray(chunk[name], dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"observation {name} 必须有限")
    if np.any(np.asarray(chunk["contribution_weight"], dtype=np.float64) < 0.0):
        raise ValueError("contribution_weight 必须非负")
    positive = np.asarray(chunk["positive_observation"], dtype=np.float64)
    negative = np.asarray(chunk["negative_observation"], dtype=np.float64)
    if np.any((positive + negative) > 1.0 + 1e-6):
        raise ValueError("positive/negative observation 不得同时为真")
    accepted = np.asarray(chunk["mask_quality_accepted"], dtype=bool)
    probability_available = np.asarray(chunk["sam_probability_available"], dtype=bool)
    if np.any((positive + negative)[~(accepted & probability_available)] != 0.0):
        raise ValueError("rejected/unavailable SAM observation 不得伪装成正负证据")
    lidar = np.asarray(chunk["lidar_support"], dtype=np.float64)
    lidar_available = np.asarray(chunk["lidar_support_available"], dtype=bool)
    if np.any(lidar[~lidar_available] != 0.0):
        raise ValueError("unavailable observation lidar_support 必须显式为 0")


def validate_edge_table(table: Mapping[str, np.ndarray], *, gaussian_count: int) -> None:
    _require_fields(table, EDGE_FIELDS | EDGE_SCALARS, "per-edge table")
    _validate_scalar_text(table, EDGE_SCALARS)
    source = np.asarray(table["source_gaussian_id"], dtype=np.int64)
    target = np.asarray(table["target_gaussian_id"], dtype=np.int64)
    count = source.size
    for name in EDGE_FIELDS - {"source_gaussian_id"}:
        if np.asarray(table[name]).shape != (count,):
            raise ValueError(f"edge {name} shape 必须为 {(count,)}")
    if np.any(source < 0) or np.any(target < 0) or np.any(source >= gaussian_count) or np.any(target >= gaussian_count):
        raise ValueError("edge Gaussian 索引越界")
    if np.any(source == target):
        raise ValueError("edge 不允许 self-loop")
    for name in ("mahalanobis_distance", "normal_distance", "motion_distance"):
        value = np.asarray(table[name], dtype=np.float64)
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"edge {name} 必须有限非负")
    for name in ("boundary_barrier", "edge_affinity"):
        _validate_probability(name, np.asarray(table[name]))


def atomic_save_npz(path: str | Path, payload: Mapping[str, np.ndarray]) -> None:
    """用固定 ZIP metadata 写出 byte-stable NPZ。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(payload):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(payload[name]), allow_pickle=False)
                entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(
                    entry,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    os.replace(temporary, target)
