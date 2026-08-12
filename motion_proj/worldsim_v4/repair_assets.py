"""M2 内容寻址 repair Gaussian 资产与临时可撤销挂载。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from motion_proj.worldsim_v33.spatial_delta import (
    atomic_save_npz,
    load_npz,
    sha256_file,
    temporary_spatial_composition,
)

from .repair_candidates import GaussianAssetBinding, REPAIR_METHODS


REPAIR_ASSET_SCHEMA_VERSION = "worldsim_v4_m2_repair_gaussian_asset_v1"
POINT_PROVENANCE = {
    "observed_cross_view": np.uint8(1),
    "native_scene_donor": np.uint8(2),
    "generated_telea": np.uint8(3),
    "generated_model": np.uint8(4),
}

_ROW_FIELDS = (
    "means",
    "raw_scales",
    "quats",
    "features_dc",
    "features_rest",
    "raw_opacities",
    "confidence",
    "point_provenance",
    "source_row_ids",
    "source_gaussian_ids",
    "source_frames",
    "source_camera_ids",
    "source_pixels_xy",
)
_META_FIELDS = ("schema_version", "candidate_id", "method", "provenance")


def _scalar_text(asset: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(asset[name])
    if value.ndim != 0:
        raise ValueError(f"repair asset {name} 必须为 scalar")
    return str(value.item())


def build_repair_asset(
    *,
    candidate_id: str,
    method: str,
    provenance: str,
    means: np.ndarray,
    raw_scales: np.ndarray,
    quats: np.ndarray,
    features_dc: np.ndarray,
    features_rest: np.ndarray,
    raw_opacities: np.ndarray,
    confidence: np.ndarray,
    source_gaussian_ids: np.ndarray | None = None,
    source_frames: np.ndarray | None = None,
    source_camera_ids: np.ndarray | None = None,
    source_pixels_xy: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """构造统一资产；缺失的 lineage/view 信息以 -1 显式记录。"""

    count = int(np.asarray(means).shape[0])
    if count <= 0:
        raise ValueError("repair asset 不允许为空")
    default_vector = np.full(count, -1, dtype=np.int64)
    default_pixels = np.full((count, 2), -1, dtype=np.int32)
    asset = {
        "schema_version": np.asarray(REPAIR_ASSET_SCHEMA_VERSION, dtype="<U64"),
        "candidate_id": np.asarray(str(candidate_id), dtype="<U128"),
        "method": np.asarray(str(method), dtype="<U16"),
        "provenance": np.asarray(str(provenance), dtype="<U64"),
        "means": np.asarray(means, dtype=np.float32),
        "raw_scales": np.asarray(raw_scales, dtype=np.float32),
        "quats": np.asarray(quats, dtype=np.float32),
        "features_dc": np.asarray(features_dc, dtype=np.float32),
        "features_rest": np.asarray(features_rest, dtype=np.float32),
        "raw_opacities": np.asarray(raw_opacities, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "point_provenance": np.full(
            count, POINT_PROVENANCE.get(str(provenance), np.uint8(0)), dtype=np.uint8
        ),
        "source_row_ids": np.arange(count, dtype=np.int64),
        "source_gaussian_ids": np.asarray(
            default_vector if source_gaussian_ids is None else source_gaussian_ids,
            dtype=np.int64,
        ),
        "source_frames": np.asarray(
            default_vector if source_frames is None else source_frames, dtype=np.int32
        ),
        "source_camera_ids": np.asarray(
            default_vector if source_camera_ids is None else source_camera_ids,
            dtype=np.int16,
        ),
        "source_pixels_xy": np.asarray(
            default_pixels if source_pixels_xy is None else source_pixels_xy,
            dtype=np.int32,
        ),
    }
    validate_repair_asset(asset)
    return asset


def validate_repair_asset(asset: Mapping[str, np.ndarray]) -> None:
    required = set(_META_FIELDS) | set(_ROW_FIELDS)
    if set(asset) != required:
        raise ValueError(f"repair asset 字段漂移: {sorted(set(asset) ^ required)}")
    if _scalar_text(asset, "schema_version") != REPAIR_ASSET_SCHEMA_VERSION:
        raise ValueError("repair asset schema version 漂移")
    candidate_id = _scalar_text(asset, "candidate_id")
    method = _scalar_text(asset, "method")
    provenance = _scalar_text(asset, "provenance")
    if not candidate_id:
        raise ValueError("repair asset candidate_id 不能为空")
    if method not in REPAIR_METHODS:
        raise ValueError(f"repair asset method 非法: {method}")
    if provenance not in POINT_PROVENANCE:
        raise ValueError(f"repair asset provenance 非法: {provenance}")

    count = int(np.asarray(asset["means"]).shape[0])
    if count <= 0 or any(np.asarray(asset[name]).shape[0] != count for name in _ROW_FIELDS):
        raise ValueError("repair asset 行数不一致或为空")
    expected = {
        "means": (count, 3),
        "raw_scales": (count, 3),
        "quats": (count, 4),
        "confidence": (count,),
        "point_provenance": (count,),
        "source_row_ids": (count,),
        "source_gaussian_ids": (count,),
        "source_frames": (count,),
        "source_camera_ids": (count,),
        "source_pixels_xy": (count, 2),
    }
    for name, shape in expected.items():
        if np.asarray(asset[name]).shape != shape:
            raise ValueError(f"repair asset {name} shape 非法")
    if np.asarray(asset["features_dc"]).ndim < 2 or np.asarray(
        asset["features_rest"]
    ).ndim < 2 or np.asarray(asset["raw_opacities"]).ndim != 2:
        raise ValueError("repair asset Gaussian feature shape 非法")
    numeric = (
        "means",
        "raw_scales",
        "quats",
        "features_dc",
        "features_rest",
        "raw_opacities",
        "confidence",
    )
    if any(not np.isfinite(np.asarray(asset[name])).all() for name in numeric):
        raise ValueError("repair asset 存在非有限数值")
    confidence = np.asarray(asset["confidence"], dtype=np.float32)
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("repair asset confidence 超出 [0,1]")
    expected_code = POINT_PROVENANCE[provenance]
    if not np.all(np.asarray(asset["point_provenance"]) == expected_code):
        raise ValueError("repair asset point provenance 与 manifest 不一致")
    rows = np.asarray(asset["source_row_ids"], dtype=np.int64)
    if not np.array_equal(rows, np.arange(count, dtype=np.int64)):
        raise ValueError("repair asset source_row_ids 必须连续唯一")
    quats = np.asarray(asset["quats"], dtype=np.float64)
    if np.any(np.linalg.norm(quats, axis=1) <= 1e-8):
        raise ValueError("repair asset quaternion 退化")


def atomic_save_repair_asset(
    path: str | Path, asset: Mapping[str, np.ndarray]
) -> GaussianAssetBinding:
    validate_repair_asset(asset)
    target = Path(path)
    atomic_save_npz(target, asset)
    return bind_repair_asset(target)


def load_repair_asset(path: str | Path) -> dict[str, np.ndarray]:
    asset = load_npz(path)
    validate_repair_asset(asset)
    return asset


def bind_repair_asset(path: str | Path) -> GaussianAssetBinding:
    target = Path(path)
    asset = load_repair_asset(target)
    return GaussianAssetBinding(
        path=str(target),
        sha256=sha256_file(target),
        bytes=target.stat().st_size,
        gaussian_count=int(asset["means"].shape[0]),
    )


def verify_repair_asset_binding(binding: GaussianAssetBinding) -> dict[str, np.ndarray]:
    target = Path(binding.path)
    asset = load_repair_asset(target)
    if target.stat().st_size != binding.bytes:
        raise ValueError("repair asset bytes 与 binding 不一致")
    if sha256_file(target) != binding.sha256:
        raise ValueError("repair asset SHA256 与 binding 不一致")
    if int(asset["means"].shape[0]) != binding.gaussian_count:
        raise ValueError("repair asset gaussian_count 与 binding 不一致")
    return asset


def _as_v33_background_delta(asset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """只在挂载边界适配旧 compositor；V4 资产本身保留真实 provenance。"""

    validate_repair_asset(asset)
    count = int(np.asarray(asset["means"]).shape[0])
    candidate_id = _scalar_text(asset, "candidate_id")
    return {
        "means": np.asarray(asset["means"]),
        "raw_scales": np.asarray(asset["raw_scales"]),
        "quats": np.asarray(asset["quats"]),
        "features_dc": np.asarray(asset["features_dc"]),
        "features_rest": np.asarray(asset["features_rest"]),
        "raw_opacities": np.asarray(asset["raw_opacities"]),
        "source_flat_indices": np.asarray(asset["source_row_ids"], dtype=np.int64),
        "source_gaussian_ids": np.arange(count, dtype=np.int64),
        "feather_weight": np.ones(count, dtype=np.float32),
        "provenance_code": np.full(count, 2, dtype=np.uint8),
        "target_role": np.asarray(["m2-repair"] * count, dtype="<U32"),
        "donor_patch_id": np.asarray([candidate_id] * count, dtype="<U128"),
        "donor_chunk_ids": np.asarray(["m2-adapter"] * count, dtype="<U256"),
    }


@contextmanager
def temporary_repair_composition(
    models: Mapping[str, Any],
    *,
    erase_delta: Mapping[str, np.ndarray],
    asset: Mapping[str, np.ndarray],
) -> Iterator[dict[str, Any]]:
    """原子执行 ERASE+INSERT_REPAIR，并继承 V3.3 的异常路径精确回滚。"""

    validate_repair_asset(asset)
    with temporary_spatial_composition(
        models,
        erase_delta=erase_delta,
        background_delta=_as_v33_background_delta(asset),
    ) as audit:
        yield {
            **audit,
            "schema_version": "worldsim_v4_m2_repair_composition_audit_v1",
            "candidate_id": _scalar_text(asset, "candidate_id"),
            "method": _scalar_text(asset, "method"),
            "provenance": _scalar_text(asset, "provenance"),
            "repair_asset_gaussian_count": int(np.asarray(asset["means"]).shape[0]),
        }
