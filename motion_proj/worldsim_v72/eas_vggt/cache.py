"""公共视觉几何缓存；原子写入且拒绝任何 target 字段。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from motion_proj.worldsim_v72.eas_vggt.types import (
    BACKBONE_GEOMETRY_SCHEMA_VERSION,
    BackboneGeometry,
)


_ARRAY_FIELDS = (
    "frame_ids",
    "image_sha256",
    "model_from_original_px",
    "points_reference",
    "confidence",
    "valid_mask",
    "reference_from_camera_opencv",
    "intrinsics_model_px",
    "feature_grid",
)


def save_backbone_geometry(value: BackboneGeometry, output_path: Path) -> tuple[Path, Path]:
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("视觉几何缓存必须使用 .npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output_path.with_suffix(".json")
    temporary_npz = output_path.with_suffix(".tmp.npz")
    temporary_json = sidecar.with_suffix(".tmp.json")
    np.savez_compressed(
        temporary_npz,
        schema_version=np.asarray(BACKBONE_GEOMETRY_SCHEMA_VERSION),
        **{name: getattr(value, name) for name in _ARRAY_FIELDS},
    )
    metadata: dict[str, Any] = {
        "schema_version": BACKBONE_GEOMETRY_SCHEMA_VERSION,
        "backbone_id": value.backbone_id,
        "checkpoint_id": value.checkpoint_id,
        "repository_commit": value.repository_commit,
        "scale_status": value.scale_status,
        "provenance": dict(value.provenance),
    }
    lowered = {str(key).lower() for key in metadata | metadata["provenance"]}
    if any(any(word in key for word in ("target", "label", "ground_truth", "heldout")) for key in lowered):
        raise ValueError("视觉几何 cache metadata 不能包含监督字段")
    temporary_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_npz.replace(output_path)
    temporary_json.replace(sidecar)
    return output_path, sidecar


def load_backbone_geometry(input_path: Path) -> BackboneGeometry:
    input_path = Path(input_path)
    metadata = json.loads(input_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BACKBONE_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("视觉几何 JSON cache 版本不匹配")
    with np.load(input_path, allow_pickle=False) as payload:
        if str(payload["schema_version"]) != BACKBONE_GEOMETRY_SCHEMA_VERSION:
            raise ValueError("视觉几何 NPZ cache 版本不匹配")
        arrays = {name: payload[name] for name in _ARRAY_FIELDS}
    return BackboneGeometry(
        backbone_id=str(metadata["backbone_id"]),
        checkpoint_id=str(metadata["checkpoint_id"]),
        repository_commit=str(metadata["repository_commit"]),
        scale_status=str(metadata["scale_status"]),
        provenance=dict(metadata["provenance"]),
        **arrays,
    )
