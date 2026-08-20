"""V5.2.1 Discovery-only 基座 badcase census 诊断封装。"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from motion_proj.worldsim_v4.evaluator import LpipsRegionProtocol, evaluate_frame
from motion_proj.worldsim_v4.region_masks import RegionMaskProtocol, build_baseline_region_masks


class CensusError(ValueError):
    """输入违反 V5.2.1 census 冻结合同。"""


@dataclass(frozen=True)
class CensusProtocol:
    metric_width: int = 800
    metric_height: int = 450
    boundary_radius_pixels: int = 3
    actor_minimum_pixels: int = 64
    boundary_minimum_pixels: int = 64
    lpips_crop_padding_pixels: int = 8
    lpips_minimum_side_pixels: int = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def load_metric_rgb(path: str | Path, protocol: CensusProtocol = CensusProtocol()) -> tuple[np.ndarray, str]:
    """以冻结 LANCZOS 路径解码并返回 canonical resized-pixel hash。"""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        expected = (protocol.metric_width, protocol.metric_height)
        if image.size != expected:
            image = image.resize(expected, Image.Resampling.LANCZOS)
        payload = _png_bytes(image)
        array = np.asarray(image, dtype=np.float64) / 255.0
    return array, sha256_bytes(payload)


def load_metric_mask(path: str | Path, protocol: CensusProtocol = CensusProtocol()) -> tuple[np.ndarray, str]:
    with Image.open(path) as opened:
        mask = opened.convert("L")
        expected = (protocol.metric_width, protocol.metric_height)
        if mask.size != expected:
            mask = mask.resize(expected, Image.Resampling.NEAREST)
        array = np.asarray(mask) > 0
        canonical = Image.fromarray(array.astype(np.uint8) * 255, mode="L")
        digest = sha256_bytes(_png_bytes(canonical))
    return array, digest


def validate_discovery_record(record: Mapping[str, Any]) -> None:
    if record.get("partition") != "discovery":
        raise CensusError("P2 只允许 Discovery record")
    if record.get("quality_decoded") not in (False, None):
        raise CensusError("P1 registry 必须来自 quality-blind freeze")
    if record.get("eligible_bases") != ["adgs", "streetgs"]:
        raise CensusError("eligible base denominator 漂移")
    if record.get("sample_token") is not None:
        raise CensusError("当前 processed cohort 必须使用 canonical_sample_index")


def geometry_undefined() -> dict[str, Any]:
    return {
        "status": "undefined_no_comparable_base_depth",
        "prediction_depth_semantics": None,
        "static_lidar_depth_mae": None,
        "actor_lidar_depth_mae": None,
        "valid_projected_lidar_count": None,
        "actor_lidar_support_count": None,
        "cross_base_ranking_allowed": False,
    }


def actor_context_union(dynamic: np.ndarray, annotation_sha256: str) -> dict[str, Any]:
    count = int(np.asarray(dynamic, dtype=bool).sum())
    return {
        "entity_kind": "dynamic_union",
        "actor_token": None,
        "region_source": "drivestudio_dynamic_masks/all",
        "is_ground_truth": False,
        "annotation_sha256": annotation_sha256,
        "overlap_rule": "union_nonzero",
        "valid_pixel_count": count,
        "image_area_ratio": count / float(dynamic.size),
        "instance_metric_status": "undefined_no_instance_region",
        "visibility_level": None,
        "category": None,
        "camera_count": None,
        "lidar_point_count": None,
        "distance_m": None,
        "world_speed_mps": None,
        "view_transition": "undefined_no_audited_actor_track_annotation",
        "occlusion_transition": "undefined_no_audited_visibility_annotation",
    }


def evaluate_discovery_view(
    *,
    base: str,
    record: Mapping[str, Any],
    prediction_path: str | Path,
    dynamic_mask_path: str | Path,
    lpips_model: Callable[[Any, Any], Any] | None,
    renderer_provenance: Mapping[str, Any],
    resource: Mapping[str, Any],
    protocol: CensusProtocol = CensusProtocol(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_discovery_record(record)
    if base not in record["eligible_bases"]:
        raise CensusError(f"base 不在 frozen denominator：{base}")
    prediction, prediction_pixel_sha = load_metric_rgb(prediction_path, protocol)
    target, target_pixel_sha = load_metric_rgb(record["target_path"], protocol)
    if prediction.shape != target.shape:
        raise CensusError("prediction/target metric shape 不一致")
    dynamic, dynamic_sha = load_metric_mask(dynamic_mask_path, protocol)
    egocar = np.zeros(dynamic.shape, dtype=bool)
    masks = build_baseline_region_masks(
        dynamic,
        egocar,
        protocol=RegionMaskProtocol(boundary_radius_pixels=protocol.boundary_radius_pixels),
    )
    evaluation = evaluate_frame(
        prediction,
        target,
        masks,
        lpips_model=lpips_model,
        lpips_protocol=LpipsRegionProtocol(
            crop_padding_pixels=protocol.lpips_crop_padding_pixels,
            minimum_side_pixels=protocol.lpips_minimum_side_pixels,
        ),
    )
    actor_context = actor_context_union(dynamic, dynamic_sha)
    identity = {
        "base": base,
        "scene": record["scene"],
        "frame": int(record["frame"]),
        "sample_token": record.get("sample_token"),
        "canonical_sample_index": int(record["canonical_sample_index"]),
        "camera": int(record["camera"]),
        "partition": "discovery",
    }
    metrics = {
        name: evaluation["regions"][name]
        for name in ("global", "static", "actor", "boundary")
    }
    metrics["geometry"] = geometry_undefined()
    metrics["temporal"] = {
        "status": "deferred_to_window_table",
        "correspondence": "none",
        "may_trigger_b_temporal": False,
    }
    base_row = {
        **identity,
        "entity_kind": "view",
        "prediction_path": str(Path(prediction_path).resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_pixel_sha256": prediction_pixel_sha,
        "target_path": str(Path(record["target_path"]).resolve()),
        "target_source_sha256": record["target_sha256"],
        "target_sha256": target_pixel_sha,
        "dynamic_mask_path": str(Path(dynamic_mask_path).resolve()),
        "dynamic_mask_sha256": dynamic_sha,
        "egocar_mask_status": "undefined_asset_zero_mask_with_provenance",
        "metrics": metrics,
        "actor_context": actor_context,
        "renderer": dict(renderer_provenance),
        "resource": dict(resource),
    }
    actor_row = {
        **identity,
        "entity_kind": "dynamic_union",
        "actor_token": None,
        "region_provenance": actor_context,
        "valid_support": {
            "pixel_count": actor_context["valid_pixel_count"],
            "minimum_pixels": protocol.actor_minimum_pixels,
            "sufficient": actor_context["valid_pixel_count"] >= protocol.actor_minimum_pixels,
        },
        "metrics": metrics["actor"],
        "undefined_reason": "undefined_no_instance_region",
    }
    return base_row, actor_row


def temporal_proxy_row(
    *,
    earlier: Mapping[str, Any],
    later: Mapping[str, Any],
    earlier_prediction: np.ndarray,
    earlier_target: np.ndarray,
    later_prediction: np.ndarray,
    later_target: np.ndarray,
    earlier_dynamic: np.ndarray,
    later_dynamic: np.ndarray,
    protocol: CensusProtocol = CensusProtocol(),
) -> dict[str, Any]:
    for row in (earlier, later):
        validate_discovery_record(row)
    if earlier["scene"] != later["scene"] or earlier["camera"] != later["camera"]:
        raise CensusError("temporal window 必须同 scene/camera")
    residual_a = np.mean(np.abs(earlier_prediction - earlier_target), axis=-1)
    residual_b = np.mean(np.abs(later_prediction - later_target), axis=-1)
    delta = np.abs(residual_b - residual_a)
    actor_union = np.asarray(earlier_dynamic, bool) | np.asarray(later_dynamic, bool)
    boundary_a = build_baseline_region_masks(
        earlier_dynamic, np.zeros_like(earlier_dynamic), protocol=RegionMaskProtocol(protocol.boundary_radius_pixels)
    )["boundary"]
    boundary_b = build_baseline_region_masks(
        later_dynamic, np.zeros_like(later_dynamic), protocol=RegionMaskProtocol(protocol.boundary_radius_pixels)
    )["boundary"]
    boundary_union = boundary_a | boundary_b

    def mean_or_none(mask: np.ndarray) -> float | None:
        return None if not mask.any() else float(np.mean(delta[mask], dtype=np.float64))

    frame_a, frame_b = int(earlier["frame"]), int(later["frame"])
    return {
        "entity_kind": "temporal_window",
        "scene": earlier["scene"],
        "camera": int(earlier["camera"]),
        "partition": "discovery",
        "window_id": f"{earlier['scene']}|c{int(earlier['camera'])}|f{frame_a:03d}-f{frame_b:03d}",
        "member_sample_tokens": [earlier.get("sample_token"), later.get("sample_token")],
        "member_canonical_sample_indices": [frame_a, frame_b],
        "frame_gap": frame_b - frame_a,
        "correspondence_provenance": "none",
        "status": "unwarped_temporal_proxy",
        "may_trigger_b_temporal": False,
        "may_trigger_b_occ": False,
        "metrics": {
            "global_residual_change_l1": float(np.mean(delta, dtype=np.float64)),
            "actor_union_residual_change_l1": mean_or_none(actor_union),
            "boundary_union_residual_change_l1": mean_or_none(boundary_union),
            "actor_crop_lpips_temporal_delta": None,
            "same_track_actor_quality_variance": None,
            "visibility_transition_quality_jump": None,
        },
        "undefined_reasons": {
            "warp": "undefined_no_frozen_flow_or_correspondence",
            "actor_track": "undefined_no_audited_actor_instance_region",
            "occlusion": "undefined_no_audited_visibility_transition",
        },
    }


def assert_unique_keys(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            raise CensusError(f"重复主键：{key}")
        seen.add(key)
