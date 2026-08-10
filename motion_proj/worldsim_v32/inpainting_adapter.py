"""V3.2 生成背景点与 StreetGS Background Gaussian 之间的显式适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v3.gaussian_ancestry import (
    GaussianAncestryLedger,
    InitSource,
    PER_GAUSSIAN_FIELDS,
    RUNNING_METRICS,
)

from .depth_guided_unseen_mask import unproject_rgbd


@dataclass(frozen=True)
class CompletionPoints:
    means: np.ndarray
    rgb: np.ndarray
    scales: np.ndarray
    confidence: np.ndarray
    observed_cross_view: np.ndarray
    source_pixels_xy: np.ndarray


def completion_points_from_view(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    observed_cross_view: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    stride: int,
    scale_multiplier: float,
    minimum_scale_m: float,
    maximum_scale_m: float,
) -> CompletionPoints:
    """按冻结深度把生成完成图反投影成背景 Gaussian 候选。"""

    if not 0 < minimum_scale_m <= maximum_scale_m:
        raise ValueError("Gaussian scale 上下界非法")
    if scale_multiplier <= 0:
        raise ValueError("scale_multiplier 必须为正")
    observed = np.asarray(observed_cross_view, dtype=bool)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (np.asarray(depth) > 1e-4)
    means, colors, pixels = unproject_rgbd(
        depth=depth,
        rgb=rgb,
        valid=valid,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        stride=stride,
    )
    if means.shape[0] == 0:
        return CompletionPoints(
            means=means.astype(np.float32),
            rgb=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            observed_cross_view=np.empty((0,), dtype=bool),
            source_pixels_xy=pixels,
        )
    k = np.asarray(intrinsics, dtype=np.float64)
    sampled_depth = np.asarray(depth)[pixels[:, 1], pixels[:, 0]]
    focal = max(float((k[0, 0] + k[1, 1]) / 2.0), 1e-6)
    footprint = sampled_depth * float(stride) / focal * float(scale_multiplier)
    footprint = np.clip(footprint, minimum_scale_m, maximum_scale_m)
    point_observed = observed[pixels[:, 1], pixels[:, 0]]
    confidence = np.where(point_observed, 0.9, 0.5).astype(np.float32)
    return CompletionPoints(
        means=means.astype(np.float32),
        rgb=colors.astype(np.float32) / 255.0,
        scales=np.repeat(footprint[:, None], 3, axis=1).astype(np.float32),
        confidence=confidence,
        observed_cross_view=point_observed,
        source_pixels_xy=pixels,
    )


def merge_completion_points(
    groups: list[CompletionPoints], *, voxel_size_m: float
) -> CompletionPoints:
    """对多个目标视图的补全点做确定性 voxel 去重。"""

    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m 必须为正")
    nonempty = [group for group in groups if group.means.shape[0]]
    if not nonempty:
        return CompletionPoints(
            means=np.empty((0, 3), dtype=np.float32),
            rgb=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            observed_cross_view=np.empty((0,), dtype=bool),
            source_pixels_xy=np.empty((0, 2), dtype=np.int64),
        )
    means = np.concatenate([group.means for group in nonempty])
    rgb = np.concatenate([group.rgb for group in nonempty])
    scales = np.concatenate([group.scales for group in nonempty])
    confidence = np.concatenate([group.confidence for group in nonempty])
    observed = np.concatenate([group.observed_cross_view for group in nonempty])
    pixels = np.concatenate([group.source_pixels_xy for group in nonempty])
    voxels = np.floor(means / float(voxel_size_m)).astype(np.int64)
    order = np.lexsort(
        (
            np.arange(means.shape[0], dtype=np.int64),
            -confidence,
            voxels[:, 2],
            voxels[:, 1],
            voxels[:, 0],
        )
    )
    sorted_voxels = voxels[order]
    first = np.r_[True, np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)]
    chosen = order[first]
    chosen.sort()
    return CompletionPoints(
        means=means[chosen],
        rgb=rgb[chosen],
        scales=scales[chosen],
        confidence=confidence[chosen],
        observed_cross_view=observed[chosen],
        source_pixels_xy=pixels[chosen],
    )


def _append_generated_ancestry(
    ledger: GaussianAncestryLedger,
    *,
    parent_rows: torch.Tensor,
    birth_step: int,
) -> tuple[int, int]:
    """扩展 V3.1 账本索引；权威生成类型由 V3.2 sidecar 提供。"""

    count = int(parent_rows.numel())
    if count <= 0:
        raise ValueError("generated ancestry 不能为空")
    parent_rows = parent_rows.to(device=ledger.device, dtype=torch.long).flatten()
    if int(parent_rows.min()) < 0 or int(parent_rows.max()) >= len(ledger):
        raise ValueError("generated ancestry 的父行越界")
    start = int(ledger.next_gaussian_id)
    ids = torch.arange(start, start + count, device=ledger.device, dtype=torch.long)
    zeros = torch.zeros(count, device=ledger.device, dtype=torch.long)
    nan = torch.full((count,), torch.nan, device=ledger.device)
    rows: dict[str, torch.Tensor] = {
        "gaussian_id": ids,
        "actor_id": torch.full_like(ids, -1),
        # V3.1 稳定编码中没有 generated-background；V3.2 sidecar 必须随候选 checkpoint 保留。
        "init_source": torch.full_like(ids, int(InitSource.UNKNOWN)),
        # V3.1 账本要求 lineage root 来自初始 Gaussian。生成点以最近的旧背景
        # Gaussian 为父节点；V3.2 sidecar 另外明确标记 GENERATED_BACKGROUND。
        "parent_id": ledger.gaussian_id[parent_rows].clone(),
        "lineage_root_id": ledger.lineage_root_id[parent_rows].clone(),
        "birth_step": torch.full_like(ids, int(birth_step)),
        "generation": ledger.generation[parent_rows].clone() + 1,
        "visibility_count": zeros.clone(),
        "nearest_lidar_distance": nan.clone(),
    }
    for metric in RUNNING_METRICS:
        rows[metric] = nan.clone()
        rows[f"{metric}_count"] = zeros.clone()
    if set(rows) != set(PER_GAUSSIAN_FIELDS):
        raise RuntimeError("generated ancestry 字段与冻结账本不一致")
    ledger._append(rows)
    ledger.next_gaussian_id += count
    ledger.validate()
    return start, start + count


def append_generated_background(
    background: Any,
    points: CompletionPoints,
    *,
    opacity: float,
    birth_step: int,
) -> dict[str, int | float]:
    """把 GENERATED_BACKGROUND 点追加到 StreetGS Background，不改动旧行。"""

    if points.means.ndim != 2 or points.means.shape[1] != 3:
        raise ValueError("completion means 必须为 (N, 3)")
    count = int(points.means.shape[0])
    if count == 0:
        raise ValueError("不允许追加空背景资产")
    if not 0 < opacity < 1:
        raise ValueError("opacity 必须介于 0 和 1")
    device = background._means.device
    dtype = background._means.dtype
    means = torch.as_tensor(points.means, device=device, dtype=dtype)
    rgb = torch.as_tensor(points.rgb, device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
    scales = torch.as_tensor(points.scales, device=device, dtype=dtype)
    if scales.shape != (count, background._scales.shape[1]):
        raise ValueError("completion scales 与 Background schema 不一致")
    if background.sh_degree > 0:
        from models.gaussians.basics import RGB2SH

        features_dc = RGB2SH(rgb)
    else:
        features_dc = torch.logit(rgb)
    features_rest = torch.zeros(
        (count,) + tuple(background._features_rest.shape[1:]),
        device=device,
        dtype=background._features_rest.dtype,
    )
    quats = torch.zeros(
        (count, background._quats.shape[1]), device=device, dtype=background._quats.dtype
    )
    quats[:, 0] = 1.0
    raw_opacity = torch.logit(
        torch.full(
            (count, background._opacities.shape[1]),
            float(opacity),
            device=device,
            dtype=background._opacities.dtype,
        )
    )
    old_count = int(background.num_points)
    if old_count <= 0:
        raise RuntimeError("Background 没有可作为 lineage anchor 的旧 Gaussian")
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - 正式环境必须有 scipy
        raise RuntimeError("追加生成背景需要 scipy.spatial.cKDTree") from exc
    old_means = background._means.detach().cpu().numpy()
    _, nearest_rows = cKDTree(old_means).query(points.means, k=1, workers=-1)
    parent_rows = torch.as_tensor(nearest_rows, device=device, dtype=torch.long)
    background._means = nn.Parameter(torch.cat([background._means.detach(), means]))
    background._features_dc = nn.Parameter(
        torch.cat([background._features_dc.detach(), features_dc])
    )
    background._features_rest = nn.Parameter(
        torch.cat([background._features_rest.detach(), features_rest])
    )
    background._opacities = nn.Parameter(
        torch.cat([background._opacities.detach(), raw_opacity])
    )
    background._scales = nn.Parameter(
        torch.cat([background._scales.detach(), torch.log(scales)])
    )
    background._quats = nn.Parameter(torch.cat([background._quats.detach(), quats]))
    ledger = getattr(background, "_a2_ancestry", None)
    if ledger is None or len(ledger) != old_count:
        raise RuntimeError("Background ancestry 缺失或与原 Gaussian 数不对齐")
    ancestry_start, ancestry_end = _append_generated_ancestry(
        ledger, parent_rows=parent_rows, birth_step=birth_step
    )
    if int(background.num_points) != old_count + count or len(ledger) != old_count + count:
        raise RuntimeError("Background/ancestry 追加后数量不对齐")
    return {
        "old_background_count": old_count,
        "generated_count": count,
        "new_background_count": int(background.num_points),
        "row_start": old_count,
        "row_end_exclusive": old_count + count,
        "ancestry_gaussian_id_start": ancestry_start,
        "ancestry_gaussian_id_end_exclusive": ancestry_end,
        "opacity": float(opacity),
    }
