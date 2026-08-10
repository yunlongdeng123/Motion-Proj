"""Asset Harvester 的确定性单视图评分与多视图集合选择。"""

from __future__ import annotations

from itertools import combinations
import io
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


VIEW_COMPONENTS = (
    "area",
    "mask",
    "sharpness",
    "visible",
    "occlusion",
    "truncation",
)


def atomic_save_deterministic_npz(
    path: str | Path, arrays: Mapping[str, np.ndarray]
) -> None:
    """用固定 entry 顺序、时间、权限和压缩参数写入 NPZ。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                entry = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
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


def clamp01(value: float) -> float:
    """把有限标量限制到 [0, 1]。"""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"view metric 非有限: {value}")
    return min(1.0, max(0.0, value))


def circular_distance_radians(first: float, second: float) -> float:
    """返回两个方位角在 [0, pi] 内的最短距离。"""
    delta = abs((float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi)
    return min(math.pi, delta)


def robust_unit_interval(values: Sequence[float]) -> np.ndarray:
    """用 5/95 分位确定性缩放，常量输入映射为 0.5。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("robust normalization 输入必须是一维有限非空数组")
    lower, upper = np.quantile(array, [0.05, 0.95], method="linear")
    if upper - lower <= 1e-12:
        return np.full(array.shape, 0.5, dtype=np.float64)
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0)


def rank_view_candidates(
    candidates: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    """补齐归一化分量并计算 Q_view；不改变输入顺序或对象。"""
    if not candidates:
        raise ValueError("view candidate 为空")
    if set(weights) != set(VIEW_COMPONENTS):
        raise ValueError(f"view score 权重字段漂移: {sorted(weights)}")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("view score 权重必须非负")

    area = robust_unit_interval([float(row["projected_area_fraction"]) for row in candidates])
    sharpness = robust_unit_interval([float(row["sharpness_laplacian_variance"]) for row in candidates])
    ranked: list[dict[str, Any]] = []
    for index, source in enumerate(candidates):
        components = {
            "area": float(area[index]),
            "mask": clamp01(float(source["mask_confidence"])),
            "sharpness": float(sharpness[index]),
            "visible": clamp01(float(source["visible_fraction"])),
            "occlusion": clamp01(float(source["occlusion_score"])),
            "truncation": clamp01(float(source["truncation_score"])),
        }
        score = (
            float(weights["area"]) * components["area"]
            + float(weights["mask"]) * components["mask"]
            + float(weights["sharpness"]) * components["sharpness"]
            + float(weights["visible"]) * components["visible"]
            - float(weights["occlusion"]) * components["occlusion"]
            - float(weights["truncation"]) * components["truncation"]
        )
        row = dict(source)
        row["view_score_components"] = components
        row["view_score"] = float(score)
        ranked.append(row)
    return ranked


def _identity(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["frame"]), int(row["camera_id"])


def _pairwise_mean(rows: Sequence[Mapping[str, Any]], fn: Any) -> float:
    if len(rows) < 2:
        return 0.0
    return float(np.mean([fn(first, second) for first, second in combinations(rows, 2)]))


def set_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    yaw_weight: float,
    temporal_weight: float,
    camera_weight: float,
    frame_span: int,
) -> dict[str, float]:
    """计算 Q_set，并返回可审计的三个 diversity 分量。"""
    if not rows:
        raise ValueError("view set 为空")
    if frame_span <= 0:
        raise ValueError("frame_span 必须为正")
    yaw = _pairwise_mean(
        rows,
        lambda a, b: circular_distance_radians(a["yaw_radians"], b["yaw_radians"])
        / math.pi,
    )
    temporal = _pairwise_mean(
        rows,
        lambda a, b: min(1.0, abs(int(a["frame"]) - int(b["frame"])) / frame_span),
    )
    camera = _pairwise_mean(
        rows,
        lambda a, b: float(int(a["camera_id"]) != int(b["camera_id"])),
    )
    quality_sum = float(sum(float(row["view_score"]) for row in rows))
    total = (
        quality_sum
        + float(yaw_weight) * yaw
        + float(temporal_weight) * temporal
        + float(camera_weight) * camera
    )
    return {
        "quality_sum": quality_sum,
        "yaw_diversity": yaw,
        "temporal_diversity": temporal,
        "camera_diversity": camera,
        "set_score": float(total),
    }


def _valid_extension(
    selected: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    minimum_same_camera_frame_gap: int,
    minimum_pairwise_yaw_degrees: float,
) -> bool:
    for current in selected:
        if _identity(current) == _identity(candidate):
            return False
        if int(current["camera_id"]) == int(candidate["camera_id"]) and abs(
            int(current["frame"]) - int(candidate["frame"])
        ) < int(minimum_same_camera_frame_gap):
            return False
        yaw_degrees = math.degrees(
            circular_distance_radians(current["yaw_radians"], candidate["yaw_radians"])
        )
        if yaw_degrees < float(minimum_pairwise_yaw_degrees):
            return False
    return True


def select_view_sets(
    candidates: Sequence[Mapping[str, Any]],
    *,
    view_counts: Sequence[int],
    yaw_weight: float,
    temporal_weight: float,
    camera_weight: float,
    frame_span: int,
    minimum_same_camera_frame_gap: int,
    minimum_pairwise_yaw_degrees: float,
    beam_width: int,
) -> dict[int, dict[str, Any]]:
    """用确定性 beam search 分别选择 1/2/4-view，不强制集合互为前缀。"""
    if not candidates:
        raise ValueError("没有可选择的 eligible view")
    counts = tuple(int(value) for value in view_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("view_counts 必须为正整数")
    if beam_width <= 0:
        raise ValueError("beam_width 必须为正")

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (-float(row["view_score"]), _identity(row)),
    )
    outputs: dict[int, dict[str, Any]] = {}
    for target_count in counts:
        beam: list[tuple[dict[str, Any], ...]] = [tuple()]
        for _ in range(target_count):
            expansions: dict[tuple[tuple[int, int], ...], tuple[dict[str, Any], ...]] = {}
            for current in beam:
                for candidate in ordered:
                    if not _valid_extension(
                        current,
                        candidate,
                        minimum_same_camera_frame_gap=minimum_same_camera_frame_gap,
                        minimum_pairwise_yaw_degrees=minimum_pairwise_yaw_degrees,
                    ):
                        continue
                    rows = tuple(sorted((*current, candidate), key=_identity))
                    key = tuple(_identity(row) for row in rows)
                    expansions[key] = rows
            if not expansions:
                raise RuntimeError(f"无法形成 {target_count}-view diversity set")
            scored = []
            for rows in expansions.values():
                metrics = set_score(
                    rows,
                    yaw_weight=yaw_weight,
                    temporal_weight=temporal_weight,
                    camera_weight=camera_weight,
                    frame_span=frame_span,
                )
                scored.append((metrics["set_score"], tuple(_identity(row) for row in rows), rows))
            scored.sort(key=lambda item: (-item[0], item[1]))
            beam = [item[2] for item in scored[: int(beam_width)]]

        winner = beam[0]
        metrics = set_score(
            winner,
            yaw_weight=yaw_weight,
            temporal_weight=temporal_weight,
            camera_weight=camera_weight,
            frame_span=frame_span,
        )
        outputs[target_count] = {
            "view_count": target_count,
            "selected_views": [dict(row) for row in winner],
            **metrics,
        }
    return outputs
