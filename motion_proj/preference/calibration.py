"""Measurement ROPE、scene-level tie split 与 paired bootstrap 校准。"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


class CalibrationError(RuntimeError):
    """校准拆分、measurement audit 或阈值计算不合法。"""


PRIMARY_COMPONENTS = ("punc", "acceleration", "curvature", "coherence")


def scene_hash_split(
    cases: Sequence[Mapping[str, Any]],
    *,
    calibration_count: int,
    salt: str,
) -> dict[str, Any]:
    """按 scene 哈希做一次性 12/10 拆分；同 scene 永不跨集合。"""
    if calibration_count <= 0 or calibration_count >= len(cases):
        raise ValueError("calibration_count 必须把 cases 拆成两个非空集合")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for raw in cases:
        row = dict(raw)
        scene = str(row.get("scene_id", ""))
        case_id = str(row.get("case_id", ""))
        if not scene or not case_id:
            raise CalibrationError("tie case 缺少 scene_id/case_id")
        by_scene.setdefault(scene, []).append(row)
    if any(len(rows) != 1 for rows in by_scene.values()):
        raise CalibrationError("当前冻结协议要求每个 reviewed tie 来自不同 scene")
    ranked = sorted(
        (hashlib.sha256(f"{salt}\0{scene}".encode("utf-8")).hexdigest(), scene, rows[0])
        for scene, rows in by_scene.items()
    )
    calibration = [row for _, _, row in ranked[:calibration_count]]
    holdout = [row for _, _, row in ranked[calibration_count:]]
    calibration_scenes = {str(row["scene_id"]) for row in calibration}
    holdout_scenes = {str(row["scene_id"]) for row in holdout}
    if calibration_scenes & holdout_scenes:
        raise CalibrationError("calibration/holdout scene 泄漏")
    payload = {
        "protocol": "scene-hash-tie-split-v1",
        "salt": salt,
        "calibration_count": len(calibration),
        "holdout_count": len(holdout),
        "calibration_case_ids": [str(row["case_id"]) for row in calibration],
        "holdout_case_ids": [str(row["case_id"]) for row in holdout],
        "calibration_scene_ids": sorted(calibration_scenes),
        "holdout_scene_ids": sorted(holdout_scenes),
        "scene_disjoint": True,
    }
    payload["split_hash"] = hashlib.sha256(
        repr((payload["salt"], payload["calibration_case_ids"], payload["holdout_case_ids"])).encode("utf-8")
    ).hexdigest()
    return payload


def measurement_ropes(
    absolute_differences: Mapping[str, Sequence[float]],
    *,
    quantile: float,
    minimums: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """只从 repeatability/benign nuisance 差异估计 component equivalence band。"""
    if not 0.5 <= quantile < 1.0:
        raise ValueError("measurement ROPE quantile 必须位于 [0.5,1)")
    result = {}
    for component in PRIMARY_COMPONENTS:
        values = torch.tensor([
            abs(float(value)) for value in absolute_differences.get(component, [])
            if math.isfinite(float(value))
        ], dtype=torch.float64)
        if not int(values.numel()):
            raise CalibrationError(f"{component} 缺少 measurement audit 数据")
        floor = float(minimums[component])
        rope = max(float(torch.quantile(values, quantile)), floor)
        result[component] = {
            "rope": rope,
            "sample_count": int(values.numel()),
            "quantile": quantile,
            "minimum": floor,
            "median_absolute_difference": float(values.median()),
            "maximum_absolute_difference": float(values.max()),
        }
    return result


def split_conformal_threshold(
    nonconformity_scores: Sequence[float],
    *,
    alpha: float,
    minimum: float = 1.0,
) -> dict[str, Any]:
    """有限样本 split-conformal 上分位数；12 个 ties、alpha=.1 时取最大值。"""
    values = sorted(float(value) for value in nonconformity_scores if math.isfinite(float(value)))
    if not values:
        raise CalibrationError("没有可比较 calibration tie score")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须位于 (0,1)")
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * (1.0 - alpha))))
    raw = values[rank - 1]
    return {
        "threshold": max(float(minimum), raw),
        "raw_quantile": raw,
        "rank": rank,
        "sample_count": len(values),
        "alpha": alpha,
        "finite_sample_coverage_lower": rank / (len(values) + 1),
    }


@dataclass(frozen=True)
class BootstrapInterval:
    component: str
    point: float | None
    lower: float | None
    upper: float | None
    p_value_two_sided: float | None
    finite_observations: int
    spatial_clusters: int
    samples: int
    seed: int

    def to_record(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "p_value_two_sided": self.p_value_two_sided,
            "finite_observations": self.finite_observations,
            "spatial_clusters": self.spatial_clusters,
            "bootstrap_samples": self.samples,
            "bootstrap_seed": self.seed,
        }


def paired_cluster_block_bootstrap(
    differences: torch.Tensor,
    cluster_ids: torch.Tensor,
    *,
    component: str,
    samples: int,
    seed: int,
    confidence: float = 0.95,
    temporal_block: int = 2,
) -> BootstrapInterval:
    """以空间 cluster 为一级、连续 temporal block 为二级的 paired bootstrap。"""
    if differences.ndim != 2:
        raise ValueError("differences 必须是 [Q,T]")
    if cluster_ids.shape != differences.shape[:1]:
        raise ValueError("cluster_ids 必须是 [Q]")
    if samples <= 0 or not 0 < confidence < 1 or temporal_block <= 0:
        raise ValueError("bootstrap 参数无效")
    finite = torch.isfinite(differences)
    finite_count = int(finite.sum())
    active_rows = torch.nonzero(finite.any(dim=1), as_tuple=False).flatten()
    clusters = torch.unique(cluster_ids[active_rows]) if int(active_rows.numel()) else torch.empty(0, dtype=torch.long)
    if not finite_count or not int(clusters.numel()):
        return BootstrapInterval(component, None, None, None, None, finite_count, 0, samples, seed)
    point = float(differences[finite].median())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    time_count = differences.shape[1]
    block_starts = list(range(max(time_count - temporal_block + 1, 1)))
    draws = []
    for _ in range(samples):
        sampled_cluster_indices = torch.randint(
            0, len(clusters), (len(clusters),), generator=generator
        )
        sampled_rows = []
        for cluster_index in sampled_cluster_indices.tolist():
            rows = torch.nonzero(cluster_ids == clusters[cluster_index], as_tuple=False).flatten()
            sampled_rows.extend(rows.tolist())
        block_count = max(1, math.ceil(time_count / temporal_block))
        sampled_times = []
        for _ in range(block_count):
            start = block_starts[int(torch.randint(0, len(block_starts), (1,), generator=generator))]
            sampled_times.extend(range(start, min(start + temporal_block, time_count)))
        sample = differences[sampled_rows][:, sampled_times]
        finite_sample = sample[torch.isfinite(sample)]
        if int(finite_sample.numel()):
            draws.append(finite_sample.median())
    if not draws:
        return BootstrapInterval(component, point, None, None, None, finite_count, int(clusters.numel()), samples, seed)
    distribution = torch.stack(draws).double()
    tail = (1.0 - confidence) * 0.5
    lower = float(torch.quantile(distribution, tail))
    upper = float(torch.quantile(distribution, 1.0 - tail))
    nonpositive = (int((distribution <= 0).sum()) + 1) / (len(distribution) + 1)
    nonnegative = (int((distribution >= 0).sum()) + 1) / (len(distribution) + 1)
    p_value = min(1.0, 2.0 * min(nonpositive, nonnegative))
    return BootstrapInterval(
        component, point, lower, upper, p_value, finite_count,
        int(clusters.numel()), len(distribution), seed,
    )


def holm_adjust(p_values: Mapping[str, float | None], *, alpha: float) -> dict[str, dict[str, Any]]:
    """Holm step-down correction；缺失 p-value 永不显著。"""
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须位于 (0,1)")
    finite = sorted(
        (float(value), str(key)) for key, value in p_values.items()
        if value is not None and math.isfinite(float(value))
    )
    total = len(finite)
    rejected: dict[str, bool] = {str(key): False for key in p_values}
    adjusted: dict[str, float | None] = {str(key): None for key in p_values}
    running = 0.0
    still_rejecting = True
    for rank, (value, key) in enumerate(finite, start=1):
        running = max(running, (total - rank + 1) * value)
        adjusted[key] = min(1.0, running)
        threshold = alpha / (total - rank + 1)
        if still_rejecting and value <= threshold:
            rejected[key] = True
        else:
            still_rejecting = False
    return {
        key: {"p_value": p_values[key], "adjusted_p_value": adjusted[key], "significant": rejected[key]}
        for key in p_values
    }
