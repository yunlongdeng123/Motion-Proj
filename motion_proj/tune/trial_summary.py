"""Optuna trial 所需的 summary 字段组装与归一化改进量。"""
from __future__ import annotations

from typing import Mapping


REQUIRED_METRIC_KEYS = (
    "static_drift",
    "track_acceleration",
    "lpips",
    "projection_eligible_fraction",
    "normalized_static_drift_improvement",
    "normalized_track_acceleration_improvement",
)


def normalized_improvement(base: float, value: float, *, eps: float = 1e-8) -> float:
    """越低越好指标相对 base 的归一化改进；(base-value)/max(|base|,eps)。"""
    return (float(base) - float(value)) / max(abs(float(base)), eps)


def build_trial_metrics(adapter: Mapping[str, float], base: Mapping[str, float]) -> dict:
    """由 adapter/base 原始指标生成 Optuna 淘汰与排序所需字段。"""
    static_drift = float(adapter["static_drift"])
    track_acceleration = float(adapter["track_acceleration"])
    lpips = float(adapter["lpips"])
    eligible = float(adapter["projection_eligible_fraction"])
    return {
        "static_drift": static_drift,
        "track_acceleration": track_acceleration,
        "lpips": lpips,
        "projection_eligible_fraction": eligible,
        "normalized_static_drift_improvement": normalized_improvement(
            float(base["static_drift"]), static_drift
        ),
        "normalized_track_acceleration_improvement": normalized_improvement(
            float(base["track_acceleration"]), track_acceleration
        ),
        "base_static_drift": float(base["static_drift"]),
        "base_track_acceleration": float(base["track_acceleration"]),
        "base_lpips": float(base["lpips"]),
    }


def merge_trial_summary(train_summary: Mapping, metrics: Mapping) -> dict:
    """把训练完成摘要与 trial 评估指标合并为 Optuna 可读的 summary.json。"""
    payload = dict(train_summary)
    payload.update(dict(metrics))
    payload["status"] = "completed"
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in payload]
    if missing:
        raise ValueError(f"trial summary 缺少字段: {missing}")
    return payload
