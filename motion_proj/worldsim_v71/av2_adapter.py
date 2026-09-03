"""V7.1 的 AV2 坐标、类别与 sensor-opportunity 适配边界。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from motion_proj.worldsim_v71.dataset_av2 import RIGID_AV2_CATEGORIES


def map_av2_category(category: str) -> str | None:
    return "rigid_vehicle" if str(category) in RIGID_AV2_CATEGORIES else None


def interpolate_translation(
    timestamp_ns: int,
    timestamps_ns: np.ndarray,
    translations_m: np.ndarray,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    translations = np.asarray(translations_m, dtype=np.float64).reshape(-1, 3)
    if len(timestamps) != len(translations) or len(timestamps) == 0:
        raise ValueError("timestamp/translation 输入无效")
    if len(timestamps) == 1:
        return translations[0].copy()
    right = int(np.searchsorted(timestamps, int(timestamp_ns), side="right"))
    right = min(max(right, 1), len(timestamps) - 1)
    left = right - 1
    span = max(int(timestamps[right] - timestamps[left]), 1)
    weight = np.clip((int(timestamp_ns) - int(timestamps[left])) / span, 0.0, 1.0)
    return (1.0 - weight) * translations[left] + weight * translations[right]


def normalize_sensor_opportunities(counts: np.ndarray, opportunities: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    opportunities = np.asarray(opportunities, dtype=np.float32)
    return counts / np.maximum(opportunities, 1.0)


def allowed_zero_shot_operations() -> Mapping[str, bool]:
    return {
        "coordinate_conversion": True,
        "unit_conversion": True,
        "category_mapping": True,
        "timestamp_interpolation": True,
        "actor_canonicalization": True,
        "sensor_opportunity_normalization": True,
        "fine_tuning": False,
        "threshold_selection": False,
        "quality_based_log_replacement": False,
    }
