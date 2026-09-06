"""AdaPoinTr 适配数据的 target-free 输入读取边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_adapointr_input(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """部署侧只返回 partial 与非 target 元数据。"""
    with np.load(path, allow_pickle=False) as payload:
        partial = np.asarray(payload["partial_normalized"], dtype=np.float32)
        metadata = {
            "scene_name": str(payload["scene_name"]),
            "log_id": str(payload["log_id"]),
            "track_id": str(payload["track_id"]),
            "scale_m": float(payload["scale_m"]),
            "native_input_point_count": int(payload["native_input_point_count"]),
        }
    return partial, metadata


def load_adapointr_training_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """训练侧显式读取 partial/target；不得在推理入口调用。"""
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["partial_normalized"], dtype=np.float32),
            np.asarray(payload["target_normalized"], dtype=np.float32),
        )


def denormalize_adapointr_output(points: np.ndarray, scale_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("AdaPoinTr 反归一化尺度必须为有限正数")
    return points * float(scale_m)
