"""把冻结 development render 映射成 WorldSim V4 scene-level 指标。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .evaluator import REGION_NAMES, evaluate_frame
from .region_masks import build_baseline_region_masks


class BaselineSceneEvaluationError(ValueError):
    """baseline render manifest 不满足 development-only 合同。"""


def _rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = image.convert("L")
        if value.size != (shape[1], shape[0]):
            value = value.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        return np.asarray(value) > 0


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise BaselineSceneEvaluationError("development records 为空")
    seen: set[tuple[int, int]] = set()
    for record in records:
        if record.get("partition") != "development":
            raise BaselineSceneEvaluationError("只允许 development record")
        key = (int(record["frame"]), int(record["camera_id"]))
        if key in seen:
            raise BaselineSceneEvaluationError(f"重复 frame/camera：{key}")
        seen.add(key)
        for name in ("prediction", "target", "dynamic_mask", "egocar_mask"):
            if not Path(record[name]).is_file():
                raise BaselineSceneEvaluationError(f"record 文件缺失：{name}")


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    status_counts: dict[str, dict[str, int]] = {
        region: defaultdict(int) for region in REGION_NAMES
    }
    for row in rows:
        for region, result in row["evaluation"]["regions"].items():
            status_counts[region][result["status"]] += 1
            for metric in ("psnr", "ssim", "lpips_alex"):
                value = result[metric]
                if value is not None:
                    values[(region, metric)].append(float(value))
    regions: dict[str, Any] = {}
    for region in REGION_NAMES:
        metrics = {}
        for metric in ("psnr", "ssim", "lpips_alex"):
            selected = values[(region, metric)]
            metrics[metric] = {
                "mean": None if not selected else float(np.mean(selected)),
                "valid_frames": len(selected),
            }
        regions[region] = {
            "metrics": metrics,
            "status_counts": dict(sorted(status_counts[region].items())),
        }
    return {"frame_rows": len(rows), "regions": regions}


def evaluate_scene_records(
    records: Sequence[Mapping[str, Any]],
    *,
    lpips_model: Callable[[Any, Any], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_records(records)
    rows: list[dict[str, Any]] = []
    for record in records:
        prediction = _rgb(record["prediction"])
        target = _rgb(record["target"])
        if prediction.shape != target.shape:
            raise BaselineSceneEvaluationError("prediction/target shape 不一致")
        shape = prediction.shape[:2]
        regions = build_baseline_region_masks(
            _mask(record["dynamic_mask"], shape),
            _mask(record["egocar_mask"], shape),
        )
        rows.append(
            {
                "frame": int(record["frame"]),
                "camera_id": int(record["camera_id"]),
                "partition": "development",
                "evaluation": evaluate_frame(
                    prediction,
                    target,
                    regions,
                    lpips_model=lpips_model,
                ),
            }
        )
    return rows, summarize_rows(rows)
