from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from motion_proj.worldsim_v4.baseline_scene_evaluator import (
    BaselineSceneEvaluationError,
    evaluate_scene_records,
)


class MeanDistance(torch.nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first - second).abs().mean().reshape(1, 1, 1, 1)


def save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    Image.fromarray((value.astype(np.uint8) * 255), mode="L").save(path)


def record(tmp_path: Path, *, partition: str = "development") -> dict:
    target = np.zeros((72, 96, 3), dtype=np.uint8)
    prediction = target.copy()
    prediction[20:40, 30:50] = 32
    dynamic = np.zeros(target.shape[:2], dtype=bool)
    dynamic[18:42, 28:52] = True
    egocar = np.zeros_like(dynamic)
    egocar[65:, :] = True
    paths = {
        "prediction": tmp_path / "prediction.png",
        "target": tmp_path / "target.png",
        "dynamic_mask": tmp_path / "dynamic.png",
        "egocar_mask": tmp_path / "egocar.png",
    }
    save_rgb(paths["prediction"], prediction)
    save_rgb(paths["target"], target)
    save_mask(paths["dynamic_mask"], dynamic)
    save_mask(paths["egocar_mask"], egocar)
    return {
        "frame": 2,
        "camera_id": 0,
        "partition": partition,
        **{name: str(path) for name, path in paths.items()},
    }


def test_evaluate_scene_records_produces_all_regions(tmp_path: Path) -> None:
    rows, summary = evaluate_scene_records([record(tmp_path)], lpips_model=MeanDistance())
    assert rows[0]["evaluation"]["ground_truth"] is True
    assert summary["frame_rows"] == 1
    assert summary["regions"]["actor"]["metrics"]["psnr"]["valid_frames"] == 1
    assert summary["regions"]["edit_roi"]["metrics"]["psnr"]["mean"] is None


def test_evaluator_rejects_heldout_or_duplicate_records(tmp_path: Path) -> None:
    with pytest.raises(BaselineSceneEvaluationError, match="development"):
        evaluate_scene_records([record(tmp_path, partition="heldout")], lpips_model=MeanDistance())
    row = record(tmp_path)
    with pytest.raises(BaselineSceneEvaluationError, match="重复"):
        evaluate_scene_records([row, row], lpips_model=MeanDistance())
