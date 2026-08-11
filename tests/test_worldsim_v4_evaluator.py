from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from motion_proj.worldsim_v4.evaluator import EvaluationError, evaluate_frame


class MeanDistance(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(left - right))


def test_regions_use_frozen_contract_and_empty_region_is_undefined() -> None:
    target = np.zeros((16, 16, 3), dtype=np.float32)
    prediction = target.copy()
    prediction[4:8, 4:8] = 0.25
    actor = np.zeros((16, 16), dtype=bool)
    actor[4:8, 4:8] = True
    empty = np.zeros_like(actor)
    result = evaluate_frame(
        prediction,
        target,
        {"static": ~actor, "actor": actor, "boundary": actor, "edit_roi": empty},
        lpips_model=MeanDistance(),
    )
    assert result["regions"]["actor"]["status"] == "done"
    assert result["regions"]["actor"]["psnr"] == pytest.approx(12.0411998)
    assert math.isinf(result["regions"]["static"]["psnr"])
    assert result["regions"]["edit_roi"]["status"] == "undefined_empty_region"
    assert result["regions"]["global"]["lpips_alex"] is not None


def test_missing_ground_truth_never_uses_generated_image_as_gt() -> None:
    result = evaluate_frame(np.zeros((8, 8, 3)), None, {})
    assert all(row["status"] == "undefined_no_ground_truth" for row in result["regions"].values())
    assert all(row["psnr"] is None for row in result["regions"].values())


def test_input_range_fails_closed() -> None:
    with pytest.raises(EvaluationError, match=r"\[0,1\]"):
        evaluate_frame(np.full((8, 8, 3), 255.0), np.zeros((8, 8, 3)), {})
