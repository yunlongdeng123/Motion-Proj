from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_worldsim_v5_m1_unary_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_m1_unary_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_frozen_mask_metrics_handle_exact_and_empty_cases() -> None:
    mask = np.asarray([[False, True], [False, True]])
    assert RUNNER._binary_iou(mask, mask) == pytest.approx(1.0)
    assert RUNNER._binary_iou(np.zeros_like(mask), np.zeros_like(mask)) == 1.0
    assert RUNNER._boundary_f1(mask, mask, tolerance=1) == pytest.approx(1.0)


def test_nll_prefers_calibrated_correct_probability() -> None:
    target = np.asarray([0.0, 1.0])
    correct = RUNNER._negative_log_likelihood(np.asarray([0.1, 0.9]), target)
    wrong = RUNNER._negative_log_likelihood(np.asarray([0.9, 0.1]), target)
    assert correct < wrong


def test_metric_aggregate_is_fieldwise_mean() -> None:
    result = RUNNER._aggregate_metrics(
        [{"iou": 0.2, "brier": 0.4}, {"iou": 0.6, "brier": 0.2}]
    )
    assert result == pytest.approx({"iou": 0.4, "brier": 0.3})


def test_collect_gaussians_sets_nearest_frozen_timeline_frame() -> None:
    class Model:
        in_test_set = False

        def set_cur_frame(self, value: int) -> None:
            self.frame = value

    class Trainer:
        normalized_timestamps = torch.asarray([0.0, 0.5, 1.0])
        in_test_set = True
        gaussian_classes = {"Background": 0, "RigidNodes": 1}
        models = {"Background": Model(), "RigidNodes": Model()}

        def process_camera(self, **kwargs):
            return kwargs

        def collect_gaussians(self, **kwargs):
            return kwargs

    trainer = Trainer()
    _, gaussians = RUNNER._collect_gaussians(
        trainer,
        {"normed_time": torch.asarray([0.49]), "img_idx": torch.asarray([7])},
        {"camera_to_world": torch.eye(4)},
    )
    assert int(trainer.cur_frame.item()) == 1
    assert int(trainer.models["RigidNodes"].frame.item()) == 1
    assert trainer.models["RigidNodes"].in_test_set is True
    assert int(gaussians["image_ids"]) == 7
