import numpy as np
import pytest
import torch

from motion_proj.dynamic_editing_v2.stress_metrics import (
    box_iou,
    interpolate_trajectory,
    masked_ssim,
    primary_failure,
    time_warp_ordinals,
)


def test_speed_time_warp_is_three_quarters_rate() -> None:
    assert time_warp_ordinals(5, "speed").tolist() == [0.0, 0.75, 1.5, 2.25, 3.0]


def test_stop_restart_freezes_then_reaches_terminal_pose() -> None:
    mapping = time_warp_ordinals(30, "stop_restart", fps=10)
    assert np.all(mapping[15:25] == 15)
    assert mapping[-1] == pytest.approx(29.0)
    assert np.all(np.diff(mapping) >= 0)


def test_quaternion_interpolation_uses_shortest_sign() -> None:
    translations = torch.tensor([[0.0, 0, 0], [2.0, 0, 0]])
    quaternions = torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    output_t, output_q = interpolate_trajectory(
        translations, quaternions, np.array([0.5])
    )
    assert output_t[0, 0].item() == pytest.approx(1.0)
    assert output_q[0, 0].abs().item() == pytest.approx(1.0)


def test_masked_ssim_identity_and_primary_priority() -> None:
    image = np.full((4, 4, 3), 80, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    assert masked_ssim(image, image, mask) == pytest.approx(1.0)
    assert primary_failure(["TEMPORAL_FLICKER", "SOURCE_RESIDUAL"]) == "SOURCE_RESIDUAL"


def test_box_iou() -> None:
    assert box_iou(np.array([0, 0, 2, 2]), np.array([1, 1, 3, 3])) == pytest.approx(1 / 7)
