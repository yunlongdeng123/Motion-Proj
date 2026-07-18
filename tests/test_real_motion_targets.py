import math

import pytest
import torch

from motion_proj.data.real_motion_targets import (
    binary_roc_auc,
    boxes_background_mask,
    flow_direction_agreement,
    sparse_ego_flow_target,
    spearman_correlation,
)


K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
I = torch.eye(4, dtype=torch.float64)


def test_sparse_ego_flow_translation_and_yaw_have_expected_sign():
    depth = torch.zeros((81, 101), dtype=torch.float64)
    depth[40, 50] = 10.0
    translated = I.clone()
    translated[0, 3] = 1.0
    flow, valid = sparse_ego_flow_target(depth, K, K, I, I, I, translated)
    assert valid[40, 50]
    assert flow[40, 50].tolist() == pytest.approx([-10.0, 0.0], abs=1e-5)

    angle = math.radians(5.0)
    yawed = I.clone()
    yawed[:3, :3] = torch.tensor(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]],
        dtype=torch.float64,
    )
    yaw_flow, yaw_valid = sparse_ego_flow_target(depth, K, K, I, I, I, yawed)
    assert yaw_valid[40, 50]
    assert yaw_flow[40, 50, 0] < 0


def test_background_mask_and_flow_direction_gate():
    mask = boxes_background_mask(10, 12, [{"xyxy": [3, 2, 6, 5]}], dilation_px=0)
    assert mask[0, 0]
    assert not mask[3, 4]
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    observed = torch.tensor([[[2.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]])
    result = flow_direction_agreement(target, observed, torch.ones((1, 3), dtype=torch.bool))
    assert result["count"] == 3
    assert result["agreement_fraction"] == pytest.approx(2 / 3)


def test_auc_and_spearman_do_not_zero_fill_missing_values():
    assert binary_roc_auc([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert binary_roc_auc([], [1.0]) is None
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

