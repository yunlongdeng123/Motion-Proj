import pytest
import torch

from motion_proj.losses import flow_warp_charbonnier_loss


def test_flow_loss_is_epsilon_for_static_consistent_video():
    video = torch.ones(2, 4, 3, 5, 7)
    flow = torch.zeros(2, 3, 5, 7, 2)
    confidence = torch.ones(2, 3, 1, 5, 7)
    loss = flow_warp_charbonnier_loss(video, flow, confidence, epsilon=1e-3)
    assert float(loss) == pytest.approx(1e-3, abs=1e-6)


def test_flow_confidence_excludes_inconsistent_pixels():
    video = torch.zeros(1, 2, 1, 3, 3)
    video[:, 1] = 10
    flow = torch.zeros(1, 1, 3, 3, 2)
    confidence = torch.zeros(1, 1, 1, 3, 3)
    assert float(flow_warp_charbonnier_loss(video, flow, confidence)) == 0.0


def test_flow_loss_uses_stop_gradient_warp_target():
    video = torch.zeros(1, 2, 1, 3, 3, requires_grad=True)
    with torch.no_grad():
        video[:, 1] = 1
    flow = torch.zeros(1, 1, 3, 3, 2)
    confidence = torch.ones(1, 1, 1, 3, 3)

    flow_warp_charbonnier_loss(video, flow, confidence).backward()

    assert torch.count_nonzero(video.grad[:, 0]) == video[:, 0].numel()
    assert torch.count_nonzero(video.grad[:, 1]) == 0
