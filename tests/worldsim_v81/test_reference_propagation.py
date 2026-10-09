"""paper-bidirectional-m4 的纯 CPU 几何与遮罩回归测试。"""

import pytest
import torch
from torch.nn import functional as F

from motion_proj.worldsim_v81.reference_propagation import (
    PROTOCOL,
    build_reference_plan,
    propagate_reference_latents,
    static_fcnet_pair_masks,
)


class CaptureRefinement:
    """仅截取官方 _post_fuse 的输入；warp 公式与固定官方代码相同。"""

    def __init__(self, *, use_fb_check=True):
        self.use_fb_check = use_fb_check
        self.calls = 0

    def backward_warp_cached(self, image, flow):
        batch, _, height, width = image.shape
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).to(image).unsqueeze(0).expand(batch, -1, -1, -1)
        grid = grid + flow.permute(0, 2, 3, 1)
        grid[..., 0] = 2 * grid[..., 0] / (width - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (height - 1) - 1
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _post_fuse(self, original, masks, future, past, future_cov, past_cov,
                   future_flow, past_flow):
        self.calls += 1
        self.last = (original, masks, future_cov, past_cov, future_flow, past_flow)
        return future, past


def translation_flows(plan, *, spatial=22):
    """图像分辨率每帧平移2像素，11² latent 每帧平移1格。"""
    forward = torch.zeros(1, len(plan.pairs), 2, spatial, spatial)
    backward = torch.zeros_like(forward)
    for k, (earlier, later) in enumerate(plan.pairs):
        forward[:, k, 0] = 2 * (later - earlier)
        backward[:, k, 0] = -2 * (later - earlier)
    return forward, backward


def video_with_only_source(source, *, value=1.0):
    latent = torch.zeros(1, 5, 4, 11, 11)
    latent[0, source, 0, 5, 3 + source] = value
    mask = torch.ones(1, 5, 1, 22, 22)
    mask[:, source] = 0
    return latent, mask


def test_pair_graph_has_direct_nearest_edges_and_fixed_order():
    plan = build_reference_plan(5, [0, 2, 4])
    assert plan.protocol == PROTOCOL
    assert plan.pairs == ((0, 2), (2, 4), (0, 1), (1, 2), (2, 3), (3, 4))
    assert len(build_reference_plan(25, [0, 4, 8, 12, 16, 20, 24]).pairs) == 42
    with pytest.raises(ValueError):
        build_reference_plan(5, [0, 4, 2])
    with pytest.raises(ValueError, match="间距"):
        build_reference_plan(7, [0, 6])


def test_nonadjacent_past_and_future_pull_and_flow_scale():
    plan = build_reference_plan(5, [0, 2, 4])
    flows = translation_flows(plan)
    module = CaptureRefinement()

    from_past, mask = video_with_only_source(0)
    future, past = propagate_reference_latents(module, from_past, *flows, mask, plan)
    assert float(past[0, 4, 0, 5, 7]) == pytest.approx(1)
    assert float(past[0, 3, 0, 5, 6]) == pytest.approx(1)
    assert float(future[0, 4, 0].abs().max()) == 0

    from_future, mask = video_with_only_source(4)
    future, past = propagate_reference_latents(module, from_future, *flows, mask, plan)
    assert float(future[0, 0, 0, 5, 3]) == pytest.approx(1)
    assert float(future[0, 1, 0, 5, 4]) == pytest.approx(1)
    assert float(past[0, 0, 0].abs().max()) == 0
    assert module.calls == 2
    assert module.last[4].shape == (1, 5, 2, 11, 11)


def test_visible_target_preserved_and_unknown_source_blocked():
    plan = build_reference_plan(5, [0, 2, 4])
    flows = translation_flows(plan)
    module = CaptureRefinement()
    latent, mask = video_with_only_source(4)
    latent[0, 1, 0, 5, 4] = 0.7
    mask[:, 1, :, 10:12, 8:10] = 0  # latent[5,4] 在 2 倍原图中的可见格
    future, _ = propagate_reference_latents(module, latent, *flows, mask, plan)
    assert float(future[0, 1, 0, 5, 4]) == pytest.approx(0.7)

    unknown = torch.zeros_like(latent)
    unknown[0, 4, 0, 5, 7] = 10  # 源帧洞区 VAE latent 明显非零
    all_holes = torch.ones_like(mask)
    future, _ = propagate_reference_latents(module, unknown, *flows, all_holes, plan)
    # 本帧条件 latent 原样保留；不允许它传播到其他目标帧。
    assert float(future[0, :4].abs().max()) == 0


def test_fractional_source_coverage_is_applied_once():
    plan = build_reference_plan(2, [0, 1])
    latent = torch.full((1, 2, 4, 11, 11), 100.0)
    latent[:, 0] = 0
    latent[:, 1, :, :, 4] = 2
    mask = torch.ones(1, 2, 1, 11, 11)
    mask[:, 1, :, :, 4] = 0
    forward = torch.zeros(1, 1, 2, 11, 11)
    forward[:, :, 0] = .5
    future, _ = propagate_reference_latents(
        CaptureRefinement(), latent, forward, -forward, mask, plan)
    # 从来源x=3.5采样，只有x=4可见：2×0.5；洞区100不能混入，也不能再乘0.5。
    assert float(future[0, 0, 0, 5, 3]) == pytest.approx(1.0)


def test_flow_pair_axis_order_is_observable_and_fb_check_is_retained():
    plan = build_reference_plan(5, [0, 2, 4])
    forward, backward = translation_flows(plan)
    latent, mask = video_with_only_source(0)
    module = CaptureRefinement()
    _, past = propagate_reference_latents(module, latent, forward, backward, mask, plan)
    assert float(past[0, 4, 0, 5, 7]) == 1

    wrong_forward, wrong_backward = forward.clone(), backward.clone()
    wrong_forward[:, [0, 2]] = wrong_forward[:, [2, 0]]
    wrong_backward[:, [0, 2]] = wrong_backward[:, [2, 0]]
    _, wrong = propagate_reference_latents(module, latent, wrong_forward,
                                           wrong_backward, mask, plan)
    assert float(wrong[0, 4, 0, 5, 7]) == 0

    broken_forward = torch.zeros_like(forward)
    _, checked = propagate_reference_latents(module, latent, broken_forward,
                                             backward, mask, plan)
    _, unchecked = propagate_reference_latents(CaptureRefinement(use_fb_check=False),
                                               latent, broken_forward, backward, mask, plan)
    assert float(checked[0, 4, 0, 5, 7]) == 0
    assert float(unchecked[0, 4, 0, 5, 7]) == 1


def test_static_fcnet_masks_require_k_plus_one_and_reject_dynamic():
    plan = build_reference_plan(25, [0, 4, 8, 12, 16, 20, 24])
    masks = torch.ones(1, 25, 1, 22, 22)
    assert static_fcnet_pair_masks(masks, plan).shape == (1, 43, 1, 22, 22)
    masks[:, 1] = 0
    with pytest.raises(ValueError, match="动态 mask"):
        static_fcnet_pair_masks(masks, plan)
