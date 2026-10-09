"""v8.1 首批 CPU 张量契约：洞区方向、流几何、可见参考融合。"""

import pytest
import torch

from motion_proj.worldsim_v81 import (
    apply_hole_mask,
    backward_warp,
    fuse_reference_latents,
    make_border_outpaint_mask,
    resize_flow,
)


def test_border_outpaint_left_and_right_use_one_third_and_hole_is_one():
    mask = make_border_outpaint_mask(6, 10)
    assert mask.shape == (1, 1, 6, 10)
    assert torch.all(mask[..., :, :3] == 1)
    assert torch.all(mask[..., :, -3:] == 1)
    assert torch.all(mask[..., :, 3:7] == 0)

    four_sides = make_border_outpaint_mask(6, 10, 0.33, 0.33, top_fraction=0.33, bottom_fraction=0.33)
    assert torch.all(four_sides[..., :2, :] == 1)
    assert torch.all(four_sides[..., -2:, :] == 1)
    assert torch.all(four_sides[..., 2:4, 3:7] == 0)


def test_masked_conditioning_removes_hidden_values_and_gradients():
    source = torch.tensor([[[[2.0, 999.0, 4.0]]]], requires_grad=True)
    mask = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    conditioned = apply_hole_mask(source, mask)
    torch.testing.assert_close(conditioned, torch.tensor([[[[2.0, 0.0, 4.0]]]]))
    conditioned.sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[[[1.0, 0.0, 1.0]]]]))
    assert apply_hole_mask(torch.tensor([[[[float("nan")]]]]), torch.ones(1, 1, 1, 1)).item() == 0


def test_identity_flow_preserves_source_and_all_pixels_are_valid():
    source = torch.arange(15.0).reshape(1, 1, 3, 5)
    warped, valid = backward_warp(source, torch.zeros(1, 2, 3, 5))
    torch.testing.assert_close(warped, source)
    assert valid.dtype == torch.bool
    assert valid.all()


def test_flow_is_target_to_source_with_strict_out_of_bounds_mask():
    source = torch.arange(5.0).reshape(1, 1, 1, 5)
    flow = torch.zeros(1, 2, 1, 5)
    flow[:, 0] = 1
    warped, valid = backward_warp(source, flow)
    torch.testing.assert_close(warped, torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 0.0]]]]))
    torch.testing.assert_close(valid, torch.tensor([[[[True, True, True, True, False]]]]))

    flow[:, 0] = -1
    warped, valid = backward_warp(source, flow)
    torch.testing.assert_close(warped, torch.tensor([[[[0.0, 0.0, 1.0, 2.0, 3.0]]]]))
    assert not valid[..., 0].any()
    assert valid[..., 1:].all()


def test_non_square_flow_resize_scales_x_and_y_independently():
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 0] = 1.5
    flow[:, 1] = -2
    resized = resize_flow(flow, (6, 8))
    assert resized.shape == (1, 2, 6, 8)
    torch.testing.assert_close(resized[:, 0], torch.full((1, 6, 8), 3.0))
    torch.testing.assert_close(resized[:, 1], torch.full((1, 6, 8), -6.0))


def test_backward_warp_has_finite_nonzero_gradient_through_flow():
    y, x = torch.meshgrid(torch.arange(4.0), torch.arange(5.0), indexing="ij")
    source = (2 * x + 3 * y).reshape(1, 1, 4, 5)
    flow = torch.full((1, 2, 4, 5), 0.25, requires_grad=True)
    warped, _ = backward_warp(source, flow)
    warped[..., 1:3, 1:4].sum().backward()
    assert torch.isfinite(flow.grad).all()
    assert flow.grad[:, 0, 1:3, 1:4].abs().sum() > 0
    assert flow.grad[:, 1, 1:3, 1:4].abs().sum() > 0


def test_reference_fusion_uses_reliability_and_preserves_visible_target():
    target = torch.tensor([[[[101.0, 5000.0, 103.0]]]])
    target_hole = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    refs = torch.tensor([[[[[2.0, 2.0, 2.0]]], [[[10.0, 10.0, 10.0]]]]])
    flows = torch.zeros(1, 2, 2, 1, 3)
    ref_holes = torch.zeros(1, 2, 1, 1, 3)
    reliability = torch.full((1, 2, 1, 1, 3), 0.25)
    reliability[:, 1] = 0.75
    output, confidence = fuse_reference_latents(target, target_hole, refs, flows, ref_holes, reliability)
    torch.testing.assert_close(output, torch.tensor([[[[101.0, 8.0, 103.0]]]]))
    torch.testing.assert_close(confidence, torch.tensor([[[[1.0, 0.5, 1.0]]]]))


def test_masked_reference_and_hidden_target_cannot_contaminate_fusion():
    target = torch.tensor([[[[7777.0, 5.0, 6.0]]]], requires_grad=True)
    target_hole = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    refs = torch.tensor([[[[[2.0, 9999.0, 6.0]]]]], requires_grad=True)
    ref_holes = torch.tensor([[[[[0.0, 1.0, 0.0]]]]])
    flow = torch.zeros(1, 1, 2, 1, 3)
    flow[:, :, 0, :, 0] = 0.5
    output, confidence = fuse_reference_latents(target, target_hole, refs, flow, ref_holes)
    torch.testing.assert_close(output, torch.tensor([[[[2.0, 5.0, 6.0]]]]))
    torch.testing.assert_close(confidence[..., 0], torch.tensor([[[0.5]]]))
    output.sum().backward()
    assert target.grad[0, 0, 0, 0] == 0
    assert refs.grad[0, 0, 0, 0, 1] == 0


def test_fusion_without_valid_reference_has_zero_hole_and_confidence():
    target = torch.tensor([[[[999.0]]]])
    target_hole = torch.ones(1, 1, 1, 1)
    refs = torch.tensor([[[[[123.0]]]]])
    flows = torch.zeros(1, 1, 2, 1, 1)
    ref_holes = torch.ones(1, 1, 1, 1, 1)
    output, confidence = fuse_reference_latents(target, target_hole, refs, flows, ref_holes)
    assert output.item() == 0
    assert confidence.item() == 0


def test_invalid_reliability_is_rejected():
    with pytest.raises(ValueError, match="reliabilities"):
        fuse_reference_latents(
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 2, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.full((1, 1, 1, 1, 1), -0.1),
        )
