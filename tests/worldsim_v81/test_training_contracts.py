"""P0 输入角色和训练范围的 CPU 契约。"""

from types import SimpleNamespace

import pytest
import torch

from motion_proj.worldsim_v81.model_bridge import configure_trainable, validate_batch
from motion_proj.worldsim_v81.train import classify_propagator_gradient, gradient_report


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = torch.nn.Linear(2, 2)
        self.temporal_transformer_block = torch.nn.Linear(2, 2)


def test_training_scope_only_enables_official_temporal_and_propagation_components():
    modules = SimpleNamespace(vae=torch.nn.Linear(2, 2), image_encoder=torch.nn.Linear(2, 2),
                              raft=torch.nn.Linear(2, 2), fcnet=torch.nn.Linear(2, 2),
                              propagator=torch.nn.Linear(2, 2), unet=TinyUNet())
    sizes = configure_trainable(modules)
    assert sizes["svd_temporal"] > 0
    assert all(not p.requires_grad for model in (modules.vae, modules.image_encoder,
                                                   modules.raft, modules.unet.spatial)
               for p in model.parameters())
    assert all(p.requires_grad for model in (modules.fcnet, modules.propagator,
                                              modules.unet.temporal_transformer_block)
               for p in model.parameters())


def test_visible_holes_must_be_zero_and_target_has_separate_role():
    mask = torch.zeros(1, 25, 1, 256, 256)
    mask[..., :85] = 1
    visible = torch.zeros(1, 25, 3, 256, 256)
    target = torch.ones_like(visible)
    validate_batch({"visible_rgb": visible, "target_rgb": target, "hole_mask": mask})
    visible[..., 0, 0] = 1
    with pytest.raises(ValueError, match="洞区"):
        validate_batch({"visible_rgb": visible, "target_rgb": target, "hole_mask": mask})


def test_gradient_report_distinguishes_real_gradient_from_missing_or_invalid():
    module = torch.nn.Linear(2, 1)
    assert gradient_report(module)["status"] == "missing"
    module(torch.ones(1, 2)).sum().backward()
    report = gradient_report(module)
    assert report["status"] == "ok" and report["finite"] and report["nonzero"]
    assert report["norm"] > 0
    module.weight.grad[0, 0] = float("inf")
    report = gradient_report(module)
    assert report["status"] == "invalid" and report["norm"] is None


def test_condition_dropout_marks_finite_zero_gradient_as_skipped_but_not_nonfinite():
    module = torch.nn.Linear(2, 1)
    (module(torch.zeros(1, 2)) * 0).sum().backward()
    report = gradient_report(module)
    assert report["finite"] and not report["nonzero"]
    assert classify_propagator_gradient(report, True)["status"] == "skipped_condition_dropout"
    module.weight.grad[0, 0] = float("inf")
    assert classify_propagator_gradient(gradient_report(module), True)["status"] == "invalid"
