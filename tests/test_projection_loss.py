import torch
from omegaconf import OmegaConf

from motion_proj.backbones.base import Conditioning, DiffusionBackbone
from motion_proj.losses import anchor_loss, projection_loss, real_loss


class MockBackbone(DiffusionBackbone):
    parameterization = "edm"

    def __init__(self, perfect_target=None):
        self.perfect = perfect_target

    def encode(self, f):
        return f

    def decode(self, l):
        return l

    def sample_sigmas(self, num, device):
        return torch.linspace(0.02, 1.0, num, device=device)

    def predict_x0(self, z, sigma, cond):
        return self.perfect if self.perfect is not None else z * 0.0

    def anchor_predict_x0(self, z, sigma, cond):
        return z * 0.0

    def build_conditioning(self, batch):
        return Conditioning({})

    def trainable_parameters(self):
        return []

    def save_adapter(self, p):
        pass

    def load_adapter(self, p):
        pass


def _data():
    B, T, C, h, w = 2, 4, 4, 8, 12
    y = torch.randn(B, T, C, h, w)
    xd = torch.randn(B, T, C, h, w)
    mask = torch.ones(B, T, 1, h, w)
    tube = OmegaConf.create({"sigma_quantile_range": [0.0, 0.4], "bound_B": 1000.0})
    return y, xd, mask, tube


def test_perfect_prediction_zero_loss():
    y, xd, mask, tube = _data()
    bk = MockBackbone(perfect_target=xd)
    out = projection_loss(bk, y, xd, mask, Conditioning({}), tube)
    assert float(out["loss"]) < 1e-6
    assert out["gate_frac"] == 1.0


def test_zero_prediction_matches_target_energy():
    y, xd, mask, tube = _data()
    bk = MockBackbone(perfect_target=torch.zeros_like(xd))
    out = projection_loss(bk, y, xd, mask, Conditioning({}), tube)
    assert abs(float(out["loss"]) - float((xd**2).mean())) < 1e-3


def test_bound_gate_can_drop_all():
    y, xd, mask, _ = _data()
    tube = OmegaConf.create({"sigma_quantile_range": [0.0, 0.4], "bound_B": 1e-6})
    bk = MockBackbone(perfect_target=xd)
    out = projection_loss(bk, y, xd, mask, Conditioning({}), tube)
    assert out["gate_frac"] == 0.0


def test_eps_form_perfect_zero():
    y, xd, mask, tube = _data()
    bk = MockBackbone(perfect_target=xd)
    bk.parameterization = "eps"
    out = projection_loss(bk, y, xd, mask, Conditioning({}), tube)
    assert float(out["loss"]) < 1e-6


def test_real_and_anchor_smoke():
    y, xd, mask, tube = _data()
    bk = MockBackbone(perfect_target=xd)
    out = projection_loss(bk, y, xd, mask, Conditioning({}), tube)
    rl = real_loss(bk, y, Conditioning({}))
    al = anchor_loss(bk, out["z"], out["sigma"], Conditioning({}), out["x0_hat"])
    assert torch.isfinite(rl["loss"]) and torch.isfinite(al)
