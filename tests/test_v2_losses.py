import torch

from motion_proj.backbones.base import Conditioning, DiffusionBackbone
from motion_proj.losses import correction_v_loss, outside_mask_preserve_v_loss, real_loss, teacher_relative_v_target


class VBackbone(DiffusionBackbone):
    def encode(self, x): return x
    def decode(self, x): return x
    def sample_sigmas(self, num, device): return torch.ones(num, device=device)
    def predict_model_output(self, z, sigma, cond): return z * 0.0
    def anchor_predict_model_output(self, z, sigma, cond): return z * 0.0
    def x0_from_model_output(self, z, sigma, model_output): return model_output
    def model_output_from_x0(self, z, sigma, x0): return x0
    def build_conditioning(self, batch): return Conditioning({})
    def trainable_parameters(self): return []
    def save_adapter(self, path): pass
    def load_adapter(self, path): pass


def test_residual_v_target_matches_closed_form_and_trust_region():
    bk = VBackbone()
    z = torch.zeros(1, 2, 1, 2, 2)
    base = torch.zeros_like(z)
    projected = torch.ones_like(z)
    mask = torch.ones(1, 2, 1, 2, 2)
    out = teacher_relative_v_target(bk, z, torch.tensor([2.0]), Conditioning({}), base, projected, mask, torch.zeros_like(mask), eta=1.0, trust_region_B=100.0)
    assert torch.allclose(out["target"], torch.full_like(z, -(5.0**0.5) / 2.0))
    clipped = teacher_relative_v_target(bk, z, torch.tensor([0.1]), Conditioning({}), base, projected, mask, torch.zeros_like(mask), eta=1.0, trust_region_B=0.02)
    assert float(clipped["eta_eff"][0]) < 1.0
    assert float(clipped["trust_region_clipping_fraction"]) == 1.0


def test_correction_and_outside_preserve_are_spatially_disjoint():
    v = torch.zeros(1, 1, 1, 5, 5)
    target = v.clone(); target[..., 2, 2] = 2.0
    mask = torch.zeros(1, 1, 1, 5, 5); mask[..., 2, 2] = 1.0
    corr = correction_v_loss(v, target, mask, torch.zeros_like(mask))
    assert float(corr["loss_static"]) > 0 and float(corr["loss_object"]) == 0.0
    preserve = outside_mask_preserve_v_loss(v, target, mask, dilation_radius=1)
    assert float(preserve["loss"]) == 0.0
    assert float(preserve["outside_mask"][..., 2, 2]) == 0.0


def test_real_loss_accepts_fixed_noise_and_sigma():
    bk = VBackbone()
    x0 = torch.ones(2, 1, 1, 2, 2)
    sigma = torch.tensor([0.5, 1.0])
    noise = torch.zeros_like(x0)
    a = real_loss(bk, x0, Conditioning({}), use_edm_weight=False, sigma=sigma, noise=noise)
    b = real_loss(bk, x0, Conditioning({}), use_edm_weight=False, sigma=sigma, noise=noise)
    assert torch.equal(a["z"], b["z"])
    assert torch.equal(a["sigma"], sigma)
