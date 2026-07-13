import pytest
import torch
from diffusers import EulerDiscreteScheduler
from omegaconf import OmegaConf

from motion_proj.backbones.base import Conditioning
from motion_proj.backbones.svd_backbone import SVDBackbone


SIGMAS = (0.02, 0.05, 0.1, 0.5, 1.0, 5.0)


def _cfg(*, lora_enabled: bool = False):
    return OmegaConf.create(
        {
            "sigma_floor": 1.0e-3,
            "num_frames": 2,
            "lora": {
                "enable": lora_enabled,
                "rank": 16,
                "alpha": 16,
                "scope": "temporal_only",
                "projections": ["to_q", "to_k", "to_v", "to_out.0"],
            },
        }
    )


@pytest.mark.parametrize("sigma_value", SIGMAS)
def test_float32_x0_v_roundtrip(sigma_value):
    torch.manual_seed(7)
    backbone = SVDBackbone(_cfg())
    z = torch.randn(2, 3, 4, 5, 6)
    x0 = torch.randn_like(z)
    raw_v = torch.randn_like(z)
    sigma = torch.full((z.shape[0],), sigma_value)

    reconstructed_x0 = backbone.x0_from_model_output(
        z, sigma, backbone.model_output_from_x0(z, sigma, x0)
    )
    reconstructed_v = backbone.model_output_from_x0(
        z, sigma, backbone.x0_from_model_output(z, sigma, raw_v)
    )

    assert float((reconstructed_x0 - x0).abs().max()) < 1.0e-5
    assert float((reconstructed_v - raw_v).abs().max()) < 1.0e-5
    assert torch.isfinite(reconstructed_x0).all()
    assert torch.isfinite(reconstructed_v).all()


@pytest.mark.parametrize("sigma_value", SIGMAS)
def test_bf16_roundtrip_has_preregistered_tolerance(sigma_value):
    torch.manual_seed(11)
    backbone = SVDBackbone(_cfg())
    z = torch.randn(2, 2, 4, 3, 3, dtype=torch.bfloat16)
    raw_v = torch.randn_like(z)
    sigma = torch.full((z.shape[0],), sigma_value, dtype=torch.float32)

    reconstructed = backbone.model_output_from_x0(
        z, sigma, backbone.x0_from_model_output(z, sigma, raw_v)
    )
    relative_error = (reconstructed.float() - raw_v.float()).abs() / raw_v.float().abs().clamp_min(0.05)

    # bf16 输入、fp32 sigma 是训练路径的真实组合；容差在实验前固定。
    assert float(relative_error.max()) < 2.0e-3
    assert torch.isfinite(reconstructed).all()


def test_sigma_floor_and_nonfinite_guard():
    backbone = SVDBackbone(_cfg())
    z = torch.randn(1, 2, 4, 3, 3)
    raw_v = torch.randn_like(z)
    at_zero = backbone.x0_from_model_output(z, torch.tensor([0.0]), raw_v)
    at_floor = backbone.x0_from_model_output(z, torch.tensor([1.0e-3]), raw_v)

    torch.testing.assert_close(at_zero, at_floor, rtol=0.0, atol=0.0)
    assert torch.isfinite(at_zero).all()
    with pytest.raises(ValueError, match="NaN/Inf"):
        backbone.x0_from_model_output(z, torch.tensor([float("nan")]), raw_v)


class _ToggleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters_enabled = True

    def enable_adapters(self):
        self.adapters_enabled = True

    def disable_adapters(self):
        self.adapters_enabled = False

    def forward(self, model_input, *_args, **_kwargs):
        value = 2.0 if self.adapters_enabled else 1.0
        output = model_input[:, :, :4] * 0.0 + value
        return (output,)


def _conditioning():
    return Conditioning(
        {
            "image_latents": torch.zeros(1, 2, 4, 2, 2),
            "image_embeds": torch.zeros(1, 1, 4),
            "added_time_ids": torch.zeros(1, 3),
        }
    )


def test_anchor_raw_output_is_detached_and_restores_enabled_state():
    backbone = SVDBackbone(_cfg(lora_enabled=True))
    backbone.unet = _ToggleUNet()
    backbone.dtype = torch.float32
    backbone._lora_enabled = True
    z = torch.randn(1, 2, 4, 2, 2, requires_grad=True)
    sigma = torch.tensor([0.5])

    student = backbone.predict_model_output(z, sigma, _conditioning())
    anchor = backbone.anchor_predict_model_output(z, sigma, _conditioning())

    assert torch.all(student == 2.0)
    assert torch.all(anchor == 1.0)
    assert not anchor.requires_grad
    assert backbone._lora_enabled is True
    assert backbone.unet.adapters_enabled is True


def test_anchor_preserves_preexisting_disabled_state():
    backbone = SVDBackbone(_cfg(lora_enabled=True))
    backbone.unet = _ToggleUNet()
    backbone.dtype = torch.float32
    backbone._lora_enabled = True
    backbone._set_lora_enabled(False)

    anchor = backbone.anchor_predict_model_output(
        torch.zeros(1, 2, 4, 2, 2), torch.tensor([0.5]), _conditioning()
    )

    assert torch.all(anchor == 1.0)
    assert backbone._lora_enabled is False
    assert backbone.unet.adapters_enabled is False


def test_raw_v_algebra_matches_official_euler_scheduler_step():
    """The inverse-pair tests alone cannot detect duplicated preconditioning."""
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000,
        prediction_type="v_prediction",
        timestep_type="continuous",
        use_karras_sigmas=True,
        sigma_min=0.002,
        sigma_max=700.0,
    )
    scheduler.set_timesteps(25)
    generator = torch.Generator().manual_seed(19)
    sample = torch.randn(2, 4, 3, 5, generator=generator)
    raw_v = torch.randn(sample.shape, generator=generator)
    backbone = SVDBackbone(_cfg())

    for index, timestep in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[index]
        expected_input = sample / torch.sqrt(1.0 + sigma.square())
        scheduler_input = scheduler.scale_model_input(sample, timestep)
        torch.testing.assert_close(scheduler_input, expected_input, rtol=1.0e-6, atol=1.0e-6)
        torch.testing.assert_close(timestep, 0.25 * sigma.log(), rtol=1.0e-6, atol=1.0e-6)

        official_x0 = scheduler.step(raw_v, timestep, sample).pred_original_sample
        code_x0 = backbone.x0_from_model_output(
            sample.unsqueeze(0), sigma.reshape(1), raw_v.unsqueeze(0)
        ).squeeze(0)
        torch.testing.assert_close(code_x0, official_x0, rtol=1.0e-6, atol=1.0e-6)
