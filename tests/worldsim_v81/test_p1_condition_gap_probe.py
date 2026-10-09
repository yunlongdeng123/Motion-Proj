"""用实际Euler实现核对诊断公式；不加载模型、不宣称生成能力通过。"""

from types import SimpleNamespace

import pytest
import torch
from diffusers import EulerDiscreteScheduler

from motion_proj.worldsim_v81.p1_condition_gap_probe import (
    clip_pixels_from_visible, latent_errors, measure_fixed_input, schedule_levels,
)


def _scheduler():
    # 独立数值夹具；不是已验证的远端XT1.1模型配置。
    return EulerDiscreteScheduler(prediction_type='v_prediction',
        timestep_type='continuous', use_karras_sigmas=True,
        sigma_min=.002, sigma_max=700, timestep_spacing='leading')


class _FixedPrediction(torch.nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))
        self.prediction = prediction
        self.inputs = []

    def forward(self, x, t, clip, added_time_ids):
        self.inputs.append((x.clone(), t.clone(), clip.clone(), added_time_ids.clone()))
        return SimpleNamespace(sample=self.prediction)


@pytest.mark.parametrize('level_index', [0, 1, 2])
def test_denoised_matches_diffusers_euler_pred_original_at_all_noise_levels(level_index):
    scheduler = _scheduler()
    level = schedule_levels(scheduler)[level_index]
    generator = torch.Generator().manual_seed(7)
    target = torch.randn((1, 3, 4, 4, 4), generator=generator)
    noise = torch.randn(target.shape, generator=generator)
    prediction = torch.randn(target.shape, generator=generator)
    condition = torch.zeros_like(target)
    hole = torch.ones(1, 3, 1, 4, 4); hole[..., 1:3] = 0
    unet = _FixedPrediction(prediction)
    score, actual = measure_fixed_input(unet, target, noise, condition,
        torch.zeros(1, 1, 5), torch.tensor([[6, 127, .02]]), hole,
        sigma=level['sigma'], amp='bf16')
    # 外部scheduler公式作为独立对照，不复制本诊断的重建公式。
    noisy = target + level['sigma'] * noise
    timestep = scheduler.timesteps[level['schedule_index']]
    scaled = scheduler.scale_model_input(noisy, timestep)
    expected = scheduler.step(prediction, timestep, noisy).pred_original_sample
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(unet.inputs[0][0][:, :, :4], scaled)
    assert score['weighted_mse'] >= 0


def test_condition_cells_share_the_same_noisy_gt_and_time():
    shape = (1, 2, 4, 4, 4)
    target, noise = torch.ones(shape), torch.arange(128.).reshape(shape) / 128
    unet = _FixedPrediction(torch.zeros(shape))
    hole = torch.ones(1, 2, 1, 4, 4); hole[..., 1:3] = 0
    for value in (0., 1.):
        measure_fixed_input(unet, target, noise, torch.full(shape, value),
            torch.full((1, 1, 5), value), torch.tensor([[6,127,.02]]),
            hole, sigma=70, amp='bf16')
    first, second = unet.inputs
    assert torch.equal(first[0][:, :, :4], second[0][:, :, :4])
    assert torch.equal(first[1], second[1]) and torch.equal(first[3], second[3])
    assert not torch.equal(first[0][:, :, 4:], second[0][:, :, 4:])
    assert not torch.equal(first[2], second[2])


def test_black_hole_input_discards_hidden_pixels_and_keeps_visible_pixels():
    hole = torch.ones(1, 2, 1, 4, 4); hole[..., 1:3] = 0
    first = torch.full((1, 2, 3, 4, 4), .2)
    second = first.masked_fill(hole.expand_as(first).bool(), .9)
    a, b = (clip_pixels_from_visible(x, hole) for x in (first, second))
    assert torch.equal(a, b)
    assert (a[..., 1:3] == .2).all() and (a[..., 0] == -1).all()


def test_masked_metrics_separate_hole_and_visible_error():
    target = torch.zeros(1, 2, 4, 4, 4)
    hole = torch.ones(1, 2, 1, 4, 4); hole[..., 1:3] = 0
    denoised = torch.full_like(target, 2.)
    denoised[..., 1:3] = 3.
    score = latent_errors(denoised, target, hole, sigma=2)
    assert score == pytest.approx({'hole_mse': 4., 'known_mse': 9., 'weighted_mse': 8.125})
    with pytest.raises(ValueError, match='非空'):
        latent_errors(denoised, target, torch.ones_like(hole), sigma=2)


def test_rejects_incompatible_prediction_and_time_protocol():
    for kwargs in ({'prediction_type': 'epsilon', 'timestep_type': 'continuous'},
                   {'prediction_type': 'v_prediction', 'timestep_type': 'discrete'}):
        with pytest.raises(ValueError, match='continuous/v_prediction'):
            schedule_levels(EulerDiscreteScheduler(**kwargs))
