import pytest
import torch
from omegaconf import OmegaConf

from motion_proj.backbones.svd_backbone import resolve_svd_generation_settings
from motion_proj.diagnostics.svd_conditioning_parity import (
    as_torch_device,
    compare_candidate_conditioning,
    compare_generation_traces,
    tensor_difference,
)


def _trace(value: float = 0.0):
    tensor = torch.full((1, 2, 3), value)
    return {
        "added_time_ids": tensor[:, :1, :],
        "condition_noise": tensor,
        "initial_video_latents": tensor,
        "scheduler_timesteps": [torch.tensor(1.0), torch.tensor(0.5)],
        "scheduler_inputs": [tensor, tensor],
        "scaled_model_inputs": [tensor, tensor],
        "unet_inputs": [tensor, tensor],
        "raw_model_outputs": [tensor, tensor],
        "unconditional_raw_model_outputs": [tensor, tensor],
        "conditional_raw_model_outputs": [tensor, tensor],
        "cfg_outputs": [tensor, tensor],
        "scheduler_step_outputs": [tensor, tensor],
        "post_step_latents": [tensor, tensor],
        "final_latent": tensor,
        "decoded_frames": tensor,
    }


def test_generation_settings_version_and_legacy_default_are_explicit():
    legacy = resolve_svd_generation_settings(OmegaConf.create({}))
    assert legacy["protocol"] == "svd_legacy_unversioned"
    official = resolve_svd_generation_settings(OmegaConf.create({"generation": {"protocol": "svd_official_v1"}}))
    assert official["protocol"] == "svd_official_v1"
    assert official["fps"] == 7
    with pytest.raises(ValueError, match="未知 SVD generation.protocol"):
        resolve_svd_generation_settings(OmegaConf.create({"generation": {"protocol": "unknown"}}))


def test_tensor_difference_reports_shape_and_exactness():
    exact = tensor_difference(torch.zeros(2), torch.zeros(2))
    assert exact["exact"]
    assert exact["max_abs"] == 0.0
    mismatch = tensor_difference(torch.zeros(2), torch.zeros(3))
    assert not mismatch["shape_match"]
    assert mismatch["max_abs"] == float("inf")


def test_candidate_pipeline_device_is_a_torch_device():
    assert as_torch_device("cuda") == torch.device("cuda")
    device = torch.device("cpu")
    assert as_torch_device(device) is device


def test_generation_trace_comparison_finds_first_step_mismatch():
    reference = _trace()
    candidate = _trace()
    candidate["raw_model_outputs"][1] = candidate["raw_model_outputs"][1] + 1.0e-3
    result = compare_generation_traces(
        reference,
        candidate,
        raw_tolerance=1.0e-4,
        final_latent_rms_tolerance=1.0e-4,
        rgb_tolerance=1.0e-4,
    )
    assert not result["passed"]
    assert result["first_mismatch"] == "raw_model_outputs"


def test_candidate_conditioning_requires_exact_official_tensors():
    official = {
        "condition_noise": torch.zeros(1),
        "noisy_condition_image": torch.zeros(1),
        "image_embeds": torch.zeros(1),
        "image_latents": torch.zeros(1),
        "added_time_ids": torch.zeros(1),
        "initial_video_latents": torch.zeros(1),
    }
    candidate = {key: value.clone() for key, value in official.items()}
    assert compare_candidate_conditioning(official, candidate)["passed"]
    candidate["added_time_ids"] += 1
    result = compare_candidate_conditioning(official, candidate)
    assert not result["passed"]
    assert result["first_mismatch"] == "added_time_ids"
