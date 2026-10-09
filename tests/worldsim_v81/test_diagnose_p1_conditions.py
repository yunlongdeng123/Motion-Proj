"""条件诊断CPU契约：不加载权重、不触发CUDA。"""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from scripts.worldsim_v81.diagnose_p1_conditions import (
    FRAMES, SIZE, decode_unscaled_latent, draw_shared_randomness,
    masked_model_tensor, per_frame_delta, repeated_first_frames,
)


class DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(scaling_factor=2.0)


class DummyPipeline:
    def __init__(self):
        self.vae = DummyVAE()
        self.prepared = 0
        self.decode_calls = []

    def prepare_latents(self, batch, frames, channels, height, width,
                        dtype, device, generator):
        self.prepared += 1
        return torch.randn((batch, frames, channels // 2, height // 8, width // 8),
                           dtype=dtype, device=device, generator=generator)

    def decode_latents(self, latent, *, num_frames, decode_chunk_size):
        self.decode_calls.append((latent.clone(), num_frames, decode_chunk_size))
        return torch.zeros((1, 3, num_frames, 2, 2), dtype=latent.dtype)


class DummyScheduler:
    def __init__(self):
        self.calls = []

    def set_timesteps(self, steps, *, device):
        self.calls.append((steps, device))


def test_repeat_uses_only_masked_first_frame_and_hides_outer_rgb():
    originals = []
    for index in range(FRAMES):
        image = np.full((SIZE, SIZE, 3), index, dtype=np.uint8)
        image[:, :84] = 255  # 故意放入应被mask遮住的非零值。
        originals.append(Image.fromarray(image))
    repeated = repeated_first_frames(originals)
    assert len(repeated) == FRAMES
    assert all(np.array_equal(np.asarray(item), np.asarray(originals[0]))
               for item in repeated)
    hole = torch.zeros(1, FRAMES, 1, SIZE, SIZE)
    hole[..., :84] = 1
    hole[..., 172:] = 1
    tensor = masked_model_tensor(originals, hole)
    assert torch.count_nonzero(tensor[..., :84]) == 0
    assert torch.count_nonzero(tensor[..., 172:]) == 0
    assert not torch.equal(tensor[:, 0, :, :, 84:172],
                           tensor[:, 1, :, :, 84:172])
    repeated_tensor = masked_model_tensor(repeated, hole)
    assert torch.equal(tensor[:, 0], repeated_tensor[:, 0])
    assert all(torch.equal(repeated_tensor[:, 0], repeated_tensor[:, index])
               for index in range(FRAMES))


def test_rng_draws_one_shared_observation_and_one_initial_latent():
    pipeline, scheduler = DummyPipeline(), DummyScheduler()
    visible = torch.zeros(1, FRAMES, 3, SIZE, SIZE)
    embedding = torch.ones(1, 1, 4)
    noise, initial = draw_shared_randomness(
        pipeline, scheduler, visible, embedding, seed=2026, steps=25,
        unet_channels=8,
    )
    generator = torch.Generator().manual_seed(2026)
    expected_noise = torch.randn(visible.shape, generator=generator)
    expected_initial = torch.randn(initial.shape, generator=generator)
    assert torch.equal(noise, expected_noise)
    assert torch.equal(initial, expected_initial)
    assert pipeline.prepared == 1
    assert scheduler.calls == [(25, visible.device), (25, visible.device)]


def test_unscaled_condition_multiplied_before_official_decode():
    pipeline = DummyPipeline()
    latent = torch.ones(1, FRAMES, 4, 2, 2)
    frames = decode_unscaled_latent(pipeline, latent)
    assert frames.shape == (FRAMES, 3, 2, 2)
    assert [call[1:] for call in pipeline.decode_calls] == [
        (8, 8), (8, 8), (8, 8), (1, 8),
    ]
    assert all(torch.equal(call[0], torch.full_like(call[0], 2.0))
               for call in pipeline.decode_calls)
    assert torch.all(frames == 0.5)


def test_response_delta_is_per_frame_and_not_a_quality_score():
    normal = torch.zeros(FRAMES, 3, 2, 2)
    repeated = normal.clone()
    repeated[12] = 0.25
    delta = per_frame_delta(normal, repeated)
    assert len(delta) == FRAMES
    assert delta[0] == 0 and delta[12] == 0.25 and delta[24] == 0
