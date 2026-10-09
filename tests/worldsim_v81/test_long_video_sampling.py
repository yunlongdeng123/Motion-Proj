"""全视频窗口采样的CPU契约，不加载模型权重或CUDA。"""

from types import SimpleNamespace

import pytest
import torch

from motion_proj.worldsim_v81.long_video_sampling import (
    get_views, iter_decoded_frame_chunks, sample_global_latents,
)


class FakePipeline:
    def __init__(self, *, zeros=False):
        self.zeros = zeros
        self.prepare_calls = []
        self.decode_calls = []

    def prepare_latents(self, batch, frames, channels, height, width,
                        dtype, device, generator):
        self.prepare_calls.append((batch, frames, channels, height, width))
        shape = (batch, frames, channels // 2, height // 8, width // 8)
        if self.zeros:
            return torch.zeros(shape, dtype=dtype, device=device)
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    def decode_latents(self, latents, *, num_frames, decode_chunk_size):
        self.decode_calls.append((num_frames, decode_chunk_size))
        assert num_frames == latents.shape[1]
        return latents.permute(0, 2, 1, 3, 4) * 2 - 1


class FakeScheduler:
    def __init__(self):
        self.set_calls = []
        self.step_calls = []
        self.timesteps = ()

    def set_timesteps(self, steps, *, device):
        self.set_calls.append((steps, device))
        self.timesteps = tuple(range(steps - 1, -1, -1))

    def step(self, prediction, timestep, latents):
        self.step_calls.append((timestep, prediction.clone()))
        return SimpleNamespace(prev_sample=latents - prediction * 0.25)


def _conditions(frames):
    condition = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1)
    return condition, torch.ones(1, 1, 2), torch.tensor([[6.0, 127.0, 0.02]])


def _predict(latents, condition, embedding, ids, timestep, scheduler):
    assert condition.shape[1] == latents.shape[1]
    assert embedding.shape[0] == ids.shape[0] == latents.shape[0]
    unconditional = latents * 0.05
    conditional = latents * 0.1 + condition * 0.2 + timestep * 0.01
    return torch.cat((unconditional, conditional), dim=0)


@pytest.mark.parametrize("frames, expected", [
    (1, [(0, 1)]),
    (24, [(0, 24)]),
    (25, [(0, 25)]),
    (26, [(0, 25), (1, 26)]),
    (41, [(0, 25), (16, 41)]),
    (42, [(0, 25), (16, 41), (17, 42)]),
])
def test_public_get_views_covers_tail_and_short_video(frames, expected):
    views = get_views(frames)
    assert views == expected
    assert {index for start, end in views for index in range(start, end)} == set(range(frames))
    assert len(views) == len(set(views))


def test_global_noise_once_and_overlap_prediction_mean():
    condition, embedding, ids = _conditions(26)
    pipeline, scheduler = FakePipeline(zeros=True), FakeScheduler()
    starts = []

    def local_index_predictor(latents, cond, image_embedding, time_ids, timestep, sched):
        starts.append(int(cond[0, 0, 0, 0, 0]))
        local = torch.arange(cond.shape[1], dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return torch.cat((torch.zeros_like(latents), local.expand_as(latents)), dim=0)

    output = sample_global_latents(
        pipeline, scheduler, condition, embedding, ids,
        num_channels_latents=2, height=8, width=8, num_inference_steps=2,
        generator=torch.Generator().manual_seed(7), predict_noise=local_index_predictor,
    )
    assert output.shape == (1, 26, 1, 1, 1)
    assert pipeline.prepare_calls == [(1, 26, 2, 8, 8)]
    assert len(scheduler.set_calls) == 2
    assert starts == [0, 1, 0, 1]
    assert [step for step, _ in scheduler.step_calls] == [1, 0]
    guidance = torch.linspace(1.0, 3.0, 26)
    # 全局帧1来自两个窗口的local 1/0；帧25只来自末窗local 24。
    assert scheduler.step_calls[0][1][0, 1, 0, 0, 0] == pytest.approx(guidance[1].item() * 0.5)
    assert scheduler.step_calls[0][1][0, 25, 0, 0, 0] == pytest.approx(guidance[25].item() * 24)


def test_25_frames_matches_existing_single_window_call_order_and_values():
    condition, embedding, ids = _conditions(25)
    pipeline, scheduler = FakePipeline(), FakeScheduler()
    actual = sample_global_latents(
        pipeline, scheduler, condition, embedding, ids,
        num_channels_latents=2, height=8, width=8, num_inference_steps=3,
        generator=torch.Generator().manual_seed(2026), predict_noise=_predict,
    )

    # infer_p1.generate 的现行25帧单窗口采样顺序。
    legacy_pipeline, legacy_scheduler = FakePipeline(), FakeScheduler()
    legacy_scheduler.set_timesteps(3, device=condition.device)
    expected = legacy_pipeline.prepare_latents(
        1, 25, 2, 8, 8, embedding.dtype, condition.device,
        torch.Generator().manual_seed(2026),
    )
    legacy_scheduler.set_timesteps(3, device=condition.device)
    guidance = torch.linspace(1.0, 3.0, 25).view(1, 25, 1, 1, 1)
    for timestep in legacy_scheduler.timesteps:
        prediction = _predict(expected, condition, embedding, ids, timestep, legacy_scheduler)
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + guidance * (conditional - unconditional)
        expected = legacy_scheduler.step(prediction, timestep, expected).prev_sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [t for t, _ in scheduler.step_calls] == [2, 1, 0]


def test_short_video_is_one_unpadded_window():
    condition, embedding, ids = _conditions(7)
    lengths = []

    def record(*args):
        lengths.append(args[0].shape[1])
        return _predict(*args)

    result = sample_global_latents(
        FakePipeline(), FakeScheduler(), condition, embedding, ids,
        num_channels_latents=2, height=8, width=8, num_inference_steps=2,
        generator=torch.Generator().manual_seed(1), predict_noise=record,
    )
    assert result.shape[1] == 7
    assert lengths == [7, 7]


def test_decode_streams_all_25_frames_in_existing_chunk_sizes():
    latents = torch.linspace(0, 1, 25).view(1, 25, 1, 1, 1)
    pipeline = FakePipeline()
    chunks = list(iter_decoded_frame_chunks(pipeline, latents, decode_chunk_size=8))
    assert [start for start, _ in chunks] == [0, 8, 16, 24]
    assert [frames.shape[0] for _, frames in chunks] == [8, 8, 8, 1]
    assert pipeline.decode_calls == [(8, 8), (8, 8), (8, 8), (1, 8)]
    actual = torch.cat([frames for _, frames in chunks], dim=0)
    # 对照现行generate整段decode后的同一归一化算式，避免浮点往返误差。
    full = pipeline.decode_latents(latents, num_frames=25, decode_chunk_size=8)[0]
    expected = ((full.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cpu"
