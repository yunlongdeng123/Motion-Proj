"""P1 全视频高斯采样：固定25帧重叠窗口、全局latent逐步更新。

条件、参考传播及初始噪声须在调用前对整段视频各构造一次。
``predict_noise`` 应复用 infer_p1._unet_noise 的标准 CFG 双支调用，
并返回 [unconditional, conditional] 顺序的预测张量。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch


def get_views(total_frames: int, window_size: int = 25, stride: int = 16) -> list[tuple[int, int]]:
    """对应固定公开 test.py 的 get_views，末窗锚定视频结尾。"""
    if total_frames < 1:
        raise ValueError("全视频至少需要1帧")
    if window_size < 1 or not 1 <= stride <= window_size:
        raise ValueError("窗口大小必须为正，步长必须在[1, window_size]内")
    if total_frames <= window_size:
        return [(0, total_frames)]

    views = []
    start = 0
    while start + window_size <= total_frames:
        views.append((start, start + window_size))
        start += stride
    if views[-1][1] < total_frames:
        views.append((total_frames - window_size, total_frames))
    return list(dict.fromkeys(views))


@torch.no_grad()
def sample_global_latents(
    pipeline,
    scheduler,
    condition: torch.Tensor,
    image_embedding: torch.Tensor,
    time_ids: torch.Tensor,
    *,
    num_channels_latents: int,
    height: int,
    width: int,
    num_inference_steps: int,
    generator: torch.Generator,
    predict_noise: Callable,
    window_size: int = 25,
    stride: int = 16,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
) -> torch.Tensor:
    """一次抽取全局噪声，每步平均各窗口CFG预测并更新全局latent一次。

    ``predict_noise(latent_window, condition_window, image_embedding,
    time_ids, timestep, scheduler)`` 必须返回 [2B,L,C,H,W]；它负责
    调用 scheduler.scale_model_input 和 U-Net，与 infer_p1._unet_noise 一致。
    """
    if condition.ndim != 5 or not 1 <= condition.shape[1]:
        raise ValueError("condition须为非空[B,T,C,H,W]")
    batch_size, total_frames = condition.shape[:2]
    if image_embedding.shape[0] != batch_size or time_ids.shape[0] != batch_size:
        raise ValueError("CLIP和时间条件batch须匹配全局条件")
    if num_inference_steps < 1:
        raise ValueError("扩散步数必须为正")
    views = get_views(total_frames, window_size, stride)
    device = condition.device

    # 保留当前25帧generate的 set_timesteps -> prepare_latents -> set_timesteps 顺序。
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = pipeline.prepare_latents(
        batch_size, total_frames, num_channels_latents, height, width,
        image_embedding.dtype, device, generator,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    if latents.shape != condition.shape:
        raise ValueError("pipeline返回的全局latent形状与传播条件不一致")

    guidance = torch.linspace(
        min_guidance_scale, max_guidance_scale, total_frames, device=device, dtype=latents.dtype,
    ).view(1, total_frames, 1, 1, 1)
    for timestep in scheduler.timesteps:
        values = torch.zeros_like(latents)
        counts = torch.zeros_like(latents)
        for start, end in views:
            prediction = predict_noise(
                latents[:, start:end], condition[:, start:end],
                image_embedding, time_ids, timestep, scheduler,
            )
            expected = (2 * batch_size, end - start, *latents.shape[2:])
            if prediction.shape != expected:
                raise ValueError(f"CFG预测形状{tuple(prediction.shape)}，预期{expected}")
            unconditional, conditional = prediction.chunk(2, dim=0)
            scale = guidance[:, start:end]
            guided = unconditional + scale * (conditional - unconditional)
            values[:, start:end] += guided
            counts[:, start:end] += 1
        if torch.any(counts == 0):
            raise RuntimeError("窗口未覆盖全部视频帧")
        latents = scheduler.step(values / counts, timestep, latents).prev_sample
    return latents


@torch.no_grad()
def iter_decoded_frame_chunks(
    pipeline, latents: torch.Tensor, *, decode_chunk_size: int = 8,
) -> Iterator[tuple[int, torch.Tensor]]:
    """沿时间轴分块VAE解码并立即转CPU，避免整段RGB驻留GPU。"""
    if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] < 1:
        raise ValueError("解码要求非空[B=1,T,C,H,W]全局latent")
    if decode_chunk_size < 1:
        raise ValueError("decode_chunk_size必须为正")
    total_frames = latents.shape[1]
    for start in range(0, total_frames, decode_chunk_size):
        end = min(start + decode_chunk_size, total_frames)
        decoded = pipeline.decode_latents(
            latents[:, start:end], num_frames=end - start,
            decode_chunk_size=decode_chunk_size,
        )[0]
        if decoded.ndim != 4 or decoded.shape[1] != end - start:
            raise ValueError("VAE解码帧数与输入chunk不一致")
        frames = ((decoded.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1).cpu()
        del decoded
        yield start, frames
