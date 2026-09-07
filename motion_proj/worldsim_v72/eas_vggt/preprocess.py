"""为 VGGT 与 Pi3X 生成相同像素域的可追溯输入。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from motion_proj.worldsim_v72.data.camera_schema import CameraWindow


def resize_camera_window(
    window: CameraWindow,
    *,
    maximum_pixels: int = 255_000,
    patch_multiple: int = 14,
) -> tuple[torch.Tensor, np.ndarray]:
    """保持全图宽高比，只做缩放并记录 original→model 像素矩阵。"""
    if maximum_pixels <= 0 or patch_multiple <= 0:
        raise ValueError("maximum_pixels 与 patch_multiple 必须为正")
    first_width, first_height = map(int, window.frames[0].original_size_wh)
    ratio = (maximum_pixels / float(first_width * first_height)) ** 0.5
    target_width = max(1, round(first_width * ratio / patch_multiple)) * patch_multiple
    target_height = max(1, round(first_height * ratio / patch_multiple)) * patch_multiple
    while target_width * target_height > maximum_pixels:
        if target_width / target_height > first_width / first_height:
            target_width -= patch_multiple
        else:
            target_height -= patch_multiple
    tensors: list[torch.Tensor] = []
    transforms: list[np.ndarray] = []
    for frame in window.frames:
        width, height = map(int, frame.original_size_wh)
        if (width, height) != (first_width, first_height):
            raise ValueError("同一模型 batch 当前要求相机图像尺寸一致")
        with Image.open(Path(frame.image_path)) as image:
            rgb = image.convert("RGB").resize((target_width, target_height), Image.Resampling.LANCZOS)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        transforms.append(
            np.asarray(
                [[target_width / width, 0.0, 0.0], [0.0, target_height / height, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        )
    return torch.stack(tensors), np.stack(transforms)
