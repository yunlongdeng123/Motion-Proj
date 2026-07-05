"""用于定性检查的可视化辅助工具（光流、检测框、拼图面板）。"""
from __future__ import annotations

import numpy as np
import torch


def flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    """将 ``[H,W,2]`` 光流场用颜色编码（HSV 色轮）-> ``[H,W,3]`` uint8。"""
    import cv2

    f = flow.detach().float().cpu().numpy()
    mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
    hsv = np.zeros((*f.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def draw_boxes(img_uint8: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """在 ``[H,W,3]`` uint8 图像上绘制 2D 检测框 ``[N,4]`` (u0,v0,u1,v1)。"""
    import cv2

    out = img_uint8.copy()
    for b in np.asarray(boxes_xyxy).reshape(-1, 4):
        u0, v0, u1, v1 = [int(round(x)) for x in b]
        cv2.rectangle(out, (u0, v0), (u1, v1), color, 2)
    return out


def hstack_panels(*imgs: np.ndarray) -> np.ndarray:
    """水平拼接 ``[H,W,3]`` uint8 图像（填充到最大高度 H）。"""
    h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] != h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
            im = np.concatenate([im, pad], axis=0)
        padded.append(im)
    return np.concatenate(padded, axis=1)


def make_comparison_panel(y: torch.Tensor, x_dagger: torch.Tensor, mask: torch.Tensor | None = None):
    """构建逐帧的 [y | x_dagger | mask] 对比条带，输出为 uint8 视频。

    Args:
        y, x_dagger: ``[T,3,H,W]``，取值范围 [-1,1] 或 [0,1]。
        mask: 可选的 ``[T,1,H,W]`` 或 ``[T,H,W]`` 可靠性掩码，取值范围 [0,1]。
    Returns:
        ``[T,H,3W,3]`` uint8 数组。
    """
    from .io import to_uint8_video

    yv = to_uint8_video(y)
    xv = to_uint8_video(x_dagger)
    frames = []
    for t in range(yv.shape[0]):
        panels = [yv[t], xv[t]]
        if mask is not None:
            m = mask[t]
            if m.dim() == 3:
                m = m[0]
            m = (m.detach().float().cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
            panels.append(np.repeat(m[..., None], 3, axis=-1))
        frames.append(hstack_panels(*panels))
    return np.stack(frames, axis=0)
