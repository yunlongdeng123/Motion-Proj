"""IO 辅助工具：张量的（反）序列化、json 读写，以及简单的视频写出。"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_tensor(t: torch.Tensor, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(t.detach().cpu(), path)


def load_tensor(path: str, map_location: str = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=map_location)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not json serializable: {type(o)}")


def write_video(frames_uint8: np.ndarray, path: str, fps: int = 4) -> None:
    """从 [T, H, W, 3] uint8 数组写出视频。若没有可用的 ffmpeg 后端，
    则静默回退（仍会将帧保存为 .npy）。"""
    ensure_dir(os.path.dirname(path))
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, list(frames_uint8), fps=fps, macro_block_size=None)
    except Exception:
        np.save(os.path.splitext(path)[0] + ".npy", frames_uint8)


def to_uint8_video(frames: torch.Tensor) -> np.ndarray:
    """[T,3,H,W]（取值范围 [-1,1] 或 [0,1]）-> [T,H,W,3] uint8。"""
    x = frames.detach().float().cpu()
    if x.min() < -0.01:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()
