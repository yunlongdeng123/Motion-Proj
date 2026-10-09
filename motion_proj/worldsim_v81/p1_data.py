"""官方 YouTube-VOS train 采样：随机视频、随机连续25帧、固定双侧33%掩码。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .masks import apply_hole_mask, make_border_outpaint_mask


class YouTubeVOSP1Dataset(Dataset):
    def __init__(self, data_root: Path, *, seed: int = 123, samples: int = 100_000,
                 frames: int = 25, size: int = 256, mask_ratio: float = 0.33):
        root = data_root.expanduser().resolve()
        rgb_root = root if root.name == "JPEGImages" else root / "JPEGImages"
        if not rgb_root.is_dir():
            raise FileNotFoundError(f"必须显式指定 YouTube-VOS train/JPEGImages: {rgb_root}")
        if samples < 1 or frames != 25 or size != 256 or mask_ratio != 0.33:
            raise ValueError("P1 固定 25帧、256²、左右各0.33；samples 必须大于0")
        self.videos: list[tuple[str, tuple[Path, ...]]] = []
        self.rejected_short = 0
        for directory in sorted(rgb_root.iterdir()):
            if not directory.is_dir():
                continue
            images = tuple(sorted(path for path in directory.iterdir()
                                  if path.is_file() and path.suffix.lower() == ".jpg"))
            # 官方公开实现采用 len(os.listdir(video_dir)) > sample_frames。
            if len(images) <= frames:
                self.rejected_short += 1
                continue
            self.videos.append((directory.name, images))
        if not self.videos:
            raise ValueError(f"没有超过25张 JPG 的训练视频: {rgb_root}")
        self.rgb_root = rgb_root
        self.seed, self.samples, self.frames, self.size = seed, samples, frames, size
        self.mask = make_border_outpaint_mask(size, size, mask_ratio, mask_ratio)

    def __len__(self) -> int:
        return self.samples

    def profile(self) -> dict:
        return {"rgb_root": str(self.rgb_root), "videos": len(self.videos),
                "rejected_short": self.rejected_short,
                "available_windows": sum(len(images) - self.frames + 1 for _, images in self.videos),
                "virtual_samples": self.samples, "seed": self.seed,
                "frames": self.frames, "size": self.size, "mask_ratio_each_side": 0.33}

    def choice(self, step: int) -> tuple[int, int]:
        if step < 0 or step >= self.samples:
            raise IndexError(step)
        # 固定 step 的样本选择可在任一 checkpoint 精确恢复。
        rng = random.Random(self.seed + step)
        video_index = rng.randrange(len(self.videos))
        start = rng.randrange(len(self.videos[video_index][1]) - self.frames + 1)
        return video_index, start

    def __getitem__(self, step: int) -> dict:
        video_index, start = self.choice(step)
        video_id, files = self.videos[video_index]
        decoded = []
        for filename in files[start:start + self.frames]:
            with Image.open(filename) as image:
                image = image.convert("RGB")
                scale = max(self.size / image.width, self.size / image.height)
                width, height = round(image.width * scale), round(image.height * scale)
                image = image.resize((width, height), Image.Resampling.BICUBIC)
                x, y = (width - self.size) // 2, (height - self.size) // 2
                image = image.crop((x, y, x + self.size, y + self.size))
                decoded.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1)
        target = torch.stack(decoded)
        mask = self.mask.expand(self.frames, 1, self.size, self.size)
        return {"target_rgb": target, "visible_rgb": apply_hole_mask(target, mask),
                "hole_mask": mask, "video_id": video_id, "clip_start": start}
