"""P0 视频清单与读取：完整 RGB 只作为训练真值，可见输入独立遮蔽。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .masks import apply_hole_mask, make_border_outpaint_mask


def discover_videos(root: Path, split: str, frames: int = 25) -> tuple[list[dict], list[dict]]:
    """root 必须是该 split 的 JPEGImages；不猜测或混合 train/test 目录。"""
    if split not in {"train", "val", "test"}:
        raise ValueError("split 必须显式指定 train/val/test")
    if not root.is_dir() or frames < 2:
        raise ValueError("RGB 目录不存在或帧数无效")
    accepted, rejected = [], []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        row = {"video_id": directory.name, "split": split, "frames": [str(p.resolve()) for p in images]}
        if len(images) < frames:
            rejected.append({"video_id": directory.name, "reason": "too_few_frames", "count": len(images)})
        else:
            accepted.append(row)
    return accepted, rejected


class OutpaintingDataset(Dataset):
    """每视频一条确定性起步 clip；随机 100K clip 采样留给正式训练入口。"""
    def __init__(self, manifest: Path, frames: int = 25, size: tuple[int, int] = (256, 256),
                 left_fraction: float = 0.33, right_fraction: float = 0.33):
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("清单为空")
        self.frames, self.size = frames, size
        self.left_fraction, self.right_fraction = left_fraction, right_fraction
        for row in self.rows:
            if row["split"] != "train" or len(row["frames"]) < frames:
                raise ValueError("此训练读取入口只接受足够长的 train clip")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        images = []
        height, width = self.size
        for filename in row["frames"][:self.frames]:
            with Image.open(filename) as image:
                image = image.convert("RGB")
                # 公开代码先等比缩放至覆盖目标，再中心裁切。
                scale = max(width / image.width, height / image.height)
                new_width, new_height = round(image.width * scale), round(image.height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
                x, y = (new_width - width) // 2, (new_height - height) // 2
                image = image.crop((x, y, x + width, y + height))
                images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1)
        target = torch.stack(images)
        hole = make_border_outpaint_mask(height, width, self.left_fraction, self.right_fraction)
        hole = hole.expand(self.frames, 1, height, width)
        return {"target_rgb": target, "visible_rgb": apply_hole_mask(target, hole),
                "hole_mask": hole, "video_id": row["video_id"]}
