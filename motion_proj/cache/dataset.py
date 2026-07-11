"""用于训练、加载投影缓存的数据集。"""
from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import Dataset

from ..utils.io import load_json, load_tensor
from .writer import COMPLETE


class ProjectionCacheDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.dirs = sorted(
            d for d in glob.glob(os.path.join(cache_dir, "*"))
            if os.path.isfile(os.path.join(d, "metadata.json")) and os.path.isfile(os.path.join(d, COMPLETE))
        )
        if not self.dirs:
            raise FileNotFoundError(
                f"no cache entries under {cache_dir}; run motion_proj.cache.build_cache first"
            )

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> dict:
        d = self.dirs[idx]
        item = {
            "y": load_tensor(os.path.join(d, "y.pt")),
            "x_dagger": load_tensor(os.path.join(d, "x_dagger.pt")),
            "mask": load_tensor(os.path.join(d, "mask.pt")),
            "metadata": load_json(os.path.join(d, "metadata.json")),
        }
        ctx_path = os.path.join(d, "context.pt")
        if os.path.isfile(ctx_path):
            item["context"] = torch.load(ctx_path, map_location="cpu")
        return item


def cache_collate(batch: list[dict]) -> dict:
    out: dict = {}
    out["y"] = torch.stack([b["y"] for b in batch], 0)
    out["x_dagger"] = torch.stack([b["x_dagger"] for b in batch], 0)
    out["mask"] = torch.stack([b["mask"] for b in batch], 0)
    out["metadata"] = [b["metadata"] for b in batch]
    if "context" in batch[0]:
        keys = batch[0]["context"].keys()
        out["context"] = {k: torch.stack([b["context"][k] for b in batch], 0) for k in keys}
    return out
