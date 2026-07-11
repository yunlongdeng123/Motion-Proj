"""用于训练、加载投影缓存的数据集。"""
from __future__ import annotations

import json
import glob
import os

import torch
from torch.utils.data import Dataset

from ..utils.io import load_json, load_tensor
from .writer import COMPLETE


class ProjectionCacheDataset(Dataset):
    def __init__(self, cache_dir: str, expected_fingerprint: str | None = None):
        self.cache_dir = cache_dir
        candidates = sorted(
            d for d in glob.glob(os.path.join(cache_dir, "*"))
            if ".stale-" not in os.path.basename(d)
            and os.path.isfile(os.path.join(d, "metadata.json"))
            and os.path.isfile(os.path.join(d, COMPLETE))
        )
        if expected_fingerprint is not None:
            mismatched = []
            matched = []
            for directory in candidates:
                with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
                    fingerprint = json.load(handle).get("cache_fingerprint")
                if fingerprint == expected_fingerprint:
                    matched.append(directory)
                else:
                    mismatched.append((directory, fingerprint))
            if mismatched:
                preview = ", ".join(os.path.basename(path) for path, _ in mismatched[:3])
                raise RuntimeError(
                    f"cache 目录混入 {len(mismatched)} 个 fingerprint 不匹配样本，"
                    f"示例: {preview}"
                )
            candidates = matched
        self.dirs = candidates
        if not candidates:
            raise FileNotFoundError(
                f"no cache entries under {cache_dir}; run motion_proj.cache.build_cache first"
            )

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> dict:
        d = self.dirs[idx]
        item = {
            "clean": load_tensor(os.path.join(d, "clean.pt")),
            "y": load_tensor(os.path.join(d, "y.pt")),
            "x_dagger": load_tensor(os.path.join(d, "x_dagger.pt")),
            "mask": load_tensor(os.path.join(d, "mask.pt")),
            "metadata": load_json(os.path.join(d, "metadata.json")),
        }
        ctx_path = os.path.join(d, "context.pt")
        if os.path.isfile(ctx_path):
            item["context"] = torch.load(ctx_path, map_location="cpu", weights_only=True)
        return item


def cache_collate(batch: list[dict]) -> dict:
    out: dict = {}
    out["clean"] = torch.stack([b["clean"] for b in batch], 0)
    out["y"] = torch.stack([b["y"] for b in batch], 0)
    out["x_dagger"] = torch.stack([b["x_dagger"] for b in batch], 0)
    out["mask"] = torch.stack([b["mask"] for b in batch], 0)
    out["metadata"] = [b["metadata"] for b in batch]
    if "context" in batch[0]:
        keys = batch[0]["context"].keys()
        out["context"] = {k: torch.stack([b["context"][k] for b in batch], 0) for k in keys}
    return out
