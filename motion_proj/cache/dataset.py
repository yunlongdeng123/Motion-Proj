"""用于训练、加载投影缓存的数据集。"""
from __future__ import annotations

import json
import glob
import os

import torch
from torch.utils.data import Dataset

from ..utils.io import load_json, load_tensor
from .writer import CACHE_SCHEMA_VERSION, COMPLETE


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
        stale_schema = []
        for directory in candidates:
            with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
                version = json.load(handle).get("cache_schema_version")
            if version != CACHE_SCHEMA_VERSION:
                stale_schema.append((directory, version))
        if stale_schema:
            preview = ", ".join(os.path.basename(path) for path, _ in stale_schema[:3])
            raise RuntimeError(
                f"cache 目录包含 {len(stale_schema)} 个非 schema v{CACHE_SCHEMA_VERSION} 样本，示例: {preview}"
            )
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
        for key, filename in (("latent_flow", "latent_flow.pt"),
                              ("flow_confidence", "flow_confidence.pt")):
            path = os.path.join(d, filename)
            if os.path.isfile(path):
                item[key] = torch.load(path, map_location="cpu", weights_only=True)
        return item


class MixedProjectionCacheDataset(Dataset):
    """按固定整数比例构造确定性虚拟 epoch，不复制 cache 文件。"""

    def __init__(self, datasets: dict[str, ProjectionCacheDataset], ratios: dict[str, int],
                 epoch_size: int | None = None):
        if set(datasets) != set(ratios) or not datasets:
            raise ValueError("datasets 与 ratios 必须具有相同的非空 source 集合")
        if any(int(value) <= 0 for value in ratios.values()):
            raise ValueError("cache 混合比例必须为正整数")
        self.datasets = datasets
        self.ratios = {key: int(value) for key, value in ratios.items()}
        self.cycle = [key for key in sorted(ratios) for _ in range(self.ratios[key])]
        requested = int(epoch_size or sum(len(ds) for ds in datasets.values()))
        cycles = max(1, (requested + len(self.cycle) - 1) // len(self.cycle))
        self.epoch_size = cycles * len(self.cycle)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        source = self.cycle[idx % len(self.cycle)]
        occurrence = idx // len(self.cycle) * self.ratios[source]
        occurrence += self.cycle[:idx % len(self.cycle)].count(source)
        item = dict(self.datasets[source][occurrence % len(self.datasets[source])])
        item["cache_source"] = source
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
    for key in ("latent_flow", "flow_confidence"):
        present = [key in item for item in batch]
        if any(present) and not all(present):
            raise ValueError(f"同一 batch 的 {key} 不得部分缺失")
        if all(present):
            out[key] = torch.stack([b[key] for b in batch], 0)
    out["cache_source"] = [b.get("cache_source", b["metadata"].get("source")) for b in batch]
    return out
