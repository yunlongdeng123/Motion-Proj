"""投影缓存写入器（方案第 9 节，第 2 阶段）。

每个样本的目录结构::

    <cache_dir>/<sample_id>/
        y.pt              # latent [T,C,h,w] 或 rgb [T,3,H,W]
        x_dagger.pt       # 与 y 相同的形状/空间
        mask.pt           # 可靠性 [T,1,h,w]（latent 分辨率）或 [T,1,H,W]（rgb）
        context.pt        # 骨干网络条件（仅 latent 存储时）
        metadata.json
"""
from __future__ import annotations

import os

import torch

from ..utils.io import ensure_dir, save_json, save_tensor
from ..utils.logging import get_logger

log = get_logger(__name__)


class ProjectionCacheWriter:
    def __init__(self, cache_dir: str, store: str = "latent", overwrite: bool = False):
        assert store in ("latent", "rgb")
        self.cache_dir = ensure_dir(cache_dir)
        self.store = store
        self.overwrite = overwrite

    def sample_dir(self, sample_id: str) -> str:
        return os.path.join(self.cache_dir, sample_id)

    def exists(self, sample_id: str) -> bool:
        return os.path.isfile(os.path.join(self.sample_dir(sample_id), "metadata.json"))

    def write(
        self,
        sample_id: str,
        y: torch.Tensor,
        x_dagger: torch.Tensor,
        mask: torch.Tensor,
        metadata: dict,
        context: dict | None = None,
    ) -> str:
        d = ensure_dir(self.sample_dir(sample_id))
        save_tensor(y, os.path.join(d, "y.pt"))
        save_tensor(x_dagger, os.path.join(d, "x_dagger.pt"))
        save_tensor(mask, os.path.join(d, "mask.pt"))
        if context is not None:
            torch.save({k: v.detach().cpu() for k, v in context.items()}, os.path.join(d, "context.pt"))
        meta = dict(metadata)
        meta["store"] = self.store
        save_json(meta, os.path.join(d, "metadata.json"))
        return d
