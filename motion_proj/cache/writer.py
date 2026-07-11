"""逐样本临时写入、校验后原子提交的 projection cache。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import torch

from ..runtime.atomic import atomic_directory, atomic_write_json
from ..utils.io import ensure_dir

CACHE_SCHEMA_VERSION = 2
COMPLETE = "COMPLETE"


class ProjectionCacheWriter:
    def __init__(self, cache_dir: str, store: str = "latent", overwrite: bool = False,
                 fingerprint: str | None = None):
        if store not in ("latent", "rgb"):
            raise ValueError("store 必须是 latent 或 rgb")
        self.cache_dir = ensure_dir(cache_dir)
        self.store = store
        self.overwrite = bool(overwrite)
        self.fingerprint = fingerprint

    def sample_dir(self, sample_id: str) -> str:
        return os.path.join(self.cache_dir, sample_id)

    def exists(self, sample_id: str) -> bool:
        directory = self.sample_dir(sample_id)
        try:
            with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            return False
        required = ["y.pt", "x_dagger.pt", "mask.pt", "metadata.json", COMPLETE]
        if self.store == "latent":
            required.append("context.pt")
        return (
            all(os.path.isfile(os.path.join(directory, name)) for name in required)
            and meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and (self.fingerprint is None or meta.get("cache_fingerprint") == self.fingerprint)
        )

    @staticmethod
    def _validate(y: torch.Tensor, x_dagger: torch.Tensor, mask: torch.Tensor, context: dict | None) -> None:
        if y.shape != x_dagger.shape:
            raise ValueError("y 与 x_dagger shape 不一致")
        if mask.shape[0] != y.shape[0] or mask.shape[-2:] != y.shape[-2:]:
            raise ValueError("mask 与目标的时间/空间 shape 不一致")
        for name, tensor in (("y", y), ("x_dagger", x_dagger), ("mask", mask)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} 包含 NaN/Inf")
        if bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ValueError("mask 超出 [0,1]")
        if context is not None and not all(bool(torch.isfinite(v).all()) for v in context.values()):
            raise ValueError("context 包含 NaN/Inf")

    def write(self, sample_id: str, y: torch.Tensor, x_dagger: torch.Tensor, mask: torch.Tensor,
              metadata: dict, context: dict | None = None) -> str:
        if self.exists(sample_id) and not self.overwrite:
            return self.sample_dir(sample_id)
        self._validate(y, x_dagger, mask, context)
        target = self.sample_dir(sample_id)
        if os.path.exists(target):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            os.replace(target, f"{target}.stale-{stamp}")
        with atomic_directory(target) as tmp:
            torch.save(y.detach().cpu(), os.path.join(tmp, "y.pt"))
            torch.save(x_dagger.detach().cpu(), os.path.join(tmp, "x_dagger.pt"))
            torch.save(mask.detach().cpu(), os.path.join(tmp, "mask.pt"))
            if context is not None:
                torch.save({key: value.detach().cpu() for key, value in context.items()}, os.path.join(tmp, "context.pt"))
            meta = dict(metadata)
            meta.update({"store": self.store, "cache_schema_version": CACHE_SCHEMA_VERSION,
                         "cache_fingerprint": self.fingerprint})
            atomic_write_json(os.path.join(tmp, "metadata.json"), meta)
            with open(os.path.join(tmp, COMPLETE), "w", encoding="utf-8") as handle:
                handle.write((self.fingerprint or "unversioned") + "\n")
        return target
