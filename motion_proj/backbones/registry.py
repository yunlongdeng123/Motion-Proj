"""骨干网络注册表，使训练器可以与具体骨干网络无关。"""
from __future__ import annotations

from typing import Any

import torch

from .base import DiffusionBackbone


def build_backbone(cfg: Any, load: bool = True, device: str = "cuda", dtype=torch.bfloat16) -> DiffusionBackbone:
    """根据 ``model`` 子配置构建一个骨干网络。"""
    name = cfg.name
    if name == "svd":
        from .svd_backbone import build_svd_backbone

        return build_svd_backbone(cfg, load=load, device=device, dtype=dtype)
    raise ValueError(f"unknown backbone: {name}")
