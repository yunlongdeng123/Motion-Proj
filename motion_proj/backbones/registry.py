"""可扩展骨干注册表；OpenDWM adapter 无需修改 trainer。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .base import DiffusionBackbone

Builder = Callable[..., DiffusionBackbone]
_BUILDERS: dict[str, Builder] = {}


def register_backbone(name: str, builder: Builder, *, replace: bool = False) -> None:
    if name in _BUILDERS and not replace:
        raise KeyError(f"backbone 已注册: {name}")
    _BUILDERS[name] = builder


def _svd_builder(cfg, **kwargs):
    from .svd_backbone import build_svd_backbone

    return build_svd_backbone(cfg, **kwargs)


register_backbone("svd", _svd_builder)


def build_backbone(cfg: Any, load: bool = True, device: str = "cuda",
                   dtype=torch.bfloat16) -> DiffusionBackbone:
    name = str(cfg.name)
    try:
        builder = _BUILDERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backbone: {name}; registered={sorted(_BUILDERS)}") from exc
    backbone = builder(cfg, load=load, device=device, dtype=dtype)
    if backbone.parameterization not in backbone.capabilities.parameterizations:
        raise ValueError(f"backbone capability 未声明 parameterization={backbone.parameterization}")
    return backbone
