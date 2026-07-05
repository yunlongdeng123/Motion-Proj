"""Motion-Proj 的配置加载。

配置文件为纯 OmegaConf YAML。训练配置可以声明一个 ``defaults_chain``
列表（相对路径），这些路径会在配置自身的键之前按顺序合并，从而在不引入
额外依赖的情况下模拟轻量级的 Hydra 风格组合。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _resolve_relative(base_file: str, rel: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(base_file), rel))


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """加载配置文件，解析其 ``defaults_chain`` 以及命令行覆盖项。

    Args:
        path: （叶子）yaml 配置的路径。
        overrides: 可选的 dotlist 覆盖项，例如 ``["train.lr=1e-4"]``。
    """
    path = os.path.abspath(path)
    leaf = OmegaConf.load(path)

    merged = OmegaConf.create({})
    for rel in leaf.get("defaults_chain", []) or []:
        parent_path = _resolve_relative(path, rel)
        merged = OmegaConf.merge(merged, load_config(parent_path))

    leaf = OmegaConf.create({k: v for k, v in leaf.items() if k != "defaults_chain"})
    merged = OmegaConf.merge(merged, leaf)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))

    OmegaConf.resolve(merged)
    return merged  # type: ignore[return-value]


@dataclass
class ResolvedPaths:
    data_root: str
    cache_dir: str
    ckpt_dir: str
    log_dir: str


def get_paths(cfg: DictConfig) -> ResolvedPaths:
    p = cfg.paths
    paths = ResolvedPaths(
        data_root=p.data_root,
        cache_dir=p.cache_dir,
        ckpt_dir=p.ckpt_dir,
        log_dir=p.log_dir,
    )
    for d in (paths.cache_dir, paths.ckpt_dir, paths.log_dir):
        os.makedirs(d, exist_ok=True)
    return paths


def to_container(cfg: Any) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/train/motionproj_v1.yaml")
    print(OmegaConf.to_yaml(cfg))
