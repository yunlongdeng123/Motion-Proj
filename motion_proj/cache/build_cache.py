"""第 2 阶段命令行工具：构建投影缓存。

对每个片段：审计 -> 投影 -> （可选地编码为 latent）-> 写入。
整个过程都不计算梯度。当 ``cache.store=latent`` 时，使用骨干网络的
VAE/条件模块来存储 latent + 条件（体积更小、训练更快）。当
``cache.store=rgb`` 时无需骨干网络，latent 在训练时再计算。
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from ..auditor import MotionAuditor
from ..config import get_paths, load_config
from ..data import NuScenesFutureVideoDataset
from ..projector import DynamicsProjector
from ..projector.mask import downsample_mask_to_latent
from ..utils.logging import get_logger
from .writer import ProjectionCacheWriter

log = get_logger(__name__)


def build_cache(cfg) -> None:
    paths = get_paths(cfg)
    device = cfg.device
    store = cfg.cache.store
    writer = ProjectionCacheWriter(paths.cache_dir, store=store, overwrite=cfg.cache.overwrite)

    dataset = NuScenesFutureVideoDataset(cfg.data)
    auditor = MotionAuditor(device=device, enable_depth=True)
    projector = DynamicsProjector()

    backbone = None
    if store == "latent":
        from ..backbones import build_backbone

        backbone = build_backbone(cfg.model, load=True, device=device)
        scale = int(cfg.model.vae_scale_factor)

    n = len(dataset)
    if cfg.cache.get("max_samples"):
        n = min(n, int(cfg.cache.max_samples))
    log.info("Building %s cache for %d clips -> %s", store, n, paths.cache_dir)

    for i in tqdm(range(n)):
        sample = dataset[i]
        sid = sample["sample_id"]
        if writer.exists(sid) and not cfg.cache.overwrite:
            continue

        state = auditor.audit(sample)
        res = projector.project(sample["frames"].to(device), state)

        if store == "rgb":
            writer.write(
                sid, res.y.cpu(), res.x_dagger.cpu(), res.mask.cpu(), res.metadata
            )
        else:
            batch = {"cond_frame": sample["cond_frame"].unsqueeze(0)}
            cond = backbone.build_conditioning(batch)
            y_lat = backbone.encode(res.y.unsqueeze(0))[0]
            xd_lat = backbone.encode(res.x_dagger.unsqueeze(0))[0]
            mask_lat = downsample_mask_to_latent(res.mask.to(device), scale)
            context = {
                "image_embeds": cond.data["image_embeds"][0],
                "image_latents": cond.data["image_latents"][0],
                "added_time_ids": cond.data["added_time_ids"][0],
            }
            writer.write(sid, y_lat.cpu(), xd_lat.cpu(), mask_lat.cpu(), res.metadata, context)

    log.info("Cache build complete.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. cache.store=rgb")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    build_cache(cfg)


if __name__ == "__main__":
    main()
