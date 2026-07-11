"""第 2 阶段命令行工具：构建投影缓存。

对每个片段：审计 -> 投影 -> （可选地编码为 latent）-> 写入。
整个过程都不计算梯度。当 ``cache.store=latent`` 时，使用骨干网络的
VAE/条件模块来存储 latent + 条件（体积更小、训练更快）。当
``cache.store=rgb`` 时无需骨干网络，latent 在训练时再计算。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..auditor import MotionAuditor
from ..config import cache_config_fingerprint, cache_stage_fingerprint, get_paths, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..eval.synthetic_corrupt_nuscenes import project_synthetic_sample
from ..projector import DynamicsProjector
from ..projector.mask import downsample_mask_to_latent
from ..runtime.fingerprint import environment_fingerprint, git_state
from ..utils.logging import get_logger
from ..runtime.stage import StageManifest
from .writer import ProjectionCacheWriter

log = get_logger(__name__)


def _flow_to_resolution(flow: torch.Tensor, confidence: torch.Tensor, height: int, width: int):
    """把像素空间 RAFT flow 缩放到 cache 分辨率并保持像素单位一致。"""
    source_h, source_w = flow.shape[1:3]
    value = F.interpolate(flow.permute(0, 3, 1, 2), size=(height, width),
                          mode="bilinear", align_corners=True)
    value[:, 0] *= width / source_w
    value[:, 1] *= height / source_h
    conf = F.interpolate(confidence, size=(height, width), mode="area").clamp(0, 1)
    return value.permute(0, 2, 3, 1).contiguous(), conf


def build_cache(cfg) -> None:
    paths = get_paths(cfg)
    device = cfg.device
    store = cfg.cache.store
    source = str(cfg.cache.get("source", "synthetic"))
    fingerprint = cache_config_fingerprint(cfg)
    stage_fingerprint = cache_stage_fingerprint(cfg)
    writer = ProjectionCacheWriter(paths.cache_dir, store=store, overwrite=cfg.cache.overwrite,
                                   fingerprint=fingerprint)
    stage = StageManifest(os.path.join(paths.cache_dir, "_stage"), "cache", stage_fingerprint)
    if stage.is_complete() and not cfg.cache.overwrite:
        log.info("Cache stage fingerprint 已完成，跳过: %s", fingerprint[:12])
        return
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 cache 构建拒绝在 dirty worktree 上运行")
    stage.begin(
        {
            "store": store,
            "source": source,
            "sample_fingerprint": fingerprint,
            "git": git,
            "environment": environment_fingerprint(),
            "data_split": f"{cfg.data.version}:{cfg.data.get('split', 'all')}",
            "command": list(sys.argv),
        }
    )
    save_resolved_config(cfg, os.path.join(paths.cache_dir, "_stage", "resolved.yaml"))

    dataset = NuScenesFutureVideoDataset(cfg.data)
    auditor = MotionAuditor(device=device, enable_depth=True) if source == "synthetic" else None
    projector = DynamicsProjector() if source == "synthetic" else None

    backbone = None
    if store == "latent":
        from ..backbones import build_backbone

        backbone = build_backbone(cfg.model, load=True, device=device)
        scale = int(cfg.model.vae_scale_factor)

    n = len(dataset)
    if cfg.cache.get("max_samples"):
        n = min(n, int(cfg.cache.max_samples))
    log.info("Building %s cache for %d clips -> %s", store, n, paths.cache_dir)

    try:
        for i in tqdm(range(n)):
            sample = dataset[i]
            sid = sample["sample_id"]
            if writer.exists(sid) and not cfg.cache.overwrite:
                continue

            clean = sample["frames"]
            if source == "synthetic":
                assert auditor is not None and projector is not None
                payload = project_synthetic_sample(
                    sample,
                    auditor,
                    projector,
                    case_index=i,
                    seed=int(cfg.seed),
                    settings=cfg.cache,
                    clip_index=i,
                )
                y = payload.pop("y_corrupted")
                target = payload.pop("x_dagger")
                mask = payload.pop("mask")
                flow = payload.pop("latent_flow")
                flow_confidence = payload.pop("flow_confidence")
                metadata = payload
            elif source == "clean":
                y = clean
                target = clean
                mask = torch.zeros(
                    clean.shape[0],
                    1,
                    clean.shape[-2],
                    clean.shape[-1],
                    dtype=clean.dtype,
                )
                metadata = {"sample_id": sid, "source": "clean"}
                flow = None
                flow_confidence = None
            else:
                raise NotImplementedError("replay cache 尚未接入；不得回退为 clean projection")

            if store == "rgb":
                writer.write(
                    sid,
                    y.cpu(),
                    target.cpu(),
                    mask.cpu(),
                    metadata,
                    clean=clean.cpu(),
                    latent_flow=flow.cpu() if flow is not None else None,
                    flow_confidence=flow_confidence.cpu() if flow_confidence is not None else None,
                    source=source,
                    generation_seed=int(cfg.seed),
                    source_fingerprint=fingerprint,
                )
            else:
                batch = {"cond_frame": sample["cond_frame"].unsqueeze(0)}
                cond = backbone.build_conditioning(batch)
                clean_lat = backbone.encode(clean.to(device).unsqueeze(0))[0]
                y_lat = backbone.encode(y.to(device).unsqueeze(0))[0]
                xd_lat = backbone.encode(target.to(device).unsqueeze(0))[0]
                mask_lat = downsample_mask_to_latent(mask.to(device), scale)
                flow_lat, confidence_lat = (None, None)
                if flow is not None and flow_confidence is not None:
                    flow_lat, confidence_lat = _flow_to_resolution(
                        flow.to(device), flow_confidence.to(device), y_lat.shape[-2], y_lat.shape[-1]
                    )
                context = {
                    "image_embeds": cond.data["image_embeds"][0],
                    "image_latents": cond.data["image_latents"][0],
                    "added_time_ids": cond.data["added_time_ids"][0],
                }
                writer.write(
                    sid,
                    y_lat.cpu(),
                    xd_lat.cpu(),
                    mask_lat.cpu(),
                    metadata,
                    context,
                    clean=clean_lat.cpu(),
                    latent_flow=flow_lat.cpu() if flow_lat is not None else None,
                    flow_confidence=confidence_lat.cpu() if confidence_lat is not None else None,
                    source=source,
                    generation_seed=int(cfg.seed),
                    source_fingerprint=fingerprint,
                )
        stage.complete({"samples": n})
    except Exception as exc:
        stage.fail(repr(exc))
        raise

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
