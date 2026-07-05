"""阶段 4：回放挖掘（方案第 9 节，阶段 4）。

从 *当前* 模型（无梯度）采样未来帧，离线对其进行审计，保留高误差
（高静态漂移）的生成结果，对其中可修复的样本做投影，并将它们加入
投影缓存。这弥合了合成腐蚀与真实模型失败之间的差距。

V1 范围：挖掘针对的是静态漂移类失败（生成帧上没有 GT 框，因此在未加入
学习式检测器之前，物体层是空的）。源片段的自车轨迹 / 标定被复用作为
条件生成时所期望的几何。基于学习式检测器的物体挖掘推迟到可选的
``motionproj-mm`` 环境中实现。
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from ..auditor import MotionAuditor
from ..auditor.state import MotionState
from ..backbones import build_backbone
from ..config import get_paths, load_config
from ..data import NuScenesFutureVideoDataset
from ..projector import DynamicsProjector
from ..projector.mask import downsample_mask_to_latent
from ..utils.logging import get_logger
from ..cache.writer import ProjectionCacheWriter

log = get_logger(__name__)


def _audit_generated(auditor: MotionAuditor, gen_frames: torch.Tensor, src_sample: dict) -> MotionState:
    """审计一个生成的片段，复用源片段的几何信息，不使用 GT 框。"""
    sample = {
        "frames": gen_frames,
        "boxes": [[] for _ in range(gen_frames.shape[0])],  # 生成帧上没有 GT
        "intrinsics": src_sample["intrinsics"],
        "cam2ego": src_sample["cam2ego"],
        "ego2global": src_sample["ego2global"],
        "sample_id": src_sample["sample_id"] + "_gen",
    }
    return auditor.audit(sample)


def mine(cfg, adapter: str | None, drift_thresh: float, max_samples: int) -> None:
    paths = get_paths(cfg)
    device = cfg.device
    ds = NuScenesFutureVideoDataset(cfg.data)
    backbone = build_backbone(cfg.model, load=True, device=device)
    if adapter:
        backbone.load_adapter(adapter)
    auditor = MotionAuditor(device=device)
    projector = DynamicsProjector()
    writer = ProjectionCacheWriter(paths.cache_dir, store=cfg.cache.store, overwrite=True)
    scale = int(cfg.model.vae_scale_factor)

    n = min(max_samples, len(ds)) if max_samples else len(ds)
    kept = 0
    for i in tqdm(range(n)):
        src = ds[i]
        gen = backbone.generate(src["cond_frame"], num_frames=int(cfg.data.num_frames))
        state = _audit_generated(auditor, gen, src)
        drift = auditor.static_drift_score(state)
        if drift < drift_thresh:
            continue  # 生成结果没问题，无需修复
        res = projector.project(gen, state)
        sid = src["sample_id"] + "_gen"
        if cfg.cache.store == "rgb":
            writer.write(sid, res.y.cpu(), res.x_dagger.cpu(), res.mask.cpu(), res.metadata)
        else:
            cond = backbone.build_conditioning({"cond_frame": src["cond_frame"].unsqueeze(0)})
            y_lat = backbone.encode(res.y.unsqueeze(0))[0]
            xd_lat = backbone.encode(res.x_dagger.unsqueeze(0))[0]
            mask_lat = downsample_mask_to_latent(res.mask.to(device), scale)
            context = {
                "image_embeds": cond.data["image_embeds"][0],
                "image_latents": cond.data["image_latents"][0],
                "added_time_ids": cond.data["added_time_ids"][0],
            }
            writer.write(sid, y_lat.cpu(), xd_lat.cpu(), mask_lat.cpu(), res.metadata, context)
        kept += 1
    log.info("Replay mining done: kept %d / %d generated clips (drift>=%.3f)", kept, n, drift_thresh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="path to trained LoRA adapter")
    ap.add_argument("--drift-thresh", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    mine(cfg, args.adapter, args.drift_thresh, args.max_samples)


if __name__ == "__main__":
    main()
