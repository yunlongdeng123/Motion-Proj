"""YouTube-VOS P1 训练入口：官方25帧采样与优化预算，复用真实组件训练步。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate

from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           encode_video, load_components, load_trainable_state,
                           raft_flows, time_ids, trainable_state, validate_batch)
from .p1_data import YouTubeVOSP1Dataset
from .train import classify_propagator_gradient, gradient_report, parameter_counts


FORMAT = "worldsim_v81_seen_to_scene_p1_youtube_vos"
DEFAULT_PROPAGATION_PROTOCOL = "reference-m4"
PROPAGATION_PROTOCOLS = ("reference-m4", "literal-allframes")
DEFAULT_DATA = Path("/root/autodl-tmp/data/worldsim_v81/youtube_vos_2019/train/JPEGImages")


def _official_util(external: Path):
    """官方采样和抗混叠实现，与固定 checkout 的 train.py 保持一致。"""
    import sys
    if str(external) not in sys.path:
        sys.path.insert(0, str(external))
    return importlib.import_module("utils.util")


@torch.no_grad()
def _official_image_embedding(components, target: torch.Tensor, util) -> torch.Tensor:
    image = util._resize_with_antialiasing(target[:, 0].float(), (224, 224))
    image = (image + 1) / 2
    pixels = components.feature_extractor(images=image.cpu(), do_normalize=True,
                                          do_center_crop=False, do_resize=False,
                                          do_rescale=False, return_tensors="pt").pixel_values
    pixels = pixels.to(device=target.device, dtype=next(components.image_encoder.parameters()).dtype)
    return components.image_encoder(pixels).image_embeds.unsqueeze(1)


def reference_pairs_from_visible(visible: torch.Tensor, mask: torch.Tensor) -> list[tuple[int, int]]:
    """只用可见中心选 m=4 参考链；复用 QUERY 的固定官方构造。"""
    if visible.shape[0] != 1 or visible.shape[1] != 25:
        raise ValueError("P1 参考链目前只支持 batch=1、25帧")
    if not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError("P1 参考链训练要求全视频静态 mask")
    known = torch.where(mask[0, 0, 0, 0] == 0)[0]
    if len(known) == 0 or not torch.equal(
        known, torch.arange(known[0], known[-1] + 1, device=known.device)
    ):
        raise ValueError("P1 可见区域必须是连续中央带")
    left, right = int(known[0]), int(known[-1]) + 1
    centers = []
    for frame in visible[0]:
        rgb = ((frame[:, :, left:right].permute(1, 2, 0).float() + 1) * 127.5)
        centers.append(Image.fromarray(rgb.round().clamp(0, 255).byte().cpu().numpy()))
    # infer_p1 顶层引用本模块的 FORMAT，故在函数中导入以免循环导入。
    from .infer_p1 import build_pairs_for_all_frames, select_reference_frame_indices
    refs = select_reference_frame_indices(centers, reference_window_size=4)
    chain, to_frame = build_pairs_for_all_frames(25, refs)
    pairs = chain + to_frame
    if len(pairs) != 24 or len({target for _, target in pairs}) != 24:
        raise ValueError("m=4 参考链必须为25帧构造24条唯一父边")
    return pairs


@torch.no_grad()
def raft_flows_for_pairs(raft, target: torch.Tensor, pairs: list[tuple[int, int]],
                         *, iters: int, pair_chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    forward, backward = [], []
    for start in range(0, len(pairs), pair_chunk):
        fw, bw = raft.forward_pairs(target, pairs[start:start + pair_chunk],
                                    iters=iters, bidirectional=True)
        forward.append(fw)
        backward.append(bw)
    return torch.cat(forward, dim=1), torch.cat(backward, dim=1)


def paired_flow_loss(flow_loss, completed, ground_truth, masks: torch.Tensor,
                     frames: torch.Tensor, pairs: list[tuple[int, int]]):
    """沿用官方 L1/ternary 公式，但让每条 flow 对应真实的 (source,target)。"""
    source = [s for s, _ in pairs]
    target = [t for _, t in pairs]
    height, width = frames.shape[-2:]
    losses = []
    warps = []
    for predicted, truth, mask, current, shift in (
        (completed[0], ground_truth[0], masks[:, source], frames[:, source], frames[:, target]),
        (completed[1], ground_truth[1], masks[:, target], frames[:, target], frames[:, source]),
    ):
        if predicted.shape != truth.shape or predicted.shape[1] != len(pairs):
            raise ValueError("参考对与光流数量不匹配")
        combined = predicted * mask + truth * (1 - mask)
        losses.append(flow_loss.l1_criterion(predicted * mask, truth * mask) / mask.mean()
                      + flow_loss.l1_criterion(predicted * (1 - mask), truth * (1 - mask))
                      / (1 - mask).mean())
        warps.append(flow_loss.ternary_loss(combined.reshape(-1, 2, height, width),
                                           truth.reshape(-1, 2, height, width),
                                           mask.reshape(-1, 1, height, width),
                                           current.reshape(-1, 3, height, width),
                                           shift.reshape(-1, 3, height, width)))
    return sum(losses), sum(warps)


def validate_resume_protocol(state: dict, requested: str) -> str:
    """旧版 P1 断点未记录协议，但其训练路径确定为 All Frames。"""
    previous = state.get("propagation_protocol", "literal-allframes")
    if previous not in PROPAGATION_PROTOCOLS:
        raise ValueError(f"断点传播协议未知: {previous}")
    if previous != requested:
        raise ValueError(f"禁止跨传播协议恢复: 断点={previous}, 本次={requested}; 应从原始权重新启训练")
    return previous


def train_step_p1(components, batch: dict, optimizer: torch.optim.Optimizer,
                  scaler: torch.amp.GradScaler, *, amp: str, raft_iters: int,
                  pair_chunk: int, propagation_protocol: str, util,
                  dropout_generator: torch.Generator) -> dict:
    """保留公开训练条件；传播在论文 m=4 与公开 All Frames 间显式选择。"""
    validate_batch(batch)
    if propagation_protocol not in PROPAGATION_PROTOCOLS:
        raise ValueError(f"未知传播协议: {propagation_protocol}")
    pairs = (reference_pairs_from_visible(batch["visible_rgb"], batch["hole_mask"])
             if propagation_protocol == "reference-m4" else None)
    device = next(components.unet.parameters()).device
    target = batch["target_rgb"].to(device, non_blocking=True)
    visible = batch["visible_rgb"].to(device, non_blocking=True)
    mask = batch["hole_mask"].to(device, non_blocking=True)
    components.fcnet.train()
    components.propagator.train()
    components.unet.train()
    for module in (components.vae, components.image_encoder, components.raft):
        module.eval()
        module.to(device)
    optimizer.zero_grad(set_to_none=True)

    # 对齐官方 train.py 顺序：GT flow -> FCNet -> GT VAE -> noise -> cond sigma
    # -> masked VAE -> latent propagation -> diffusion sigma -> CLIP -> CFG dropout。
    gt_flow = (raft_flows_for_pairs(components.raft, target, pairs,
                                    iters=raft_iters, pair_chunk=pair_chunk)
               if pairs is not None else
               raft_flows(components.raft, target, iters=raft_iters, pair_chunk=pair_chunk))
    flow_pred, _ = components.fcnet.forward_bidirect_flow(gt_flow, mask)
    flow = components.fcnet.combine_flow(gt_flow, flow_pred, mask)
    target_latent = encode_video(components.vae, target, scaled=True, sample=True)
    batch_size = target_latent.shape[0]
    noise = torch.randn_like(target_latent)
    cond_sigma = util.rand_log_normal([batch_size], loc=-3.0, scale=0.5).to(target_latent)
    noise_aug_strength = cond_sigma[0]
    conditional_pixels = torch.randn_like(visible) * cond_sigma[:, None, None, None, None] + visible
    conditional_latent = encode_video(components.vae, conditional_pixels, scaled=False, sample=True)
    # 固定官方 LatentPropagation 当前签名不接收 orig_lats；仅修调用，不把 GT latent 塞入条件。
    _, _, condition = components.propagator(conditional_latent, flow[0], flow[1], mask,
                                             flow_pairs_info=pairs)
    sigma = util.rand_log_normal([batch_size], loc=0.7, scale=1.6).to(target_latent.device)
    sigma = sigma[:, None, None, None, None]
    noisy = noise * sigma + target_latent
    timestep = 0.25 * sigma.log().flatten()
    model_input = noisy / (sigma.square() + 1).sqrt()
    clip_embedding = _official_image_embedding(components, target, util)
    added_time_ids = time_ids(components.unet, batch_size, device,
                              clip_embedding.dtype, noise_aug_strength)
    added_time_ids[:, 0] = 7  # 官方训练常数，P0 桥接默认 6 不适用于 P1。
    # 官方代码使用独立 CUDA generator 对 CFG 进行可恢复采样。
    dropout = torch.rand(batch_size, device=device, generator=dropout_generator)
    clip_embedding = torch.where((dropout < 0.2)[:, None, None],
                                 torch.zeros_like(clip_embedding), clip_embedding)
    condition_dropped = (dropout >= 0.1) & (dropout < 0.3)
    condition = torch.where(condition_dropped[:, None, None, None, None],
                            torch.zeros_like(condition), condition)
    model_input = torch.cat((model_input, condition), dim=2)
    for module in (components.vae, components.image_encoder, components.raft):
        if any(p.requires_grad for p in module.parameters()):
            raise RuntimeError("P1 观测编码器必须冻结")
        module.to("cpu")

    cast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    with torch.autocast(device_type=device.type, dtype=cast_dtype,
                        enabled=device.type == "cuda" and cast_dtype is not None):
        predicted = components.unet(model_input, timestep, clip_embedding,
                                     added_time_ids=added_time_ids).sample
    denoised = predicted.float() * (-sigma / (sigma.square() + 1).sqrt()) + noisy / (sigma.square() + 1)
    weight = (1 + sigma.square()) / sigma.square()
    diffusion_loss = (weight * (denoised - target_latent).square()).flatten(1).mean(1).mean()
    flow_l1, ternary_warp = (paired_flow_loss(components.flow_loss, flow, gt_flow,
                                             mask, target, pairs)
                            if pairs is not None else
                            components.flow_loss(flow, gt_flow, mask, target))
    loss = diffusion_loss + flow_l1 + ternary_warp
    if not torch.isfinite(loss):
        raise FloatingPointError(f"P1 训练损失非有限值: {float(loss)}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = {name: gradient_report(module) for name, module in
                 (("fcnet", components.fcnet), ("propagator", components.propagator),
                  ("svd_temporal", components.unet))}
    gradients["propagator"] = classify_propagator_gradient(
        gradients["propagator"], bool(condition_dropped.all()))
    frozen = [p for module in (components.vae, components.image_encoder,
                               components.raft, components.unet)
              for p in module.parameters() if not p.requires_grad]
    frozen_grad_tensors = sum(p.grad is not None for p in frozen)
    if frozen_grad_tensors:
        raise RuntimeError(f"P1 冻结参数出现梯度: {frozen_grad_tensors}")
    if any(row["status"] not in {"ok", "skipped_condition_dropout"}
           for row in gradients.values()):
        raise RuntimeError(f"P1 真实训练组件缺少有效梯度: {gradients}")
    scaler.step(optimizer)
    scaler.update()
    return {"loss": float(loss.detach()), "diffusion": float(diffusion_loss.detach()),
            "flow_l1": float(flow_l1.detach()), "ternary_warp": float(ternary_warp.detach()),
            "gradients": gradients, "frozen_grad_tensors": frozen_grad_tensors,
            "frozen_observation_encoders_offloaded": True,
            "train_conditioning": "official_full_rgb_flow_and_first_frame_clip",
            "propagation_protocol": propagation_protocol}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA,
                        help="完整 YouTube-VOS train/JPEGImages 或其 train 父目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--amp", choices=("fp16", "bf16", "none"), default="fp16")
    parser.add_argument("--propagation-protocol", choices=PROPAGATION_PROTOCOLS,
                        default=DEFAULT_PROPAGATION_PROTOCOL)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--raft-pair-chunk", type=int, default=2)
    parser.add_argument("--inspect-data-only", action="store_true",
                        help="CPU 扫描训练视频池并输出采样统计，不加载模型")
    return parser.parse_args()


def save_checkpoint(path: Path, components, optimizer, scheduler, scaler,
                    dropout_generator: torch.Generator,
                    step: int, args: argparse.Namespace, data_profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": FORMAT, "propagation_protocol": args.propagation_protocol,
             "step": step, "models": trainable_state(components),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
             "python_rng": random.getstate(), "cfg_rng": dropout_generator.get_state(),
             "data_profile": data_profile,
             "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def trim_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(output_dir.glob("p1-checkpoint-*.pt"))
    for old in checkpoints[:-keep]:
        old.unlink()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.max_steps > 100_000 or args.save_every < 1 or args.keep_checkpoints < 1:
        raise ValueError("P1 max-steps 在 1..100000，save-every/keep-checkpoints 为正")
    if args.lr != 1e-5:
        raise ValueError("P1 固定官方学习率 1e-5")
    dataset = YouTubeVOSP1Dataset(args.data_root, seed=args.seed, samples=100_000)
    profile = dataset.profile()
    print(json.dumps({"event": "p1_data", **profile}), flush=True)
    if args.inspect_data_only:
        return
    from diffusers.optimization import get_scheduler

    if not torch.cuda.is_available():
        raise RuntimeError("真实 P1 训练需要 GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                  args.external, device="cuda")
    components.unet.enable_gradient_checkpointing()
    print(json.dumps({"event": "parameters", "counts": parameter_counts(components)}), flush=True)
    parameters = [p for module in (components.unet, components.propagator, components.fcnet)
                  for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=1e-2, eps=1e-8)
    # 官方 config 选择 constant；其默认 warmup500 在此调度类型下不生效。
    scheduler = get_scheduler("constant", optimizer=optimizer,
                              num_warmup_steps=500, num_training_steps=100_000)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    util = _official_util(args.external)
    dropout_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT:
            raise ValueError("仅可恢复 P1 原始权重训练断点；不可载入 P0/微调成品")
        validate_resume_protocol(state, args.propagation_protocol)
        if state["args"]["data_root"] != str(args.data_root) or state["args"]["seed"] != args.seed:
            raise ValueError("恢复时 YouTube-VOS 数据根和 seed 必须一致")
        if state["data_profile"]["videos"] != profile["videos"] or \
           state["data_profile"]["available_windows"] != profile["available_windows"]:
            raise ValueError("恢复时视频池数量或连续窗口数变化")
        load_trainable_state(components, state["models"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        dropout_generator.set_state(state["cfg_rng"])
        step = int(state["step"])
        print(json.dumps({"event": "resume", "step": step, "checkpoint": str(args.resume)}), flush=True)
    while step < args.max_steps:
        started = time.monotonic()
        batch = default_collate([dataset[step]])
        torch.cuda.reset_peak_memory_stats()
        metrics = train_step_p1(components, batch, optimizer, scaler, amp=args.amp,
                                raft_iters=args.raft_iters, pair_chunk=args.raft_pair_chunk,
                                propagation_protocol=args.propagation_protocol,
                                util=util, dropout_generator=dropout_generator)
        scheduler.step()
        step += 1
        metrics["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        metrics["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024 ** 3)
        metrics["wall_seconds"] = time.monotonic() - started
        print(json.dumps({"event": "train_step", "step": step, "video_id": batch["video_id"][0],
                          "clip_start": int(batch["clip_start"][0]), "lr": scheduler.get_last_lr()[0],
                          **metrics}), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.output_dir / f"p1-checkpoint-{step:06d}.pt", components,
                            optimizer, scheduler, scaler, dropout_generator, step, args, profile)
            trim_checkpoints(args.output_dir, args.keep_checkpoints)


if __name__ == "__main__":
    main()
