"""Seen-to-Scene P0：25 帧可见输入、三项官方损失、可恢复真实训练。"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .data import OutpaintingDataset
from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           completed_flows, conditioning, encode_video, image_embedding,
                           load_components, load_trainable_state, raft_flows, time_ids,
                           trainable_state, validate_batch)


def parameter_counts(components) -> dict:
    groups = {"vae": components.vae, "clip": components.image_encoder,
              "raft": components.raft, "fcnet": components.fcnet,
              "propagator": components.propagator, "svd_unet": components.unet}
    counts = {name: {"trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
                     "frozen": sum(p.numel() for p in module.parameters() if not p.requires_grad)}
              for name, module in groups.items()}
    counts["total"] = {kind: sum(row[kind] for row in counts.values())
                       for kind in ("trainable", "frozen")}
    return counts


def gradient_report(module: torch.nn.Module) -> dict:
    """真实、未缩放的可训练参数梯度；无梯度时不伪装成通过。"""
    parameters = [p for p in module.parameters() if p.requires_grad]
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return {"status": "missing", "norm": None, "finite": None, "nonzero": False,
                "tensors_with_grad": 0, "trainable_tensors": len(parameters)}
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    squared_norm = sum((g.float().square().sum() for g in grads),
                       torch.zeros((), device=grads[0].device))
    norm = float(squared_norm.sqrt()) if finite else None
    finite = finite and math.isfinite(norm)
    if not finite:
        norm = None
    nonzero = finite and norm > 0
    return {"status": "ok" if nonzero else "invalid", "norm": norm,
            "finite": finite, "nonzero": nonzero,
            "tensors_with_grad": len(grads), "trainable_tensors": len(parameters)}


def classify_propagator_gradient(report: dict, dropped: bool) -> dict:
    if dropped and report["finite"] is not False and not report["nonzero"]:
        return {**report, "status": "skipped_condition_dropout"}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--raft-pair-chunk", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    return parser.parse_args()


def train_step(components, batch: dict, optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler, *, amp: str, raft_iters: int,
               pair_chunk: int) -> dict:
    validate_batch(batch)
    device = next(components.unet.parameters()).device
    visible = batch["visible_rgb"].to(device, non_blocking=True)
    target = batch["target_rgb"].to(device, non_blocking=True)
    mask = batch["hole_mask"].to(device, non_blocking=True)
    components.fcnet.train()
    components.propagator.train()
    components.unet.train()
    # eval 冻结的子模型，避免 UNet 空间层/编码器发生状态更新。
    components.vae.eval()
    components.image_encoder.eval()
    components.raft.eval()
    optimizer.zero_grad(set_to_none=True)

    # teacher 仅用于监督；conditioned flow 仅从已遮蔽视频取得。
    teacher_flow = raft_flows(components.raft, target, iters=raft_iters, pair_chunk=pair_chunk)
    target_latent = encode_video(components.vae, target, scaled=True, sample=True)
    clip_embedding = image_embedding(components, visible)
    noise_strength = torch.randn((), device=device).mul_(0.5).add_(-3.0).exp()
    flow = completed_flows(components, visible, mask, raft_iters=raft_iters,
                           pair_chunk=pair_chunk)
    condition = conditioning(components, visible, mask, flow, noise_strength, sample=True)
    batch_size = target_latent.shape[0]
    # 官方 SVD 训练的条件丢弃，供推理时的 classifier-free guidance 使用。
    dropout = torch.rand(batch_size, device=device)
    clip_embedding = torch.where((dropout < 0.2)[:, None, None],
                                 torch.zeros_like(clip_embedding), clip_embedding)
    condition = torch.where(((dropout >= 0.1) & (dropout < 0.3))[:, None, None, None, None],
                            torch.zeros_like(condition), condition)
    sigma = torch.randn(batch_size, device=device).mul_(1.6).add_(0.7).exp()
    sigma = sigma[:, None, None, None, None]
    noisy = target_latent + torch.randn_like(target_latent) * sigma
    timestep = 0.25 * sigma.log().flatten()
    model_input = torch.cat((noisy / (sigma.square() + 1).sqrt(), condition), dim=2)
    added_time_ids = time_ids(components.unet, batch_size, device,
                              clip_embedding.dtype, noise_strength)
    cast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    with torch.autocast(device_type=device.type, dtype=cast_dtype,
                        enabled=device.type == "cuda" and cast_dtype is not None):
        predicted = components.unet(model_input, timestep, clip_embedding,
                                     added_time_ids=added_time_ids).sample
    denoised = predicted.float() * (-sigma / (sigma.square() + 1).sqrt()) + noisy / (sigma.square() + 1)
    weight = (1 + sigma.square()) / sigma.square()
    diffusion_loss = (weight * (denoised - target_latent).square()).mean()
    flow_l1, ternary_warp = components.flow_loss(flow, teacher_flow, mask, target)
    loss = diffusion_loss + flow_l1 + ternary_warp
    if not torch.isfinite(loss):
        raise FloatingPointError(f"训练损失非有限值: {loss.item()}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = {}
    for name, module in (("fcnet", components.fcnet), ("propagator", components.propagator),
                         ("svd_temporal", components.unet)):
        gradients[name] = gradient_report(module)
    dropped_propagation = bool(((dropout >= 0.1) & (dropout < 0.3)).all())
    gradients["propagator"] = classify_propagator_gradient(gradients["propagator"],
                                                            dropped_propagation)
    frozen = [p for module in (components.vae, components.image_encoder,
                               components.raft, components.unet)
              for p in module.parameters() if not p.requires_grad]
    frozen_grad_tensors = sum(p.grad is not None for p in frozen)
    if frozen_grad_tensors:
        raise RuntimeError(f"冻结组件出现梯度: {frozen_grad_tensors} 个张量")
    accepted = {"ok", "skipped_condition_dropout"}
    if any(report["status"] not in accepted for report in gradients.values()):
        raise RuntimeError(f"真实训练组件缺少有效梯度: {gradients}")
    scaler.step(optimizer)
    scaler.update()
    return {"loss": float(loss.detach()), "diffusion": float(diffusion_loss.detach()),
            "flow_l1": float(flow_l1.detach()), "ternary_warp": float(ternary_warp.detach()),
            "gradients": gradients, "frozen_grad_tensors": frozen_grad_tensors}


def save_checkpoint(path: Path, components, optimizer, scaler, step: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": "worldsim_v81_seen_to_scene_p0", "step": step,
             "models": trainable_state(components), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
             "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.save_every < 1:
        raise ValueError("max-steps 与 save-every 必须大于 0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("真实 SVD/RAFT/FCNet 训练需要 GPU")
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                  args.external, device=device)
    print(json.dumps({"event": "parameters", "counts": parameter_counts(components)}), flush=True)
    components.unet.enable_gradient_checkpointing()
    parameters = [p for module in (components.unet, components.propagator, components.fcnet)
                  for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("format") != "worldsim_v81_seen_to_scene_p0":
            raise ValueError("checkpoint 格式不匹配")
        if state["args"]["manifest"] != str(args.manifest):
            raise ValueError("恢复训练必须使用相同 manifest")
        load_trainable_state(components, state["models"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        step = int(state["step"])
    dataset = OutpaintingDataset(args.manifest, frames=25, size=(256, 256))
    while step < args.max_steps:
        epoch, offset = divmod(step, len(dataset))
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + epoch))
        batch = default_collate([dataset[int(order[offset])]])
        torch.cuda.reset_peak_memory_stats()
        metrics = train_step(components, batch, optimizer, scaler, amp=args.amp,
                             raft_iters=args.raft_iters, pair_chunk=args.raft_pair_chunk)
        metrics["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        metrics["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024 ** 3)
        step += 1
        print(json.dumps({"step": step, "video_id": batch["video_id"][0], **metrics}), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.output_dir / f"checkpoint-{step:06d}.pt",
                            components, optimizer, scaler, step, args)


if __name__ == "__main__":
    main()
