"""单一训练片段容量诊断；产物不能作为独立验证集或论文指标。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from .infer_p1 import _official_pipeline_class, generate, save_outputs
from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           encode_video, load_components, load_trainable_state,
                           time_ids, trainable_state)
from .p1_data import YouTubeVOSP1Dataset
from .reference_propagation import build_reference_plan, propagate_reference_latents, static_fcnet_pair_masks
from .train_p1 import (DEFAULT_DATA, FORMAT, _official_image_embedding, _official_util,
                       raft_flows_for_pairs, reference_indices_from_visible, train_step_p1,
                       validate_resume_protocol)

VIDEO_ID = "0fc958cde2"
CLIP_START = 2
SOURCE_STEP = 100
MAX_UPDATES = 64
PROBE_FORMAT = "worldsim_v81_fixed_clip_capacity_probe_v1"


def validate_probe_contract(state: dict, profile: dict, batch: dict, *,
                            data_root: Path, updates: int) -> str:
    """在装载模型前阻止断点、数据或预算静默漂移。"""
    if not 1 <= updates <= MAX_UPDATES:
        raise ValueError(f"容量诊断只允许 1..{MAX_UPDATES} 次更新")
    if state.get("format") != FORMAT or state.get("step") != SOURCE_STEP:
        raise ValueError("只接受 paper-bidirectional-m4 的第100步 P1 checkpoint")
    validate_resume_protocol(state, "paper-bidirectional-m4")
    if state.get("optimizer_name") != "Adam":
        raise ValueError("容量诊断必须沿用 Adam 优化器")
    groups = state.get("optimizer", {}).get("param_groups", [])
    if not groups or any(float(group["lr"]) != 1e-5 or
                         float(group["weight_decay"]) != 0 for group in groups):
        raise ValueError("checkpoint 优化器必须为 lr=1e-5、weight_decay=0")
    saved_args = state.get("args", {})
    if Path(saved_args.get("data_root", "/missing")).resolve() != data_root.resolve():
        raise ValueError("容量诊断必须使用第100步相同的数据根")
    if float(saved_args.get("lr", -1)) != 1e-5:
        raise ValueError("checkpoint 学习率不匹配")
    amp = saved_args.get("amp")
    if amp not in {"bf16", "fp16"}:
        raise ValueError("checkpoint 精度模式必须是 bf16 或 fp16")
    saved_profile = state.get("data_profile", {})
    if any(saved_profile.get(key) != profile.get(key)
           for key in ("videos", "available_windows")):
        raise ValueError("数据池与 checkpoint 不一致")
    if batch["video_id"] != [VIDEO_ID] or int(batch["clip_start"][0]) != CLIP_START:
        raise ValueError("固定训练片段必须是 0fc958cde2/start2")
    return amp


def _rgb_image(frame: torch.Tensor) -> Image.Image:
    rgb = ((frame.detach().float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
    return Image.fromarray(np.rint(rgb).clip(0, 255).astype(np.uint8))


def query_images_from_visible(batch: dict) -> tuple[list[Image.Image], list[Image.Image], tuple[int, int]]:
    """QUERY 条件仅由已遮蔽 RGB 制作；GT 只返回给输出对照。"""
    visible, mask = batch["visible_rgb"][0], batch["hole_mask"][0]
    known = torch.where(mask[0, 0, 0] == 0)[0]
    left, right = int(known[0]), int(known[-1]) + 1
    visible_images = []
    for frame, hole in zip(visible, mask):
        image = np.asarray(_rgb_image(frame)).copy()
        image[hole[0].bool().cpu().numpy()] = 0  # 官方 PIL 输入中洞区为黑色
        visible_images.append(Image.fromarray(image))
    targets = [_rgb_image(frame) for frame in batch["target_rgb"][0]]
    return targets, visible_images, (left, right)


@torch.no_grad()
def prepare_teacher_cache(components, batch: dict, *, seed: int, util) -> dict:
    """冻结固定 σ、噪声和 GT 条件；每次评估仅重跑可训练模块。"""
    device = torch.device("cuda")
    target = batch["target_rgb"].to(device)
    visible = batch["visible_rgb"].to(device)
    mask = batch["hole_mask"].to(device)
    refs = reference_indices_from_visible(batch["visible_rgb"], batch["hole_mask"])
    plan = build_reference_plan(25, refs)
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed + 1)  # VAE posterior sample 固定；独立于训练随机流
    gt_flow = raft_flows_for_pairs(components.raft, target, list(plan.pairs),
                                   iters=20, pair_chunk=2)
    target_latent = encode_video(components.vae, target, scaled=True, sample=True)
    cond_sigma = math.exp(-3.0)
    cond_noise = torch.randn(visible.shape, generator=generator, device=device)
    cond_latent = encode_video(components.vae, visible + cond_sigma * cond_noise,
                               scaled=False, sample=True)
    diffusion_noise = torch.randn(target_latent.shape, generator=generator, device=device)
    clip = _official_image_embedding(components, target, util)
    ids = time_ids(components.unet, 1, device, clip.dtype, cond_sigma)
    ids[:, 0] = 7
    return {"plan": plan, "flow": tuple(x.cpu() for x in gt_flow),
            "target_latent": target_latent.cpu(), "cond_latent": cond_latent.cpu(),
            "noise": diffusion_noise.cpu(), "clip": clip.cpu(), "time_ids": ids.cpu(),
            "mask": mask.cpu(), "sigma": math.exp(0.7), "cond_sigma": cond_sigma}


@torch.no_grad()
def build_teacher_condition(components, cache: dict, device: torch.device) -> torch.Tensor:
    """每次评估重跑 FCNet 和传播；只缓存冻结的 GT RAFT/VAE/CLIP。"""
    plan = cache["plan"]
    mask = cache["mask"].to(device)
    flow_gt = tuple(x.to(device) for x in cache["flow"])
    pair_masks = static_fcnet_pair_masks(mask, plan)
    predicted_flow, _ = components.fcnet.forward_bidirect_flow(flow_gt, pair_masks)
    flow = components.fcnet.combine_flow(flow_gt, predicted_flow, pair_masks)
    _, _, condition = propagate_reference_latents(
        components.propagator, cache["cond_latent"].to(device),
        flow[0], flow[1], mask, plan)
    return condition


@torch.no_grad()
def measure_teacher(components, cache: dict, *, amp: str) -> tuple[dict, torch.Tensor]:
    """带噪 GT x0 的单步去噪；这是 BUILD 容量值，不是自由生成质量。"""
    device = torch.device("cuda")
    components.fcnet.eval()
    components.propagator.eval()
    components.unet.eval()
    condition = build_teacher_condition(components, cache, device)
    mask = cache["mask"].to(device)
    sigma = torch.tensor(cache["sigma"], device=device).reshape(1, 1, 1, 1, 1)
    target = cache["target_latent"].to(device)
    noisy = target + sigma * cache["noise"].to(device)
    model_input = torch.cat((noisy / (1 + sigma.square()).sqrt(), condition), dim=2)
    with torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[amp]):
        predicted = components.unet(model_input, 0.25 * sigma.log().flatten(),
                                    cache["clip"].to(device),
                                    added_time_ids=cache["time_ids"].to(device)).sample
    denoised = predicted.float() * (-sigma / (1 + sigma.square()).sqrt()) + noisy / (1 + sigma.square())
    error = (denoised - target).square()
    holes = F.interpolate(mask.flatten(0, 1), size=error.shape[-2:], mode="nearest")
    holes = holes.reshape(mask.shape[0], mask.shape[1], 1, *error.shape[-2:])
    channels = error.shape[2]
    score = {"weighted_mse": float(((1 + sigma.square()) / sigma.square() * error).mean()),
             "hole_mse": float((error * holes).sum() / (holes.sum() * channels)),
             "known_mse": float((error * (1 - holes)).sum() / ((1 - holes).sum() * channels))}
    if not all(math.isfinite(value) for value in score.values()):
        raise FloatingPointError(f"teacher-forced 指标非有限值: {score}")
    return score, denoised.cpu()


@torch.no_grad()
def save_teacher_frames(pipeline, components, denoised: torch.Tensor,
                        batch: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    components.vae.to("cuda")
    decoded = pipeline.decode_latents(denoised.to("cuda"), num_frames=25,
                                      decode_chunk_size=8)[0]
    decoded = ((decoded.permute(1, 0, 2, 3).float().cpu() + 1) / 2).clamp(0, 1)
    components.vae.to("cpu")
    sheet = Image.new("RGB", (3 * 256, 2 * 282), "white")
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate((0, 12, 24)):
        prediction = Image.fromarray(np.rint(decoded[index].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        prediction.save(output_dir / f"denoised_f{index:02d}.png")
        gt = _rgb_image(batch["target_rgb"][0, index])
        draw.text((column * 256 + 8, 5), f"GT f{index:02d}", fill="black")
        draw.text((column * 256 + 8, 287), f"teacher denoised f{index:02d}", fill="black")
        sheet.paste(gt, (column * 256, 26))
        sheet.paste(prediction, (column * 256, 308))
    sheet.save(output_dir / "contact.jpg", quality=92)


@torch.no_grad()
def run_query(components, pipeline, batch: dict, *, refs: tuple[int, ...],
              seed: int, steps: int, amp: str, output_dir: Path) -> dict:
    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet):
        module.to("cuda")
    targets, visible_images, edges = query_images_from_visible(batch)
    visible = batch["visible_rgb"].to("cuda")
    hole = batch["hole_mask"].to("cuda")
    pairs = list(build_reference_plan(25, list(refs)).pairs)
    prediction = generate(components, pipeline, visible_images, visible, hole, pairs,
                          seed=seed, steps=steps, mode="paper-feedforward", amp=amp,
                          propagation_protocol="paper-bidirectional-m4", refs=list(refs))
    return save_outputs(output_dir, targets, visible_images, prediction, edges)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-steps", type=int, default=25)
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    return parser.parse_args(argv)


def save_probe_state(path: Path, components, optimizer, scheduler, scaler,
                     dropout_generator: torch.Generator, *, source_checkpoint: Path,
                     updates: int, seed: int, data_profile: dict,
                     input_frames: list[str], amp: str) -> None:
    """只保存探针复核状态；不同格式不能被正式 train_p1 恢复。"""
    state = {"format": PROBE_FORMAT, "source_checkpoint": str(source_checkpoint.resolve()),
             "source_step": SOURCE_STEP, "probe_updates": updates,
             "propagation_protocol": "paper-bidirectional-m4",
             "optimizer_name": "Adam", "amp": amp, "seed": seed,
             "data_profile": data_profile,
             "diagnostic_profile": {"video_id": VIDEO_ID, "clip_start": CLIP_START,
                                    "input_frames": input_frames,
                                    "teacher": "fixed_noisy_gt_gtflow_full_first_frame_clip",
                                    "query": "masked_visible_rgb_pure_gaussian_same_train_clip"},
             "models": trainable_state(components),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
             "cfg_rng": dropout_generator.get_state()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.query_steps < 1 or not 1 <= args.updates <= MAX_UPDATES:
        raise ValueError("query-steps 必须为正；updates 只允许 1..64")
    if not torch.cuda.is_available():
        raise RuntimeError("仅由主运行队列显式启动 GPU 容量诊断")
    dataset = YouTubeVOSP1Dataset(args.data_root)
    batch = default_collate([dataset[0]])
    video_index, start = dataset.choice(0)
    input_frames = [str(path) for path in dataset.videos[video_index][1][start:start + 25]]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    amp = validate_probe_contract(state, dataset.profile(), batch,
                                  data_root=args.data_root, updates=args.updates)
    if args.output_dir.exists():
        raise FileExistsError(f"独立容量诊断输出目录必须是新目录: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                  args.external, device="cuda")
    load_trainable_state(components, state["models"])
    components.unet.enable_gradient_checkpointing()
    pipeline_class = _official_pipeline_class(args.external)
    pipeline = pipeline_class(vae=components.vae, image_encoder=components.image_encoder,
                              unet=components.unet, scheduler=components.scheduler,
                              feature_extractor=components.feature_extractor,
                              fix_raft=components.raft, vo_flow_complete=components.fcnet,
                              lat_bi_propagator=components.propagator)
    util = _official_util(args.external)
    cache = prepare_teacher_cache(components, batch, seed=args.seed, util=util)
    metadata = {"status": "running", "kind": "fixed_training_clip_capacity_only",
                "source_checkpoint": str(args.checkpoint.resolve()), "source_step": SOURCE_STEP,
                "protocol": "paper-bidirectional-m4", "video_id": VIDEO_ID,
                "clip_start": CLIP_START, "updates": args.updates, "seed": args.seed,
                "query_seed": args.seed + 10, "query_steps": args.query_steps,
                "amp": amp, "optimizer": "Adam", "lr": 1e-5, "weight_decay": 0,
                "teacher_sigma": cache["sigma"], "teacher_cond_sigma": cache["cond_sigma"],
                "teacher_role": "fixed noisy GT x0; full GT RAFT flow and full GT first-frame CLIP; not QUERY",
                "query_role": "same training clip, visible RGB only to CLIP/RAFT/VAE; pure Gaussian paper-feedforward; not held-out",
                "formal_eval": False, "generalization_claim": False,
                "input_frames": input_frames,
                "reference_indices": list(cache["plan"].refs), "flow_pairs": list(cache["plan"].pairs)}
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    initial, decoded = measure_teacher(components, cache, amp=amp)
    save_teacher_frames(pipeline, components, decoded, batch, args.output_dir / "teacher_before")
    run_query(components, pipeline, batch, refs=cache["plan"].refs,
              seed=args.seed + 10, steps=args.query_steps, amp=amp,
              output_dir=args.output_dir / "query_before")
    for module in (components.fcnet, components.propagator, components.unet):
        module.to("cuda")
    parameters = [parameter for module in (components.unet, components.propagator, components.fcnet)
                  for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=0)
    optimizer.load_state_dict(state["optimizer"])
    from diffusers.optimization import get_scheduler
    scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=500,
                              num_training_steps=100_000)
    scheduler.load_state_dict(state["scheduler"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    scaler.load_state_dict(state["scaler"])
    dropout_generator = torch.Generator(device="cuda").manual_seed(args.seed + 20)
    rows = [{"update": 0, "teacher": initial}]
    log = args.output_dir / "updates.jsonl"
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[0]) + "\n")
        for update in range(1, args.updates + 1):
            step_metrics = train_step_p1(components, batch, optimizer, scaler, amp=amp,
                                         raft_iters=20, pair_chunk=2,
                                         propagation_protocol="paper-bidirectional-m4",
                                         util=util, dropout_generator=dropout_generator)
            scheduler.step()
            row = {"update": update, "train": step_metrics}
            if update % 16 == 0 or update == args.updates:
                teacher_score, decoded = measure_teacher(components, cache, amp=amp)
                row["teacher"] = teacher_score
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({"event": "capacity_update", **row}), flush=True)
    probe_state = args.output_dir / "probe_state_final.pt"
    save_probe_state(probe_state, components, optimizer, scheduler, scaler,
                     dropout_generator, source_checkpoint=args.checkpoint,
                     updates=args.updates, seed=args.seed,
                     data_profile=dataset.profile(), input_frames=input_frames, amp=amp)
    save_teacher_frames(pipeline, components, decoded, batch, args.output_dir / "teacher_after")
    run_query(components, pipeline, batch, refs=cache["plan"].refs,
              seed=args.seed + 10, steps=args.query_steps, amp=amp,
              output_dir=args.output_dir / "query_after")
    metadata["status"] = "complete"
    metadata["teacher_before"] = initial
    metadata["teacher_after"] = rows[-1]["teacher"]
    metadata["updates_log"] = str(log)
    metadata["probe_state"] = str(probe_state)
    metadata["probe_state_bytes"] = probe_state.stat().st_size
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
