"""Seen-to-Scene P1：固定官方组件、m=4 参考帧、25 帧可见输入推理。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           encode_video, load_components, load_trainable_state,
                           official_modules)
from .train_p1 import FORMAT


FRAMES = 25
SIZE = 256
REFERENCE_WINDOW = 4


def effective_amp(mode: str, requested: str) -> str:
    """公开反演内部硬编码 FP16，外层须同精度以免 autocast 缓存混型。"""
    return "fp16" if mode == "literal-public" else requested


def compute_structure_term(im1_gray: np.ndarray, im2_gray: np.ndarray) -> float:
    """逐式对应固定官方 test.py 的结构项，不把遮蔽区域带入选择。"""
    x = im1_gray.astype(np.float32) / 255.0
    y = im2_gray.astype(np.float32) / 255.0
    mu_x = x.mean()
    mu_y = y.mean()
    x_c = x - mu_x
    y_c = y - mu_y
    sigma_x = np.sqrt((x_c ** 2).mean() + 1e-8)
    sigma_y = np.sqrt((y_c ** 2).mean() + 1e-8)
    sigma_xy = (x_c * y_c).mean()
    C3 = 1e-4
    denom = sigma_x * sigma_y + C3
    if denom == 0:
        return 0.0
    return float((sigma_xy + C3) / denom)


def select_reference_frame_indices(images: list[Image.Image], reference_window_size: int = 4) -> list[int]:
    total = len(images)
    if total == 0:
        return []
    grays = [np.array(image.convert("L")) for image in images]
    selected = [0]
    current = 0
    while current < total - 1:
        remaining = total - 1 - current
        if remaining < reference_window_size:
            selected.append(total - 1)
            break
        best_idx = None
        best_score = None
        for index in range(current + 1, current + reference_window_size + 1):
            score = compute_structure_term(grays[current], grays[index])
            if best_score is None or score < best_score:
                best_score = score
                best_idx = index
        selected.append(best_idx)
        current = best_idx
    if selected[-1] != total - 1:
        selected.append(total - 1)
    return selected


def build_pairs_for_all_frames(total: int, ref_indices: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    refs = sorted(ref_indices)
    if not refs:
        raise ValueError("至少需要一个参考帧")
    pairs_chain = [(refs[j], refs[j - 1]) for j in range(len(refs) - 1, 0, -1)]
    pairs_to_frame = []
    for target in range(total):
        segment_ref = next((reference for reference in refs if reference >= target), refs[-1])
        if segment_ref != target:
            pairs_to_frame.append((segment_ref, target))
    return pairs_chain, pairs_to_frame


def _resize_center_crop(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    scale = max(SIZE / image.width, SIZE / image.height)
    width, height = round(image.width * scale), round(image.height * scale)
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    x, y = (width - SIZE) // 2, (height - SIZE) // 2
    return image.crop((x, y, x + SIZE, y + SIZE))


def load_sequence(data_root: Path, sequence_id: str, side_ratio: float):
    if side_ratio not in (0.125, 0.33):
        raise ValueError("P1 side_ratio 仅支持 0.125 或 0.33")
    folder = data_root / sequence_id
    if not folder.is_dir():
        raise FileNotFoundError(f"序列不存在: {folder}")
    paths = sorted(path for path in folder.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) < FRAMES:
        raise ValueError(f"序列 {sequence_id} 仅有 {len(paths)} 帧，P1 必须有连续 25 帧")
    paths = paths[:FRAMES]
    target = []
    for path in paths:
        with Image.open(path) as image:
            target.append(_resize_center_crop(image))
    left = round(side_ratio * SIZE)
    right = round((1 - side_ratio) * SIZE)
    visible = []
    for image in target:
        array = np.asarray(image).copy()
        array[:, :left] = 0
        array[:, right:] = 0
        visible.append(Image.fromarray(array))
    # 只允许可见中心参与参考选择。隐藏两侧 RGB 不进入任何模型条件。
    centers = [image.crop((left, 0, right, SIZE)) for image in visible]
    refs = select_reference_frame_indices(centers, REFERENCE_WINDOW)
    chain, to_frame = build_pairs_for_all_frames(FRAMES, refs)
    return paths, target, visible, (left, right), refs, chain + to_frame


def _official_pipeline_class(external: Path):
    official_modules(external)
    path = external / "test.py"
    if not path.is_file():
        raise FileNotFoundError(f"官方推理源码缺失: {path}")
    spec = importlib.util.spec_from_file_location("seen_to_scene_public_test_p1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StableVideoDiffusionPipeline


@torch.no_grad()
def _flows_for_pairs(raft, visible: torch.Tensor, pairs: list[tuple[int, int]], chunk: int = 2):
    forward, backward = [], []
    for offset in range(0, len(pairs), chunk):
        subset = pairs[offset:offset + chunk]
        flow_fw, flow_bw = raft.forward_pairs(visible, subset, iters=20, bidirectional=True)
        forward.append(flow_fw)
        backward.append(flow_bw)
    return torch.cat(forward, dim=1), torch.cat(backward, dim=1)


def _unet_noise(components, latents, condition, image_embedding, time_ids,
                timestep, scheduler, *, cfg: bool, amp_dtype: torch.dtype):
    input_latents = torch.cat((latents, latents)) if cfg else latents
    input_latents = scheduler.scale_model_input(input_latents, timestep)
    input_condition = torch.cat((torch.zeros_like(condition), condition)) if cfg else condition
    inputs = torch.cat((input_latents, input_condition), dim=2)
    embeds = torch.cat((torch.zeros_like(image_embedding), image_embedding)) if cfg else image_embedding
    ids = time_ids.repeat(2, 1) if cfg else time_ids
    with torch.autocast("cuda", dtype=amp_dtype):
        return components.unet(inputs, timestep, embeds, added_time_ids=ids).sample.float()


@torch.no_grad()
def generate(components, pipeline, visible_images: list[Image.Image],
             visible: torch.Tensor, hole: torch.Tensor, pairs: list[tuple[int, int]],
             *, seed: int, steps: int, mode: str, amp: str):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    cast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet):
        module.eval()
    if mode == "literal-public":
        # 直接运行固定公开 test.py。保留其中 B=1→2 的 inversion 广播、
        # 其后 B=2 去噪及仅解码第一个样本，绝不重解释为标准 CFG。
        mask = (hole[0, 0, 0].cpu().numpy() * 255).astype(np.uint8)
        masks = [Image.fromarray(mask, mode="L") for _ in range(FRAMES)]
        # test.py 的 inversion 内部固定 autocast(fp16)。外层也用 FP16，
        # 并禁用跨嵌套上下文的权重缓存，防止 BF16/Half F.linear 混型。
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            output = pipeline.__call_seen_to_scene__(
                images=visible_images, masks=masks, refer_idx=pairs,
                height=SIZE, width=SIZE, window_size=FRAMES, stride=5,
                num_inference_steps=steps, min_guidance_scale=1.0,
                max_guidance_scale=3.0, fps=7, motion_bucket_id=127,
                noise_aug_strength=0.02, decode_chunk_size=8,
                num_videos_per_prompt=1, generator=generator,
                output_type="pil", return_dict=True)
        return torch.from_numpy(np.stack([np.asarray(image) for image in output.frames]).copy()).permute(0, 3, 1, 2).float() / 255

    # 论文仅描述高斯噪声前向采样；这个显式可选路径修正 CFG 维度，
    # 不能当作公开 test.py 的精确复现结果。
    # 公开 pipeline 的 CLIP/PIL 与 VAE 解码辅助方法，输入严格为已遮蔽 RGB。
    image_embedding = pipeline._encode_image(visible_images[0], device, 1, False)
    flow_input = _flows_for_pairs(components.raft, visible, pairs)
    dilated = F.max_pool2d(hole.flatten(0, 1), 3, 1, 1).reshape_as(hole)
    flow_pred, _ = components.fcnet.forward_bidirect_flow(flow_input, dilated)
    flows = components.fcnet.combine_flow(flow_input, flow_pred, dilated)
    observation_noise = torch.randn(visible.shape, generator=generator, device=device,
                                    dtype=visible.dtype)
    condition_input = visible + 0.02 * observation_noise
    condition_latent = encode_video(components.vae, condition_input, scaled=False,
                                    sample=False)
    _, _, condition = components.propagator(condition_latent, flows[0], flows[1], hole,
                                            flow_pairs_info=pairs)
    time_ids = torch.tensor([[6, 127, 0.02]], device=device, dtype=image_embedding.dtype)
    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator):
        module.to("cpu")
    torch.cuda.empty_cache()

    scheduler = components.scheduler
    scheduler.set_timesteps(steps, device=device)
    latents = pipeline.prepare_latents(1, FRAMES, components.unet.config.in_channels,
                                       SIZE, SIZE, image_embedding.dtype, device, generator)
    scheduler.set_timesteps(steps, device=device)
    guidance = torch.linspace(1.0, 3.0, FRAMES, device=device).view(1, FRAMES, 1, 1, 1)
    for timestep in scheduler.timesteps:
        prediction = _unet_noise(components, latents, condition, image_embedding,
                                 time_ids, timestep, scheduler, cfg=True,
                                 amp_dtype=cast_dtype)
        uncond, cond = prediction.chunk(2)
        prediction = uncond + guidance * (cond - uncond)
        latents = scheduler.step(prediction, timestep, latents).prev_sample
    components.unet.to("cpu")
    components.vae.to(device)
    decoded = pipeline.decode_latents(latents, num_frames=FRAMES, decode_chunk_size=8)[0]
    return ((decoded.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1).cpu()


def _save_video(images: list[Image.Image], path: Path):
    imageio.mimsave(path, [np.asarray(image) for image in images], fps=7,
                    codec="libx264", quality=8)


def save_outputs(output_dir: Path, target: list[Image.Image], visible: list[Image.Image],
                 prediction: torch.Tensor, edges: tuple[int, int]) -> dict:
    left, right = edges
    prediction_images = [Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                         for frame in prediction]
    composite = []
    for visible_image, predicted_image in zip(visible, prediction_images):
        result = np.asarray(predicted_image).copy()
        observed = np.asarray(visible_image)
        result[:, left:right] = observed[:, left:right]
        composite.append(Image.fromarray(result))
    videos = {"gt": target, "visible": visible, "pred": prediction_images, "comp": composite}
    paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in videos.items():
        frame_dir = output_dir / f"frames_{name}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            frame.save(frame_dir / f"{index:05d}.png")
        mp4 = output_dir / f"{name}.mp4"
        _save_video(frames, mp4)
        paths[name] = {"frames": str(frame_dir), "mp4": str(mp4)}
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="train/valid/JPEGImages 或 DAVIS/JPEGImages/480p")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side-ratio", type=float, choices=(0.125, 0.33), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--mode", choices=("literal-public", "paper-feedforward"), default="literal-public")
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps 必须为正")
    if not torch.cuda.is_available():
        raise RuntimeError("P1 真实推理需要 CUDA")
    paths, target, visible_images, edges, refs, pairs = load_sequence(
        args.data_root, args.sequence_id, args.side_ratio)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("仅接受本项目 P1 原始组件训练 checkpoint")
    pipeline_class = _official_pipeline_class(args.external)
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                  args.external, device="cuda")
    load_trainable_state(components, checkpoint["models"])
    pipeline = pipeline_class(vae=components.vae, image_encoder=components.image_encoder,
                              unet=components.unet, scheduler=components.scheduler,
                              feature_extractor=components.feature_extractor,
                              fix_raft=components.raft, vo_flow_complete=components.fcnet,
                              lat_bi_propagator=components.propagator)
    rgb = np.stack([np.asarray(image) for image in visible_images])
    visible = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).unsqueeze(0).float().cuda() / 127.5 - 1
    # 洞区在 [-1,1] 约定下仍必须为 0，而不是黑像素的 -1。
    hole = torch.zeros(1, FRAMES, 1, SIZE, SIZE, device="cuda")
    hole[..., :edges[0]] = 1
    hole[..., edges[1]:] = 1
    visible *= 1 - hole
    started = time.monotonic()
    prediction = generate(components, pipeline, visible_images, visible, hole, pairs,
                          seed=args.seed, steps=args.steps, mode=args.mode, amp=args.amp)
    output = args.output_dir / args.sequence_id / f"side_{args.side_ratio:g}"
    saved = save_outputs(output, target, visible_images, prediction, edges)
    provenance = {
        "status": "complete", "sequence_id": args.sequence_id,
        "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"],
        "checkpoint_format": FORMAT, "data_root": str(args.data_root),
        "source_frames": [str(path) for path in paths], "frame_selection": "first_25_sorted_start_0",
        "preprocessing": "bicubic_resize_center_crop_256_square",
        "side_ratio_each": args.side_ratio, "mask_pixels_left": edges[0],
        "mask_pixels_right": SIZE - edges[1], "reference_window": REFERENCE_WINDOW,
        "reference_selection_input": "visible_center_only", "reference_indices": refs,
        "flow_pairs": pairs, "mode": args.mode, "steps": args.steps,
        "seed": args.seed, "amp": args.amp, "requested_amp": args.amp,
        "effective_amp": effective_amp(args.mode, args.amp), "fps": 7,
        "min_guidance": 1.0, "max_guidance": 3.0, "noise_aug_strength": 0.02,
        "motion_bucket_id": 127, "external_source": str(args.external),
        "external_revision": "2a9dfc9888e44c7fd00b08af41ef967ae46b6323",
        "mode_note": "direct fixed public test.py pipeline: B=1→2 inversion broadcast, B=2 denoise, first sample decode; internal inversion autocast fp16"
                     if args.mode == "literal-public" else "paper Gaussian feedforward with conventional CFG; public test.py uses inversion",
        "input_role": "only masked visible RGB reaches CLIP, RAFT, VAE condition, and reference selector; GT is saved for scoring",
        "output_paths": saved, "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"event": "p1_inference_complete", "run": str(output / "run.json"),
                      "checkpoint_step": checkpoint["step"]}), flush=True)


if __name__ == "__main__":
    main()
