"""P1 条件响应诊断：固定25帧/同噪声，对照真实可见序列与首帧重复序列。

本脚本只消费遮蔽后的 visible RGB，GT 不进入 CLIP、RAFT、VAE 或传播。
输出是工程诊断，不是正式评价；两组各自重算 flow 与参考图，响应差不等于质量收益。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from motion_proj.worldsim_v81.infer_p1 import (  # noqa: E402
    FRAMES, SIZE, _flows_for_pairs, _official_pipeline_class, _unet_noise,
    load_sequence, select_reference_frame_indices,
)
from motion_proj.worldsim_v81.model_bridge import (  # noqa: E402
    DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
    encode_video, load_components, load_trainable_state,
)
from motion_proj.worldsim_v81.reference_propagation import (  # noqa: E402
    PROTOCOL, build_reference_plan, propagate_reference_latents,
    static_fcnet_pair_masks,
)
from motion_proj.worldsim_v81.train_p1 import FORMAT  # noqa: E402

SEQUENCE_ID = "00f88c4f0a"
SIDE_RATIO = 0.33
DEFAULT_DATA_ROOT = Path(
    "/root/autodl-tmp/data/worldsim_v81/youtube_vos_2019/valid/JPEGImages"
)


def repeated_first_frames(images: list[Image.Image]) -> list[Image.Image]:
    if len(images) != FRAMES:
        raise ValueError(f"诊断需要恰好{FRAMES}帧遮蔽输入")
    return [images[0].copy() for _ in range(FRAMES)]


def reference_plan_from_visible(images: list[Image.Image], edges: tuple[int, int]):
    if len(images) != FRAMES:
        raise ValueError(f"参考图需要恰好{FRAMES}帧")
    centers = [image.crop((edges[0], 0, edges[1], SIZE)) for image in images]
    refs = select_reference_frame_indices(centers, 4)
    return build_reference_plan(FRAMES, refs)


def masked_model_tensor(images: list[Image.Image], hole: torch.Tensor) -> torch.Tensor:
    if len(images) != FRAMES or hole.shape != (1, FRAMES, 1, SIZE, SIZE):
        raise ValueError("可见帧或洞区mask形状错误")
    rgb = np.stack([np.asarray(image.convert("RGB")) for image in images]).copy()
    visible = torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0)
    visible = visible.to(device=hole.device, dtype=torch.float32) / 127.5 - 1
    return visible * (1 - hole)


def draw_shared_randomness(pipeline, scheduler, visible: torch.Tensor,
                           embedding: torch.Tensor, *, seed: int, steps: int,
                           unet_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """匹配短窗正式路径的 RNG 消费：增强噪声一次，再抽全局初始latent一次。"""
    generator = torch.Generator(device=visible.device).manual_seed(seed)
    observation_noise = torch.randn(
        visible.shape, generator=generator, device=visible.device,
        dtype=visible.dtype,
    )
    scheduler.set_timesteps(steps, device=visible.device)
    initial = pipeline.prepare_latents(
        1, FRAMES, unet_channels, SIZE, SIZE, embedding.dtype,
        visible.device, generator,
    )
    scheduler.set_timesteps(steps, device=visible.device)
    expected = (1, FRAMES, unet_channels // 2, SIZE // 8, SIZE // 8)
    if initial.shape != expected:
        raise ValueError(f"初始latent形状{tuple(initial.shape)}，预期{expected}")
    return observation_noise, initial


@torch.no_grad()
def decode_unscaled_latent(pipeline, latent: torch.Tensor,
                           *, chunk: int = 8) -> torch.Tensor:
    """传播条件未乘VAE scaling_factor；先乘回再用官方 decode_latents。"""
    return decode_scaled_latent(
        pipeline, latent * pipeline.vae.config.scaling_factor, chunk=chunk,
    )


@torch.no_grad()
def decode_scaled_latent(pipeline, latent: torch.Tensor,
                         *, chunk: int = 8) -> torch.Tensor:
    """扩散输出已带SVD缩放；按官方8帧块直接解码。"""
    if latent.ndim != 5 or latent.shape[:3] != (1, FRAMES, 4) or chunk < 1:
        raise ValueError("latent须为[1,25,4,H,W]，chunk须为正")
    decoded_parts = []
    for start in range(0, FRAMES, chunk):
        piece = latent[:, start:start + chunk].to(next(pipeline.vae.parameters()).device)
        decoded = pipeline.decode_latents(
            piece, num_frames=piece.shape[1], decode_chunk_size=chunk,
        )[0]
        if decoded.ndim != 4 or decoded.shape[1] != piece.shape[1]:
            raise ValueError("VAE条件解码帧数不符")
        decoded_parts.append(
            ((decoded.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1).cpu()
        )
    return torch.cat(decoded_parts, dim=0)


def rgb_tensor_to_frames(video: torch.Tensor) -> torch.Tensor:
    if video.shape != (1, FRAMES, 3, SIZE, SIZE):
        raise ValueError("RGB张量须为[1,25,3,256,256]")
    return ((video[0].detach().float().cpu() + 1) / 2).clamp(0, 1)


def save_frames(folder: Path, frames: torch.Tensor) -> None:
    if frames.shape != (FRAMES, 3, SIZE, SIZE) or not torch.isfinite(frames).all():
        raise ValueError("待保存帧须为有限的[25,3,256,256]")
    folder.mkdir(parents=True, exist_ok=False)
    thumbs = []
    for index, frame in enumerate(frames):
        array = (frame.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
        image = Image.fromarray(array)
        image.save(folder / f"{index:05d}.png")
        thumbs.append(image.resize((192, 192), Image.Resampling.BICUBIC))
    sheet = Image.new("RGB", (5 * 192, 5 * 216), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(thumbs):
        x, y = (index % 5) * 192, (index // 5) * 216
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + 194), f"f{index:02d}", fill="black")
    sheet.save(folder.parent / f"{folder.name}_contact.png")


def _shape_stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    return {
        "shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
        "min": float(value.min()), "max": float(value.max()),
        "rms": float(value.square().mean().sqrt()),
    }


@torch.no_grad()
def build_branch_conditions(components, visible: torch.Tensor, hole: torch.Tensor,
                            plan, observation_noise: torch.Tensor) -> dict[str, torch.Tensor]:
    if visible.shape != observation_noise.shape or visible.shape[:2] != (1, FRAMES):
        raise ValueError("两分支增强噪声必须逐位置复用")
    raw_flow = _flows_for_pairs(components.raft, visible, list(plan.pairs))
    dilated = F.max_pool2d(hole.flatten(0, 1), 3, 1, 1).reshape_as(hole)
    pair_masks = static_fcnet_pair_masks(dilated, plan)
    flow_pred, _ = components.fcnet.forward_bidirect_flow(raw_flow, pair_masks)
    flows = components.fcnet.combine_flow(raw_flow, flow_pred, pair_masks)
    augmented = visible + 0.02 * observation_noise
    pre = encode_video(components.vae, augmented, scaled=False, sample=False)
    past, future, fused = propagate_reference_latents(
        components.propagator, pre, flows[0], flows[1], hole, plan,
    )
    del raw_flow, dilated, pair_masks, flow_pred, flows
    plain = encode_video(components.vae, visible, scaled=False, sample=False)
    result = {
        "vae_plain": plain.cpu(), "pre_propagation_augmented": pre.cpu(),
        "propagated_past": past.cpu(), "propagated_future": future.cpu(),
        "propagated_fused": fused.cpu(),
    }
    if any(value.shape != (1, FRAMES, 4, SIZE // 8, SIZE // 8)
           or not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("条件latent形状或有限性检查失败")
    return result


@torch.no_grad()
def sample_from_shared_initial(components, condition: torch.Tensor,
                               embedding: torch.Tensor, time_ids: torch.Tensor,
                               initial: torch.Tensor, *, steps: int,
                               amp_dtype: torch.dtype) -> torch.Tensor:
    if condition.shape != initial.shape or condition.shape[1] != FRAMES:
        raise ValueError("传播条件须与共享初始latent形状一致")
    scheduler = components.scheduler
    scheduler.set_timesteps(steps, device=initial.device)
    latents = initial.clone()
    guidance = torch.linspace(1.0, 3.0, FRAMES, device=initial.device,
                              dtype=initial.dtype).view(1, FRAMES, 1, 1, 1)
    for timestep in scheduler.timesteps:
        prediction = _unet_noise(
            components, latents, condition, embedding, time_ids, timestep,
            scheduler, cfg=True, amp_dtype=amp_dtype,
        )
        unconditional, conditional = prediction.chunk(2)
        guided = unconditional + guidance * (conditional - unconditional)
        latents = scheduler.step(guided, timestep, latents).prev_sample
    if not torch.isfinite(latents).all():
        raise FloatingPointError("最终latent非有限值")
    return latents.cpu()


def per_frame_delta(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    if left.shape != right.shape or left.shape[0] != FRAMES:
        raise ValueError("对照帧数或形状不一致")
    return [float(value) for value in (left - right).abs().float().flatten(1).mean(1)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="paper_bidirectional_m4 的 step000100 checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="仓库外全新诊断目录")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.output_dir.resolve().is_relative_to(REPO):
        raise ValueError("steps须为正，诊断输出须在仓库外")
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有诊断目录: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("真实诊断需由GPU串行调度；CPU仅做脚本测试")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("format") != FORMAT or checkpoint.get("step") != 100
            or checkpoint.get("propagation_protocol") != PROTOCOL):
        raise ValueError("仅接受 paper-bidirectional-m4 的 step000100 P1 checkpoint")

    paths, target, normal_images, edges, normal_refs, _ = load_sequence(
        args.data_root, SEQUENCE_ID, SIDE_RATIO,
    )
    del target  # GT只在load_sequence内部形成遮蔽输入；从此不进入QUERY。
    repeated_images = repeated_first_frames(normal_images)
    branch_images = {"normal_visible_25": normal_images,
                     "repeat_first_after_f0": repeated_images}
    plans = {name: reference_plan_from_visible(images, edges)
             for name, images in branch_images.items()}
    if list(plans["normal_visible_25"].refs) != normal_refs:
        raise AssertionError("正常分支参考帧与正式load_sequence不一致")

    device = torch.device("cuda")
    hole = torch.zeros(1, FRAMES, 1, SIZE, SIZE, device=device)
    hole[..., :edges[0]] = 1
    hole[..., edges[1]:] = 1
    visible = {name: masked_model_tensor(images, hole)
               for name, images in branch_images.items()}
    if not torch.equal(visible["normal_visible_25"][:, 0],
                       visible["repeat_first_after_f0"][:, 0]):
        raise AssertionError("两分支首帧输入不一致")

    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                 args.external, device="cuda")
    load_trainable_state(components, checkpoint["models"])
    del checkpoint  # 原训练断点含optimizer状态，加载权重后尽早释放CPU内存。
    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet):
        module.eval()
    pipeline_class = _official_pipeline_class(args.external)
    pipeline = pipeline_class(
        vae=components.vae, image_encoder=components.image_encoder,
        unet=components.unet, scheduler=components.scheduler,
        feature_extractor=components.feature_extractor, fix_raft=components.raft,
        vo_flow_complete=components.fcnet, lat_bi_propagator=components.propagator,
    )
    # 与正式短窗一致：首帧遮蔽PIL进入CLIP，其他帧不进入CLIP。
    embedding = pipeline._encode_image(normal_images[0], device, 1, False)
    time_ids = torch.tensor([[6, 127, 0.02]], device=device, dtype=embedding.dtype)
    # 只抽一次增强噪声和初始latent。mode()编码无随机消费；两分支共享这些张量。
    observation_noise, initial = draw_shared_randomness(
        pipeline, components.scheduler, visible["normal_visible_25"], embedding,
        seed=args.seed, steps=args.steps,
        unet_channels=components.unet.config.in_channels,
    )

    args.output_dir.mkdir(parents=True)
    torch.save({"observation_noise": observation_noise.cpu(),
                "initial_latent": initial.cpu(), "clip": embedding.cpu(),
                "time_ids": time_ids.cpu()}, args.output_dir / "shared_inputs.pt")
    conditions = {}
    for name, images in branch_images.items():
        branch_dir = args.output_dir / name
        branch_dir.mkdir()
        conditions[name] = build_branch_conditions(
            components, visible[name], hole, plans[name], observation_noise,
        )
        torch.save(conditions[name], branch_dir / "condition_latents_unscaled.pt")
        rgb = torch.from_numpy(np.stack([np.asarray(image) for image in images]).copy())
        rgb = rgb.permute(0, 3, 1, 2).float() / 255
        save_frames(branch_dir / "masked_visible_rgb", rgb)
        save_frames(branch_dir / "model_visible_rgb", rgb_tensor_to_frames(visible[name]))
        save_frames(branch_dir / "augmented_model_input_rgb",
                    rgb_tensor_to_frames(visible[name] + 0.02 * observation_noise))

    del visible, observation_noise, hole  # 后续只用已保存的条件latent。
    for module in (components.raft, components.fcnet, components.propagator,
                   components.image_encoder):
        module.to("cpu")
    torch.cuda.empty_cache()
    decoded = {}
    latent_stages = (
        ("vae_reconstruction_no_aug", "vae_plain"),
        ("pre_propagation_augmented", "pre_propagation_augmented"),
        ("propagated_past", "propagated_past"),
        ("propagated_future", "propagated_future"),
        ("propagated_fused", "propagated_fused"),
    )
    for name in branch_images:
        decoded[name] = {}
        for display_name, latent_name in latent_stages:
            frames = decode_unscaled_latent(
                pipeline, conditions[name][latent_name], chunk=8,
            )
            save_frames(args.output_dir / name / display_name, frames)
            decoded[name][display_name] = frames
    components.vae.to("cpu")
    torch.cuda.empty_cache()

    initial_copy = initial.clone()
    final_latents = {}
    for name in branch_images:
        final_latents[name] = sample_from_shared_initial(
            components, conditions[name]["propagated_fused"].to(device),
            embedding, time_ids, initial, steps=args.steps,
            amp_dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp],
        )
    initial_identical = bool(torch.equal(initial, initial_copy))
    if not initial_identical:
        raise AssertionError("采样过程修改了共享初始latent")
    components.unet.to("cpu")
    components.vae.to(device)
    torch.cuda.empty_cache()
    for name in branch_images:
        frames = decode_scaled_latent(pipeline, final_latents[name])
        save_frames(args.output_dir / name / "native_final", frames)
        decoded[name]["native_final"] = frames

    normal, repeat = "normal_visible_25", "repeat_first_after_f0"
    stage_delta = {
        stage: per_frame_delta(decoded[normal][stage], decoded[repeat][stage])
        for stage in (*[display for display, _ in latent_stages], "native_final")
    }
    latent_delta = {
        key: [float(value) for value in
              (conditions[normal][key] - conditions[repeat][key]).abs()
              .float().flatten(2).mean(2)[0]]
        for key in conditions[normal]
    }
    report = {
        "status": "complete", "diagnostic_only": True,
        "quality_benefit_from_response": "not_inferred",
        "checkpoint": str(args.checkpoint), "checkpoint_step": 100,
        "sequence_id": SEQUENCE_ID, "source_frames": [str(path) for path in paths],
        "frame_selection": "first_25_sorted_start_0", "side_ratio": SIDE_RATIO,
        "mask_left_pixels": edges[0], "mask_right_pixels": SIZE - edges[1],
        "seed": args.seed, "steps": args.steps, "amp": args.amp,
        "noise_aug_strength": 0.02, "vae_encode": "mode; unscaled latent",
        "intermediate_decode": "multiply unscaled latent by VAE scaling_factor before pipeline.decode_latents",
        "shared_first_frame_clip": True, "shared_time_ids": True,
        "shared_observation_noise_exact": True,
        "shared_initial_latent_exact": initial_identical,
        "sampling": "paper-feedforward 25-frame standard CFG Euler; same initial latent and schedule",
        "query_source": "only masked visible RGB; GT not passed to models or saved here",
        "branch_flow_and_references": "independently recomputed; response differences are not an isolated quality effect",
        "branches": {
            name: {"reference_indices": list(plans[name].refs),
                   "flow_pairs": [list(pair) for pair in plans[name].pairs],
                   "condition_stats": {key: _shape_stats(value)
                                       for key, value in conditions[name].items()}}
            for name in branch_images
        },
        "pixel_mae_by_frame_between_branches": stage_delta,
        "latent_mae_by_frame_between_branches": latent_delta,
        "limitations": "Single fixed validation window; CLIP sees the same first frame, but later RGB, RAFT flow and reference graph differ. Response magnitude is not image quality or causal attribution to one component.",
    }
    (args.output_dir / "diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"event": "p1_condition_diagnostic_prepared",
                      "report": str(args.output_dir / "diagnostic.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
