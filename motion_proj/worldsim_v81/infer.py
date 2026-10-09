"""Seen-to-Scene P0 短片推理：单 CAM_FRONT 25 帧，输出原始生成帧。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .masks import apply_hole_mask, make_border_outpaint_mask
from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           completed_flows, conditioning, image_embedding,
                           load_components, load_trainable_state, time_ids, validate_batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--raft-pair-chunk", type=int, default=2)
    parser.add_argument("--decode-chunk", type=int, default=4)
    return parser.parse_args()


def load_visible_clip(manifest: Path, video_id: str | None = None) -> tuple[str, dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = [row for row in rows if video_id is None or row["video_id"] == video_id]
    if not chosen:
        raise ValueError(f"清单中没有视频: {video_id}")
    row = chosen[0]
    if len(row["frames"]) < 25:
        raise ValueError("短片推理需要连续 25 帧")
    frames = []
    for filename in row["frames"][:25]:
        with Image.open(filename) as image:
            image = image.convert("RGB")
            scale = max(256 / image.width, 256 / image.height)
            width, height = round(image.width * scale), round(image.height * scale)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            x, y = (width - 256) // 2, (height - 256) // 2
            image = image.crop((x, y, x + 256, y + 256))
            frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1)
    target = torch.stack(frames).unsqueeze(0)
    mask = make_border_outpaint_mask(256, 256).expand(1, 25, 1, 256, 256)
    visible = apply_hole_mask(target, mask)
    # 不返回完整 RGB，后续推理路径没有真值引用。
    batch = {"visible_rgb": visible, "hole_mask": mask}
    validate_batch(batch, require_target=False)
    return row["video_id"], batch


@torch.inference_mode()
def generate(components, batch: dict, *, steps: int = 25, seed: int = 2026,
             raft_iters: int = 20, pair_chunk: int = 2, decode_chunk: int = 4) -> torch.Tensor:
    validate_batch(batch, require_target=False)
    if steps < 1 or decode_chunk < 1:
        raise ValueError("steps 和 decode_chunk 必须大于 0")
    device = next(components.unet.parameters()).device
    visible = batch["visible_rgb"].to(device)
    mask = batch["hole_mask"].to(device)
    for module in (components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet):
        module.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    clip = image_embedding(components, visible)
    flow = completed_flows(components, visible, mask, raft_iters=raft_iters, pair_chunk=pair_chunk)
    condition = conditioning(components, visible, mask, flow, 0.02)
    batch_size, frames, channels, height, width = condition.shape
    components.scheduler.set_timesteps(steps, device=device)
    latents = torch.randn((batch_size, frames, channels, height, width),
                          generator=generator, device=device) * components.scheduler.init_noise_sigma
    guidance = torch.linspace(1.0, 3.0, frames, device=device).reshape(1, frames, 1, 1, 1)
    added = time_ids(components.unet, batch_size, device, clip.dtype, 0.02)
    for timestep in components.scheduler.timesteps:
        scaled = components.scheduler.scale_model_input(latents, timestep)
        model_input = torch.cat((scaled, condition), dim=2)
        # 与官方 SVD 推理一致：空 CLIP/传播条件与可见条件做 classifier-free guidance。
        combined_input = torch.cat((torch.cat((scaled, torch.zeros_like(condition)), dim=2),
                                    model_input), dim=0)
        combined_clip = torch.cat((torch.zeros_like(clip), clip), dim=0)
        combined_time = torch.cat((added, added), dim=0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = components.unet(combined_input, timestep, combined_clip,
                                         added_time_ids=combined_time).sample
        uncond, cond = prediction.chunk(2)
        prediction = uncond + guidance * (cond - uncond)
        latents = components.scheduler.step(prediction, timestep, latents).prev_sample
    flat = (latents / components.vae.config.scaling_factor).flatten(0, 1)
    decoded = []
    for start in range(0, len(flat), decode_chunk):
        chunk = flat[start:start + decode_chunk]
        decoded.append(components.vae.decode(chunk, num_frames=len(chunk)).sample.float().cpu())
    return torch.cat(decoded).reshape(batch_size, frames, 3, 256, 256).clamp(-1, 1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("真实 SVD 短片推理需要 GPU")
    video_id, batch = load_visible_clip(args.manifest, args.video_id)
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                  args.external, device="cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "worldsim_v81_seen_to_scene_p0":
        raise ValueError("checkpoint 格式不匹配")
    load_trainable_state(components, state["models"])
    frames = generate(components, batch, steps=args.steps, seed=args.seed,
                      raft_iters=args.raft_iters, pair_chunk=args.raft_pair_chunk,
                      decode_chunk=args.decode_chunk)
    out = args.output_dir / video_id
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames[0]):
        array = ((frame.permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
        Image.fromarray(array).save(out / f"{i:05d}.png")
    (out / "run.json").write_text(json.dumps({"video_id": video_id, "frames": len(frames[0]),
                                                "steps": args.steps, "seed": args.seed,
                                                "checkpoint": str(args.checkpoint)}, indent=2),
                                   encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
