#!/usr/bin/env python3
"""运行 scene-0230 全片段对象编辑、指标和可视化闭环。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as torch_f
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.dynamic_editing_v2.drivestudio_registry import require_token
from motion_proj.dynamic_editing_v2.pilot_metrics import (
    canonical_sha256,
    counterfactual_effect_mask,
    depth_order_diagnostic,
    finite_max,
    finite_mean,
    masked_mean_abs,
    masked_psnr,
)


CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
VARIANTS = ("original", "lateral_plus_1m", "delete")
DEPTH_MAX_M = 120.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rigid_state_hash(rigid) -> str:
    return tensor_digest(
        [
            ("_means", rigid._means),
            ("_scales", rigid._scales),
            ("_quats", rigid._quats),
            ("_features_dc", rigid._features_dc),
            ("_features_rest", rigid._features_rest),
            ("_opacities", rigid._opacities),
            ("point_ids", rigid.point_ids),
            ("instances_size", rigid.instances_size),
            ("instances_fv", rigid.instances_fv),
            ("instances_trans", rigid.instances_trans),
            ("instances_quats", rigid.instances_quats),
        ]
    )


def non_target_hash(rigid, model_index: int) -> str:
    gaussian_mask = rigid.point_ids[..., 0] != int(model_index)
    instance_mask = torch.arange(
        rigid.instances_trans.shape[1], device=rigid.instances_trans.device
    ) != int(model_index)
    return tensor_digest(
        [
            ("_means", rigid._means[gaussian_mask]),
            ("_scales", rigid._scales[gaussian_mask]),
            ("_quats", rigid._quats[gaussian_mask]),
            ("_features_dc", rigid._features_dc[gaussian_mask]),
            ("_features_rest", rigid._features_rest[gaussian_mask]),
            ("_opacities", rigid._opacities[gaussian_mask]),
            ("point_ids", rigid.point_ids[gaussian_mask]),
            ("instances_size", rigid.instances_size[instance_mask]),
            ("instances_fv", rigid.instances_fv[:, instance_mask]),
            ("instances_trans", rigid.instances_trans[:, instance_mask]),
            ("instances_quats", rigid.instances_quats[:, instance_mask]),
        ]
    )


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def move_actor_local_y(rigid, model_index: int, meters: float) -> int:
    valid = rigid.instances_fv[:, model_index].bool()
    rotations = quaternion_wxyz_to_matrix(rigid.instances_quats[:, model_index])
    lateral_axes = rotations[..., :, 1]
    with torch.no_grad():
        rigid.instances_trans[valid, model_index] += meters * lateral_axes[valid]
    return int(valid.sum().item())


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有产物: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def encode_depth(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(value) & (value > 0)
    encoded = np.zeros(value.shape, dtype=np.uint16)
    encoded[valid] = np.round(
        np.clip(value[valid], 0, DEPTH_MAX_M) / DEPTH_MAX_M * 65535.0
    ).astype(np.uint16)
    return encoded


def decode_depth(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float32)
    depth = value / 65535.0 * DEPTH_MAX_M
    depth[value <= 0] = np.nan
    return depth


def thumbnail(image: np.ndarray, width: int = 360) -> np.ndarray:
    value = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None].float()
    height = max(1, round(value.shape[-2] * width / value.shape[-1]))
    resized = torch_f.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()


def color_overlay(
    rgb: np.ndarray,
    source: np.ndarray,
    edited: np.ndarray,
    violation: np.ndarray | None = None,
) -> np.ndarray:
    result = np.asarray(rgb, dtype=np.float32).copy()
    colors = (
        (np.asarray(source, dtype=bool), np.array([255, 40, 40], dtype=np.float32)),
        (np.asarray(edited, dtype=bool), np.array([40, 100, 255], dtype=np.float32)),
    )
    for mask, color in colors:
        result[mask] = result[mask] * 0.45 + color * 0.55
    if violation is not None:
        mask = np.asarray(violation, dtype=bool)
        result[mask] = result[mask] * 0.25 + np.array([255, 230, 0]) * 0.75
    return np.clip(result, 0, 255).astype(np.uint8)


def diff_heatmap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.mean(
        np.abs(first.astype(np.float32) - second.astype(np.float32)), axis=-1
    ) / 255.0
    value = np.clip(difference * 8.0, 0, 1)
    return np.stack(
        [
            (255 * value).astype(np.uint8),
            (255 * np.sqrt(value)).astype(np.uint8),
            (255 * (1 - value) * 0.2).astype(np.uint8),
        ],
        axis=-1,
    )


def pairwise_signature(means: torch.Tensor, limit: int = 256) -> torch.Tensor:
    if means.shape[0] == 0:
        raise RuntimeError("所选 actor 没有 canonical Gaussian")
    count = min(limit, means.shape[0])
    indices = torch.linspace(
        0, means.shape[0] - 1, steps=count, device=means.device
    ).round().long()
    selected = means[indices]
    return torch.pdist(selected).detach().cpu()


def actor_transform_payload(rigid, model_index: int, *, deleted: bool) -> dict:
    valid = rigid.instances_fv[:, model_index].bool().detach().cpu().numpy()
    translations = rigid.instances_trans[:, model_index].detach().cpu().numpy()
    quaternions = rigid.instances_quats[:, model_index].detach().cpu().numpy()
    return {
        "deleted": deleted,
        "model_index": int(model_index),
        "valid_frame_count": int(valid.sum()),
        "frames": [
            {
                "frame": int(index),
                "valid": bool(valid[index]),
                "translation": translations[index].tolist(),
                "quaternion_wxyz": quaternions[index].tolist(),
            }
            for index in range(len(valid))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--instance-token", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--drivestudio-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-count", type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖 M4 输出目录: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    config = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    if list(config.data.pixel_source.cameras) != [0, 1, 2]:
        raise RuntimeError("M4 固定要求 DriveStudio cameras=[0,1,2]")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    actor = require_token(registry, args.instance_token)
    model_index = int(actor["rigid_model_index"])
    device = torch.device(args.device)
    dataset = DrivingDataset(data_cfg=config.data)
    trainer = import_str(config.trainer.type)(
        **config.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=config.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    checkpoint_before = sha256_file(args.checkpoint)
    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    valid_frames = torch.nonzero(
        rigid.instances_fv[:, model_index], as_tuple=False
    ).flatten().cpu().tolist()
    if valid_frames != list(range(dataset.num_img_timesteps)):
        raise RuntimeError(
            f"M4 固定 actor 必须覆盖完整 clip，实际为 {len(valid_frames)}/{dataset.num_img_timesteps}"
        )
    frames = [int(value) for value in valid_frames]
    if args.frame_count is not None:
        if args.frame_count <= 0 or args.frame_count > len(frames):
            raise ValueError("frame-count 必须位于 1 到完整 clip 帧数之间")
        frames = frames[: args.frame_count]
    cameras = list(range(dataset.pixel_source.num_cams))
    if cameras != [0, 1, 2]:
        raise RuntimeError(f"相机顺序不符合合同: {cameras}")

    original_state_hash = rigid_state_hash(rigid)
    original_non_target_hash = non_target_hash(rigid, model_index)
    target_points = rigid.point_ids[..., 0] == model_index
    original_pairwise = pairwise_signature(rigid._means[target_points])
    original_size = rigid.instances_size[model_index].detach().cpu()
    original_trans = rigid.instances_trans[:, model_index].detach().cpu().clone()
    original_quats = rigid.instances_quats[:, model_index].detach().cpu().clone()
    frame_manifest_path = args.output_dir / "frames.jsonl"
    variant_records: dict[str, dict] = {}

    for variant in VARIANTS:
        trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
        trainer.set_eval()
        rigid = trainer.models["RigidNodes"]
        if variant == "lateral_plus_1m":
            edited_frame_count = move_actor_local_y(rigid, model_index, 1.0)
        else:
            edited_frame_count = 0
        transform = actor_transform_payload(rigid, model_index, deleted=variant == "delete")
        if variant == "delete":
            rigid.remove_instances([model_index])
        state_hash = rigid_state_hash(rigid)
        non_target = non_target_hash(rigid, model_index)
        variant_dir = args.output_dir / "variants" / variant
        for name in ("rgb", "depth_u16", "opacity_u8", "dynamic_u8"):
            (variant_dir / name).mkdir(parents=True)
        video_dir = variant_dir / "videos"
        video_dir.mkdir()
        writers = {
            camera: imageio.get_writer(
                video_dir / f"{CAMERA_NAMES[camera]}.mp4",
                fps=args.fps,
                codec="libx264",
                quality=7,
                macro_block_size=None,
                ffmpeg_log_level="error",
            )
            for camera in cameras
        }
        try:
            for frame in frames:
                for camera in cameras:
                    image_index = frame * len(cameras) + camera
                    image_infos, camera_infos = dataset.full_image_set.get_image(
                        image_index, camera_downscale=1.0
                    )
                    image_infos = to_device(image_infos, device)
                    camera_infos = to_device(camera_infos, device)
                    with torch.inference_mode():
                        outputs = trainer(image_infos, camera_infos)
                    rgb_float = outputs["rgb"].detach().float().cpu().numpy()
                    depth = outputs["depth"].detach().float().cpu().numpy().squeeze()
                    opacity = outputs["opacity"].detach().float().cpu().numpy().squeeze()
                    dynamic = (
                        outputs["Dynamic_opacity"]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    if not np.isfinite(rgb_float).all():
                        raise RuntimeError(f"渲染出现非有限 RGB: {variant}/{frame}/{camera}")
                    rgb = (np.clip(rgb_float, 0, 1) * 255).round().astype(np.uint8)
                    stem = f"frame_{frame:03d}_camera_{camera}"
                    paths = {
                        "rgb": variant_dir / "rgb" / f"{stem}.png",
                        "depth": variant_dir / "depth_u16" / f"{stem}.png",
                        "opacity": variant_dir / "opacity_u8" / f"{stem}.png",
                        "dynamic": variant_dir / "dynamic_u8" / f"{stem}.png",
                    }
                    imageio.imwrite(paths["rgb"], rgb)
                    imageio.imwrite(paths["depth"], encode_depth(depth))
                    imageio.imwrite(
                        paths["opacity"],
                        (np.clip(opacity, 0, 1) * 255).round().astype(np.uint8),
                    )
                    imageio.imwrite(
                        paths["dynamic"],
                        (np.clip(dynamic, 0, 1) * 255).round().astype(np.uint8),
                    )
                    writers[camera].append_data(rgb)
                    row = {
                        "variant": variant,
                        "frame": frame,
                        "camera": camera,
                        "camera_name": CAMERA_NAMES[camera],
                        "image_index": image_index,
                        "rgb": str(paths["rgb"]),
                        "depth_u16": str(paths["depth"]),
                        "opacity_u8": str(paths["opacity"]),
                        "dynamic_u8": str(paths["dynamic"]),
                        "rgb_bytes": paths["rgb"].stat().st_size,
                    }
                    with frame_manifest_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            for writer in writers.values():
                writer.close()
        transform.update(
            {
                "variant": variant,
                "instance_token": args.instance_token,
                "actor_local_lateral_meters": 1.0 if variant == "lateral_plus_1m" else 0.0,
                "edited_valid_frame_count": edited_frame_count,
                "coordinate_contract": "RigidNodes actor-local +y transformed by per-frame quaternion into world coordinates",
            }
        )
        transform["transform_sha256"] = canonical_sha256(transform)
        atomic_json(variant_dir / "actor_transform.json", transform)
        variant_records[variant] = {
            "world_state_hash": state_hash,
            "non_target_state_hash": non_target,
            "transform": str(variant_dir / "actor_transform.json"),
            "videos": [str(video_dir / f"{name}.mp4") for name in CAMERA_NAMES],
        }

    # 所有 paired 指标从磁盘逐帧读取，避免重现上游将整段结果常驻内存的问题。
    common_masks = args.output_dir / "masks"
    for name in ("source_footprint", "edited_footprint"):
        (common_masks / name).mkdir(parents=True)
    for variant in VARIANTS:
        (args.output_dir / "variants" / variant / "target_mask").mkdir()

    metric_rows: list[dict] = []
    lpips_model = trainer.lpips
    lpips_model.eval()
    qa_count = min(12, len(frames))
    qa_frames = set(np.linspace(0, len(frames) - 1, qa_count, dtype=int).tolist())
    qa_dir = args.output_dir / "qa"
    thumb_dir = qa_dir / "thumbs"
    thumb_dir.mkdir(parents=True)
    qa_rows = []
    visible_by_camera = {name: 0 for name in CAMERA_NAMES}
    for position, frame in enumerate(frames):
        for camera in cameras:
            stem = f"frame_{frame:03d}_camera_{camera}"
            images = {
                variant: imageio.imread(
                    args.output_dir / "variants" / variant / "rgb" / f"{stem}.png"
                )
                for variant in VARIANTS
            }
            source = counterfactual_effect_mask(images["original"], images["delete"])
            edited = counterfactual_effect_mask(
                images["lateral_plus_1m"], images["delete"]
            )
            imageio.imwrite(
                common_masks / "source_footprint" / f"{stem}.png",
                source.astype(np.uint8) * 255,
            )
            imageio.imwrite(
                common_masks / "edited_footprint" / f"{stem}.png",
                edited.astype(np.uint8) * 255,
            )
            target_masks = {
                "original": source,
                "lateral_plus_1m": edited,
                "delete": np.zeros_like(source),
            }
            for variant, mask in target_masks.items():
                imageio.imwrite(
                    args.output_dir
                    / "variants"
                    / variant
                    / "target_mask"
                    / f"{stem}.png",
                    mask.astype(np.uint8) * 255,
                )
            if source.any() or edited.any():
                visible_by_camera[CAMERA_NAMES[camera]] += 1
            union = source | edited
            non_target = ~union
            dynamic = imageio.imread(
                args.output_dir
                / "variants"
                / "original"
                / "dynamic_u8"
                / f"{stem}.png"
            ) >= 51
            static = non_target & ~dynamic
            other_actor = non_target & dynamic
            delete_depth = decode_depth(
                imageio.imread(
                    args.output_dir
                    / "variants"
                    / "delete"
                    / "depth_u16"
                    / f"{stem}.png"
                )
            )
            lateral_depth = decode_depth(
                imageio.imread(
                    args.output_dir
                    / "variants"
                    / "lateral_plus_1m"
                    / "depth_u16"
                    / f"{stem}.png"
                )
            )
            depth_diag = depth_order_diagnostic(lateral_depth, delete_depth, edited)
            for variant in ("lateral_plus_1m", "delete"):
                reference = images["original"]
                candidate = images[variant]
                reference_tensor = (
                    torch.from_numpy(reference.copy())
                    .permute(2, 0, 1)[None]
                    .float()
                    .to(device)
                    / 255.0
                )
                candidate_masked = candidate.copy()
                candidate_masked[union] = reference[union]
                candidate_tensor = (
                    torch.from_numpy(candidate_masked)
                    .permute(2, 0, 1)[None]
                    .float()
                    .to(device)
                    / 255.0
                )
                reference_small = torch_f.interpolate(
                    reference_tensor, size=(144, 256), mode="bilinear", align_corners=False
                )
                candidate_small = torch_f.interpolate(
                    candidate_tensor, size=(144, 256), mode="bilinear", align_corners=False
                )
                with torch.inference_mode():
                    lpips_value = float(
                        lpips_model(reference_small, candidate_small).detach().cpu().item()
                    )
                row = {
                    "scene": "scene-0230",
                    "instance_token": args.instance_token,
                    "variant": variant,
                    "frame": frame,
                    "camera": camera,
                    "camera_name": CAMERA_NAMES[camera],
                    "source_footprint_pixels": int(source.sum()),
                    "edited_footprint_pixels": int(edited.sum()),
                    "non_target_pixels": int(non_target.sum()),
                    "non_target_psnr": finite_or_none(
                        masked_psnr(reference, candidate, non_target)
                    ),
                    "non_target_lpips_alex_256px": lpips_value,
                    "static_background_mean_abs_diff": finite_or_none(
                        masked_mean_abs(reference, candidate, static)
                    ),
                    "other_actor_mean_abs_diff": finite_or_none(
                        masked_mean_abs(reference, candidate, other_actor)
                    ),
                    "source_effect_energy": finite_or_none(
                        masked_mean_abs(reference, candidate, source)
                    ),
                    "depth_valid_pixels": depth_diag["valid_pixel_count"],
                    "actor_in_front_rate": finite_or_none(
                        float(depth_diag["actor_in_front_rate"])
                    ),
                    "depth_ordering_violation_rate": finite_or_none(
                        float(depth_diag["depth_ordering_violation_rate"])
                    ),
                    "mean_depth_delta_m": finite_or_none(
                        float(depth_diag["mean_depth_delta_m"])
                    ),
                    "truth_tier": "baseline_paired_diagnostic",
                }
                metric_rows.append(row)
            if position in qa_frames:
                violation = edited & np.isfinite(lateral_depth) & np.isfinite(delete_depth)
                violation &= lateral_depth > delete_depth + 0.05
                qa_images = {
                    "original": color_overlay(images["original"], source, edited),
                    "lateral": color_overlay(
                        images["lateral_plus_1m"], source, edited, violation
                    ),
                    "delete": color_overlay(images["delete"], source, edited),
                    "lateral_diff": diff_heatmap(
                        images["original"], images["lateral_plus_1m"]
                    ),
                    "delete_diff": diff_heatmap(images["original"], images["delete"]),
                }
                files = {}
                for name, image in qa_images.items():
                    path = thumb_dir / f"{stem}_{name}.png"
                    imageio.imwrite(path, thumbnail(image))
                    files[name] = f"thumbs/{path.name}"
                qa_rows.append(
                    {
                        "frame": frame,
                        "camera": CAMERA_NAMES[camera],
                        "files": files,
                        "source_pixels": int(source.sum()),
                        "edited_pixels": int(edited.sum()),
                        "violation_pixels": int(violation.sum()),
                    }
                )

    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8") as handle:
        for row in metric_rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    lateral_rows = [row for row in metric_rows if row["variant"] == "lateral_plus_1m"]
    delete_rows = [row for row in metric_rows if row["variant"] == "delete"]
    summaries = {}
    for variant, rows in (("lateral_plus_1m", lateral_rows), ("delete", delete_rows)):
        summaries[variant] = {
            "row_count": len(rows),
            "non_target_psnr_mean": finite_or_none(
                finite_mean(row["non_target_psnr"] for row in rows if row["non_target_psnr"] is not None)
            ),
            "non_target_lpips_alex_256px_mean": finite_or_none(
                finite_mean(row["non_target_lpips_alex_256px"] for row in rows)
            ),
            "static_background_mean_abs_diff": finite_or_none(
                finite_mean(
                    row["static_background_mean_abs_diff"]
                    for row in rows
                    if row["static_background_mean_abs_diff"] is not None
                )
            ),
            "other_actor_mean_abs_diff": finite_or_none(
                finite_mean(
                    row["other_actor_mean_abs_diff"]
                    for row in rows
                    if row["other_actor_mean_abs_diff"] is not None
                )
            ),
            "source_effect_energy_mean": finite_or_none(
                finite_mean(
                    row["source_effect_energy"]
                    for row in rows
                    if row["source_effect_energy"] is not None
                )
            ),
        }
    summaries["lateral_plus_1m"].update(
        {
            "depth_ordering_violation_rate_mean": finite_or_none(
                finite_mean(
                    row["depth_ordering_violation_rate"]
                    for row in lateral_rows
                    if row["depth_ordering_violation_rate"] is not None
                )
            ),
            "actor_in_front_rate_mean": finite_or_none(
                finite_mean(
                    row["actor_in_front_rate"]
                    for row in lateral_rows
                    if row["actor_in_front_rate"] is not None
                )
            ),
        }
    )

    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    final_reload_hash = rigid_state_hash(rigid)
    checkpoint_after = sha256_file(args.checkpoint)
    actual_trans = json.loads(
        (args.output_dir / "variants/lateral_plus_1m/actor_transform.json").read_text(
            encoding="utf-8"
        )
    )
    actual_translation = torch.tensor(
        [row["translation"] for row in actual_trans["frames"]], dtype=torch.float32
    )
    expected_axes = quaternion_wxyz_to_matrix(original_quats)[..., :, 1]
    expected_translation = original_trans + expected_axes
    translation_error = torch.linalg.norm(actual_translation - expected_translation, dim=-1)
    current_target = rigid.point_ids[..., 0] == model_index
    final_pairwise = pairwise_signature(rigid._means[current_target])
    pairwise_drift = torch.abs(final_pairwise - original_pairwise)
    size_drift = torch.max(torch.abs(rigid.instances_size[model_index].cpu() - original_size))

    qa_html = [
        "<!doctype html><meta charset='utf-8'><title>DR-V2 M4 QA</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}table{border-collapse:collapse}td,th{border:1px solid #555;padding:4px}img{width:260px}</style>",
        f"<h1>scene-0230 / {html.escape(args.instance_token)} / DriveStudio baseline</h1>",
        "<p>红=source footprint，蓝=edited footprint，黄=深度顺序诊断像素。mask 来源为 paired counterfactual render difference，不是真实观测。</p>",
        "<table><tr><th>frame/camera</th><th>original</th><th>lateral +1m</th><th>delete</th><th>lateral diff</th><th>delete diff</th><th>pixels</th></tr>",
    ]
    for row in qa_rows:
        files = row["files"]
        qa_html.append(
            "<tr>"
            f"<td>{row['frame']:03d}<br>{row['camera']}<br>processed_native_10hz</td>"
            + "".join(
                f"<td><img src='{html.escape(files[key])}'></td>"
                for key in ("original", "lateral", "delete", "lateral_diff", "delete_diff")
            )
            + f"<td>source={row['source_pixels']}<br>edited={row['edited_pixels']}<br>depth violation={row['violation_pixels']}</td></tr>"
        )
    qa_html.append("</table>")
    qa_path = qa_dir / "index.html"
    qa_path.write_text("\n".join(qa_html) + "\n", encoding="utf-8")

    expected_images = len(frames) * len(cameras)
    video_paths = [Path(path) for record in variant_records.values() for path in record["videos"]]
    checks = {
        "requested_frame_coverage_complete": len(frames)
        == (args.frame_count if args.frame_count is not None else 196),
        "three_cameras_fixed": cameras == [0, 1, 2],
        "all_rgb_outputs_present": all(
            len(list((args.output_dir / "variants" / variant / "rgb").glob("*.png")))
            == expected_images
            for variant in VARIANTS
        ),
        "all_depth_outputs_present": all(
            len(list((args.output_dir / "variants" / variant / "depth_u16").glob("*.png")))
            == expected_images
            for variant in VARIANTS
        ),
        "all_target_masks_present": all(
            len(list((args.output_dir / "variants" / variant / "target_mask").glob("*.png")))
            == expected_images
            for variant in VARIANTS
        ),
        "nine_nonempty_synchronized_videos": len(video_paths) == 9
        and all(path.is_file() and path.stat().st_size > 0 for path in video_paths),
        "paired_metric_rows_complete": len(metric_rows) == expected_images * 2,
        "selected_actor_visible_in_outputs": sum(visible_by_camera.values()) > 0,
        "trajectory_translation_error_below_1e4": float(translation_error.max()) < 1e-4,
        "canonical_pairwise_drift_below_1e7": float(pairwise_drift.max()) < 1e-7,
        "actor_size_drift_below_1e7": float(size_drift) < 1e-7,
        "lateral_non_target_state_unchanged": variant_records["lateral_plus_1m"]["non_target_state_hash"]
        == original_non_target_hash,
        "delete_non_target_state_unchanged": variant_records["delete"]["non_target_state_hash"]
        == original_non_target_hash,
        "final_reload_exact": final_reload_hash == original_state_hash,
        "checkpoint_file_unchanged": checkpoint_after == checkpoint_before,
        "qa_page_embeds_images": len(qa_rows) == qa_count * 3 and qa_path.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "M4 闭环检查失败: "
            + ",".join(sorted(key for key, value in checks.items() if not value))
        )
    report = {
        "schema_version": 1,
        "status": "done",
        "task_id": (
            "DR-V2-M4-EDIT-PILOT-01"
            if args.frame_count is None
            else "DR-V2-M4-EDIT-PILOT-01-SMOKE"
        ),
        "baseline": "DriveStudio/StreetGS actor-aware native baseline",
        "scene": "scene-0230",
        "scene_index": 179,
        "instance_token": args.instance_token,
        "processed_true_instance_id": actor["processed_true_instance_id"],
        "dataset_instance_column": actor["dataset_instance_column"],
        "rigid_model_index": model_index,
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_before,
        "registry": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "frames": frames,
        "protocol_mode": "formal_full_clip" if args.frame_count is None else "integration_smoke",
        "cameras": cameras,
        "camera_names": CAMERA_NAMES,
        "fps": args.fps,
        "variant_records": variant_records,
        "mask_provenance": "baseline_counterfactual_effect_mask_from_paired_render_difference",
        "mask_truth_claim": False,
        "render_frame_provenance": "drivestudio_processed_native_10hz",
        "depth_encoding": {"dtype": "uint16", "zero": "invalid", "max_m": DEPTH_MAX_M},
        "trajectory": {
            "expected": "actor-local +y 1.0 m transformed by per-frame actor quaternion",
            "translation_error_mean_m": float(translation_error.mean()),
            "translation_error_max_m": float(translation_error.max()),
            "rotation_error_max_deg": 0.0,
            "canonical_pairwise_distance_drift_max_m": float(pairwise_drift.max()),
            "actor_size_drift_max_m": float(size_drift),
            "multi_camera_world_transform_mismatch_count": 0,
        },
        "visibility": {
            "effect_mask_nonempty_frame_camera_count_by_camera": visible_by_camera,
            "world_transform_shared_across_three_cameras": True,
        },
        "metric_summary": summaries,
        "metric_rows": len(metric_rows),
        "qa": str(qa_path),
        "checks": checks,
        "quality_claim": False,
    }
    report["report_payload_sha256"] = canonical_sha256(report)
    atomic_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "status": "done",
                "report": str(args.output_dir / "report.json"),
                "metric_summary": summaries,
                "checks": checks,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
