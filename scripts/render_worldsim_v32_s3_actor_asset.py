#!/usr/bin/env python
"""把生成 actor 接回冻结 StreetGS 轨迹并真实渲染 original/lateral/delete。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.dynamic_editing_v2.pilot_metrics import counterfactual_effect_mask
from motion_proj.worldsim_v32.asset_harvester_adapter import inject_actor_asset
from motion_proj.worldsim_v32.actor_asset_schema import validate_actor_asset
from motion_proj.worldsim_v32.semantic_schema import sha256_file
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data,
    load_model_checkpoint_read_only,
    release_trainer_render_info,
)
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.run_dr_v2_m4_pilot import move_actor_local_y


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def uint8_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("S3 render 含非有限 RGB")
    return np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)


def render_variant(
    *,
    trainer: Any,
    dataset: Any,
    checkpoint: Path,
    actor_index: int,
    asset: dict[str, np.ndarray],
    frame: int,
    camera: int,
    variant: str,
    lateral_offset_m: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int]]:
    load_model_checkpoint_read_only(trainer, checkpoint, device)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    counts = {
        "removed_gaussians": int((rigid.point_ids[..., 0] == actor_index).sum()),
        "inserted_gaussians": 0,
    }
    if variant == "delete":
        rigid.remove_instances([actor_index])
    elif variant in {"original", "lateral"}:
        counts = inject_actor_asset(rigid, actor_index, asset)
        if variant == "lateral":
            move_actor_local_y(rigid, actor_index, lateral_offset_m)
    else:
        raise ValueError(f"未知 S3 render variant: {variant}")
    image_infos, camera_infos, *_ = get_view_data(dataset, frame, camera, device)
    with torch.inference_mode():
        outputs = trainer(image_infos, camera_infos)
    image = uint8_rgb(outputs["rgb"])
    release_trainer_render_info(trainer)
    return image, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sample",
        choices=["high_support_1view", "high_support_2view"],
        default=None,
    )
    parser.add_argument("--source-view-index", type=int, default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    asset_manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    actor = config["actor"]
    if asset_manifest["instance_token"] != actor["instance_token"]:
        raise RuntimeError("S3 render actor token 错配")
    asset_path = Path(asset_manifest["asset"]["path"])
    if sha256_file(asset_path) != asset_manifest["asset"]["sha256"]:
        raise RuntimeError("S3 render actor asset SHA 漂移")
    with np.load(asset_path, allow_pickle=False) as arrays:
        asset = {name: arrays[name].copy() for name in arrays.files}
    validate_actor_asset(asset)

    sample_name = args.sample or config["smoke"]["sample"]
    view_index = (
        int(args.source_view_index)
        if args.source_view_index is not None
        else int(config["smoke"]["source_view_index"])
    )
    sample = next(row for row in inputs["samples"] if row["sample"] == sample_name)
    if view_index < 0 or view_index >= len(sample["views"]):
        raise IndexError(
            f"S3 source view index 越界: sample={sample_name}, index={view_index}"
        )
    view = sample["views"][view_index]
    frame, camera = int(view["frame"]), int(view["camera_id"])
    checkpoint = Path(config["streetgs"]["checkpoint"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != config["streetgs"]["checkpoint_sha256"]:
        raise RuntimeError("S3 render D2 checkpoint SHA 漂移")
    runtime_config = {
        "inputs": {
            "checkpoint": str(checkpoint),
            "source_config": config["streetgs"]["source_config"],
        },
        "runtimes": {
            "drivestudio_checkout": config["streetgs"]["checkout"],
        },
    }
    if not torch.cuda.is_available():
        raise RuntimeError("S3 trajectory render 需要可见 CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    dataset, trainer = build_runtime(runtime_config, device)
    images: dict[str, np.ndarray] = {}
    counts = {}
    for variant in ("original", "lateral", "delete"):
        image, variant_counts = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            actor_index=int(actor["rigid_model_index"]),
            asset=asset,
            frame=frame,
            camera=camera,
            variant=variant,
            lateral_offset_m=float(config["smoke"]["lateral_offset_m"]),
            device=device,
        )
        path = args.output_dir / f"{variant}.png"
        imageio.imwrite(path, image)
        images[variant] = image
        counts[variant] = variant_counts
    effects = {
        "original": counterfactual_effect_mask(
            images["original"], images["delete"], threshold_uint8=2, dilation_radius=2
        ),
        "lateral": counterfactual_effect_mask(
            images["lateral"], images["delete"], threshold_uint8=2, dilation_radius=2
        ),
    }
    minimum_pixels = int(config["smoke"]["minimum_effect_pixels"])
    for name, mask in effects.items():
        if int(mask.sum()) < minimum_pixels:
            raise RuntimeError(
                f"S3 {name} trajectory render effect 太小: {int(mask.sum())}"
            )
        imageio.imwrite(args.output_dir / f"{name}_effect.png", mask.astype(np.uint8) * 255)
    if len({sha256_file(args.output_dir / f"{name}.png") for name in images}) != 3:
        raise RuntimeError("S3 original/lateral/delete 输出未发生三路差异")
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("S3 trajectory render 修改了 D2 checkpoint")
    summary = {
        "schema_version": "worldsim_v32_s3_trajectory_render_v1",
        "task_id": config["task_id"],
        "status": "done",
        "implementation": {
            "renderer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "adapter": {
                "path": str(
                    PROJECT / "motion_proj/worldsim_v32/asset_harvester_adapter.py"
                ),
                "sha256": sha256_file(
                    PROJECT / "motion_proj/worldsim_v32/asset_harvester_adapter.py"
                ),
            },
            "actor_asset_schema": {
                "path": str(
                    PROJECT / "motion_proj/worldsim_v32/actor_asset_schema.py"
                ),
                "sha256": sha256_file(
                    PROJECT / "motion_proj/worldsim_v32/actor_asset_schema.py"
                ),
            },
        },
        "asset_sample": asset_manifest["sample"],
        "evaluation_view_sample": sample_name,
        "source_view_index": view_index,
        "frame": frame,
        "camera_id": camera,
        "camera_name": view["camera_name"],
        "instance_token": actor["instance_token"],
        "rigid_model_index": int(actor["rigid_model_index"]),
        "lateral_offset_m": float(config["smoke"]["lateral_offset_m"]),
        "variant_counts": counts,
        "effect_pixels": {name: int(mask.sum()) for name, mask in effects.items()},
        "effect_masks": {
            name: {
                "path": str(args.output_dir / f"{name}_effect.png"),
                "sha256": sha256_file(args.output_dir / f"{name}_effect.png"),
            }
            for name in effects
        },
        "images": {
            name: {
                "path": str(args.output_dir / f"{name}.png"),
                "sha256": sha256_file(args.output_dir / f"{name}.png"),
            }
            for name in images
        },
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "asset_manifest": str(args.asset_manifest.resolve()),
        "asset_manifest_sha256": sha256_file(args.asset_manifest),
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    atomic_json(args.output_dir / "render_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
