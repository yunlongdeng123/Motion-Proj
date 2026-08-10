#!/usr/bin/env python
"""从 S1 canonical masks 构造 Asset Harvester 1/2-view 冻结输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file
from motion_proj.dynamic_editing_v2.pilot_metrics import (
    counterfactual_effect_mask,
)
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    release_trainer_render_info,
)
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.materialize_worldsim_v3_a3_s_b_sidecar import render_variant


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("dilation radius 必须非负")
    if radius == 0:
        return mask.astype(bool, copy=True)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0


def choose_views(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("S3 没有通过 QC 的 high-support mask")

    def is_unclipped(row: dict[str, Any]) -> bool:
        box = row.get("projected_box_xyxy")
        if box is None:
            return False
        x0, y0, x1, y1 = (float(value) for value in box)
        return (
            x0 > 1.0
            and y0 > 1.0
            and x1 < float(row["width"]) - 1.0
            and y1 < float(row["height"]) - 1.0
        )

    # S3 资产输入比语义 lift 更怕传播漂移：优先从无遮挡、未裁边的
    # 直接 prompt 帧开始，再在同一相机/传播块中选择间隔最远的稳定视图。
    direct = [
        row
        for row in rows
        if int(row["frame"]) == int(row["prompt_frame"]) and is_unclipped(row)
    ]
    if not direct:
        raise RuntimeError("S3 找不到未裁边的直接 prompt mask")
    direct.sort(
        key=lambda row: (
            -float(row["quality_metrics"].get("prompt_bbox_iou") or 0.0),
            -int(row["positive_pixels"]),
            int(row["frame"]),
            int(row["camera_id"]),
        )
    )
    first = direct[0]
    same_track = [
        row
        for row in rows
        if row is not first
        and is_unclipped(row)
        and int(row["camera_id"]) == int(first["camera_id"])
        and int(row["prompt_frame"]) == int(first["prompt_frame"])
        and abs(int(row["frame"]) - int(first["frame"])) >= 5
    ]
    same_track.sort(
        key=lambda row: (
            -abs(int(row["frame"]) - int(first["frame"])),
            -float(row["quality_metrics"].get("temporal_iou") or 0.0),
            -int(row["positive_pixels"]),
        )
    )
    if same_track:
        second = same_track[0]
    else:
        fallback = [
            row
            for row in rows
            if row is not first
            and is_unclipped(row)
            and abs(int(row["frame"]) - int(first["frame"])) >= 5
        ]
        fallback.sort(
            key=lambda row: (
                -float(row["quality_metrics"].get("prompt_bbox_iou") or 0.0),
                -int(row["positive_pixels"]),
                int(row["frame"]),
                int(row["camera_id"]),
            )
        )
        if not fallback:
            raise RuntimeError("S3 找不到冻结的第二个无遮挡差异视图")
        second = fallback[0]
    if int(second["frame"]) == int(first["frame"]) and int(
        second["camera_id"]
    ) == int(first["camera_id"]):
        raise RuntimeError("S3 找不到冻结的第二个差异视图")
    return [first, second]


def choose_requested_views(
    rows: list[dict[str, Any]], requested: list[str]
) -> list[dict[str, Any]]:
    """按人工 QA 后冻结的 frame:camera 唯一选择两个视图。"""
    if len(requested) != 2:
        raise ValueError("--view 必须恰好指定两次")
    selected: list[dict[str, Any]] = []
    for value in requested:
        frame_text, separator, camera_text = value.partition(":")
        if not separator:
            raise ValueError(f"--view 格式必须为 FRAME:CAMERA: {value}")
        frame = int(frame_text)
        matches = [
            row
            for row in rows
            if int(row["frame"]) == frame
            and (
                str(row["camera_id"]) == camera_text
                or str(row["camera_name"]) == camera_text
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"冻结视图必须唯一命中 accepted high-support mask: {value}, "
                f"matches={len(matches)}"
            )
        row = matches[0]
        if int(row["frame"]) != int(row["prompt_frame"]):
            raise RuntimeError(f"冻结视图不是直接 prompt 帧: {value}")
        selected.append(row)
    identities = {(int(row["frame"]), int(row["camera_id"])) for row in selected}
    if len(identities) != 2:
        raise RuntimeError("冻结的两个 S3 视图重复")
    return selected


def square_crop(
    image: Image.Image,
    mask: np.ndarray,
    padding_fraction: float,
    output_size: int,
) -> tuple[Image.Image, Image.Image, list[int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("S3 输入 mask 为空")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    side = max(2, int(math.ceil(side * (1.0 + 2.0 * padding_fraction))))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(math.floor(center_x - side / 2.0))
    top = int(math.floor(center_y - side / 2.0))
    right, bottom = left + side, top + side

    rgb_canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
    mask_canvas = Image.new("L", (side, side), color=0)
    source_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    source_box = (
        max(left, 0),
        max(top, 0),
        min(right, image.width),
        min(bottom, image.height),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        raise RuntimeError("S3 crop 与源图无交集")
    destination = (source_box[0] - left, source_box[1] - top)
    rgb_canvas.paste(image.crop(source_box), destination)
    mask_canvas.paste(source_mask.crop(source_box), destination)
    rgb_output = rgb_canvas.resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    mask_output = mask_canvas.resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )
    return rgb_output, mask_output, [left, top, right, bottom]


def annotation_box_size(
    instances: dict[str, Any], dataset_instance_id: int, frame: int
) -> list[float]:
    actor = instances[str(dataset_instance_id)]["frame_annotations"]
    frame_indices = [int(value) for value in actor["frame_idx"]]
    if frame not in frame_indices:
        raise RuntimeError(f"actor frame annotation 缺失: {frame}")
    index = frame_indices.index(frame)
    return [float(value) for value in actor["box_size"][index]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-config", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding-fraction", type=float, default=0.25)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--effect-dilation-radius", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--view",
        action="append",
        default=[],
        metavar="FRAME:CAMERA",
        help="人工 QA 后冻结视图；必须指定两次，CAMERA 可为编号或相机名",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"S3 input output 已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.s1_config.read_text(encoding="utf-8"))
    masks = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    if masks["config_sha256"] != sha256_file(args.s1_config):
        raise RuntimeError("S3 输入与 S1 config SHA 不一致")
    if not masks.get("heldout_excluded"):
        raise RuntimeError("S3 输入未证明 heldout 排除")
    heldout = {int(value) for value in masks["heldout_frames"]}
    candidates = [
        row
        for row in masks["masks"]
        if row["role"] == "high_support"
        and bool(row["accepted"])
        and int(row["positive_pixels"]) > 0
        and int(row["frame"]) not in heldout
    ]
    selected = (
        choose_requested_views(candidates, args.view)
        if args.view
        else choose_views(candidates)
    )
    scene_dir = Path(config["scene"]["processed_scene_dir"])
    instances_path = scene_dir / "instances/instances_info.json"
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    actor_config = config["actors"]["high_support"]
    checkpoint = Path(config["inputs"]["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != config["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("S3 D2 prior checkpoint SHA 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("S3 prior-guided input preparation 需要 CUDA")
    device = torch.device(args.device)
    dataset, trainer = build_runtime(config, device)
    audit_dir = args.output_dir / "audit"
    samples_dir = args.output_dir / "samples"
    audit_dir.mkdir()
    samples_dir.mkdir()
    rows: list[dict[str, Any]] = []
    rendered: list[tuple[Image.Image, Image.Image, dict[str, Any]]] = []
    for row in selected:
        image_path = Path(row["source_image"])
        if sha256_file(image_path) != row["source_image_sha256"]:
            raise RuntimeError(f"S3 source image SHA 漂移: {image_path}")
        mask_path = Path(row["mask"])
        if sha256_file(mask_path) != row["mask_sha256"]:
            raise RuntimeError(f"S3 mask SHA 漂移: {mask_path}")
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with np.load(mask_path, allow_pickle=False) as arrays:
            mask_small = arrays["binary"].astype(bool)
        original = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=int(row["frame"]),
            camera=int(row["camera_id"]),
            model_index=int(actor_config["rigid_model_index"]),
            variant="original",
            device=device,
        )["rgb"]
        release_trainer_render_info(trainer)
        deleted = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=int(row["frame"]),
            camera=int(row["camera_id"]),
            model_index=int(actor_config["rigid_model_index"]),
            variant="delete",
            device=device,
        )["rgb"]
        release_trainer_render_info(trainer)
        effect = counterfactual_effect_mask(
            original, deleted, threshold_uint8=2, dilation_radius=2
        )
        effect = dilate_binary(effect, args.effect_dilation_radius)
        if effect.shape != mask_small.shape:
            raise RuntimeError(
                f"S3 SAM/D2 prior shape 不一致: {mask_small.shape} != {effect.shape}"
            )
        combined_small = mask_small & effect
        if not combined_small.any():
            raise RuntimeError(
                f"S3 SAM/D2 prior 无交集: frame={row['frame']} camera={row['camera_id']}"
            )
        stem = f"f{int(row['frame']):03d}_c{int(row['camera_id'])}"
        audit_paths = {
            "original": audit_dir / f"{stem}_d2_original.png",
            "delete": audit_dir / f"{stem}_d2_delete.png",
            "sam": audit_dir / f"{stem}_sam.png",
            "effect": audit_dir / f"{stem}_d2_effect.png",
            "combined": audit_dir / f"{stem}_combined.png",
        }
        Image.fromarray(original, mode="RGB").save(audit_paths["original"])
        Image.fromarray(deleted, mode="RGB").save(audit_paths["delete"])
        Image.fromarray(mask_small.astype(np.uint8) * 255, mode="L").save(
            audit_paths["sam"]
        )
        Image.fromarray(effect.astype(np.uint8) * 255, mode="L").save(
            audit_paths["effect"]
        )
        Image.fromarray(combined_small.astype(np.uint8) * 255, mode="L").save(
            audit_paths["combined"]
        )
        mask = np.asarray(
            Image.fromarray(combined_small.astype(np.uint8) * 255, mode="L").resize(
                image.size, Image.Resampling.NEAREST
            )
        ) > 0
        crop_image, crop_mask, crop_xyxy = square_crop(
            image, mask, args.padding_fraction, args.output_size
        )
        source = {
            "frame": int(row["frame"]),
            "camera_id": int(row["camera_id"]),
            "camera_name": row["camera_name"],
            "source_image": str(image_path),
            "source_image_sha256": row["source_image_sha256"],
            "source_mask": str(mask_path),
            "source_mask_sha256": row["mask_sha256"],
            "input_mask_provenance": "SAM2_BINARY_INTERSECT_D2_COUNTERFACTUAL_PRIOR",
            "sam_positive_pixels_native": int(mask_small.sum()),
            "d2_effect_pixels_native": int(effect.sum()),
            "combined_positive_pixels_native": int(combined_small.sum()),
            "combined_to_sam_ratio": float(combined_small.sum() / mask_small.sum()),
            "d2_original_array_sha256": array_sha256(original),
            "d2_delete_array_sha256": array_sha256(deleted),
            "audit": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in audit_paths.items()
            },
            "crop_xyxy_source": crop_xyxy,
            "box_size_source": annotation_box_size(
                instances,
                int(actor_config["dataset_instance_id"]),
                int(row["frame"]),
            ),
        }
        rendered.append((crop_image, crop_mask, source))

    for view_count in (1, 2):
        sample_dir = samples_dir / f"high_support_{view_count}view"
        sample_dir.mkdir()
        sample_rows = []
        for index, (image, mask, source) in enumerate(rendered[:view_count]):
            image_output = sample_dir / f"frame_{index}.jpeg"
            mask_output = sample_dir / f"mask_{index}.png"
            image.save(image_output, format="JPEG", quality=95, subsampling=0)
            mask.save(mask_output, format="PNG")
            mask_array = np.asarray(mask) > 0
            if int(mask_array.sum()) == 0:
                raise RuntimeError(f"S3 crop mask 为空: {mask_output}")
            sample_rows.append(
                {
                    **source,
                    "image": str(image_output),
                    "image_sha256": sha256_file(image_output),
                    "mask": str(mask_output),
                    "mask_sha256": sha256_file(mask_output),
                    "positive_pixels": int(mask_array.sum()),
                }
            )
        rows.append(
            {
                "sample": sample_dir.name,
                "view_count": view_count,
                "views": sample_rows,
            }
        )

    manifest = {
        "schema_version": "worldsim_v32_s3_asset_harvester_input_v1",
        "task_id": "WS-V32-S3-ASSET-HARVEST-01",
        "scene": config["scene"]["name"],
        "role": "high_support",
        "instance_token": actor_config["instance_token"],
        "dataset_instance_id": int(actor_config["dataset_instance_id"]),
        "rigid_model_index": int(actor_config["rigid_model_index"]),
        "provenance": "SAM_SEMANTIC_WITH_D2_COUNTERFACTUAL_PRIOR",
        "camera_source": "asset_harvester_estimated",
        "heldout_excluded": True,
        "s1_config": str(args.s1_config.resolve()),
        "s1_config_sha256": sha256_file(args.s1_config),
        "mask_manifest": str(args.mask_manifest.resolve()),
        "mask_manifest_sha256": sha256_file(args.mask_manifest),
        "instances_info_sha256": sha256_file(instances_path),
        "d2_checkpoint": str(checkpoint.resolve()),
        "d2_checkpoint_sha256": checkpoint_sha256,
        "output_size": args.output_size,
        "padding_fraction": args.padding_fraction,
        "effect_dilation_radius": args.effect_dilation_radius,
        "view_selection": {
            "method": "explicit_visual_qa" if args.view else "automatic_v1",
            "requested": list(args.view),
            "direct_prompt_required": bool(args.view),
        },
        "samples_dir": str(samples_dir.resolve()),
        "samples": rows,
    }
    atomic_json(args.output_dir / "input_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
