#!/usr/bin/env python
"""Evaluate actor-region and boundary quality for a completed WorldSim V3 A0 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as torch_f
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity

from motion_proj.worldsim_v3.actor_metrics import (
    boundary_band,
    counterfactual_effect_mask,
    finite_mean,
    psnr_from_sums,
    region_error_sums,
)


CAMERA_NAMES = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
ROLE_NAMES = ("high-support", "boundary-support")
EFFECT_THRESHOLD_UINT8 = 2
EFFECT_DILATION_RADIUS = 2
BOUNDARY_RADIUS = 3
LPIPS_HEIGHT = 144
LPIPS_WIDTH = 256
LPIPS_CROP_PADDING = 8


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def uint8_rgb(value: np.ndarray) -> np.ndarray:
    return (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def ssim_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    _, full = structural_similarity(
        target,
        prediction,
        data_range=1.0,
        channel_axis=-1,
        full=True,
    )
    value = np.asarray(full, dtype=np.float64)
    if value.ndim == 3:
        value = value.mean(axis=-1)
    if value.ndim != 2:
        raise RuntimeError(f"unexpected SSIM map shape: {value.shape}")
    return value


def masked_lpips(
    lpips_model,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> float:
    selected = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(selected)
    if rows.size == 0:
        raise ValueError("masked LPIPS requires a non-empty mask")
    height, width = selected.shape
    row_start = max(0, int(rows.min()) - LPIPS_CROP_PADDING)
    row_stop = min(height, int(rows.max()) + LPIPS_CROP_PADDING + 1)
    column_start = max(0, int(columns.min()) - LPIPS_CROP_PADDING)
    column_stop = min(width, int(columns.max()) + LPIPS_CROP_PADDING + 1)
    target_crop = np.asarray(target, dtype=np.float32)[
        row_start:row_stop, column_start:column_stop
    ].copy()
    prediction_crop = np.asarray(prediction, dtype=np.float32)[
        row_start:row_stop, column_start:column_stop
    ]
    mask_crop = selected[row_start:row_stop, column_start:column_stop]
    candidate = target_crop.copy()
    candidate[mask_crop] = prediction_crop[mask_crop]
    expected = torch.from_numpy(target_crop)
    actual = torch.from_numpy(candidate)
    expected = expected.permute(2, 0, 1)[None].to(device)
    actual = actual.permute(2, 0, 1)[None].to(device)
    expected = torch_f.interpolate(
        expected,
        size=(LPIPS_HEIGHT, LPIPS_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    actual = torch_f.interpolate(
        actual,
        size=(LPIPS_HEIGHT, LPIPS_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    with torch.inference_mode():
        result = float(lpips_model(expected, actual).detach().cpu().item())
    if hasattr(lpips_model, "reset"):
        lpips_model.reset()
    return result


class RegionAccumulator:
    def __init__(self) -> None:
        self.image_count = 0
        self.pixel_count = 0
        self.value_count = 0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_values: list[float] = []

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        image_ssim: np.ndarray,
        lpips_value: float,
    ) -> dict[str, float | int | None]:
        sums = region_error_sums(prediction, target, mask)
        count = int(sums["pixel_count"])
        if count <= 0:
            raise ValueError("cannot update a region accumulator with an empty mask")
        self.image_count += 1
        self.pixel_count += count
        self.value_count += int(sums["value_count"])
        self.squared_error_sum += float(sums["squared_error_sum"])
        self.absolute_error_sum += float(sums["absolute_error_sum"])
        selected_ssim = float(np.asarray(image_ssim)[mask].sum())
        self.ssim_sum += selected_ssim
        self.lpips_values.append(float(lpips_value))
        return {
            "pixel_count": count,
            "psnr": finite_or_none(
                psnr_from_sums(sums["squared_error_sum"], sums["value_count"])
            ),
            "mean_absolute_error": float(sums["absolute_error_sum"])
            / int(sums["value_count"]),
            "ssim": selected_ssim / count,
            "masked_lpips_alex_tight_crop_256px": finite_or_none(lpips_value),
        }

    def summary(self) -> dict[str, object]:
        if self.pixel_count == 0:
            return {"status": "ABSTAIN", "reason": "ZERO_REGION_PIXELS"}
        return {
            "status": "done",
            "visible_image_count": self.image_count,
            "pixel_count": self.pixel_count,
            "psnr": finite_or_none(
                psnr_from_sums(self.squared_error_sum, self.value_count)
            ),
            "mean_absolute_error": self.absolute_error_sum / self.value_count,
            "ssim": self.ssim_sum / self.pixel_count,
            "masked_lpips_alex_tight_crop_256px": finite_or_none(
                finite_mean(self.lpips_values)
            ),
        }


def compact_actor(actor: dict[str, object] | None) -> dict[str, object] | None:
    if actor is None:
        return None
    return {
        "availability": actor.get("availability"),
        "instance_token": actor.get("instance_token"),
        "class_name": actor.get("class_name"),
        "rigid_model_index": actor.get("rigid_model_index"),
        "gaussian_count": actor.get("gaussian_count"),
    }


def role_positions(
    test_indices: list[int], valid_frames: set[int], num_cameras: int
) -> list[int]:
    return [
        position
        for position, full_index in enumerate(test_indices)
        if int(full_index) // num_cameras in valid_frames
    ]


def render_sample(dataset, trainer, position: int, device: torch.device):
    image_infos, camera_infos = dataset.test_image_set.get_image(
        position, camera_downscale=1.0
    )
    target = image_infos["pixels"].detach().float().cpu().numpy()
    image_infos = to_device(image_infos, device)
    camera_infos = to_device(camera_infos, device)
    with torch.inference_mode():
        output = trainer(image_infos, camera_infos)["rgb"]
    prediction = output.detach().float().cpu().numpy()
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError(f"non-finite RGB for test position {position}")
    return prediction, target


def qa_image(
    prediction: np.ndarray,
    deleted: np.ndarray,
    mask: np.ndarray,
    boundary: np.ndarray,
) -> np.ndarray:
    original = uint8_rgb(prediction)
    without = uint8_rgb(deleted)
    overlay = original.copy()
    overlay[mask] = (
        0.45 * overlay[mask].astype(np.float32)
        + 0.55 * np.array([255, 45, 45], dtype=np.float32)
    ).round().astype(np.uint8)
    overlay[boundary] = np.array([255, 235, 0], dtype=np.uint8)
    return np.concatenate((original, without, overlay), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drivestudio-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-images-per-role",
        type=int,
        help="Optional engineering-smoke cap; omit for a formal full-split run.",
    )
    args = parser.parse_args()

    if args.max_images_per_role is not None and args.max_images_per_role <= 0:
        raise ValueError("max-images-per-role must be positive")

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "masks").mkdir()
    (args.output_dir / "qa").mkdir()
    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if source.get("status") != "done":
        raise RuntimeError("source A0 summary is not terminal done")
    checkpoint = Path(source["checkpoint"]["checkpoint"])
    expected_checkpoint_sha = source["checkpoint"]["sha256"]
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != expected_checkpoint_sha:
        raise RuntimeError("source checkpoint SHA-256 does not match its summary")
    registry_path = Path(source["registry"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("checkpoint_sha256") != checkpoint_before:
        raise RuntimeError("actor registry points to a different checkpoint")

    config_path = checkpoint.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)
    cameras = list(config.data.pixel_source.cameras)
    if cameras != [0, 1, 2]:
        raise RuntimeError(f"A0 actor diagnostic requires cameras=[0,1,2], got {cameras}")
    device = torch.device(args.device)
    dataset = DrivingDataset(data_cfg=config.data)
    num_cameras = int(dataset.pixel_source.num_cams)
    if num_cameras != len(CAMERA_NAMES):
        raise RuntimeError(f"unexpected camera count: {num_cameras}")
    test_indices = [int(value) for value in dataset.test_image_set.split_indices]

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
    trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
    trainer.set_eval()

    role_specs: dict[str, dict[str, object]] = {}
    selected = source.get("selected_actors", {})
    rigid = trainer.models["RigidNodes"]
    for role in ROLE_NAMES:
        actor = selected.get(role)
        if actor is None or actor.get("availability") != "available":
            role_specs[role] = {
                "status": "ABSTAIN",
                "reason": "ACTOR_UNAVAILABLE_IN_SOURCE_CONTRACT",
                "actor": compact_actor(actor),
                "candidate_heldout_image_count": 0,
            }
            continue
        model_index = int(actor["rigid_model_index"])
        valid_frames = set(
            torch.nonzero(rigid.instances_fv[:, model_index], as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        positions = role_positions(test_indices, valid_frames, num_cameras)
        if args.max_images_per_role is not None:
            positions = positions[: args.max_images_per_role]
        role_specs[role] = {
            "status": "pending" if positions else "ABSTAIN",
            "reason": None if positions else "NO_HELDOUT_VALID_FRAMES",
            "actor": compact_actor(actor),
            "valid_processed_frame_count": len(valid_frames),
            "candidate_heldout_image_count": len(positions),
            "positions": positions,
        }

    required_positions = sorted(
        {
            position
            for spec in role_specs.values()
            if spec["status"] == "pending"
            for position in spec["positions"]
        }
    )
    original_cache: dict[int, np.ndarray] = {}
    for ordinal, position in enumerate(required_positions, start=1):
        prediction, _ = render_sample(dataset, trainer, position, device)
        original_cache[position] = uint8_rgb(prediction)
        print(
            f"original {ordinal}/{len(required_positions)} position={position}",
            flush=True,
        )

    rows_path = args.output_dir / "per_image_metrics.jsonl"
    role_results: dict[str, dict[str, object]] = {}
    for role in ROLE_NAMES:
        spec = role_specs[role]
        if spec["status"] == "ABSTAIN":
            role_results[role] = {key: value for key, value in spec.items() if key != "positions"}
            continue
        actor = spec["actor"]
        model_index = int(actor["rigid_model_index"])
        trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
        trainer.set_eval()
        rigid = trainer.models["RigidNodes"]
        points_before = int(rigid._means.shape[0])
        target_points = int((rigid.point_ids[..., 0] == model_index).sum().item())
        rigid.remove_instances([model_index])
        points_after = int(rigid._means.shape[0])
        if points_before - points_after != target_points:
            raise RuntimeError(
                f"actor removal count mismatch for {role}: "
                f"before={points_before} after={points_after} target={target_points}"
            )

        actor_accumulator = RegionAccumulator()
        boundary_accumulator = RegionAccumulator()
        visible_by_camera = {name: 0 for name in CAMERA_NAMES}
        total_mask_pixels = 0
        total_image_pixels = 0
        qa_count = 0
        for ordinal, position in enumerate(spec["positions"], start=1):
            deleted_float, target = render_sample(dataset, trainer, position, device)
            deleted_uint8 = uint8_rgb(deleted_float)
            original_uint8 = original_cache[position]
            original = original_uint8.astype(np.float32) / 255.0
            effect = counterfactual_effect_mask(
                original_uint8,
                deleted_uint8,
                threshold_uint8=EFFECT_THRESHOLD_UINT8,
                dilation_radius=EFFECT_DILATION_RADIUS,
            )
            boundary = boundary_band(effect, radius=BOUNDARY_RADIUS)
            full_index = test_indices[position]
            frame = full_index // num_cameras
            camera = full_index % num_cameras
            stem = f"frame_{frame:03d}_camera_{camera}"
            total_image_pixels += int(effect.size)
            total_mask_pixels += int(effect.sum())
            row: dict[str, object] = {
                "role": role,
                "instance_token": actor["instance_token"],
                "rigid_model_index": model_index,
                "test_position": position,
                "full_image_index": full_index,
                "frame": frame,
                "camera": camera,
                "camera_name": CAMERA_NAMES[camera],
                "effect_pixels": int(effect.sum()),
                "boundary_pixels": int(boundary.sum()),
            }
            if effect.any():
                visible_by_camera[CAMERA_NAMES[camera]] += 1
                image_ssim = ssim_map(original, target)
                actor_lpips = masked_lpips(
                    trainer.lpips, original, target, effect, device
                )
                boundary_lpips = masked_lpips(
                    trainer.lpips, original, target, boundary, device
                )
                row["actor_region"] = actor_accumulator.update(
                    original, target, effect, image_ssim, actor_lpips
                )
                row["boundary_band"] = boundary_accumulator.update(
                    original, target, boundary, image_ssim, boundary_lpips
                )
                imageio.imwrite(
                    args.output_dir / "masks" / f"{role}__{stem}.png",
                    effect.astype(np.uint8) * 255,
                )
                if qa_count < 12:
                    imageio.imwrite(
                        args.output_dir / "qa" / f"{role}__{stem}.jpg",
                        qa_image(original, deleted_float, effect, boundary),
                        quality=90,
                    )
                    qa_count += 1
            else:
                row["actor_region"] = None
                row["boundary_band"] = None
            append_jsonl(rows_path, row)
            print(
                f"{role} {ordinal}/{len(spec['positions'])} position={position} "
                f"effect_pixels={int(effect.sum())}",
                flush=True,
            )

        actor_summary = actor_accumulator.summary()
        boundary_summary = boundary_accumulator.summary()
        status = "done" if actor_summary["status"] == "done" else "ABSTAIN"
        role_results[role] = {
            "status": status,
            "reason": None if status == "done" else "ZERO_COUNTERFACTUAL_EFFECT",
            "actor": actor,
            "valid_processed_frame_count": spec["valid_processed_frame_count"],
            "candidate_heldout_image_count": spec["candidate_heldout_image_count"],
            "visible_effect_image_count": actor_accumulator.image_count,
            "visible_by_camera": visible_by_camera,
            "effect_pixel_count": total_mask_pixels,
            "candidate_pixel_count": total_image_pixels,
            "effect_pixel_coverage": (
                total_mask_pixels / total_image_pixels if total_image_pixels else None
            ),
            "removed_gaussian_count": target_points,
            "actor_region": actor_summary,
            "boundary_band": boundary_summary,
        }

    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint changed during read-only actor evaluation")
    result = {
        "status": "done",
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "source_summary": str(args.source_summary),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "heldout_split": {
            "test_image_stride": int(config.data.pixel_source.test_image_stride),
            "test_image_count": len(test_indices),
            "test_full_image_indices": test_indices,
            "max_images_per_role": args.max_images_per_role,
            "formal_full_split": args.max_images_per_role is None,
        },
        "mask_contract": {
            "truth_tier": "model_counterfactual_diagnostic_not_ground_truth_segmentation",
            "source": "paired original and actor-deleted uint8 renders",
            "effect_threshold_uint8": EFFECT_THRESHOLD_UINT8,
            "effect_dilation_radius_pixels": EFFECT_DILATION_RADIUS,
            "boundary_band_radius_pixels": BOUNDARY_RADIUS,
            "masked_lpips_resolution": [LPIPS_HEIGHT, LPIPS_WIDTH],
            "masked_lpips_crop_padding_pixels": LPIPS_CROP_PADDING,
            "render_metric_quantization": "original render rounded to uint8 before [0,1] metric conversion",
            "aggregation": "pixel-weighted PSNR/SSIM/MAE; visible-image mean masked LPIPS",
        },
        "roles": role_results,
        "per_image_metrics": str(rows_path),
        "qa_directory": str(args.output_dir / "qa"),
    }
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
