#!/usr/bin/env python
"""Run the M3 original/remove/+1 m actor-local lateral DriveStudio smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.dynamic_editing_v2.drivestudio_registry import require_token


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--instance-token", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drivestudio-root", type=Path, default=Path("/root/autodl-tmp/third_party/drivestudio"))
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--lateral-meters", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"edit smoke output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    config = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    if list(config.data.pixel_source.cameras) != [0, 1, 2]:
        raise RuntimeError(f"M3 requires camera ids [0,1,2], got {list(config.data.pixel_source.cameras)}")
    registry = json.loads(args.registry.read_text())
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
    checkpoint_before = file_sha256(args.checkpoint)
    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models.get("RigidNodes")
    if rigid is None:
        raise RuntimeError("loaded trainer has no RigidNodes")
    valid_frames = torch.nonzero(rigid.instances_fv[:, model_index], as_tuple=False).flatten().cpu().numpy()
    if len(valid_frames) < args.frames:
        raise RuntimeError(f"selected actor has only {len(valid_frames)} valid frames")
    selected_positions = np.linspace(0, len(valid_frames) - 1, args.frames, dtype=int)
    frames = [int(valid_frames[position]) for position in selected_positions]
    cameras = list(range(dataset.pixel_source.num_cams))
    if cameras != [0, 1, 2]:
        raise RuntimeError(f"camera index contract failed: {cameras}")

    original_state_hash = rigid_state_hash(rigid)
    non_target_original_hash = non_target_hash(rigid, model_index)
    rendered: dict[str, dict[str, np.ndarray]] = {}

    def render_variant(name: str) -> list[dict]:
        variant_dir = args.output_dir / name
        variant_dir.mkdir()
        rendered[name] = {}
        rows = []
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
                rgb = outputs["rgb"].detach().float().cpu().numpy()
                if not np.isfinite(rgb).all() or rgb.size == 0:
                    raise RuntimeError(f"non-finite or empty render: {name} f{frame} c{camera}")
                image = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
                filename = f"frame_{frame:03d}_camera_{camera}.png"
                output = variant_dir / filename
                imageio.imwrite(output, image)
                key = f"{frame:03d}:{camera}"
                rendered[name][key] = image
                rows.append(
                    {
                        "variant": name,
                        "frame": frame,
                        "camera": camera,
                        "image_index": image_index,
                        "path": str(output),
                        "bytes": output.stat().st_size,
                        "sha256": file_sha256(output),
                        "shape": list(image.shape),
                    }
                )
        return rows

    image_rows = render_variant("original")
    edited_frame_count = move_actor_local_y(rigid, model_index, args.lateral_meters)
    lateral_non_target_hash = non_target_hash(rigid, model_index)
    image_rows.extend(render_variant("lateral_plus_1m"))

    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    reload_after_lateral_hash = rigid_state_hash(rigid)
    reload_after_lateral_non_target_hash = non_target_hash(rigid, model_index)
    rigid.remove_instances([model_index])
    remove_non_target_hash = non_target_hash(rigid, model_index)
    removed_gaussian_count = int(
        actor["checkpoint_tensor_slice"]["gaussian_count"]
    )
    if int((rigid.point_ids[..., 0] == model_index).sum().item()) != 0:
        raise RuntimeError("remove_instances left target Gaussians in the model")
    image_rows.extend(render_variant("remove"))

    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    final_reload_hash = rigid_state_hash(rigid)
    checkpoint_after = file_sha256(args.checkpoint)

    diffs = {}
    for variant in ("lateral_plus_1m", "remove"):
        values = [
            float(np.mean(np.abs(rendered[variant][key].astype(np.float32) - rendered["original"][key].astype(np.float32))) / 255.0)
            for key in sorted(rendered["original"])
        ]
        diffs[variant] = {
            "mean_abs_rgb_diff": float(np.mean(values)),
            "max_frame_camera_abs_rgb_diff": float(np.max(values)),
            "nonzero_frame_camera_count": sum(value > 0 for value in values),
        }

    expected_per_variant = len(frames) * len(cameras)
    checks = {
        "original_render_nonempty": len(rendered["original"]) == expected_per_variant,
        "lateral_render_nonempty": len(rendered["lateral_plus_1m"]) == expected_per_variant,
        "remove_render_nonempty": len(rendered["remove"]) == expected_per_variant,
        "three_camera_time_sync": all(
            sum(row["variant"] == variant and row["frame"] == frame for row in image_rows) == 3
            for variant in rendered
            for frame in frames
        ),
        "lateral_effect_nonzero": diffs["lateral_plus_1m"]["nonzero_frame_camera_count"] > 0,
        "remove_effect_nonzero": diffs["remove"]["nonzero_frame_camera_count"] > 0,
        "lateral_non_target_unchanged": lateral_non_target_hash == non_target_original_hash,
        "remove_non_target_unchanged": remove_non_target_hash == non_target_original_hash,
        "reload_after_lateral_exact": reload_after_lateral_hash == original_state_hash,
        "reload_after_lateral_non_target_exact": reload_after_lateral_non_target_hash == non_target_original_hash,
        "final_reload_exact": final_reload_hash == original_state_hash,
        "checkpoint_file_unchanged": checkpoint_before == checkpoint_after,
        "actor_local_lateral_exact_meters": args.lateral_meters == 1.0,
        "selected_actor_has_checkpoint_gaussians": removed_gaussian_count > 0,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"M3 edit smoke checks failed: {failed}")

    report = {
        "schema_version": 1,
        "status": "done",
        "baseline": "DriveStudio/StreetGS actor-aware native baseline",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "registry": str(args.registry),
        "instance_token": args.instance_token,
        "processed_true_instance_id": actor["processed_true_instance_id"],
        "dataset_instance_column": actor["dataset_instance_column"],
        "rigid_model_index": model_index,
        "removed_gaussian_count": removed_gaussian_count,
        "edited_valid_frame_count": edited_frame_count,
        "selected_frames": frames,
        "cameras": cameras,
        "lateral_meters_actor_local_y": args.lateral_meters,
        "original_state_hash": original_state_hash,
        "non_target_original_hash": non_target_original_hash,
        "lateral_non_target_hash": lateral_non_target_hash,
        "remove_non_target_hash": remove_non_target_hash,
        "reload_after_lateral_hash": reload_after_lateral_hash,
        "final_reload_hash": final_reload_hash,
        "diffs": diffs,
        "checks": checks,
        "images": image_rows,
    }
    atomic_json(args.output_dir / "edit_smoke_report.json", report)
    print(json.dumps({"status": "done", "checks": checks, "diffs": diffs}, sort_keys=True))


if __name__ == "__main__":
    main()
