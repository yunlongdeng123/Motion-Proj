#!/usr/bin/env python3
"""在冻结 StreetGS checkpoint 上串行渲染 R3 偏移与 actor 编辑。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_numpy(value):
    return value.detach().cpu().numpy()


def save_render(path: Path, outputs: dict) -> None:
    values = {
        "rgb": tensor_numpy(outputs["rgb"]).astype(np.float16),
        "depth": tensor_numpy(outputs["depth"]).astype(np.float32),
        "opacity": tensor_numpy(outputs["opacity"]).astype(np.float16),
        "dynamic_opacity": tensor_numpy(outputs["Dynamic_opacity"]).astype(np.float16),
        "dynamic_depth": tensor_numpy(outputs["Dynamic_depth"]).astype(np.float32),
    }
    np.savez_compressed(path, **values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--offsets", required=True)
    parser.add_argument("--forward-extension", required=True, type=float)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    offsets = [float(value) for value in args.offsets.split(",")]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    run_root = checkpoint.parent
    backup = run_root / "backup"
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(backup))
    sys.path.append(str(args.upstream_root.resolve()))

    import torch
    from datasets.driving_dataset import DrivingDataset
    from omegaconf import OmegaConf
    from utils.misc import import_str

    before = sha256_file(checkpoint)
    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.data.preload_device = "cpu"
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    started = time.monotonic()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"),
    )
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    camera_downscale = trainer._get_downscale_factor()
    rows = []

    # 导出训练相机 support，不读取 confirmation 分区。
    training_poses = []
    for camera_id in sorted(dataset.pixel_source.camera_data):
        camera = dataset.pixel_source.camera_data[camera_id]
        for frame_index in dataset.train_timesteps:
            training_poses.append(
                {
                    "camera_id": int(camera_id),
                    "frame_index": int(frame_index),
                    "camera_to_world": tensor_numpy(camera.cam_to_worlds[frame_index]).astype(float).tolist(),
                }
            )
    (output / "TRAINING_CAMERA_SUPPORT.json").write_text(
        json.dumps(training_poses, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with torch.inference_mode():
        for frame_index in frames:
            support_arrays = {}
            for camera_id in (0, 1, 2):
                image_infos, camera_infos = dataset.full_image_set.get_image(
                    frame_index * 3 + camera_id, camera_downscale
                )
                prefix = f"cam{camera_id}"
                support_arrays[f"{prefix}_rgb"] = np.clip(
                    tensor_numpy(image_infos["pixels"]) * 255.0, 0, 255
                ).astype(np.uint8)
                support_arrays[f"{prefix}_lidar_depth"] = tensor_numpy(
                    image_infos["lidar_depth_map"]
                ).astype(np.float32)
                support_arrays[f"{prefix}_dynamic_mask"] = tensor_numpy(
                    image_infos["dynamic_masks"]
                ).astype(bool)
                support_arrays[f"{prefix}_intrinsics"] = tensor_numpy(
                    camera_infos["intrinsics"]
                ).astype(np.float32)
                support_arrays[f"{prefix}_camera_to_world"] = tensor_numpy(
                    camera_infos["camera_to_world"]
                ).astype(np.float32)
            np.savez_compressed(output / f"support_frame_{frame_index:03d}.npz", **support_arrays)

            for lateral_offset in offsets:
                image_infos, camera_infos = dataset.full_image_set.get_image(
                    frame_index * 3, camera_downscale
                )
                right_world = camera_infos["camera_to_world"][:3, 0].clone()
                shift = right_world * lateral_offset
                image_infos["origins"] = image_infos["origins"] + shift
                camera_infos["camera_to_world"] = camera_infos["camera_to_world"].clone()
                camera_infos["camera_to_world"][:3, 3] += shift
                for values in (image_infos, camera_infos):
                    for key, value in values.items():
                        if isinstance(value, torch.Tensor):
                            values[key] = value.cuda(non_blocking=True)
                result = trainer(image_infos, camera_infos)
                name = f"frame{frame_index:03d}_lat{lateral_offset:g}m.npz"
                save_render(output / name, result)
                rows.append(
                    {
                        "scene": args.scene,
                        "frontend": "streetgs",
                        "frame_index": frame_index,
                        "camera_id": 0,
                        "variant": "camera_lateral",
                        "lateral_offset_m": lateral_offset,
                        "path": name,
                        "sha256": sha256_file(output / name),
                    }
                )

            # 短前向外推与横向 profile 分开保存，不混入横向排序统计。
            image_infos, camera_infos = dataset.full_image_set.get_image(
                frame_index * 3, camera_downscale
            )
            forward_world = camera_infos["camera_to_world"][:3, 2].clone()
            shift = forward_world * args.forward_extension
            image_infos["origins"] = image_infos["origins"] + shift
            camera_infos["camera_to_world"] = camera_infos["camera_to_world"].clone()
            camera_infos["camera_to_world"][:3, 3] += shift
            for values in (image_infos, camera_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            result = trainer(image_infos, camera_infos)
            name = f"frame{frame_index:03d}_fwd{args.forward_extension:g}m.npz"
            save_render(output / name, result)
            rows.append(
                {
                    "scene": args.scene,
                    "frontend": "streetgs",
                    "frame_index": frame_index,
                    "camera_id": 0,
                    "variant": "camera_forward_extension",
                    "lateral_offset_m": 0.0,
                    "forward_offset_m": args.forward_extension,
                    "path": name,
                    "sha256": sha256_file(output / name),
                }
            )

            # actor 编辑只在 logged camera 上执行，并在每次渲染后精确恢复参数。
            image_infos, camera_infos = dataset.full_image_set.get_image(
                frame_index * 3, camera_downscale
            )
            right_world = camera_infos["camera_to_world"][:3, 0].clone().cuda()
            for values in (image_infos, camera_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            rigid = trainer.models["RigidNodes"]
            edits = []
            opacity_before = rigid._opacities.detach().clone()
            rigid._opacities.fill_(-100.0)
            edits.append(("actor_remove_all", trainer(image_infos, camera_infos)))
            rigid._opacities.copy_(opacity_before)

            translation_before = rigid.instances_trans[frame_index].detach().clone()
            rigid.instances_trans[frame_index].add_(right_world[None, :])
            edits.append(("actor_translate_all_local_x_1m", trainer(image_infos, camera_infos)))
            rigid.instances_trans[frame_index].copy_(translation_before)

            shifted_frame = min(frame_index + 2, rigid.instances_trans.shape[0] - 1)
            quaternion_before = rigid.instances_quats[frame_index].detach().clone()
            rigid.instances_trans[frame_index].copy_(rigid.instances_trans[shifted_frame])
            rigid.instances_quats[frame_index].copy_(rigid.instances_quats[shifted_frame])
            edits.append(("actor_trajectory_time_shift_plus_2_frames", trainer(image_infos, camera_infos)))
            rigid.instances_trans[frame_index].copy_(translation_before)
            rigid.instances_quats[frame_index].copy_(quaternion_before)
            for variant, result in edits:
                name = f"frame{frame_index:03d}_{variant}.npz"
                save_render(output / name, result)
                rows.append(
                    {
                        "scene": args.scene,
                        "frontend": "streetgs",
                        "frame_index": frame_index,
                        "camera_id": 0,
                        "variant": variant,
                        "lateral_offset_m": 0.0,
                        "path": name,
                        "sha256": sha256_file(output / name),
                    }
                )

    after = sha256_file(checkpoint)
    if before != after:
        raise RuntimeError("StreetGS checkpoint before/after SHA 漂移")
    (output / "RENDER_MAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "worldsim_v6.r3_streetgs_worker.v1",
        "scene": args.scene,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "source_backup": str(backup),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "frames": frames,
        "offsets_m": offsets,
        "forward_extension_m": args.forward_extension,
        "development_content_read": True,
        "confirmation_content_read": False,
        "training_started": False,
        "render_count": len(rows),
        "wall_seconds": time.monotonic() - started,
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
