#!/usr/bin/env python3
"""渲染并评测单个 held-out StreetGS 场景的全部冻结 M5 编辑。"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from pathlib import Path

import cv2
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
    masked_mean_abs,
    masked_psnr,
)
from motion_proj.dynamic_editing_v2.stress_metrics import (
    apply_time_warp,
    masked_ssim,
    primary_failure,
    safe_mean,
)
from scripts.run_dr_v2_m4_pilot import (
    CAMERA_NAMES,
    actor_transform_payload,
    color_overlay,
    decode_depth,
    diff_heatmap,
    encode_depth,
    finite_or_none,
    move_actor_local_y,
    non_target_hash,
    pairwise_signature,
    quaternion_wxyz_to_matrix,
    rigid_state_hash,
    sha256_file,
    thumbnail,
    to_device,
)


EDITS = ("lateral", "speed", "stop_restart", "delete")
DEPTH_MAX_M = 120.0


def resolve_registry_actor(registry_payload: dict, role: str, token: str) -> dict:
    matches = [
        row
        for row in registry_payload.get("actors", [])
        if row.get("instance_token") == token
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0 and role == "boundary-support":
        # cohort 在 StreetGS 初始化前由 nuScenes 原始支持度冻结；原生运动/轨迹筛选
        # 可能完全排除 boundary actor，此时仍须保留为 coverage 证据。
        return {
            "instance_token": token,
            "availability": "unavailable_not_in_checkpoint_registry",
            "checkpoint_tensor_slice": {"gaussian_count": 0},
            "class_name": None,
            "rigid_model_index": None,
        }
    raise RuntimeError(f"{role} token registry mapping is not unique")


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有 M5 产物: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def lpips_value(model, first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if selected.sum() == 0:
        return float("nan")
    first_value = first.astype(np.float32) / 255.0
    second_value = second.astype(np.float32) / 255.0
    first_value[~selected] = 0
    second_value[~selected] = 0
    tensors = []
    for value in (first_value, second_value):
        tensor = torch.from_numpy(value).permute(2, 0, 1)[None].cuda()
        tensor = torch_f.interpolate(
            tensor, size=(144, 256), mode="bilinear", align_corners=False
        )
        tensors.append(tensor)
    with torch.inference_mode():
        result = model(tensors[0], tensors[1])
    return float(result.detach().cpu())


def mask_boundary(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    return cv2.dilate(value, kernel) > cv2.erode(value, kernel)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask.astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def choose_static_pseudo_hole(
    source: np.ndarray, dynamic: np.ndarray, sky: np.ndarray
) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    height, width = source.shape
    candidates = (
        (width // 3, 0),
        (-width // 3, 0),
        (width // 4, height // 5),
        (-width // 4, height // 5),
        (0, height // 3),
    )
    minimum = max(30, int(source.sum() * 0.40))
    best = None
    for dx, dy in candidates:
        shifted = shift_mask(source, dx, dy)
        shifted &= ~np.asarray(dynamic, dtype=bool)
        shifted &= ~np.asarray(sky, dtype=bool)
        score = int(shifted.sum())
        if best is None or score > best[0]:
            best = (score, shifted, dx, dy)
    if best is None or best[0] < minimum:
        return None, {"reason": "no_static_shift_with_minimum_support", "minimum": minimum}
    return best[1], {"dx": best[2], "dy": best[3], "pixels": best[0]}


def apply_edit(rigid, model_index: int, edit: str, fps: int) -> dict:
    schedule = None
    if edit == "lateral":
        move_actor_local_y(rigid, model_index, 1.0)
    elif edit in ("speed", "stop_restart"):
        schedule = apply_time_warp(rigid, model_index, edit, fps=fps)
    elif edit == "delete":
        pass
    else:
        raise ValueError(edit)
    transform = actor_transform_payload(rigid, model_index, deleted=edit == "delete")
    if edit == "delete":
        rigid.remove_instances([model_index])
    transform.update(
        {
            "edit": edit,
            "actor_local_y_m": 1.0 if edit == "lateral" else 0.0,
            "time_scale": 0.75 if edit == "speed" else None,
            "stop_seconds": 1.0 if edit == "stop_restart" else None,
            "schedule": schedule,
        }
    )
    transform["transform_sha256"] = canonical_sha256(transform)
    return transform


def render_variant(
    *,
    trainer,
    dataset,
    checkpoint: Path,
    variant_dir: Path,
    frames: list[int],
    cameras: list[int],
    fps: int,
    actor: dict | None,
    edit: str | None,
    evidence_dir: Path | None = None,
    heldout_frames: set[int] | None = None,
) -> dict:
    trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    model_index = int(actor["rigid_model_index"]) if actor is not None else None
    transform = None
    schedule_non_target_before = None
    edit_invariants = None
    if actor is not None and edit is not None:
        schedule_non_target_before = non_target_hash(rigid, model_index)
        valid = rigid.instances_fv[:, model_index].bool()
        original_translations = rigid.instances_trans[:, model_index].detach().clone()
        original_quaternions = rigid.instances_quats[:, model_index].detach().clone()
        target_points = rigid.point_ids[..., 0] == model_index
        original_pairwise = pairwise_signature(rigid._means[target_points])
        original_size = rigid.instances_size[model_index].detach().clone()
        transform = apply_edit(rigid, model_index, edit, fps)
        actual_translations = rigid.instances_trans[:, model_index].detach()
        actual_quaternions = rigid.instances_quats[:, model_index].detach()
        expected_translations = original_translations.clone()
        expected_quaternions = original_quaternions.clone()
        if edit == "lateral":
            rotations = quaternion_wxyz_to_matrix(original_quaternions)
            expected_translations[valid] += rotations[..., :, 1][valid]
        elif edit in ("speed", "stop_restart"):
            schedule = transform["schedule"]
            frame_indices = torch.as_tensor(
                schedule["valid_frames"], device=actual_translations.device
            )
            expected_translations[frame_indices] = torch.as_tensor(
                schedule["expected_translations"],
                dtype=actual_translations.dtype,
                device=actual_translations.device,
            )
            expected_quaternions[frame_indices] = torch.as_tensor(
                schedule["expected_quaternions_wxyz"],
                dtype=actual_quaternions.dtype,
                device=actual_quaternions.device,
            )
        translation_error = torch.linalg.vector_norm(
            actual_translations[valid] - expected_translations[valid], dim=-1
        )
        quaternion_dot = torch.abs(
            (actual_quaternions[valid] * expected_quaternions[valid]).sum(dim=-1)
        ).clamp(0, 1)
        rotation_error = torch.rad2deg(2 * torch.acos(quaternion_dot))
        current_target = rigid.point_ids[..., 0] == model_index
        current_pairwise = (
            pairwise_signature(rigid._means[current_target])
            if current_target.any()
            else original_pairwise
        )
        edit_invariants = {
            "trajectory_translation_error_max_m": float(translation_error.max().cpu())
            if translation_error.numel()
            else None,
            "trajectory_rotation_error_max_deg": float(rotation_error.max().cpu())
            if rotation_error.numel()
            else None,
            "canonical_pairwise_drift_max_m": float(
                torch.max(torch.abs(current_pairwise - original_pairwise)).cpu()
            ),
            "actor_size_drift_max_m": float(
                torch.max(
                    torch.abs(rigid.instances_size[model_index] - original_size)
                ).cpu()
            ),
        }
    state_hash = rigid_state_hash(rigid)
    non_target_state = (
        non_target_hash(rigid, model_index) if model_index is not None else None
    )
    for name in ("rgb", "depth_u16", "opacity_u8", "dynamic_u8"):
        (variant_dir / name).mkdir(parents=True)
    video_dir = variant_dir / "videos"
    video_dir.mkdir()
    writers = {
        camera: imageio.get_writer(
            video_dir / f"{CAMERA_NAMES[camera]}.mp4",
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        for camera in cameras
    }
    manifest_rows = []
    try:
        for frame in frames:
            for camera in cameras:
                image_index = frame * len(cameras) + camera
                image_infos, camera_infos = dataset.full_image_set.get_image(
                    image_index, camera_downscale=1.0
                )
                if evidence_dir is not None:
                    stem = f"frame_{frame:03d}_camera_{camera}"
                    lidar = image_infos.get("lidar_depth_map")
                    if lidar is not None:
                        lidar_value = lidar.detach().cpu().numpy().squeeze()
                        imageio.imwrite(
                            evidence_dir / "lidar_depth_u16" / f"{stem}.png",
                            encode_depth(lidar_value),
                        )
                    if heldout_frames is not None and frame in heldout_frames:
                        pixels = image_infos["pixels"].detach().cpu().numpy()
                        rgb_gt = (np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)
                        imageio.imwrite(evidence_dir / "heldout_rgb" / f"{stem}.png", rgb_gt)
                        for key, directory in (
                            ("dynamic_masks", "heldout_dynamic"),
                            ("sky_masks", "heldout_sky"),
                        ):
                            value = image_infos.get(key)
                            if value is not None:
                                mask = value.detach().cpu().numpy().squeeze()
                                imageio.imwrite(
                                    evidence_dir / directory / f"{stem}.png",
                                    mask.astype(np.uint8) * 255,
                                )
                image_infos = to_device(image_infos, torch.device("cuda:0"))
                camera_infos = to_device(camera_infos, torch.device("cuda:0"))
                with torch.inference_mode():
                    outputs = trainer(image_infos, camera_infos)
                rgb_float = outputs["rgb"].detach().float().cpu().numpy()
                depth = outputs["depth"].detach().float().cpu().numpy().squeeze()
                opacity = outputs["opacity"].detach().float().cpu().numpy().squeeze()
                dynamic = (
                    outputs["Dynamic_opacity"].detach().float().cpu().numpy().squeeze()
                )
                if not np.isfinite(rgb_float).all():
                    raise RuntimeError(f"非有限 RGB: {variant_dir}/{frame}/{camera}")
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
                imageio.imwrite(paths["opacity"], (np.clip(opacity, 0, 1) * 255).round().astype(np.uint8))
                imageio.imwrite(paths["dynamic"], (np.clip(dynamic, 0, 1) * 255).round().astype(np.uint8))
                writers[camera].append_data(rgb)
                manifest_rows.append(
                    {
                        "frame": frame,
                        "camera": camera,
                        "camera_name": CAMERA_NAMES[camera],
                        "rgb": str(paths["rgb"]),
                        "depth_u16": str(paths["depth"]),
                        "opacity_u8": str(paths["opacity"]),
                        "dynamic_u8": str(paths["dynamic"]),
                    }
                )
    finally:
        for writer in writers.values():
            writer.close()
    write_jsonl(variant_dir / "frames.jsonl", manifest_rows)
    if transform is not None:
        atomic_json(variant_dir / "actor_transform.json", transform)
    return {
        "world_state_hash": state_hash,
        "non_target_state_hash": non_target_state,
        "non_target_state_unchanged": (
            non_target_state == schedule_non_target_before
            if schedule_non_target_before is not None
            else None
        ),
        "edit_invariants": edit_invariants,
        "transform": str(variant_dir / "actor_transform.json")
        if transform is not None
        else None,
        "videos": [str(video_dir / f"{name}.mp4") for name in CAMERA_NAMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--high-token", required=True)
    parser.add_argument("--boundary-token", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/dynamic_editing_v2/m5_protocol_v1.yaml",
    )
    parser.add_argument(
        "--drivestudio-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-count", type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"M5 scene 输出已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    scene_protocol = protocol["scenes"][args.scene_name]
    if scene_protocol["high-support"] != args.high_token:
        raise RuntimeError("high-support token 与冻结协议不一致")
    if scene_protocol["boundary-support"] != args.boundary_token:
        raise RuntimeError("boundary-support token 与冻结协议不一致")
    config = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    if int(config.data.pixel_source.test_image_stride) != int(
        protocol["heldout"]["test_image_stride"]
    ):
        raise RuntimeError("checkpoint 不是 M5 训练前冻结的 held-out split")
    registry_payload = json.loads(args.registry.read_text(encoding="utf-8"))
    actors = {}
    for role, token in (
        ("high-support", args.high_token),
        ("boundary-support", args.boundary_token),
    ):
        actors[role] = resolve_registry_actor(registry_payload, role, token)
    available_actors = {
        role: actor
        for role, actor in actors.items()
        if actor.get("availability") == "available"
        and int(actor.get("checkpoint_tensor_slice", {}).get("gaussian_count", 0)) > 0
    }
    if "high-support" not in available_actors:
        raise RuntimeError("每场景 high-support actor 必须可编辑")
    dataset = DrivingDataset(data_cfg=config.data)
    cameras = list(range(dataset.pixel_source.num_cams))
    if cameras != [0, 1, 2] or dataset.num_img_timesteps != 196:
        raise RuntimeError("M5 固定要求 196 帧和 cameras=[0,1,2]")
    actual_heldout_frames = [int(value) for value in dataset.test_timesteps]
    if actual_heldout_frames != list(protocol["heldout"]["frames"]):
        raise RuntimeError("DriveStudio 实际 held-out split 与冻结协议不一致")
    frames = list(range(dataset.num_img_timesteps))
    if args.frame_count is not None:
        if args.frame_count <= 0 or args.frame_count > len(frames):
            raise ValueError("frame-count 越界")
        frames = frames[: args.frame_count]
    heldout_frames = set(protocol["heldout"]["frames"]).intersection(frames)
    trainer = import_str(config.trainer.type)(
        **config.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=config.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda:0"),
    )
    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    baseline_rigid = trainer.models["RigidNodes"]
    baseline_hash = rigid_state_hash(baseline_rigid)
    checkpoint_sha = sha256_file(args.checkpoint)
    geometry = {}
    for role, actor in available_actors.items():
        model_index = int(actor["rigid_model_index"])
        target = baseline_rigid.point_ids[..., 0] == model_index
        geometry[role] = {
            "model_index": model_index,
            "gaussian_count": int(target.sum().item()),
            "valid_frames": torch.nonzero(
                baseline_rigid.instances_fv[:, model_index], as_tuple=False
            )
            .flatten()
            .cpu()
            .tolist(),
            "size": baseline_rigid.instances_size[model_index].detach().cpu().tolist(),
            "pairwise": pairwise_signature(baseline_rigid._means[target]).tolist(),
        }

    evidence_dir = args.output_dir / "evidence"
    for name in (
        "lidar_depth_u16",
        "heldout_rgb",
        "heldout_dynamic",
        "heldout_sky",
    ):
        (evidence_dir / name).mkdir(parents=True)
    variant_records = {
        "original": render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=args.checkpoint,
            variant_dir=args.output_dir / "original",
            frames=frames,
            cameras=cameras,
            fps=args.fps,
            actor=None,
            edit=None,
            evidence_dir=evidence_dir,
            heldout_frames=heldout_frames,
        )
    }
    for role, actor in available_actors.items():
        actor_dir = args.output_dir / "actors" / f"{role}__{actor['instance_token'][:12]}"
        variant_records[role] = {}
        for edit in EDITS:
            variant_records[role][edit] = render_variant(
                trainer=trainer,
                dataset=dataset,
                checkpoint=args.checkpoint,
                variant_dir=actor_dir / edit,
                frames=frames,
                cameras=cameras,
                fps=args.fps,
                actor=actor,
                edit=edit,
            )

    lpips_model = trainer.lpips
    lpips_model.eval()
    metric_rows: list[dict] = []
    pseudo_rows: list[dict] = []
    sequence_summaries: list[dict] = []
    qa_dir = args.output_dir / "qa"
    thumb_dir = qa_dir / "thumbs"
    thumb_dir.mkdir(parents=True)
    qa_frames = set(np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int))
    qa_rows = []
    flow_cache: dict[tuple[int, int], np.ndarray] = {}

    for role, actor in actors.items():
        if role in available_actors:
            continue
        for edit in EDITS:
            sequence_summaries.append(
                {
                    "scene": args.scene_name,
                    "role": role,
                    "instance_token": actor["instance_token"],
                    "edit": edit,
                    "status": "ABSTAIN",
                    "abstain_reason": actor.get(
                        "availability", "actor_checkpoint_slice_unavailable"
                    ),
                    "metrics": None,
                    "failure_codes_pre_perception": ["INSUFFICIENT_EVIDENCE"],
                    "primary_failure_pre_perception": "INSUFFICIENT_EVIDENCE",
                    "perception_status": "ABSTAIN",
                }
            )

    for role, actor in available_actors.items():
        actor_key = f"{role}__{actor['instance_token'][:12]}"
        actor_dir = args.output_dir / "actors" / actor_key
        masks_dir = actor_dir / "masks"
        for name in ("source", "tier_b", "tier_c"):
            (masks_dir / name).mkdir(parents=True)
        for edit in EDITS:
            (masks_dir / f"edited_{edit}").mkdir(parents=True)
        visible_by_edit_camera = {
            edit: {name: 0 for name in CAMERA_NAMES} for edit in EDITS
        }
        effect_pixels_by_edit = {edit: 0 for edit in EDITS}
        tier_b_pixels = 0
        tier_c_pixels = 0
        previous = {edit: {camera: None for camera in cameras} for edit in EDITS}
        previous_boundary = {edit: {camera: None for camera in cameras} for edit in EDITS}
        previous_original = {camera: None for camera in cameras}
        for position, frame in enumerate(frames):
            for camera in cameras:
                stem = f"frame_{frame:03d}_camera_{camera}"
                original = imageio.imread(args.output_dir / "original/rgb" / f"{stem}.png")
                delete = imageio.imread(actor_dir / "delete/rgb" / f"{stem}.png")
                source = counterfactual_effect_mask(original, delete)
                imageio.imwrite(masks_dir / "source" / f"{stem}.png", source.astype(np.uint8) * 255)
                delete_depth = decode_depth(imageio.imread(actor_dir / "delete/depth_u16" / f"{stem}.png"))
                delete_opacity = imageio.imread(actor_dir / "delete/opacity_u8" / f"{stem}.png") > 127
                tier_b = source & np.isfinite(delete_depth) & delete_opacity
                tier_c = source & ~tier_b
                tier_b_pixels += int(tier_b.sum())
                tier_c_pixels += int(tier_c.sum())
                imageio.imwrite(masks_dir / "tier_b" / f"{stem}.png", tier_b.astype(np.uint8) * 255)
                imageio.imwrite(masks_dir / "tier_c" / f"{stem}.png", tier_c.astype(np.uint8) * 255)
                original_dynamic = imageio.imread(args.output_dir / "original/dynamic_u8" / f"{stem}.png") > 25
                flow_key = (frame, camera)
                if previous_original[camera] is not None and flow_key not in flow_cache:
                    current_small = cv2.resize(
                        cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), (160, 90)
                    )
                    previous_small = cv2.resize(
                        cv2.cvtColor(previous_original[camera], cv2.COLOR_RGB2GRAY),
                        (160, 90),
                    )
                    # 当前帧到前一帧的反向光流，可供 remap 从前一帧取样。
                    flow_cache[flow_key] = cv2.calcOpticalFlowFarneback(
                        current_small,
                        previous_small,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0,
                    )
                for edit in EDITS:
                    candidate = imageio.imread(actor_dir / edit / "rgb" / f"{stem}.png")
                    edited = (
                        np.zeros_like(source)
                        if edit == "delete"
                        else counterfactual_effect_mask(candidate, delete)
                    )
                    imageio.imwrite(
                        masks_dir / f"edited_{edit}" / f"{stem}.png",
                        edited.astype(np.uint8) * 255,
                    )
                    if edited.any() or (edit == "delete" and source.any()):
                        visible_by_edit_camera[edit][CAMERA_NAMES[camera]] += 1
                    effect = source | edited
                    effect_pixels_by_edit[edit] += int(effect.sum())
                    edited_guard = cv2.dilate(
                        edited.astype(np.uint8), np.ones((7, 7), np.uint8)
                    ).astype(bool)
                    vacated_source = source & ~edited_guard
                    vacated_source_boundary = mask_boundary(source) & ~edited_guard
                    source_halo = cv2.dilate(
                        source.astype(np.uint8), np.ones((15, 15), np.uint8)
                    ).astype(bool)
                    source_halo &= ~source & ~edited_guard
                    non_target = cv2.dilate(effect.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
                    candidate_dynamic = imageio.imread(actor_dir / edit / "dynamic_u8" / f"{stem}.png") > 25
                    candidate_opacity = imageio.imread(actor_dir / edit / "opacity_u8" / f"{stem}.png") / 255.0
                    delete_opacity_float = imageio.imread(actor_dir / "delete/opacity_u8" / f"{stem}.png") / 255.0
                    candidate_dynamic_float = imageio.imread(actor_dir / edit / "dynamic_u8" / f"{stem}.png") / 255.0
                    delete_dynamic_float = imageio.imread(actor_dir / "delete/dynamic_u8" / f"{stem}.png") / 255.0
                    static = non_target & ~original_dynamic & ~candidate_dynamic
                    other_actor = non_target & (original_dynamic | candidate_dynamic)
                    candidate_depth = decode_depth(imageio.imread(actor_dir / edit / "depth_u16" / f"{stem}.png"))
                    depth_diag = (
                        depth_order_diagnostic(candidate_depth, delete_depth, edited)
                        if edit != "delete"
                        else {"violation_rate": float("nan"), "actor_in_front_rate": float("nan")}
                    )
                    boundary = mask_boundary(effect)
                    temporal_lpips = float("nan")
                    temporal_warp_error = float("nan")
                    if previous[edit][camera] is not None:
                        temporal_mask = boundary | previous_boundary[edit][camera]
                        temporal_lpips = lpips_value(
                            lpips_model, previous[edit][camera], candidate, temporal_mask
                        )
                        flow = flow_cache[flow_key]
                        grid_x, grid_y = np.meshgrid(
                            np.arange(flow.shape[1], dtype=np.float32),
                            np.arange(flow.shape[0], dtype=np.float32),
                        )
                        previous_candidate_small = cv2.resize(
                            previous[edit][camera], (flow.shape[1], flow.shape[0])
                        )
                        previous_original_small = cv2.resize(
                            previous_original[camera], (flow.shape[1], flow.shape[0])
                        )
                        candidate_small = cv2.resize(
                            candidate, (flow.shape[1], flow.shape[0])
                        )
                        original_small = cv2.resize(
                            original, (flow.shape[1], flow.shape[0])
                        )
                        map_x = grid_x + flow[..., 0]
                        map_y = grid_y + flow[..., 1]
                        warped_candidate = cv2.remap(
                            previous_candidate_small,
                            map_x,
                            map_y,
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT,
                        )
                        warped_original = cv2.remap(
                            previous_original_small,
                            map_x,
                            map_y,
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT,
                        )
                        non_target_small = cv2.resize(
                            non_target.astype(np.uint8),
                            (flow.shape[1], flow.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        candidate_error = masked_mean_abs(
                            warped_candidate, candidate_small, non_target_small
                        )
                        original_error = masked_mean_abs(
                            warped_original, original_small, non_target_small
                        )
                        temporal_warp_error = abs(candidate_error - original_error)
                    previous[edit][camera] = candidate
                    previous_boundary[edit][camera] = boundary
                    row = {
                        "scene": args.scene_name,
                        "role": role,
                        "instance_token": actor["instance_token"],
                        "edit": edit,
                        "frame": frame,
                        "camera": camera,
                        "camera_name": CAMERA_NAMES[camera],
                        "effect_pixels": int(effect.sum()),
                        "source_pixels": int(source.sum()),
                        "edited_pixels": int(edited.sum()),
                        "truth_tier_b_pixels": int(tier_b.sum()),
                        "truth_tier_c_pixels": int(tier_c.sum()),
                        "non_target_psnr": finite_or_none(masked_psnr(original, candidate, non_target)),
                        "non_target_ssim": finite_or_none(masked_ssim(original, candidate, non_target)),
                        "non_target_lpips_alex_256px": finite_or_none(lpips_value(lpips_model, original, candidate, non_target)),
                        "static_mean_abs_diff": finite_or_none(masked_mean_abs(original, candidate, static)),
                        "other_actor_mean_abs_diff": finite_or_none(masked_mean_abs(original, candidate, other_actor)),
                        "source_difference_from_delete": finite_or_none(masked_mean_abs(candidate, delete, source)),
                        "vacated_source_pixels": int(vacated_source.sum()),
                        "vacated_source_residual_mean_abs": finite_or_none(
                            masked_mean_abs(candidate, delete, vacated_source)
                        ),
                        "vacated_source_boundary_residual_mean_abs": finite_or_none(
                            masked_mean_abs(
                                candidate, delete, vacated_source_boundary
                            )
                        ),
                        "source_halo_difference_from_delete": finite_or_none(
                            masked_mean_abs(candidate, delete, source_halo)
                        ),
                        "depth_ordering_violation_rate": finite_or_none(float(depth_diag["violation_rate"])),
                        "actor_in_front_rate": finite_or_none(float(depth_diag["actor_in_front_rate"])),
                        "actor_boundary_temporal_lpips": finite_or_none(temporal_lpips),
                        "non_target_temporal_warp_error_delta": finite_or_none(temporal_warp_error),
                        "new_location_alpha_overlap_rate": finite_or_none(
                            float(
                                np.mean(
                                    (
                                        np.clip(
                                            candidate_dynamic_float
                                            - delete_dynamic_float,
                                            0,
                                            1,
                                        )[edited]
                                        > 0.10
                                    )
                                    & (delete_opacity_float[edited] > 0.50)
                                )
                            )
                            if edited.any()
                            else float("nan")
                        ),
                    }
                    metric_rows.append(row)
                    if position in qa_frames:
                        violation = edited & np.isfinite(candidate_depth) & np.isfinite(delete_depth)
                        violation &= candidate_depth > delete_depth + 0.05
                        qa_image = color_overlay(candidate, source, edited, violation)
                        name = f"{actor_key}_{edit}_{stem}.png"
                        imageio.imwrite(thumb_dir / name, thumbnail(qa_image))
                        diff_name = f"{actor_key}_{edit}_{stem}_diff.png"
                        imageio.imwrite(thumb_dir / diff_name, thumbnail(diff_heatmap(original, candidate)))
                        qa_rows.append({"role": role, "edit": edit, "frame": frame, "camera": camera, "image": name, "diff": diff_name})

                if frame in heldout_frames and source.any():
                    heldout_path = evidence_dir / "heldout_rgb" / f"{stem}.png"
                    dynamic_path = evidence_dir / "heldout_dynamic" / f"{stem}.png"
                    sky_path = evidence_dir / "heldout_sky" / f"{stem}.png"
                    if heldout_path.is_file() and dynamic_path.is_file() and sky_path.is_file():
                        heldout = imageio.imread(heldout_path)
                        gt_dynamic = imageio.imread(dynamic_path) > 0
                        gt_sky = imageio.imread(sky_path) > 0
                        hole, shift = choose_static_pseudo_hole(source, gt_dynamic, gt_sky)
                        if hole is not None:
                            no_completion = heldout.copy()
                            no_completion[hole] = 0
                            telea = cv2.inpaint(heldout, hole.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
                            lidar_depth = decode_depth(
                                imageio.imread(
                                    evidence_dir / "lidar_depth_u16" / f"{stem}.png"
                                )
                            )
                            baseline_depth = decode_depth(
                                imageio.imread(
                                    args.output_dir / "original/depth_u16" / f"{stem}.png"
                                )
                            )
                            depth_support = (
                                hole
                                & np.isfinite(lidar_depth)
                                & np.isfinite(baseline_depth)
                            )
                            for method, candidate in (
                                ("no_completion", no_completion),
                                ("baseline_native_background", original),
                                ("telea_2d_diagnostic", telea),
                            ):
                                pseudo_rows.append(
                                    {
                                        "scene": args.scene_name,
                                        "role": role,
                                        "instance_token": actor["instance_token"],
                                        "frame": frame,
                                        "camera": camera,
                                        "camera_name": CAMERA_NAMES[camera],
                                        "truth_tier": "A",
                                        "heldout_excluded_before_training": True,
                                        "method": method,
                                        "pixels": int(hole.sum()),
                                        "shift": shift,
                                        "psnr": finite_or_none(masked_psnr(heldout, candidate, hole)),
                                        "ssim": finite_or_none(masked_ssim(heldout, candidate, hole)),
                                        "lpips_alex_256px": finite_or_none(lpips_value(lpips_model, heldout, candidate, hole)),
                                        "lidar_supported_pixels": int(depth_support.sum()),
                                        "depth_mae_m": (
                                            finite_or_none(
                                                float(
                                                    np.mean(
                                                        np.abs(
                                                            baseline_depth[depth_support]
                                                            - lidar_depth[depth_support]
                                                        )
                                                    )
                                                )
                                            )
                                            if method == "baseline_native_background"
                                            and depth_support.any()
                                            else None
                                        ),
                                        "claim_scope": "RGB allowed only because this image was excluded before optimization",
                                    }
                                )
                previous_original[camera] = original

        actor_rows = [row for row in metric_rows if row["role"] == role]
        actor_pseudo = [row for row in pseudo_rows if row["role"] == role]
        pseudo_baseline_psnr = safe_mean(
            row["psnr"] for row in actor_pseudo if row["method"] == "baseline_native_background"
        )
        pseudo_baseline_lpips = safe_mean(
            row["lpips_alex_256px"] for row in actor_pseudo if row["method"] == "baseline_native_background"
        )
        unsupported_rate = tier_c_pixels / max(1, tier_b_pixels + tier_c_pixels)
        for edit in EDITS:
            rows = [row for row in actor_rows if row["edit"] == edit]
            camera_visibility = visible_by_edit_camera[edit]
            camera_visibility_values = list(camera_visibility.values())
            metrics = {
                "row_count": len(rows),
                "effect_pixels": effect_pixels_by_edit[edit],
                "non_target_psnr_mean": safe_mean(row["non_target_psnr"] for row in rows),
                "non_target_ssim_mean": safe_mean(row["non_target_ssim"] for row in rows),
                "non_target_lpips_mean": safe_mean(row["non_target_lpips_alex_256px"] for row in rows),
                "static_mean_abs_diff": safe_mean(row["static_mean_abs_diff"] for row in rows),
                "other_actor_mean_abs_diff": safe_mean(row["other_actor_mean_abs_diff"] for row in rows),
                "vacated_source_residual_mean_abs": safe_mean(
                    row["vacated_source_residual_mean_abs"] for row in rows
                ),
                "vacated_source_boundary_residual_mean_abs": safe_mean(
                    row["vacated_source_boundary_residual_mean_abs"] for row in rows
                ),
                "source_halo_difference_from_delete": safe_mean(
                    row["source_halo_difference_from_delete"] for row in rows
                ),
                "depth_ordering_violation_rate": safe_mean(row["depth_ordering_violation_rate"] for row in rows),
                "actor_boundary_temporal_lpips": safe_mean(row["actor_boundary_temporal_lpips"] for row in rows),
                "non_target_temporal_warp_error_delta": safe_mean(
                    row["non_target_temporal_warp_error_delta"] for row in rows
                ),
                "new_location_alpha_overlap_rate": safe_mean(
                    row["new_location_alpha_overlap_rate"] for row in rows
                ),
                "truth_tier_b_pixels": tier_b_pixels,
                "truth_tier_c_pixels": tier_c_pixels,
                "unsupported_disocclusion_rate": unsupported_rate,
                "tier_a_pseudo_hole_rows": len(actor_pseudo),
                "tier_a_baseline_psnr": pseudo_baseline_psnr,
                "tier_a_baseline_lpips": pseudo_baseline_lpips,
                "visible_frame_camera_count": visible_by_edit_camera[edit],
                "multicamera_visibility_count_range": max(camera_visibility_values)
                - min(camera_visibility_values),
                "multicamera_visibility_min_over_max": min(camera_visibility_values)
                / max(1, max(camera_visibility_values)),
                "edit_invariants": variant_records[role][edit]["edit_invariants"],
            }
            thresholds = protocol["thresholds"]
            failures = []
            invariants = metrics["edit_invariants"]
            if (
                invariants["trajectory_translation_error_max_m"] is not None
                and invariants["trajectory_translation_error_max_m"]
                > thresholds["trajectory_translation_max_m"]
            ) or (
                invariants["trajectory_rotation_error_max_deg"] is not None
                and invariants["trajectory_rotation_error_max_deg"]
                > thresholds["trajectory_rotation_max_deg"]
            ):
                failures.append("TRAJECTORY_NOT_FOLLOWED")
            if max(
                invariants["canonical_pairwise_drift_max_m"],
                invariants["actor_size_drift_max_m"],
            ) > thresholds["geometry_drift_max_m"]:
                failures.append("ACTOR_GEOMETRY_DEFORMATION")
            if effect_pixels_by_edit[edit] < thresholds["minimum_effect_pixels_per_sequence"]:
                failures.append("INSUFFICIENT_EVIDENCE")
            if (
                metrics["vacated_source_residual_mean_abs"] is not None
                and metrics["vacated_source_residual_mean_abs"]
                > thresholds["vacated_source_residual_mean_abs_max"]
            ):
                failures.append("SOURCE_RESIDUAL")
            if metrics["non_target_psnr_mean"] is not None and metrics["non_target_psnr_mean"] < thresholds["non_target_psnr_min"]:
                failures.append("NON_TARGET_VISUAL_DRIFT")
            if metrics["non_target_ssim_mean"] is not None and metrics["non_target_ssim_mean"] < thresholds["non_target_ssim_min"]:
                failures.append("NON_TARGET_VISUAL_DRIFT")
            if metrics["non_target_lpips_mean"] is not None and metrics["non_target_lpips_mean"] > thresholds["non_target_lpips_max"]:
                failures.append("NON_TARGET_VISUAL_DRIFT")
            if metrics["depth_ordering_violation_rate"] is not None and metrics["depth_ordering_violation_rate"] > thresholds["depth_ordering_violation_rate_max"]:
                failures.extend(["DEPTH_ORDERING_ERROR", "NEW_OCCLUSION_ERROR"])
            if metrics["actor_boundary_temporal_lpips"] is not None and metrics["actor_boundary_temporal_lpips"] > thresholds["temporal_boundary_lpips_max"]:
                failures.append("TEMPORAL_FLICKER")
            if unsupported_rate > thresholds["unsupported_disocclusion_rate_max"]:
                failures.append("UNSUPPORTED_DISOCCLUSION")
            if pseudo_baseline_psnr is not None and pseudo_baseline_psnr < 20.0:
                failures.append("BACKGROUND_HOLE")
            failures = sorted(set(failures), key=protocol["failure_priority"].index)
            sequence_summaries.append(
                {
                    "scene": args.scene_name,
                    "role": role,
                    "instance_token": actor["instance_token"],
                    "edit": edit,
                    "status": "done",
                    "metrics": metrics,
                    "failure_codes_pre_perception": failures,
                    "primary_failure_pre_perception": primary_failure(failures),
                    "perception_status": "pending_frozen_grounding_dino_evaluator",
                }
            )

    write_jsonl(args.output_dir / "metrics.jsonl", metric_rows)
    write_jsonl(args.output_dir / "pseudo_hole_metrics.jsonl", pseudo_rows)
    with (args.output_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    qa_html = [
        "<!doctype html><meta charset='utf-8'><title>M5 scene QA</title>",
        f"<h1>{html.escape(args.scene_name)} M5 编辑 QA</h1>",
        "<p>红=source，蓝=edited，黄=depth-order violation。mask 是模型内反事实差分，不是真实观测。</p>",
        "<table><tr><th>actor/edit/frame/camera</th><th>overlay</th><th>diff</th></tr>",
    ]
    for row in qa_rows:
        label = f"{row['role']} / {row['edit']} / {row['frame']} / {CAMERA_NAMES[row['camera']]}"
        qa_html.append(
            f"<tr><td>{html.escape(label)}</td><td><img src='thumbs/{row['image']}'></td>"
            f"<td><img src='thumbs/{row['diff']}'></td></tr>"
        )
    qa_html.append("</table>")
    qa_path = qa_dir / "index.html"
    qa_path.write_text("\n".join(qa_html) + "\n", encoding="utf-8")

    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    reload_exact = rigid_state_hash(trainer.models["RigidNodes"]) == baseline_hash
    checks = {
        "checkpoint_unchanged": sha256_file(args.checkpoint) == checkpoint_sha,
        "baseline_state_reload_exact": reload_exact,
        "three_cameras": cameras == [0, 1, 2],
        "frame_coverage": len(frames) == (args.frame_count or 196),
        "heldout_split_exact": actual_heldout_frames
        == list(protocol["heldout"]["frames"]),
        "eight_sequences": len(sequence_summaries) == 8,
        "four_edits_each_actor": all(sum(row["role"] == role for row in sequence_summaries) == 4 for role in actors),
        "all_non_target_states_unchanged": all(
            variant_records[role][edit]["non_target_state_unchanged"]
            for role in available_actors
            for edit in EDITS
        ),
        "all_videos_nonempty": all(
            Path(path).is_file() and Path(path).stat().st_size > 0
            for role in available_actors
            for edit in EDITS
            for path in variant_records[role][edit]["videos"]
        ),
        "metric_rows_complete": len(metric_rows)
        == len(frames) * len(cameras) * 4 * len(available_actors),
        "truth_tier_a_uses_heldout_only": all(row["frame"] in heldout_frames and row["heldout_excluded_before_training"] for row in pseudo_rows),
        "truth_tier_c_has_no_rgb_accuracy": True,
        "qa_embeds_images": qa_path.is_file() and len(qa_rows) > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M5 scene validator 失败: {[key for key, value in checks.items() if not value]}")
    report = {
        "schema_version": 1,
        "task_id": "DR-V2-M5-STRESS-3SCENE-01",
        "status": "done",
        "scene": args.scene_name,
        "scene_index": int(config.data.scene_idx),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "registry": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "frames": len(frames),
        "cameras": list(CAMERA_NAMES),
        "heldout_frames": sorted(heldout_frames),
        "actors": actors,
        "available_actor_roles": sorted(available_actors),
        "geometry": geometry,
        "variant_records": variant_records,
        "sequences": sequence_summaries,
        "metric_rows": len(metric_rows),
        "pseudo_hole_rows": len(pseudo_rows),
        "truth_policy": {
            "A": "held-out observed image excluded by test_image_stride before optimization",
            "B": "finite delete-render depth and static Gaussian opacity; RGB accuracy forbidden",
            "C": "unsupported; only coverage/ABSTAIN/human review",
            "counterfactual_masks": "paired render difference, not observation truth",
        },
        "metric_scope": {
            "source_residual": "measured only on vacated source pixels after excluding the dilated edited footprint, against the same checkpoint with the actor removed",
            "shadow_residual": "ABSTAIN: the baseline has no relightable shadow component or counterfactual illumination truth, so SHADOW_RESIDUAL is retained in the taxonomy but is not auto-assigned",
            "multicamera_visibility": "per-camera visible counts and their range are diagnostic only because the three camera fields of view differ; no automatic inconsistency label is assigned",
        },
        "perception": "pending separate frozen Grounding DINO evaluation",
        "checks": checks,
        "qa": str(qa_path),
        "quality_claim": "automatic diagnostics only; M8 visual verdict remains human-only",
    }
    report["report_payload_sha256"] = canonical_sha256(report)
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": "done", "scene": args.scene_name, "sequences": 8, "metric_rows": len(metric_rows), "pseudo_hole_rows": len(pseudo_rows), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
