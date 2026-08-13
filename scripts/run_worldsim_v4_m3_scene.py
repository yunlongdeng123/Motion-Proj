#!/usr/bin/env python3
"""在真实 DriveStudio 连续 clip 上运行一个 M3 场景的五臂时序编辑。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping

import cv2
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.temporal_protocol import (  # noqa: E402
    TEMPORAL_ARMS,
    apply_actor_local_translation,
    build_arm_trajectories,
    evidence_memory_weights,
    resample_se3_transforms,
)


TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"
OPERATIONS = ("REMOVE", "LATERAL", "INSERT")
GAUSSIAN_ARMS = TEMPORAL_ARMS[:-1]


class M3SceneRunError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    value = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip()
    )


def verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding["path"])).resolve()
    if not path.is_file():
        raise M3SceneRunError(f"{label} 不存在: {path}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise M3SceneRunError(
            f"{label} SHA 不匹配: expected={binding['sha256']} actual={actual}"
        )
    return path


def tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rigid_state_hash(rigid: Any) -> str:
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


def non_target_hash(rigid: Any, model_index: int) -> str:
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


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value /= np.linalg.norm(value, axis=-1, keepdims=True)
    return Rotation.from_quat(value[..., [1, 2, 3, 0]]).as_matrix()


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    value = xyzw[..., [3, 0, 1, 2]]
    # q 与 -q 等价；固定相邻符号避免插值前后的非必要翻转。
    for index in range(1, len(value)):
        if float(np.dot(value[index - 1], value[index])) < 0.0:
            value[index] *= -1.0
    return value


def actor_transforms(rigid: Any, frames: list[int], model_index: int) -> np.ndarray:
    translations = (
        rigid.instances_trans[frames, model_index].detach().float().cpu().numpy()
    )
    quaternions = (
        rigid.instances_quats[frames, model_index].detach().float().cpu().numpy()
    )
    transforms = np.repeat(np.eye(4)[None], len(frames), axis=0)
    transforms[:, :3, :3] = quaternion_wxyz_to_matrix(quaternions)
    transforms[:, :3, 3] = translations
    return transforms


def apply_dense_schedule(
    rigid: Any,
    *,
    dense_frames: list[int],
    model_index: int,
    transforms: np.ndarray,
) -> None:
    if len(dense_frames) != len(transforms):
        raise ValueError("dense frame/transform count mismatch")
    quaternions = matrix_to_quaternion_wxyz(transforms[:, :3, :3])
    device = rigid.instances_trans.device
    with torch.no_grad():
        rigid.instances_trans[dense_frames, model_index] = torch.as_tensor(
            transforms[:, :3, 3],
            dtype=rigid.instances_trans.dtype,
            device=device,
        )
        rigid.instances_quats[dense_frames, model_index] = torch.as_tensor(
            quaternions,
            dtype=rigid.instances_quats.dtype,
            device=device,
        )


def stem(frame: int, camera: int) -> str:
    return f"frame_{frame:03d}_camera_{camera}"


def rgb_path(root: Path, frame: int, camera: int) -> Path:
    return root / "rgb" / f"{stem(frame, camera)}.png"


def render_variant(
    *,
    trainer: Any,
    dataset: Any,
    checkpoint: Path,
    output_dir: Path,
    frames: list[int],
    cameras: list[int],
    model_index: int,
    dense_frames: list[int],
    dense_transforms: np.ndarray | None,
    remove: bool,
    save_ground_truth: bool,
    baseline_state_hash: str,
    baseline_non_target_hash: str,
) -> dict[str, Any]:
    trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    if dense_transforms is not None:
        apply_dense_schedule(
            rigid,
            dense_frames=dense_frames,
            model_index=model_index,
            transforms=dense_transforms,
        )
    non_target_after_schedule = non_target_hash(rigid, model_index)
    if remove:
        rigid.remove_instances([model_index])
    state_hash = rigid_state_hash(rigid)
    output_dir.joinpath("rgb").mkdir(parents=True)
    ground_truth_dir = output_dir.parent / "ground_truth"
    if save_ground_truth:
        ground_truth_dir.mkdir(exist_ok=True)
    rows = []
    device = torch.device("cuda:0")
    for frame in frames:
        for camera in cameras:
            image_index = frame * len(cameras) + camera
            image_infos, camera_infos = dataset.full_image_set.get_image(
                image_index, camera_downscale=1.0
            )
            if save_ground_truth:
                pixels = image_infos["pixels"].detach().cpu().numpy()
                ground_truth = (
                    np.clip(pixels, 0.0, 1.0) * 255.0
                ).round().astype(np.uint8)
                imageio.imwrite(
                    ground_truth_dir / f"{stem(frame, camera)}.png", ground_truth
                )
            image_infos = to_device(image_infos, device)
            camera_infos = to_device(camera_infos, device)
            with torch.inference_mode():
                outputs = trainer(image_infos, camera_infos)
            rgb = outputs["rgb"].detach().float().cpu().numpy()
            if not np.isfinite(rgb).all() or rgb.size == 0:
                raise M3SceneRunError(
                    f"非有限或空渲染: {output_dir}/{frame}/{camera}"
                )
            image = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            path = rgb_path(output_dir, frame, camera)
            imageio.imwrite(path, image)
            rows.append(
                {
                    "frame": frame,
                    "camera": camera,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "shape": list(image.shape),
                }
            )
    write_jsonl(output_dir / "frames.jsonl", rows)
    return {
        "state_hash": state_hash,
        "non_target_state_hash": non_target_after_schedule,
        "non_target_state_unchanged": non_target_after_schedule
        == baseline_non_target_hash,
        "base_state_unchanged": state_hash == baseline_state_hash,
        "remove": remove,
        "frame_count": len(frames),
        "camera_count": len(cameras),
    }


def effect_mask(left: np.ndarray, right: np.ndarray, threshold: int = 2) -> np.ndarray:
    delta = np.max(
        np.abs(left.astype(np.int16) - right.astype(np.int16)), axis=2
    )
    return delta > threshold


def flow_current_to_previous(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    current_gray = cv2.resize(
        cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), (160, 90)
    )
    previous_gray = cv2.resize(
        cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY), (160, 90)
    )
    return cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def remap_previous(value: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    small = cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(
        small,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def build_full_warp_variant(
    *,
    base_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    frames: list[int],
    cameras: list[int],
    alpha: float,
) -> None:
    output_dir.joinpath("rgb").mkdir(parents=True)
    rows = []
    for camera in cameras:
        previous_base = None
        previous_filtered_delta = None
        for frame in frames:
            base = imageio.imread(rgb_path(base_dir, frame, camera))
            evidence = imageio.imread(rgb_path(evidence_dir, frame, camera))
            current_delta = evidence.astype(np.float32) - base.astype(np.float32)
            if previous_base is None:
                filtered_delta = current_delta
            else:
                flow = flow_current_to_previous(base, previous_base)
                warped = remap_previous(previous_filtered_delta, flow)
                warped = cv2.resize(
                    warped,
                    (base.shape[1], base.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                support = np.max(np.abs(current_delta), axis=2) > 2.0
                warped_support = np.max(np.abs(warped), axis=2) > 2.0
                union = cv2.dilate(
                    (support | warped_support).astype(np.uint8),
                    np.ones((5, 5), np.uint8),
                ).astype(bool)
                filtered_delta = current_delta.copy()
                filtered_delta[union] = (
                    (1.0 - alpha) * current_delta[union] + alpha * warped[union]
                )
            image = np.clip(
                base.astype(np.float32) + filtered_delta, 0.0, 255.0
            ).round().astype(np.uint8)
            path = rgb_path(output_dir, frame, camera)
            imageio.imwrite(path, image)
            rows.append(
                {
                    "frame": frame,
                    "camera": camera,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "shape": list(image.shape),
                }
            )
            previous_base = base
            previous_filtered_delta = filtered_delta
    write_jsonl(output_dir / "frames.jsonl", rows)


def masked_psnr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    delta = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(np.square(delta[selected])))
    return 99.0 if mse == 0.0 else float(10.0 * math.log10(255.0**2 / mse))


def masked_ssim(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    left_value = left.copy()
    right_value = right.copy()
    left_value[~selected] = right_value[~selected]
    return float(
        structural_similarity(
            left_value, right_value, channel_axis=2, data_range=255
        )
    )


def masked_lpips(
    model: Any, left: np.ndarray, right: np.ndarray, mask: np.ndarray
) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    values = []
    for image in (left, right):
        value = image.astype(np.float32) / 255.0
        value[~selected] = 0.0
        tensor = torch.from_numpy(value).permute(2, 0, 1)[None].cuda()
        tensor = torch.nn.functional.interpolate(
            tensor, size=(144, 256), mode="bilinear", align_corners=False
        )
        values.append(tensor)
    if hasattr(model, "reset"):
        model.reset()
    with torch.inference_mode():
        result = float(model(values[0], values[1]).detach().cpu())
    if hasattr(model, "reset"):
        model.reset()
    return result


def boundary(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    return cv2.dilate(value, kernel) > cv2.erode(value, kernel)


def boundary_f1(left: np.ndarray, right: np.ndarray, tolerance: int = 3) -> float:
    left_boundary = boundary(left, 1)
    right_boundary = boundary(right, 1)
    if not left_boundary.any() and not right_boundary.any():
        return 1.0
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    right_near = cv2.dilate(right_boundary.astype(np.uint8), kernel).astype(bool)
    left_near = cv2.dilate(left_boundary.astype(np.uint8), kernel).astype(bool)
    precision = (
        float(np.mean(right_near[left_boundary])) if left_boundary.any() else 0.0
    )
    recall = (
        float(np.mean(left_near[right_boundary])) if right_boundary.any() else 0.0
    )
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def finite_std(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return None if not finite else float(np.std(finite))


def evaluate_sequences(
    *,
    artifact_root: Path,
    operations: tuple[str, ...],
    frames: list[int],
    cameras: list[int],
    trainer: Any,
    trajectory_audit: Mapping[str, Any],
    minimum_effect_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_dir = artifact_root / "BASE"
    remove_dir = artifact_root / "REMOVE_SHARED"
    ground_truth_dir = artifact_root / "ground_truth"
    rows: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    source_masks: dict[tuple[int, int], np.ndarray] = {}
    flow_cache: dict[tuple[int, int], np.ndarray] = {}
    for frame in frames:
        for camera in cameras:
            base = imageio.imread(rgb_path(base_dir, frame, camera))
            removed = imageio.imread(rgb_path(remove_dir, frame, camera))
            source_masks[(frame, camera)] = effect_mask(base, removed)

    for operation in operations:
        for arm in TEMPORAL_ARMS:
            variant_dir = (
                remove_dir
                if operation == "REMOVE"
                else artifact_root / operation / arm
            )
            sequence_rows: list[dict[str, Any]] = []
            iou_values: list[float] = []
            boundary_values: list[float] = []
            centroid_values: list[np.ndarray] = []
            contact_values: list[float] = []
            rendered_effect_by_frame = {frame: 0 for frame in frames}
            semantic_reintroduction = 0
            for camera in cameras:
                previous_candidate = None
                previous_base = None
                previous_effect = None
                for frame in frames:
                    base = imageio.imread(rgb_path(base_dir, frame, camera))
                    removed = imageio.imread(rgb_path(remove_dir, frame, camera))
                    candidate = imageio.imread(rgb_path(variant_dir, frame, camera))
                    ground_truth = imageio.imread(
                        ground_truth_dir / f"{stem(frame, camera)}.png"
                    )
                    source = source_masks[(frame, camera)]
                    inserted = (
                        np.zeros_like(source)
                        if operation == "REMOVE"
                        else effect_mask(candidate, removed)
                    )
                    effect = source | inserted
                    rendered_effect_by_frame[frame] += int(effect.sum())
                    guard = cv2.dilate(
                        effect.astype(np.uint8), np.ones((9, 9), np.uint8)
                    ).astype(bool)
                    non_target = ~guard
                    row = {
                        "operation": operation,
                        "arm": arm,
                        "frame": frame,
                        "camera": camera,
                        "source_pixels": int(source.sum()),
                        "inserted_pixels": int(inserted.sum()),
                        "effect_pixels": int(effect.sum()),
                        "non_target_psnr": masked_psnr(
                            candidate, ground_truth, non_target
                        ),
                        "non_target_ssim": masked_ssim(
                            candidate, ground_truth, non_target
                        ),
                        "non_target_lpips_alex": masked_lpips(
                            trainer.lpips, candidate, ground_truth, non_target
                        ),
                        "temporal_lpips": None,
                        "warp_l1_delta": None,
                        "warp_lpips_alex": None,
                    }
                    if operation == "REMOVE":
                        semantic_reintroduction += int(
                            np.any(candidate != removed, axis=2)[source].sum()
                        )
                    if effect.any():
                        ys, xs = np.nonzero(effect)
                        centroid_values.append(
                            np.asarray([float(xs.mean()), float(ys.mean())])
                        )
                        contact_values.append(float(np.percentile(ys, 90)))
                    if previous_candidate is not None:
                        key = (frame, camera)
                        if key not in flow_cache:
                            flow_cache[key] = flow_current_to_previous(base, previous_base)
                        flow = flow_cache[key]
                        warped_candidate = remap_previous(previous_candidate, flow)
                        warped_base = remap_previous(previous_base, flow)
                        current_small = cv2.resize(candidate, (160, 90))
                        base_small = cv2.resize(base, (160, 90))
                        current_delta = current_small.astype(np.float32) - base_small.astype(np.float32)
                        warped_delta = warped_candidate.astype(np.float32) - warped_base.astype(np.float32)
                        current_effect = cv2.resize(
                            effect.astype(np.uint8),
                            (160, 90),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        warped_effect = remap_previous(
                            previous_effect.astype(np.uint8) * 255, flow
                        ) > 127
                        temporal_mask = current_effect | warped_effect
                        if temporal_mask.any():
                            row["warp_l1_delta"] = float(
                                np.mean(
                                    np.abs(current_delta - warped_delta)[temporal_mask]
                                )
                                / 255.0
                            )
                            full_temporal_mask = cv2.resize(
                                temporal_mask.astype(np.uint8),
                                (candidate.shape[1], candidate.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                            warped_candidate_full = cv2.resize(
                                warped_candidate,
                                (candidate.shape[1], candidate.shape[0]),
                            )
                            row["warp_lpips_alex"] = masked_lpips(
                                trainer.lpips,
                                candidate,
                                warped_candidate_full,
                                full_temporal_mask,
                            )
                            row["temporal_lpips"] = masked_lpips(
                                trainer.lpips,
                                candidate,
                                previous_candidate,
                                boundary(effect) | boundary(previous_effect),
                            )
                        union = np.logical_or(current_effect, warped_effect).sum()
                        iou_values.append(
                            1.0
                            if union == 0
                            else float(
                                np.logical_and(current_effect, warped_effect).sum()
                                / union
                            )
                        )
                        boundary_values.append(
                            boundary_f1(current_effect, warped_effect)
                        )
                    rows.append(row)
                    sequence_rows.append(row)
                    previous_candidate = candidate
                    previous_base = base
                    previous_effect = effect
            centroids = np.asarray(centroid_values)
            centroid_jitter = None
            if len(centroids) >= 3:
                acceleration = np.diff(centroids, n=2, axis=0)
                centroid_jitter = float(
                    np.sqrt(np.mean(np.sum(acceleration**2, axis=1)))
                )
            render_success = all(
                pixels >= minimum_effect_pixels
                for pixels in rendered_effect_by_frame.values()
            )
            trajectory = trajectory_audit.get(arm, {})
            sequences.append(
                {
                    "operation": operation,
                    "arm": arm,
                    "status": "done",
                    "operation_success": bool(render_success),
                    "rendered_effect_pixels_by_frame": rendered_effect_by_frame,
                    "non_target_psnr": finite_mean(
                        row["non_target_psnr"] for row in sequence_rows
                    ),
                    "non_target_ssim": finite_mean(
                        row["non_target_ssim"] for row in sequence_rows
                    ),
                    "non_target_lpips_alex": finite_mean(
                        row["non_target_lpips_alex"] for row in sequence_rows
                    ),
                    "temporal_lpips": finite_mean(
                        row["temporal_lpips"] for row in sequence_rows
                    ),
                    "warp_l1_delta": finite_mean(
                        row["warp_l1_delta"] for row in sequence_rows
                    ),
                    "warp_lpips_alex": finite_mean(
                        row["warp_lpips_alex"] for row in sequence_rows
                    ),
                    "frame_flicker": finite_mean(
                        row["warp_l1_delta"] for row in sequence_rows
                    ),
                    "mask_iou_jitter": finite_std(iou_values),
                    "boundary_f1_jitter": finite_std(boundary_values),
                    "identity_switch": 0,
                    "trajectory_rmse_m": trajectory.get("source_rmse_m", 0.0),
                    "trajectory_acceleration_energy": trajectory.get(
                        "acceleration_energy", 0.0
                    ),
                    "centroid_jitter_pixels": centroid_jitter,
                    "contact_region_jitter_pixels": finite_std(contact_values),
                    "semantic_reintroduction_pixels": semantic_reintroduction,
                }
            )
    return rows, sequences


def output_manifest(run_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "worldsim_v4_m3_scene_manifest_v1",
        "task_id": TASK_ID,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def validate_scene_contract(
    *,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    cohort: Mapping[str, Any],
    scene: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scene not in inventory["scenes"]:
        raise M3SceneRunError(f"scene 不在 M3 inventory: {scene}")
    binding = inventory["scenes"][scene]
    if binding["partition"] not in {"development", "validation"}:
        raise M3SceneRunError("M3 scene runner 禁止读取 test partition")
    records = [
        row for row in cohort["freeze"]["scene_records"] if row["scene"] == scene
    ]
    if len(records) != 1:
        raise M3SceneRunError("D0 cohort scene 必须唯一")
    record = records[0]
    clip = record["continuous_clip"]
    expected = binding["clip"]
    if record["role"] != binding["partition"]:
        raise M3SceneRunError("scene partition 与 D0 cohort 不一致")
    if clip["actor_instance_token"] != binding["instance_token"]:
        raise M3SceneRunError("scene actor token 与 D0 cohort 不一致")
    for key in ("start_index", "end_index", "duration_s"):
        if clip[key] != expected[key]:
            raise M3SceneRunError(f"scene clip {key} 与 D0 cohort 不一致")
    if not 2.0 <= float(clip["duration_s"]) <= 4.0:
        raise M3SceneRunError("连续 clip 时长不在 [2, 4] 秒")
    if len(clip["sample_tokens"]) != 7:
        raise M3SceneRunError("M3 要求 D0 冻结的七个连续 nuScenes keyframes")
    if list(inventory["camera_ids"]) != list(config["clip"]["camera_ids"]):
        raise M3SceneRunError("M3 三相机合同不一致")
    return binding, record


def run(
    *,
    config_path: Path,
    inventory_path: Path,
    scene: str,
    run_dir: Path,
    control_points: int,
    acceleration_regularization: float,
    evidence_retention: float,
    warp_alpha: float,
    operations: tuple[str, ...],
    diagnostic: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    cohort_path = verify_binding(inventory["cohort"], "D0 cohort")
    cohort = yaml.safe_load(cohort_path.read_text(encoding="utf-8"))
    binding, cohort_record = validate_scene_contract(
        config=config,
        inventory=inventory,
        cohort=cohort,
        scene=scene,
    )
    for source in (config_path, inventory_path, cohort_path, Path(__file__)):
        shutil.copy2(source, source_snapshot / source.name)
    for source in (
        PROJECT_ROOT / "motion_proj/worldsim_v4/se3_bspline.py",
        PROJECT_ROOT / "motion_proj/worldsim_v4/temporal_protocol.py",
        PROJECT_ROOT / "motion_proj/worldsim_v4/temporal_metrics.py",
    ):
        shutil.copy2(source, source_snapshot / source.name)

    parameters = {
        "control_point_count": int(control_points),
        "acceleration_regularization": float(acceleration_regularization),
        "evidence_retention": float(evidence_retention),
        "warp_blend_alpha": float(warp_alpha),
    }
    search = config["trajectory"]["development_search"]
    if control_points not in search["control_point_count"]:
        raise M3SceneRunError("control point 值不在 development search 合同")
    if acceleration_regularization not in search["acceleration_regularization"]:
        raise M3SceneRunError("acceleration regularization 不在 development search 合同")
    if evidence_retention not in search["evidence_retention"]:
        raise M3SceneRunError("evidence retention 不在 development search 合同")
    if warp_alpha not in search["warp_blend_alpha"]:
        raise M3SceneRunError("warp alpha 不在 development search 合同")
    if set(operations) - set(OPERATIONS) or not operations:
        raise M3SceneRunError("operation 子集非法")
    if not diagnostic and set(operations) != set(OPERATIONS):
        raise M3SceneRunError("formal M3 scene 必须包含 REMOVE/LATERAL/INSERT")
    selected = config["trajectory"].get("selected_parameters")
    if binding["partition"] == "validation":
        if selected is None or parameters != selected:
            raise M3SceneRunError("validation 必须使用 development freeze 参数")
    if binding["partition"] == "development" and selected is not None and not diagnostic:
        if parameters != selected:
            raise M3SceneRunError("freeze 后 development formal 只能使用选定参数")
    if not diagnostic and git_dirty():
        raise M3SceneRunError("formal M3 scene 要求 clean git worktree")

    verified_inputs = {
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
        },
        "cohort": {
            "path": str(cohort_path),
            "sha256": sha256_file(cohort_path),
        },
    }
    if verified_inputs["inventory"]["sha256"] != config["inputs"]["scene_inventory"]["sha256"]:
        raise M3SceneRunError("M3 config 绑定的 scene inventory SHA 不匹配")
    if verified_inputs["cohort"]["sha256"] != config["inputs"]["cohort"]["sha256"]:
        raise M3SceneRunError("M3 config 绑定的 D0 cohort SHA 不匹配")
    for label in ("checkpoint", "drivestudio_config", "registry"):
        path = verify_binding(binding[label], f"{scene} {label}")
        verified_inputs[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    registry = json.loads(
        Path(binding["registry"]["path"]).read_text(encoding="utf-8")
    )
    actors = [
        actor
        for actor in registry["actors"]
        if actor["instance_token"] == binding["instance_token"]
    ]
    if len(actors) != 1:
        raise M3SceneRunError("target actor token 在 registry 中必须唯一")
    actor = actors[0]
    registry_ready = (
        actor.get("availability") == "available"
        and int(actor.get("checkpoint_tensor_slice", {}).get("gaussian_count", 0)) > 0
    )
    if registry_ready != (binding["status"] == "ready"):
        raise M3SceneRunError("scene readiness 与 registry 不一致")
    resolved = {
        "schema_version": "worldsim_v4_m3_scene_resolved_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "partition": binding["partition"],
        "diagnostic": diagnostic,
        "parameters": parameters,
        "operations": list(operations),
        "clip": cohort_record["continuous_clip"],
        "camera_ids": list(config["clip"]["camera_ids"]),
        "binding": binding,
        "test_quality_read": False,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    if binding["status"] == "abstain":
        summary = {
            "schema_version": "worldsim_v4_m3_scene_summary_v1",
            "task_id": TASK_ID,
            "scene": scene,
            "partition": binding["partition"],
            "status": "abstain",
            "reason": binding["reason"],
            "retained_in_denominator": True,
            "operations": list(operations),
            "sequence_count": 0,
            "parameters": parameters,
            "verified_inputs": verified_inputs,
            "development_content_read": False,
            "development_optimization_read": False,
            "validation_content_read": False,
            "validation_optimization_read": False,
            "test_quality_read": False,
            "project_git_head": git_head(),
            "project_git_dirty": git_dirty(),
            "duration_seconds": time.monotonic() - started,
        }
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "task_id": TASK_ID,
                "scene": scene,
                "project_git_head": git_head(),
                "project_git_dirty": git_dirty(),
                "cuda_used": False,
                "test_quality_read": False,
            },
        )
        manifest = output_manifest(run_dir)
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(
            run_dir / "status.json",
            {
                "task_id": TASK_ID,
                "scene": scene,
                "status": "done",
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            },
        )
        return summary

    if not torch.cuda.is_available():
        raise M3SceneRunError("M3 real renderer 需要 CUDA")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    drivestudio_root = Path(inventory["drivestudio_checkout"])
    sys.path.insert(0, str(drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    checkpoint = Path(binding["checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    ds_config = OmegaConf.load(binding["drivestudio_config"]["path"])
    if list(ds_config.data.pixel_source.cameras) != list(inventory["camera_ids"]):
        raise M3SceneRunError("DriveStudio checkpoint 不是三相机合同")
    dataset = DrivingDataset(data_cfg=ds_config.data)
    cameras = list(range(dataset.pixel_source.num_cams))
    if cameras != list(inventory["camera_ids"]):
        raise M3SceneRunError("DriveStudio dataset 三相机索引不一致")
    expected_timesteps = (
        int(cohort_record["sample_count"]) - 1
    ) * int(inventory["processed_keyframe_stride"]) + 1
    if int(dataset.num_img_timesteps) != expected_timesteps:
        raise M3SceneRunError("DriveStudio 10 Hz timeline 与 D0 keyframe 映射不一致")
    trainer = import_str(ds_config.trainer.type)(
        **ds_config.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=ds_config.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda:0"),
    )
    trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
    trainer.set_eval()
    if hasattr(trainer, "optimizer"):
        raise M3SceneRunError("M3 evaluator 禁止 optimizer")
    rigid = trainer.models["RigidNodes"]
    model_index = int(binding["actor"]["model_index"])
    keyframe_stride = int(inventory["processed_keyframe_stride"])
    clip = binding["clip"]
    frames = list(
        range(
            int(clip["start_index"]) * keyframe_stride,
            int(clip["end_index"]) * keyframe_stride + 1,
            keyframe_stride,
        )
    )
    dense_frames = list(range(frames[0], frames[-1] + 1))
    if len(frames) != 7:
        raise M3SceneRunError("processed keyframe 映射必须得到七帧")
    if not bool(rigid.instances_fv[frames, model_index].all()):
        raise M3SceneRunError("目标 actor 在 D0 clip keyframes 中并非持续可见")
    if int((rigid.point_ids[..., 0] == model_index).sum().item()) != int(
        binding["actor"]["gaussian_count"]
    ):
        raise M3SceneRunError("目标 actor Gaussian 数与 registry 不一致")
    baseline_state = rigid_state_hash(rigid)
    baseline_non_target = non_target_hash(rigid, model_index)
    source_keyframe_transforms = actor_transforms(rigid, frames, model_index)
    artifact_root = run_dir / "artifacts"
    base_audit = render_variant(
        trainer=trainer,
        dataset=dataset,
        checkpoint=checkpoint,
        output_dir=artifact_root / "BASE",
        frames=frames,
        cameras=cameras,
        model_index=model_index,
        dense_frames=dense_frames,
        dense_transforms=None,
        remove=False,
        save_ground_truth=True,
        baseline_state_hash=baseline_state,
        baseline_non_target_hash=baseline_non_target,
    )
    remove_audit = render_variant(
        trainer=trainer,
        dataset=dataset,
        checkpoint=checkpoint,
        output_dir=artifact_root / "REMOVE_SHARED",
        frames=frames,
        cameras=cameras,
        model_index=model_index,
        dense_frames=dense_frames,
        dense_transforms=None,
        remove=True,
        save_ground_truth=False,
        baseline_state_hash=baseline_state,
        baseline_non_target_hash=baseline_non_target,
    )
    support = []
    for frame in frames:
        pixels = 0
        for camera in cameras:
            base = imageio.imread(rgb_path(artifact_root / "BASE", frame, camera))
            removed = imageio.imread(
                rgb_path(artifact_root / "REMOVE_SHARED", frame, camera)
            )
            pixels += int(effect_mask(base, removed).sum())
        support.append(pixels)
    support_array = np.asarray(support, dtype=np.float64)
    if int(support_array.sum()) == 0:
        trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
        trainer.set_eval()
        final_state = rigid_state_hash(trainer.models["RigidNodes"])
        checkpoint_after = sha256_file(checkpoint)
        rollback_exact = (
            final_state == baseline_state and checkpoint_after == checkpoint_before
        )
        if not rollback_exact:
            raise M3SceneRunError("零支持 abstain 的 rollback 或 checkpoint 非 exact")
        write_jsonl(run_dir / "metrics.jsonl", [])
        summary = {
            "schema_version": "worldsim_v4_m3_scene_summary_v1",
            "task_id": TASK_ID,
            "scene": scene,
            "partition": binding["partition"],
            "status": "abstain",
            "reason": "ABSTAIN_NO_RENDERED_CLIP_SUPPORT",
            "retained_in_denominator": True,
            "phase": "diagnostic" if diagnostic else "formal",
            "parameters": parameters,
            "operations": list(operations),
            "clip": {
                "duration_s": clip["duration_s"],
                "nuscenes_keyframe_count": len(frames),
                "processed_keyframe_indices": frames,
                "dense_trajectory_frame_count": len(dense_frames),
                "camera_ids": cameras,
                "resolution_reduction": False,
            },
            "rendered_cross_camera_actor_support_pixels": support,
            "sequence_count": 0,
            "render_audit": {
                "BASE": base_audit,
                "REMOVE_SHARED": remove_audit,
            },
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_immutable": checkpoint_before == checkpoint_after,
            "rollback_exact": rollback_exact,
            "verified_inputs": verified_inputs,
            "development_content_read": binding["partition"] == "development",
            "development_optimization_read": False,
            "validation_content_read": binding["partition"] == "validation",
            "validation_optimization_read": False,
            "test_quality_read": False,
            "project_git_head": git_head(),
            "project_git_dirty": git_dirty(),
            "duration_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        }
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "task_id": TASK_ID,
                "scene": scene,
                "project_git_head": git_head(),
                "project_git_dirty": git_dirty(),
                "checkpoint_sha256": checkpoint_before,
                "config_sha256": sha256_file(config_path),
                "inventory_sha256": sha256_file(inventory_path),
                "cohort_sha256": sha256_file(cohort_path),
                "parameters_sha256": canonical_sha256(parameters),
                "cuda_device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "test_quality_read": False,
            },
        )
        manifest = output_manifest(run_dir)
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(
            run_dir / "status.json",
            {
                "task_id": TASK_ID,
                "scene": scene,
                "status": "done",
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
            },
        )
        return summary
    trajectories = build_arm_trajectories(
        source_keyframe_transforms,
        support=support_array,
        control_point_count=control_points,
        evidence_retention=evidence_retention,
        acceleration_regularization=acceleration_regularization,
    )
    trajectory_audit = {
        arm: {
            "source_rmse_m": trajectory.source_rmse_m,
            "acceleration_energy": trajectory.acceleration_energy,
            "control_twists": (
                None
                if trajectory.control_twists is None
                else trajectory.control_twists.tolist()
            ),
        }
        for arm, trajectory in trajectories.items()
    }
    trajectory_audit["evidence_support_pixels"] = support
    trajectory_audit["evidence_memory_weights"] = evidence_memory_weights(
        support_array, retention=evidence_retention
    ).tolist()
    atomic_json(artifact_root / "trajectory_audit.json", trajectory_audit)
    render_audit: dict[str, Any] = {
        "BASE": base_audit,
        "REMOVE_SHARED": remove_audit,
    }
    for operation in operations:
        if operation == "REMOVE":
            continue
        offset = np.asarray(
            config["operations"][operation]["actor_local_translation_m"],
            dtype=np.float64,
        )
        render_audit[operation] = {}
        for arm in GAUSSIAN_ARMS:
            edited_keyframes = apply_actor_local_translation(
                trajectories[arm].transforms, offset
            )
            dense = resample_se3_transforms(edited_keyframes, len(dense_frames))
            audit = render_variant(
                trainer=trainer,
                dataset=dataset,
                checkpoint=checkpoint,
                output_dir=artifact_root / operation / arm,
                frames=frames,
                cameras=cameras,
                model_index=model_index,
                dense_frames=dense_frames,
                dense_transforms=dense,
                remove=False,
                save_ground_truth=False,
                baseline_state_hash=baseline_state,
                baseline_non_target_hash=baseline_non_target,
            )
            if not audit["non_target_state_unchanged"]:
                raise M3SceneRunError(f"{operation}/{arm} 修改了非目标状态")
            render_audit[operation][arm] = audit
        build_full_warp_variant(
            base_dir=artifact_root / "BASE",
            evidence_dir=artifact_root
            / operation
            / "CUBIC_BSPLINE_TEMPORAL_EVIDENCE",
            output_dir=artifact_root / operation / "FULL_WARP_REGULARIZED",
            frames=frames,
            cameras=cameras,
            alpha=warp_alpha,
        )
        render_audit[operation]["FULL_WARP_REGULARIZED"] = {
            "source": "CUBIC_BSPLINE_TEMPORAL_EVIDENCE",
            "warp_blend_alpha": warp_alpha,
            "base_checkpoint_mutated": False,
            "non_target_state_unchanged": True,
        }
    atomic_json(artifact_root / "render_audit.json", render_audit)

    metric_rows, sequences = evaluate_sequences(
        artifact_root=artifact_root,
        operations=operations,
        frames=frames,
        cameras=cameras,
        trainer=trainer,
        trajectory_audit=trajectory_audit,
        minimum_effect_pixels=int(config["operations"]["minimum_rendered_effect_pixels"]),
    )
    write_jsonl(run_dir / "metrics.jsonl", metric_rows)
    trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
    trainer.set_eval()
    final_state = rigid_state_hash(trainer.models["RigidNodes"])
    checkpoint_after = sha256_file(checkpoint)
    rollback_exact = final_state == baseline_state and checkpoint_after == checkpoint_before
    if not rollback_exact:
        raise M3SceneRunError("M3 rollback 或 checkpoint 文件非 exact")
    for sequence in sequences:
        sequence["rollback_exact"] = True
    summary = {
        "schema_version": "worldsim_v4_m3_scene_summary_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "partition": binding["partition"],
        "status": "done",
        "phase": "diagnostic" if diagnostic else "formal",
        "parameters": parameters,
        "operations": list(operations),
        "arms": list(TEMPORAL_ARMS),
        "clip": {
            "duration_s": clip["duration_s"],
            "nuscenes_keyframe_count": len(frames),
            "processed_keyframe_indices": frames,
            "dense_trajectory_frame_count": len(dense_frames),
            "camera_ids": cameras,
            "resolution_reduction": False,
        },
        "sequence_count": len(sequences),
        "sequences": sequences,
        "trajectory_audit": trajectory_audit,
        "render_audit": render_audit,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": checkpoint_before == checkpoint_after,
        "rollback_exact": rollback_exact,
        "verified_inputs": verified_inputs,
        "development_content_read": binding["partition"] == "development",
        "development_optimization_read": False,
        "validation_content_read": binding["partition"] == "validation",
        "validation_optimization_read": False,
        "test_quality_read": False,
        "project_git_head": git_head(),
        "project_git_dirty": git_dirty(),
        "duration_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "fingerprint.json",
        {
            "task_id": TASK_ID,
            "scene": scene,
            "project_git_head": git_head(),
            "project_git_dirty": git_dirty(),
            "checkpoint_sha256": checkpoint_before,
            "config_sha256": sha256_file(config_path),
            "inventory_sha256": sha256_file(inventory_path),
            "cohort_sha256": sha256_file(cohort_path),
            "parameters_sha256": canonical_sha256(parameters),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "test_quality_read": False,
        },
    )
    manifest = output_manifest(run_dir)
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "scene": scene,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/worldsim_v4/m3_temporal_v1.yaml",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT / "configs/worldsim_v4/m3_scene_inventory_v1.yaml",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-points", type=int, required=True)
    parser.add_argument("--acceleration-regularization", type=float, required=True)
    parser.add_argument("--evidence-retention", type=float, required=True)
    parser.add_argument("--warp-alpha", type=float, required=True)
    parser.add_argument("--operations", nargs="+", default=list(OPERATIONS))
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(
            config_path=args.config.resolve(),
            inventory_path=args.inventory.resolve(),
            scene=args.scene,
            run_dir=args.run_dir.resolve(),
            control_points=args.control_points,
            acceleration_regularization=args.acceleration_regularization,
            evidence_retention=args.evidence_retention,
            warp_alpha=args.warp_alpha,
            operations=tuple(args.operations),
            diagnostic=args.diagnostic,
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "scene": summary["scene"],
                    "sequence_count": summary["sequence_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.run_dir / "status.json",
            {
                "task_id": TASK_ID,
                "scene": args.scene,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "test_quality_read": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
