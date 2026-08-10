#!/usr/bin/env python
"""在 StreetGS 上执行 3DGIC 原理适配的删车背景补全与只读验收。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.depth_guided_unseen_mask import splat_rgbd_to_target
from motion_proj.worldsim_v32.inpainting_adapter import (
    CompletionPoints,
    append_generated_background,
    completion_points_from_view,
    merge_completion_points,
)
from motion_proj.worldsim_v32.semantic_schema import sha256_file
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data,
    load_model_checkpoint_read_only,
    release_trainer_render_info,
    uint8_rgb,
)
from scripts.lift_worldsim_v32_semantics import build_runtime


GIB = 1024**3


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None if value == "max" else int(value)


def read_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    return {
        key: int(value)
        for key, value in (
            line.split() for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


class ResourceMonitor:
    """采样正式 S2 运行的 GPU、cgroup 与磁盘资源。"""

    def __init__(self, *, minimum_free_disk_gib: float, maximum_cgroup_fraction: float):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.minimum_free_disk_bytes = int(minimum_free_disk_gib * GIB)
        self.maximum_cgroup_fraction = float(maximum_cgroup_fraction)
        self.peak_nvidia_mib = 0
        self.peak_cgroup_current_bytes = 0
        self.cgroup_limit_bytes = read_int(Path("/sys/fs/cgroup/memory.max"))
        self.events_before = read_memory_events()
        self.disk_free_before_bytes = shutil.disk_usage("/root/autodl-tmp").free
        self.consecutive_pressure_samples = 0
        self.memory_pressure_observed = False

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            current = read_int(Path("/sys/fs/cgroup/memory.current")) or 0
            self.peak_cgroup_current_bytes = max(self.peak_cgroup_current_bytes, current)
            if (
                self.cgroup_limit_bytes
                and current >= self.maximum_cgroup_fraction * self.cgroup_limit_bytes
            ):
                self.consecutive_pressure_samples += 1
                if self.consecutive_pressure_samples >= 2:
                    self.memory_pressure_observed = True
            else:
                self.consecutive_pressure_samples = 0
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for line in output.splitlines():
                    self.peak_nvidia_mib = max(self.peak_nvidia_mib, int(line.strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def start(self) -> None:
        if self.disk_free_before_bytes < self.minimum_free_disk_bytes:
            raise RuntimeError("S2 可用磁盘低于冻结停止门")
        self._thread.start()

    def finish(self) -> dict[str, object]:
        self._stop.set()
        self._thread.join(timeout=10)
        after = read_memory_events()
        keys = set(self.events_before) | set(after)
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_nvidia_memory_mib_sampled": self.peak_nvidia_mib,
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cgroup_current_bytes_sampled": self.peak_cgroup_current_bytes,
            "cgroup_memory_max_bytes": self.cgroup_limit_bytes,
            "memory_pressure_observed": self.memory_pressure_observed,
            "memory_events_before": self.events_before,
            "memory_events_after": after,
            "memory_events_delta": {
                key: after.get(key, 0) - self.events_before.get(key, 0)
                for key in sorted(keys)
            },
            "disk_free_before_bytes": self.disk_free_before_bytes,
            "disk_free_after_bytes": shutil.disk_usage("/root/autodl-tmp").free,
        }


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, path_key, sha_key in (
        ("checkpoint", "checkpoint", "checkpoint_sha256"),
        ("source_config", "source_config", "source_config_sha256"),
        ("s1_config", "s1_config", "s1_config_sha256"),
    ):
        path = Path(config["inputs"][path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = config["inputs"][sha_key]
        if actual != expected:
            raise RuntimeError(f"S2 {name} SHA 漂移: {actual}")
        verified[name] = {"path": str(path), "sha256": actual}
    s1 = yaml.safe_load(Path(config["inputs"]["s1_config"]).read_text(encoding="utf-8"))
    if s1["split"]["heldout_frames"] != config["scene"]["heldout_frames"]:
        raise RuntimeError("S2 与 S1 held-out split 不一致")
    heldout = set(int(value) for value in config["scene"]["heldout_frames"])
    for role, target in config["targets"].items():
        for key in ("dataset_instance_id", "instance_token", "rigid_model_index"):
            if target[key] != s1["actors"][role][key]:
                raise RuntimeError(f"S2 {role} 的 {key} 与 S1 身份不一致")
        optimization_frames = [int(target["frame"])] + [
            int(view[0]) for view in target["support_views"]
        ]
        leaked = sorted(set(optimization_frames) & heldout)
        if leaked:
            raise RuntimeError(f"S2 {role} 优化视图泄漏 held-out: {leaked}")
        mask = Path(target["mask"])
        if sha256_file(mask) != target["mask_sha256"]:
            raise RuntimeError(f"S2 {role} mask SHA 漂移")
        verified[f"mask_{role}"] = {
            "path": str(mask),
            "sha256": target["mask_sha256"],
        }
    for name in ("3dgic", "inpaint360gs"):
        spec = config["third_party"][name]
        checkout = Path(spec["checkout"])
        head = git_head(checkout)
        if head != spec["commit"]:
            raise RuntimeError(f"S2 {name} commit 漂移: {head}")
        license_path = Path(spec["license"])
        license_sha = sha256_file(license_path)
        if license_sha != spec["license_sha256"]:
            raise RuntimeError(f"S2 {name} license SHA 漂移")
        verified[name] = {
            "checkout": str(checkout),
            "commit": head,
            "license": str(license_path),
            "license_sha256": license_sha,
        }
    return verified


def numpy_value(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def render_snapshot(
    *,
    trainer: Any,
    dataset: Any,
    frame: int,
    camera_id: int,
    device: torch.device,
    hide_actor: int | None = None,
) -> dict[str, Any]:
    image_infos, camera_infos, groundtruth, measured, egocar, image_index = get_view_data(
        dataset, frame, camera_id, device
    )
    rigid = trainer.models["RigidNodes"]
    original_visibility = None
    if hide_actor is not None:
        original_visibility = rigid.instances_fv[:, hide_actor].detach().clone()
        with torch.no_grad():
            rigid.instances_fv[:, hide_actor] = False
    try:
        camera = trainer.process_camera(
            camera_infos, image_infos["img_idx"].flatten()[0]
        )
        with torch.inference_mode():
            outputs = trainer(image_infos, camera_infos)
        result = {
            "rgb": uint8_rgb(outputs["rgb"]),
            "depth": numpy_value(outputs["depth"]).squeeze(),
            "opacity": numpy_value(outputs["opacity"]).squeeze(),
            "background_rgb": uint8_rgb(outputs["Background_rgb"]),
            "background_depth": numpy_value(outputs["Background_depth"]).squeeze(),
            "background_opacity": numpy_value(outputs["Background_opacity"]).squeeze(),
            "groundtruth": groundtruth,
            "measured_lidar_depth": measured,
            "dynamic_mask": image_infos["dynamic_masks"].detach().bool().cpu().numpy(),
            "egocar_mask": egocar,
            "intrinsics": numpy_value(camera.Ks),
            "camera_to_world": numpy_value(camera.camtoworlds),
            "image_index": int(image_index),
        }
    finally:
        release_trainer_render_info(trainer)
        if original_visibility is not None:
            with torch.no_grad():
                rigid.instances_fv[:, hide_actor].copy_(original_visibility)
    return result


def load_binary_mask(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "binary" not in payload.files:
            raise RuntimeError(f"S2 mask 缺 binary: {path}")
        mask = np.asarray(payload["binary"], dtype=bool)
    if mask.shape != (450, 800) or not mask.any():
        raise RuntimeError(f"S2 mask 无效: {path}")
    return mask


def combine_cross_view_splats(
    *,
    supports: list[dict[str, Any]],
    target: dict[str, Any],
    target_mask: np.ndarray,
    projection: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = target_mask.shape
    best_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    best_score = np.full((height, width), np.inf, dtype=np.float32)
    observed = np.zeros((height, width), dtype=bool)
    support_count = np.zeros((height, width), dtype=np.uint16)
    colors: list[np.ndarray] = []
    depth_residuals: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    for support in supports:
        valid = (
            ~support["dynamic_mask"]
            & ~support["egocar_mask"]
            & np.isfinite(support["background_depth"])
            & (support["background_depth"] > 1e-4)
            & (
                support["background_opacity"]
                >= float(projection["minimum_source_opacity"])
            )
        )
        splat = splat_rgbd_to_target(
            source_depth=support["background_depth"],
            source_rgb=support["groundtruth"],
            source_valid=valid,
            source_intrinsics=support["intrinsics"],
            source_camera_to_world=support["camera_to_world"],
            target_depth=target["background_depth"],
            target_mask=target_mask,
            target_intrinsics=target["intrinsics"],
            target_camera_to_world=target["camera_to_world"],
            absolute_depth_tolerance_m=float(
                projection["absolute_depth_tolerance_m"]
            ),
            relative_depth_tolerance=float(projection["relative_depth_tolerance"]),
            stride=int(projection["source_stride"]),
        )
        update = splat.observed & (splat.score < best_score)
        best_rgb[update] = splat.rgb[update]
        best_score[update] = splat.score[update]
        observed |= splat.observed
        support_count += splat.observed.astype(np.uint16)
        color = np.full((height, width, 3), np.nan, dtype=np.float32)
        color[splat.observed] = splat.rgb[splat.observed].astype(np.float32)
        colors.append(color)
        residual = np.full((height, width), np.nan, dtype=np.float32)
        valid_target = splat.observed & np.isfinite(target["background_depth"])
        residual[valid_target] = np.abs(
            splat.depth[valid_target] - target["background_depth"][valid_target]
        )
        depth_residuals.append(residual)
        per_view.append(
            {
                "frame": int(support["frame"]),
                "camera_id": int(support["camera_id"]),
                "valid_source_pixels": int(valid.sum()),
                "observed_target_pixels": int(splat.observed.sum()),
            }
        )
    observed &= target_mask
    multi = target_mask & (support_count >= 2)
    color_stack = np.stack(colors, axis=0)
    depth_stack = np.stack(depth_residuals, axis=0)
    temporal_color_std = None
    if multi.any():
        temporal_color_std = float(
            np.nanmean(np.nanstd(color_stack[:, multi, :], axis=0))
        )
    depth_error_mean = None
    if observed.any():
        depth_error_mean = float(np.nanmean(depth_stack[:, observed]))
    audit = {
        "per_support_view": per_view,
        "target_mask_pixels": int(target_mask.sum()),
        "cross_view_observed_pixels": int(observed.sum()),
        "cross_view_coverage": float(observed.sum() / target_mask.sum()),
        "multi_support_pixels": int(multi.sum()),
        "temporal_color_std_uint8_mean_multi_support": temporal_color_std,
        "depth_consistency_abs_error_m_mean_observed": depth_error_mean,
        "maximum_support_count": int(support_count.max()),
    }
    return best_rgb, observed, audit


def make_completion(
    *, target: dict[str, Any], target_mask: np.ndarray, cross_rgb: np.ndarray,
    observed: np.ndarray, persistence_mask: np.ndarray, config: Mapping[str, Any]
) -> tuple[np.ndarray, CompletionPoints, np.ndarray]:
    context = target["rgb"].copy()
    context[observed] = cross_rgb[observed]
    unseen = target_mask & ~observed
    inpainted = cv2.inpaint(
        context,
        unseen.astype(np.uint8) * 255,
        float(config["projection"]["inpaint_radius_pixels"]),
        cv2.INPAINT_TELEA,
    )
    completed = target["rgb"].copy()
    completed[observed] = cross_rgb[observed]
    completed[unseen] = inpainted[unseen]
    points = completion_points_from_view(
        rgb=completed,
        depth=target["background_depth"],
        mask=persistence_mask,
        observed_cross_view=observed,
        intrinsics=target["intrinsics"],
        camera_to_world=target["camera_to_world"],
        stride=int(config["gaussians"]["target_stride"]),
        scale_multiplier=float(config["gaussians"]["scale_multiplier"]),
        minimum_scale_m=float(config["gaussians"]["minimum_scale_m"]),
        maximum_scale_m=float(config["gaussians"]["maximum_scale_m"]),
    )
    return completed, points, unseen


def concatenate(groups: list[CompletionPoints]) -> CompletionPoints:
    return CompletionPoints(
        means=np.concatenate([group.means for group in groups]),
        rgb=np.concatenate([group.rgb for group in groups]),
        scales=np.concatenate([group.scales for group in groups]),
        confidence=np.concatenate([group.confidence for group in groups]),
        observed_cross_view=np.concatenate(
            [group.observed_cross_view for group in groups]
        ),
        source_pixels_xy=np.concatenate([group.source_pixels_xy for group in groups]),
    )


def psnr_uint8(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    values = difference[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("PSNR mask 为空")
    mse = float(np.mean(np.square(values)))
    return float("inf") if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))


def ssim_uint8(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(
        structural_similarity(
            prediction, reference, channel_axis=2, data_range=255
        )
    )


def lpips_uint8(
    trainer: Any, prediction: np.ndarray, reference: np.ndarray, device: torch.device
) -> float:
    prediction_tensor = (
        torch.from_numpy(prediction.copy()).to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)[None]
        / 255.0
    )
    reference_tensor = (
        torch.from_numpy(reference.copy()).to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)[None]
        / 255.0
    )
    trainer.lpips.reset()
    with torch.inference_mode():
        value = float(trainer.lpips(prediction_tensor, reference_tensor).item())
    trainer.lpips.reset()
    return value


def save_target_panel(
    path: Path,
    *,
    groundtruth: np.ndarray,
    source_delete: np.ndarray,
    mask: np.ndarray,
    cross_rgb: np.ndarray,
    observed: np.ndarray,
    completed: np.ndarray,
    candidate: np.ndarray | None = None,
) -> None:
    overlay = source_delete.copy()
    overlay[mask] = np.round(
        0.45 * overlay[mask].astype(np.float32)
        + 0.55 * np.array([255, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    cross = source_delete.copy()
    cross[observed] = cross_rgb[observed]
    images = [groundtruth, source_delete, overlay, cross, completed]
    if candidate is not None:
        images.append(candidate)
    imageio.imwrite(path, np.concatenate(images, axis=1))


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verified = verify_inputs(config)
    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    targets_dir = artifacts / "targets"
    targets_dir.mkdir(parents=True)
    atomic_json(
        output_dir / "status.json",
        {
            "task_id": config["task_id"],
            "stage": "S2",
            "status": "running",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
    )

    resources = ResourceMonitor(
        minimum_free_disk_gib=float(config["resources"]["minimum_free_disk_gib"]),
        maximum_cgroup_fraction=float(config["resources"]["maximum_cgroup_fraction"]),
    )
    resources.start()
    started = time.monotonic()
    source_checkpoint = Path(config["inputs"]["checkpoint"])
    source_sha_before = sha256_file(source_checkpoint)
    device = torch.device("cuda:0")
    dataset, trainer = build_runtime(config, device)
    trainer.set_eval()
    source_heldout: dict[tuple[int, int], dict[str, Any]] = {}
    target_records: dict[str, dict[str, Any]] = {}
    raw_groups: list[CompletionPoints] = []

    for role, target_cfg in config["targets"].items():
        frame = int(target_cfg["frame"])
        camera_id = int(target_cfg["camera_id"])
        target = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera_id,
            device=device,
            hide_actor=int(target_cfg["rigid_model_index"]),
        )
        supports: list[dict[str, Any]] = []
        for support_frame, support_camera in target_cfg["support_views"]:
            support = render_snapshot(
                trainer=trainer,
                dataset=dataset,
                frame=int(support_frame),
                camera_id=int(support_camera),
                device=device,
            )
            support["frame"] = int(support_frame)
            support["camera_id"] = int(support_camera)
            supports.append(support)
        mask = load_binary_mask(Path(target_cfg["mask"]))
        cross_rgb, observed, cross_audit = combine_cross_view_splats(
            supports=supports,
            target=target,
            target_mask=mask,
            projection=config["projection"],
        )
        completed, points, unseen = make_completion(
            target=target,
            target_mask=mask,
            cross_rgb=cross_rgb,
            observed=observed,
            persistence_mask=(
                observed
                if target_cfg["checkpoint_persistence"]
                == "cross_view_observed_only"
                else mask
            ),
            config=config,
        )
        if int(observed.sum()) < int(
            config["evaluation"]["minimum_cross_view_observed_pixels"]
        ):
            raise RuntimeError(f"S2 {role} cross-view 观测像素不足")
        if points.means.shape[0] == 0:
            raise RuntimeError(f"S2 {role} 没有可反投影的补全点")
        role_dir = targets_dir / role
        role_dir.mkdir()
        imageio.imwrite(role_dir / "groundtruth_with_actor.png", target["groundtruth"])
        imageio.imwrite(role_dir / "source_delete.png", target["rgb"])
        imageio.imwrite(role_dir / "target_mask.png", mask.astype(np.uint8) * 255)
        imageio.imwrite(role_dir / "cross_view_observed.png", observed.astype(np.uint8) * 255)
        imageio.imwrite(role_dir / "unseen_generated.png", unseen.astype(np.uint8) * 255)
        imageio.imwrite(role_dir / "completion_reference.png", completed)
        save_target_panel(
            role_dir / "synthesis_panel.png",
            groundtruth=target["groundtruth"],
            source_delete=target["rgb"],
            mask=mask,
            cross_rgb=cross_rgb,
            observed=observed,
            completed=completed,
        )
        atomic_npz(
            role_dir / "completion_points_raw.npz",
            means=points.means,
            rgb=points.rgb,
            scales=points.scales,
            confidence=points.confidence,
            observed_cross_view=points.observed_cross_view,
            source_pixels_xy=points.source_pixels_xy,
        )
        target_records[role] = {
            "config": target_cfg,
            "mask_pixels": int(mask.sum()),
            "observed_pixels": int(observed.sum()),
            "unseen_generated_pixels": int(unseen.sum()),
            "checkpoint_persistence": target_cfg["checkpoint_persistence"],
            "persisted_cross_view_pixels": int(
                observed.sum()
                if target_cfg["checkpoint_persistence"]
                == "cross_view_observed_only"
                else observed.sum()
            ),
            "persisted_unseen_pixels": int(
                0
                if target_cfg["checkpoint_persistence"]
                == "cross_view_observed_only"
                else unseen.sum()
            ),
            "raw_completion_points": int(points.means.shape[0]),
            "cross_view_audit": cross_audit,
            "target": target,
            "mask": mask,
            "cross_rgb": cross_rgb,
            "observed": observed,
            "completed": completed,
        }
        raw_groups.append(points)
        print(
            f"S2 synthesis {role}: mask={int(mask.sum())} observed={int(observed.sum())} "
            f"unseen={int(unseen.sum())} points={points.means.shape[0]}",
            flush=True,
        )

    for frame, camera_id in config["evaluation"]["heldout_views"]:
        key = (int(frame), int(camera_id))
        source_heldout[key] = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=key[0],
            camera_id=key[1],
            device=device,
        )

    merged = merge_completion_points(
        raw_groups, voxel_size_m=float(config["gaussians"]["voxel_size_m"])
    )
    all_raw = concatenate(raw_groups)
    origin_tree = cKDTree(all_raw.means)
    _, origin_index = origin_tree.query(merged.means, k=1, workers=-1)
    raw_target_codes = np.concatenate(
        [
            np.full(group.means.shape[0], index, dtype=np.uint8)
            for index, group in enumerate(raw_groups)
        ]
    )
    target_codes = raw_target_codes[np.asarray(origin_index, dtype=np.int64)]
    background = trainer.models["Background"]
    old_background_means = background._means.detach().cpu().numpy().copy()
    append_audit = append_generated_background(
        background,
        merged,
        opacity=float(config["gaussians"]["opacity"]),
        birth_step=int(trainer.step),
    )
    if not np.array_equal(
        background._means.detach()[: int(append_audit["old_background_count"])]
        .float()
        .cpu()
        .numpy(),
        old_background_means,
    ):
        raise RuntimeError("S2 追加 generated rows 时修改了旧 Background means")
    row_indices = np.arange(
        int(append_audit["row_start"]),
        int(append_audit["row_end_exclusive"]),
        dtype=np.int64,
    )
    provenance_npz = artifacts / "generated_background_provenance.npz"
    atomic_npz(
        provenance_npz,
        background_row_index=row_indices,
        provenance_code=np.ones(row_indices.shape[0], dtype=np.uint8),
        means=merged.means,
        rgb=merged.rgb,
        scales=merged.scales,
        confidence=merged.confidence,
        observed_cross_view=merged.observed_cross_view,
        source_pixels_xy=merged.source_pixels_xy,
        target_code=target_codes,
    )
    candidate = artifacts / "checkpoint_s2_generated_background.pth"
    candidate_partial = candidate.with_suffix(candidate.suffix + f".partial.{os.getpid()}")
    torch.save(trainer.state_dict(only_model=True), candidate_partial)
    os.replace(candidate_partial, candidate)
    candidate_sha = sha256_file(candidate)
    del old_background_means
    torch.cuda.empty_cache()

    # 从磁盘严格重载候选，验证模型行与 V3.1 ancestry 都能对齐。
    load_model_checkpoint_read_only(trainer, candidate, device)
    trainer.set_eval()
    reloaded_background = trainer.models["Background"]
    if int(reloaded_background.num_points) != int(append_audit["new_background_count"]):
        raise RuntimeError("S2 候选重载后的 Background 行数漂移")
    reloaded_background._a2_ancestry.validate(
        expected_actor_ids=torch.full(
            (int(reloaded_background.num_points),),
            -1,
            device=device,
            dtype=torch.long,
        )
    )
    reloaded_means = (
        reloaded_background._means.detach()[
            int(append_audit["row_start"]) : int(append_audit["row_end_exclusive"])
        ]
        .float()
        .cpu()
        .numpy()
    )
    if not np.array_equal(reloaded_means, merged.means):
        raise RuntimeError("S2 候选重载后的 generated means 非精确一致")

    target_evaluations: dict[str, Any] = {}
    for role, record in target_records.items():
        target_cfg = record["config"]
        candidate_render = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=int(target_cfg["frame"]),
            camera_id=int(target_cfg["camera_id"]),
            device=device,
            hide_actor=int(target_cfg["rigid_model_index"]),
        )
        mask = record["mask"]
        source_delete = record["target"]["rgb"]
        completed = record["completed"]
        candidate_rgb = candidate_render["rgb"]
        masked_candidate = completed.copy()
        masked_candidate[mask] = candidate_rgb[mask]
        effect = np.max(
            np.abs(candidate_rgb.astype(np.int16) - source_delete.astype(np.int16)),
            axis=2,
        )
        outside = ~mask
        outside_l1 = float(
            np.mean(
                np.abs(
                    candidate_rgb[outside].astype(np.float32)
                    - source_delete[outside].astype(np.float32)
                )
            )
        )
        metrics = {
            "candidate_effect_pixels_in_mask": int((effect[mask] >= 2).sum()),
            "candidate_effect_pixels_global": int((effect >= 2).sum()),
            "outside_mask_l1_uint8": outside_l1,
            "source_delete_vs_completion_psnr_mask_db": psnr_uint8(
                source_delete, completed, mask
            ),
            "candidate_vs_completion_psnr_mask_db": psnr_uint8(
                candidate_rgb, completed, mask
            ),
            "candidate_vs_completion_ssim_masked_context": ssim_uint8(
                masked_candidate, completed
            ),
            "candidate_vs_completion_lpips_masked_context": lpips_uint8(
                trainer, masked_candidate, completed, device
            ),
        }
        target_evaluations[role] = metrics
        role_dir = targets_dir / role
        imageio.imwrite(role_dir / "candidate_delete.png", candidate_rgb)
        save_target_panel(
            role_dir / "candidate_panel.png",
            groundtruth=record["target"]["groundtruth"],
            source_delete=source_delete,
            mask=mask,
            cross_rgb=record["cross_rgb"],
            observed=record["observed"],
            completed=completed,
            candidate=candidate_rgb,
        )

    heldout_dir = artifacts / "heldout"
    heldout_dir.mkdir()
    heldout_rows: list[dict[str, Any]] = []
    for key, source in source_heldout.items():
        candidate_render = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=key[0],
            camera_id=key[1],
            device=device,
        )
        static = ~source["dynamic_mask"] & ~source["egocar_mask"]
        row = {
            "frame": key[0],
            "camera_id": key[1],
            "source_psnr_global_db": psnr_uint8(
                source["rgb"], source["groundtruth"], np.ones(static.shape, bool)
            ),
            "candidate_psnr_global_db": psnr_uint8(
                candidate_render["rgb"], source["groundtruth"], np.ones(static.shape, bool)
            ),
            "source_psnr_static_db": psnr_uint8(
                source["rgb"], source["groundtruth"], static
            ),
            "candidate_psnr_static_db": psnr_uint8(
                candidate_render["rgb"], source["groundtruth"], static
            ),
            "source_ssim": ssim_uint8(source["rgb"], source["groundtruth"]),
            "candidate_ssim": ssim_uint8(
                candidate_render["rgb"], source["groundtruth"]
            ),
            "source_lpips": lpips_uint8(
                trainer, source["rgb"], source["groundtruth"], device
            ),
            "candidate_lpips": lpips_uint8(
                trainer, candidate_render["rgb"], source["groundtruth"], device
            ),
        }
        row["psnr_delta_db"] = row["candidate_psnr_global_db"] - row["source_psnr_global_db"]
        row["ssim_delta"] = row["candidate_ssim"] - row["source_ssim"]
        row["lpips_delta"] = row["candidate_lpips"] - row["source_lpips"]
        heldout_rows.append(row)
        imageio.imwrite(
            heldout_dir / f"f{key[0]:03d}_c{key[1]}_source_candidate_gt.png",
            np.concatenate(
                [source["rgb"], candidate_render["rgb"], source["groundtruth"]],
                axis=1,
            ),
        )

    heldout_mean = {
        key: float(np.mean([row[key] for row in heldout_rows]))
        for key in (
            "source_psnr_global_db",
            "candidate_psnr_global_db",
            "source_psnr_static_db",
            "candidate_psnr_static_db",
            "source_ssim",
            "candidate_ssim",
            "source_lpips",
            "candidate_lpips",
            "psnr_delta_db",
            "ssim_delta",
            "lpips_delta",
        )
    }
    gates = {
        "cross_view_observed": all(
            int(record["observed_pixels"])
            >= int(config["evaluation"]["minimum_cross_view_observed_pixels"])
            for record in target_records.values()
        ),
        "candidate_effect": all(
            metrics["candidate_effect_pixels_in_mask"]
            >= int(config["evaluation"]["minimum_candidate_effect_pixels"])
            for metrics in target_evaluations.values()
        ),
        "outside_l1": all(
            metrics["outside_mask_l1_uint8"]
            <= float(config["evaluation"]["maximum_target_outside_l1_uint8"])
            for metrics in target_evaluations.values()
        ),
        "heldout_psnr": heldout_mean["psnr_delta_db"]
        >= -float(config["evaluation"]["maximum_heldout_psnr_degradation_db"]),
        "heldout_ssim": heldout_mean["ssim_delta"]
        >= -float(config["evaluation"]["maximum_heldout_ssim_degradation"]),
        "heldout_lpips": heldout_mean["lpips_delta"]
        <= float(config["evaluation"]["maximum_heldout_lpips_increase"]),
    }
    candidate_selected = all(gates.values())
    source_sha_after = sha256_file(source_checkpoint)
    if source_sha_after != source_sha_before:
        raise RuntimeError("S2 执行后 D2 source checkpoint 被修改")
    resource_summary = resources.finish()
    if resource_summary["memory_pressure_observed"]:
        raise RuntimeError("S2 cgroup 内存达到连续停止门")
    if resource_summary["memory_events_delta"].get("oom", 0) or resource_summary[
        "memory_events_delta"
    ].get("oom_kill", 0):
        raise RuntimeError("S2 运行新增 cgroup OOM 事件")

    # 清理不可 JSON 序列化的大数组，只在正式 artifact 中保留。
    synthesis_summary = {
        role: {
            key: value
            for key, value in record.items()
            if key
            not in {
                "target",
                "mask",
                "cross_rgb",
                "observed",
                "completed",
            }
        }
        for role, record in target_records.items()
    }
    evaluation = {
        "schema_version": "worldsim_v32_s2_evaluation_v1",
        "target_completion": target_evaluations,
        "heldout_views_read_only_after_generation": heldout_rows,
        "heldout_mean": heldout_mean,
        "gates": gates,
        "candidate_selected": candidate_selected,
        "accuracy_claim_scope": {
            "observed_heldout": "PSNR/SSIM/LPIPS against observed ground truth",
            "unseen_completion": "no accuracy claim; provenance, view consistency and artifacts only",
        },
    }
    atomic_json(artifacts / "evaluation.json", evaluation)
    provenance = {
        "schema_version": "worldsim_v32_generated_background_v1",
        "provenance_label": "GENERATED_BACKGROUND",
        "authoritative_sidecar": str(provenance_npz),
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": candidate_sha,
        "append_audit": append_audit,
        "target_code_names": {
            str(index): role for index, role in enumerate(config["targets"])
        },
        "method": {
            "name": "3DGIC-adapted StreetGS background completion",
            "upstream_claim": "depth-guided cross-view principle and RGB-D unprojection; not an untouched upstream checkpoint run",
            "external_2d_inpainter": "OpenCV Telea deterministic",
            "official_3dgic_checkpoint_schema": config["third_party"]["3dgic"][
                "official_checkpoint_schema"
            ],
            "source_checkpoint_schema": config["third_party"]["3dgic"][
                "source_checkpoint_schema"
            ],
        },
        "v31_ancestry_compatibility": {
            "init_source_code": "unknown",
            "generated_rows_link_to_nearest_prior_background_lineage": True,
            "reason": "frozen V3.1 enum has no generated-background code; this sidecar is authoritative",
        },
        "inpaint360gs_disposition": config["third_party"]["inpaint360gs"],
    }
    atomic_json(artifacts / "provenance.json", provenance)
    summary = {
        "schema_version": "worldsim_v32_s2_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "method_completed": True,
        "candidate_selected": candidate_selected,
        "selected_checkpoint": str(candidate) if candidate_selected else str(source_checkpoint),
        "selected_checkpoint_sha256": candidate_sha if candidate_selected else source_sha_after,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "verified_inputs": verified,
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": source_sha_after,
        "synthesis": synthesis_summary,
        "append_audit": append_audit,
        "generated_background_provenance_npz": str(provenance_npz),
        "generated_background_provenance_npz_sha256": sha256_file(provenance_npz),
        "evaluation": evaluation,
        "resource_audit": {
            **resource_summary,
            "wall_time_seconds": float(time.monotonic() - started),
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "status.json",
        {
            "task_id": config["task_id"],
            "stage": "S2",
            "status": "done",
            "candidate_selected": candidate_selected,
            "summary": str(output_dir / "summary.json"),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"正式 S2 output 已存在: {args.output_dir}")
    random.seed(20260810)
    np.random.seed(20260810)
    torch.manual_seed(20260810)
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.manual_seed_all(20260810)
    torch.cuda.reset_peak_memory_stats()
    try:
        summary = run(args.config.resolve(), args.output_dir.resolve())
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output_dir / "status.json",
            {
                "task_id": "WS-V32-S2-BACKGROUND-INPAINT-01",
                "stage": "S2",
                "status": "rejected",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
