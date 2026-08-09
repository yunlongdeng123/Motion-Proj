#!/usr/bin/env python
"""执行 A4-P1 贡献度扫描、候选物化、质量评估与运行时画像。"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as torch_f


PROJECT = Path("/root/autodl-tmp/motion_proj")
DRIVESTUDIO = Path("/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml"
CAMERA_NAMES = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
ARM_FRACTIONS = {"p1-source": 0.0, "p1-b05": 0.05, "p1-b10": 0.10, "p1-b20": 0.20}
_ACTIVE_RUN_DIR: Path | None = None


from motion_proj.worldsim_v3.actor_metrics import boundary_band
from motion_proj.worldsim_v3.contribution_prune import (
    array_sha256,
    build_candidate_masks,
    build_candidate_registry,
    compare_metric_group,
    prune_checkpoint_state,
    tensor_sha256,
)
from scripts.eval_worldsim_v3_a0_actor_metrics import (
    RegionAccumulator,
    masked_lpips,
    ssim_map,
    to_device,
    uint8_rgb,
)
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data,
    load_model_checkpoint_read_only,
    release_trainer_render_info,
)
from scripts.run_worldsim_v3_a4_p0_profile import ResourceSampler, nearest_rank, rgb_sha256
from scripts.run_worldsim_v3_a4_p1_prune import (
    atomic_json,
    directory_digest,
    load_stage,
    nvidia_compute_rows,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p1_contribution_prune_protocol import (
    validate_inputs,
    validate_schema,
)


def source_inputs_unchanged(protocol: Mapping[str, Any]) -> bool:
    selected = protocol["selected_asset"]
    return all(
        sha256_file(Path(selected[name]["path"])) == selected[name]["sha256"]
        for name in ("checkpoint", "source_config", "actor_registry")
    )


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """使用同目录临时文件原子写入压缩 NPZ。"""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    """将候选 checkpoint 一次性写入并禁止覆盖。"""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def stage_resource_summary(device: torch.device, sampler: ResourceSampler) -> dict[str, Any]:
    sampled = sampler.summary()
    if (
        sampled["sampling_errors"]
        or sampled["peak_nvidia_process_memory_mib_sampled"] is None
        or sampled["peak_cgroup_memory_bytes_sampled"] is None
    ):
        raise RuntimeError(f"A4-P1 resource sampling incomplete: {sampled}")
    return {
        "peak_torch_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "peak_torch_reserved_mib": float(
            torch.cuda.max_memory_reserved(device) / (1024**2)
        ),
        **sampled,
    }


def build_runtime(protocol: Mapping[str, Any], device: torch.device):
    """构造无 optimizer 的 DriveStudio 数据集与只读 trainer。"""
    sys.path.insert(0, str(DRIVESTUDIO))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    config = OmegaConf.load(protocol["selected_asset"]["source_config"]["path"])
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
    if hasattr(trainer, "optimizer"):
        raise RuntimeError("A4-P1 read-only worker constructed optimizer")
    return config, dataset, trainer


def runtime_model_counts(trainer: Any) -> dict[str, int]:
    return {
        name: int(model._means.shape[0])
        for name, model in trainer.models.items()
        if name in {"Background", "RigidNodes"}
    }


def runtime_point_ids(trainer: Any) -> np.ndarray:
    rigid = trainer.models["RigidNodes"]
    if not hasattr(rigid, "point_ids"):
        raise RuntimeError("A4-P1 RigidNodes runtime point_ids missing")
    return rigid.point_ids[:, 0].detach().cpu().numpy().astype(np.int64, copy=True)


def runtime_gaussian_ids(model: Any) -> np.ndarray:
    ledger = getattr(model, "_a2_ancestry", None)
    if ledger is None or not hasattr(ledger, "gaussian_id"):
        raise RuntimeError("A4-P1 runtime ancestry gaussian_id missing")
    return ledger.gaussian_id.detach().cpu().numpy().astype(np.int64, copy=True)


def intersection_alpha_weights(info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """按冻结的近到远相交顺序计算每次命中的 ``T_before * alpha``。"""
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    height = int(info["height"])
    width = int(info["width"])
    means2d = info["means2d"]
    device = means2d.device
    transmittances = torch.ones(
        (int(info["n_cameras"]), height, width), device=device, dtype=means2d.dtype
    )
    gaussian_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
        0,
        2**31 - 1,
        transmittances,
        means2d,
        info["conics"],
        info["opacities"],
        width,
        height,
        int(info["tile_size"]),
        info["isect_offsets"],
        info["flatten_ids"],
    )
    if gaussian_ids.numel() == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    ray_ids = camera_ids * (height * width) + pixel_ids
    order = torch.argsort(ray_ids, stable=True)
    gaussian_ids = gaussian_ids[order]
    pixel_ids = pixel_ids[order]
    camera_ids = camera_ids[order]
    ray_ids = ray_ids[order]

    projected = means2d[camera_ids, gaussian_ids]
    conics = info["conics"][camera_ids, gaussian_ids]
    opacity = info["opacities"][camera_ids, gaussian_ids]
    x = pixel_ids.remainder(width).to(projected.dtype) + 0.5
    y = torch.div(pixel_ids, width, rounding_mode="floor").to(projected.dtype) + 0.5
    dx = projected[:, 0] - x
    dy = projected[:, 1] - y
    sigma = 0.5 * (conics[:, 0] * dx.square() + conics[:, 2] * dy.square())
    sigma = sigma + conics[:, 1] * dx * dy
    alpha = torch.minimum(opacity * torch.exp(-sigma), opacity.new_tensor(0.999))

    alpha64 = alpha.to(torch.float64)
    log_survival = torch.log1p(-alpha64)
    prefix = torch.cumsum(log_survival, dim=0)
    starts = torch.ones_like(ray_ids, dtype=torch.bool)
    starts[1:] = ray_ids[1:] != ray_ids[:-1]
    start_indices = torch.nonzero(starts, as_tuple=False).flatten()
    bases = torch.zeros(start_indices.numel(), device=device, dtype=torch.float64)
    if start_indices.numel() > 1:
        bases[1:] = prefix[start_indices[1:] - 1]
    segment_ids = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    log_before = prefix - log_survival - bases[segment_ids]
    weights = torch.exp(log_before) * alpha64
    result_ids = gaussian_ids.detach().cpu().numpy().astype(np.int64, copy=True)
    result_weights = weights.detach().cpu().numpy().astype(np.float64, copy=True)
    return result_ids, result_weights


def score_one_view(
    trainer: Any,
    dataset: Any,
    *,
    frame: int,
    camera: int,
    device: torch.device,
    background_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    image_infos, camera_infos, _, _, _, _ = get_view_data(
        dataset, frame, camera, device
    )
    try:
        with torch.inference_mode():
            trainer(image_infos, camera_infos)
        radii = trainer.info["radii"][0].detach().cpu().numpy() > 0
        ids, weights = intersection_alpha_weights(trainer.info)
        total_count = int(trainer.info["means2d"].shape[1])
        if total_count != radii.shape[0]:
            raise RuntimeError("A4-P1 projected radii and Gaussian count drift")
        accumulated = np.zeros(total_count, dtype=np.float64)
        np.add.at(accumulated, ids, weights)
        return (
            accumulated[:background_count],
            accumulated[background_count:],
            radii,
            len(ids),
            float(weights.sum(dtype=np.float64)),
        )
    finally:
        release_trainer_render_info(trainer)


def run_contribution_scan(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any], device: torch.device
) -> None:
    load_stage(run_dir, manifest, "input_audit")
    if nvidia_compute_rows():
        raise RuntimeError("A4-P1 contribution scan GPU preflight not idle")
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with ResourceSampler(os.getpid()) as sampler:
        _, dataset, trainer = build_runtime(protocol, device)
        checkpoint = Path(protocol["selected_asset"]["checkpoint"]["path"])
        load_model_checkpoint_read_only(trainer, checkpoint, device)
        trainer.set_eval()
        counts = runtime_model_counts(trainer)
        background_count = counts["Background"]
        rigid_count = counts["RigidNodes"]
        expected = protocol["selected_asset"]["inventory"]
        if counts != {
            "Background": int(expected["background_gaussians"]),
            "RigidNodes": int(expected["rigid_gaussians"]),
        }:
            raise RuntimeError(f"A4-P1 source count drift: {counts}")

        arrays = {
            "background_train_alpha_weight_sum": np.zeros(background_count, dtype=np.float64),
            "background_train_visible_view_count": np.zeros(background_count, dtype=np.int64),
            "background_heldout_alpha_weight_sum": np.zeros(background_count, dtype=np.float64),
            "background_heldout_visible_view_count": np.zeros(background_count, dtype=np.int64),
            "rigid_train_alpha_weight_sum": np.zeros(rigid_count, dtype=np.float64),
            "rigid_train_visible_view_count": np.zeros(rigid_count, dtype=np.int64),
            "rigid_heldout_alpha_weight_sum": np.zeros(rigid_count, dtype=np.float64),
            "rigid_heldout_visible_view_count": np.zeros(rigid_count, dtype=np.int64),
            "background_learned_opacity": trainer.models["Background"].get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float64),
            "background_gaussian_ids": runtime_gaussian_ids(trainer.models["Background"]),
            "background_model_flat_index": np.arange(background_count, dtype=np.int64),
            "rigid_learned_opacity": trainer.models["RigidNodes"].get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float64),
            "rigid_gaussian_ids": runtime_gaussian_ids(trainer.models["RigidNodes"]),
            "rigid_model_flat_index": np.arange(rigid_count, dtype=np.int64),
            "rigid_point_ids": runtime_point_ids(trainer),
        }
        contract = protocol["contribution_contract"]
        partitions = (
            ("train", [int(value) for value in contract["training_discovery_frames"]]),
            ("heldout", [int(value) for value in contract["heldout_audit_frames"]]),
        )
        cameras = [int(value) for value in contract["cameras"]]
        view_rows = []
        for partition, frames in partitions:
            for frame in frames:
                for camera in cameras:
                    background, rigid, visible, intersections, weight_sum = score_one_view(
                        trainer,
                        dataset,
                        frame=frame,
                        camera=camera,
                        device=device,
                        background_count=background_count,
                    )
                    arrays[f"background_{partition}_alpha_weight_sum"] += background
                    arrays[f"rigid_{partition}_alpha_weight_sum"] += rigid
                    arrays[f"background_{partition}_visible_view_count"] += visible[:background_count]
                    arrays[f"rigid_{partition}_visible_view_count"] += visible[background_count:]
                    view_rows.append(
                        {
                            "partition": partition,
                            "frame": frame,
                            "camera": camera,
                            "intersection_count": intersections,
                            "alpha_weight_sum": weight_sum,
                            "visible_gaussian_count": int(visible.sum()),
                        }
                    )
                    print(
                        f"A4-P1 contribution {partition} frame={frame} camera={camera} intersections={intersections}",
                        flush=True,
                    )
        artifact = run_dir / contract["score_artifact"]
        atomic_npz(artifact, arrays)
        array_records = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
            for name, value in arrays.items()
        }
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset
        torch.cuda.empty_cache()
    resources = stage_resource_summary(device, sampler)
    duration = time.perf_counter() - started
    required_arrays = set(contract["score_artifact_required_arrays"])
    if not required_arrays.issubset(arrays):
        raise RuntimeError("A4-P1 contribution score schema missing frozen arrays")
    stage = {
        "status": "done",
        "stage": "contribution_scan",
        "duration_seconds": duration,
        "score_artifact": {
            "path": str(artifact.relative_to(run_dir)),
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
            "arrays": array_records,
        },
        "required_arrays_exact": set(contract["score_artifact_required_arrays"])
        == {name for name in arrays if name in required_arrays},
        "train_frames": [int(value) for value in contract["training_discovery_frames"]],
        "heldout_audit_frames": [int(value) for value in contract["heldout_audit_frames"]],
        "cameras": [int(value) for value in contract["cameras"]],
        "ranking_partition": "train_only",
        "heldout_influenced_ranking": False,
        "accumulation": "per_gaussian_cpu_float64_np_add_at_stable_intersection_order",
        "view_rows": view_rows,
        "model_gaussian_counts": counts,
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "raw_render_media_written": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "contribution_scan_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "contribution_scan", stage)


def recursive_tensor_hashes(value: Any, prefix: str = "") -> dict[str, str]:
    rows: dict[str, str] = {}
    if torch.is_tensor(value):
        rows[prefix] = tensor_sha256(value)
    elif isinstance(value, Mapping):
        for name, item in value.items():
            child = f"{prefix}.{name}" if prefix else str(name)
            rows.update(recursive_tensor_hashes(item, child))
    return rows


def invariant_hashes(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    all_hashes = recursive_tensor_hashes(checkpoint)
    rigid_exact = {
        "models.RigidNodes.instances_quats",
        "models.RigidNodes.instances_trans",
        "models.RigidNodes.instances_size",
        "models.RigidNodes.instances_fv",
        "step",
    }
    return {
        name: digest
        for name, digest in all_hashes.items()
        if name.startswith("lpips.")
        or name.startswith("models.Sky.")
        or name in rigid_exact
    }


def row_alignment_audit(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for model_name in ("Background", "RigidNodes"):
        state = checkpoint["models"][model_name]
        count = int(state["_means"].shape[0])
        required = [
            "_means",
            "_scales",
            "_quats",
            "_features_dc",
            "_features_rest",
            "_opacities",
        ]
        if model_name == "RigidNodes":
            required.append("points_ids")
        field_rows = {name: int(state[name].shape[0]) for name in required}
        ancestry_rows = {
            name: int(value.shape[0])
            for name, value in state["worldsim_a2_ancestry"]["fields"].items()
        }
        exact = all(value == count for value in field_rows.values()) and all(
            value == count for value in ancestry_rows.values()
        )
        rows[model_name] = {
            "count": count,
            "field_rows": field_rows,
            "ancestry_rows": ancestry_rows,
            "exact": exact,
        }
    rows["exact"] = all(rows[name]["exact"] for name in ("Background", "RigidNodes"))
    return rows


def run_materialize(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: dict[str, Any],
    arm: str,
) -> None:
    if arm not in {"p1-b05", "p1-b10", "p1-b20"}:
        raise ValueError(f"A4-P1 materialize arm invalid: {arm}")
    load_stage(run_dir, manifest, "contribution_scan")
    started = time.perf_counter()
    score_stage = load_stage(run_dir, manifest, "contribution_scan")
    score_path = run_dir / score_stage["score_artifact"]["path"]
    if sha256_file(score_path) != score_stage["score_artifact"]["sha256"]:
        raise RuntimeError("A4-P1 contribution score artifact drift")
    with np.load(score_path, allow_pickle=False) as loaded:
        scores = {name: loaded[name].copy() for name in loaded.files}
    fraction = ARM_FRACTIONS[arm]
    background_scores = {
        "train_alpha_weight_sum": scores["background_train_alpha_weight_sum"],
        "train_visible_view_count": scores["background_train_visible_view_count"],
        "learned_opacity": scores["background_learned_opacity"],
        "gaussian_ids": scores["background_gaussian_ids"],
    }
    rigid_scores = {
        "train_alpha_weight_sum": scores["rigid_train_alpha_weight_sum"],
        "train_visible_view_count": scores["rigid_train_visible_view_count"],
        "learned_opacity": scores["rigid_learned_opacity"],
        "gaussian_ids": scores["rigid_gaussian_ids"],
    }
    background_keep, rigid_keep, removal_rows = build_candidate_masks(
        background_scores=background_scores,
        rigid_scores=rigid_scores,
        rigid_point_ids=scores["rigid_point_ids"],
        prune_fraction=fraction,
        decimal_places=int(protocol["contribution_contract"]["score_quantization"]["decimal_places"]),
    )
    source_path = Path(protocol["selected_asset"]["checkpoint"]["path"])
    source_sha_before = sha256_file(source_path)
    source_checkpoint = torch.load(source_path, map_location="cpu")
    before_invariants = invariant_hashes(source_checkpoint)
    candidate = prune_checkpoint_state(
        source_checkpoint,
        torch.from_numpy(background_keep),
        torch.from_numpy(rigid_keep),
    )
    after_invariants = invariant_hashes(candidate)
    candidate_path = run_dir / "artifacts" / "candidates" / f"{arm}.pth"
    atomic_torch_save(candidate_path, candidate)
    candidate_sha = sha256_file(candidate_path)
    candidate_registry = build_candidate_registry(
        json.loads(
            Path(protocol["selected_asset"]["actor_registry"]["path"]).read_text(
                encoding="utf-8"
            )
        ),
        candidate["models"]["RigidNodes"]["points_ids"][:, 0].tolist(),
        candidate_sha,
    )
    registry_path = run_dir / "artifacts" / "candidates" / f"{arm}_actor_registry.json"
    atomic_json(registry_path, candidate_registry)
    alignment = row_alignment_audit(candidate)
    source_counts = {
        "Background": int(source_checkpoint["models"]["Background"]["_means"].shape[0]),
        "RigidNodes": int(source_checkpoint["models"]["RigidNodes"]["_means"].shape[0]),
    }
    candidate_counts = {
        "Background": int(candidate["models"]["Background"]["_means"].shape[0]),
        "RigidNodes": int(candidate["models"]["RigidNodes"]["_means"].shape[0]),
    }
    every_removal_exact = all(
        int(row["removed_count"]) == math.floor(int(row["source_count"]) * fraction)
        and int(row["remaining_count"]) >= 1
        for row in removal_rows
    )
    unavailable = [
        row
        for row in candidate_registry["actors"]
        if row["availability"] == "unavailable_empty_checkpoint_slice"
    ]
    source_sha_after = sha256_file(source_path)
    duration = time.perf_counter() - started
    stage_name = f"materialize_{arm.replace('-', '_')}"
    stage = {
        "status": "done",
        "stage": stage_name,
        "duration_seconds": duration,
        "arm": arm,
        "prune_fraction": fraction,
        "source_checkpoint_sha256": source_sha_before,
        "candidate_checkpoint": {
            "path": str(candidate_path.relative_to(run_dir)),
            "sha256": candidate_sha,
            "bytes": candidate_path.stat().st_size,
            "write_count": 1,
        },
        "candidate_registry": {
            "path": str(registry_path.relative_to(run_dir)),
            "sha256": sha256_file(registry_path),
            "bytes": registry_path.stat().st_size,
            "embedded_sha256": candidate_registry["actor_registry_sha256"],
        },
        "source_and_candidate_model_counts": {
            "source": source_counts,
            "candidate": candidate_counts,
        },
        "per_asset_removed_count": removal_rows,
        "candidate_grid_and_removal_counts_exact": len(removal_rows)
        == 1 + int(protocol["selected_asset"]["inventory"]["available_actor_count"])
        and every_removal_exact,
        "invariant_field_hashes_before_after": {
            "before": before_invariants,
            "after": after_invariants,
            "exact": before_invariants == after_invariants,
        },
        "checkpoint_schema_exact": list(source_checkpoint.keys()) == list(candidate.keys())
        and list(source_checkpoint["models"].keys()) == list(candidate["models"].keys()),
        "row_alignment_audit": alignment,
        "unavailable_actor_remains_explicitly_empty": len(unavailable)
        == int(protocol["selected_asset"]["inventory"]["unavailable_actor_count"]),
        "source_checkpoint_unchanged": source_sha_before == source_sha_after,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "training_optimizer_or_render_performed": False,
        "minimum_rerun_unit": f"{stage_name}_and_downstream",
    }
    del candidate, source_checkpoint, scores
    write_stage(run_dir, manifest, stage_name, stage)


def mean_or_minus_one(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else -1.0


def torch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((-10 * torch.log10(torch_f.mse_loss(prediction, target))).item())


def global_metric_name(short_name: str) -> str:
    return f"image_metrics/test/{short_name}"


def finite_metric_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_metric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_metric_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def evaluate_quality_arm(
    *,
    protocol: Mapping[str, Any],
    dataset: Any,
    trainer: Any,
    arm: str,
    checkpoint: Path,
    registry_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """一次渲染完整 held-out split，并复用帧缓存计算冻结区域指标。"""
    load_started = time.perf_counter()
    load_model_checkpoint_read_only(trainer, checkpoint, device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    trainer.set_eval()
    if hasattr(trainer.lpips, "reset"):
        trainer.lpips.reset()
    model_counts = runtime_model_counts(trainer)
    point_ids = runtime_point_ids(trainer)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    actor_counts_exact = all(
        int((point_ids == int(actor["rigid_model_index"])).sum())
        == int(actor["checkpoint_tensor_slice"]["gaussian_count"])
        for actor in registry["actors"]
    )
    test_indices = [int(value) for value in dataset.test_image_set.split_indices]
    expected_frames = [int(value) for value in protocol["contribution_contract"]["full_quality_heldout_frames"]]
    expected_indices = [frame * 3 + camera for frame in expected_frames for camera in (0, 1, 2)]
    if test_indices != expected_indices:
        raise RuntimeError("A4-P1 heldout quality split drift")

    metric_lists: dict[str, list[float]] = {
        name: []
        for name in (
            "psnr",
            "ssim",
            "lpips",
            "occupied_psnr",
            "occupied_ssim",
            "masked_psnr",
            "masked_ssim",
            "human_psnr",
            "human_ssim",
            "vehicle_psnr",
            "vehicle_ssim",
        )
    }
    prediction_cache: dict[int, np.ndarray] = {}
    target_cache: dict[int, np.ndarray] = {}
    for position in range(len(test_indices)):
        image_infos, camera_infos = dataset.test_image_set.get_image(
            position, camera_downscale=trainer._get_downscale_factor()
        )
        target_cpu = image_infos["pixels"].detach().float().cpu().numpy()
        image_infos = to_device(image_infos, device)
        camera_infos = to_device(camera_infos, device)
        try:
            with torch.inference_mode():
                output = trainer(image_infos, camera_infos)
            raw_rgb = output["rgb"]
            rgb = raw_rgb.clamp(0.0, 1.0)
            target = image_infos["pixels"]
            prediction = rgb.detach().float().cpu().numpy()
            metric_lists["psnr"].append(torch_psnr(rgb, target))
            metric_lists["ssim"].append(
                float(structural_similarity(target_cpu, prediction, data_range=1.0, channel_axis=-1))
            )
            metric_lists["lpips"].append(
                float(
                    trainer.lpips(
                        rgb[None].permute(0, 3, 1, 2),
                        target[None].permute(0, 3, 1, 2),
                    ).detach().cpu().item()
                )
            )
            masks = (
                ("occupied", "sky_masks", True),
                ("masked", "dynamic_masks", False),
                ("human", "human_masks", False),
                ("vehicle", "vehicle_masks", False),
            )
            for prefix, key, invert in masks:
                if key not in image_infos:
                    continue
                selected = image_infos[key].detach().bool()
                if invert:
                    selected = ~selected
                selected_np = selected.cpu().numpy().squeeze().astype(bool)
                if int(selected_np.sum()) == 0:
                    continue
                metric_lists[f"{prefix}_psnr"].append(torch_psnr(rgb[selected], target[selected]))
                full_ssim = structural_similarity(
                    target_cpu,
                    prediction,
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1]
                metric_lists[f"{prefix}_ssim"].append(float(np.asarray(full_ssim)[selected_np].mean()))
            prediction_cache[position] = uint8_rgb(raw_rgb.detach().float().cpu().numpy())
            target_cache[position] = target_cpu
        finally:
            release_trainer_render_info(trainer)
        print(f"A4-P1 quality arm={arm} render={position + 1}/{len(test_indices)}", flush=True)

    global_metrics = {
        global_metric_name(name): mean_or_minus_one(values)
        for name, values in metric_lists.items()
    }
    if hasattr(trainer.lpips, "reset"):
        trainer.lpips.reset()
    masks_dir = Path(protocol["baseline_quality"]["actor_masks"]["path"])
    historical_actor = json.loads(
        Path(protocol["baseline_quality"]["actor_metrics_summary"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    role_accumulators = {
        role: {
            "actor_region": RegionAccumulator(),
            "boundary_band": RegionAccumulator(),
        }
        for role in protocol["quality_contract"]["actor_endpoints"]["roles"]
    }
    union_masks: dict[int, np.ndarray] = {
        position: np.zeros(target.shape[:2], dtype=bool)
        for position, target in target_cache.items()
    }
    frozen_mask_files = 0
    for role, regions in role_accumulators.items():
        for position, full_index in enumerate(test_indices):
            frame, camera = divmod(full_index, 3)
            mask_path = masks_dir / f"{role}__frame_{frame:03d}_camera_{camera}.png"
            if not mask_path.is_file():
                continue
            effect = np.asarray(imageio.imread(mask_path)) > 0
            frozen_mask_files += 1
            union_masks[position] |= effect
            boundary = boundary_band(effect, radius=3)
            prediction = prediction_cache[position].astype(np.float32) / 255.0
            target = target_cache[position]
            image_ssim = ssim_map(prediction, target)
            for region_name, selected in (("actor_region", effect), ("boundary_band", boundary)):
                lpips_value = masked_lpips(
                    trainer.lpips, prediction, target, selected, device
                )
                regions[region_name].update(
                    prediction, target, selected, image_ssim, lpips_value
                )
    role_metrics = {
        role: {
            "actor": historical_actor["roles"][role]["actor"],
            **{name: accumulator.summary() for name, accumulator in regions.items()},
        }
        for role, regions in role_accumulators.items()
    }
    non_target = RegionAccumulator()
    for position in range(len(test_indices)):
        selected = ~union_masks[position]
        prediction = prediction_cache[position].astype(np.float32) / 255.0
        target = target_cache[position]
        image_ssim = ssim_map(prediction, target)
        lpips_value = masked_lpips(trainer.lpips, prediction, target, selected, device)
        non_target.update(prediction, target, selected, image_ssim, lpips_value)
    non_target_quality = non_target.summary()
    result = {
        "arm": arm,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "checkpoint_load_seconds": load_seconds,
        "checkpoint_reload_exact": actor_counts_exact,
        "model_gaussian_counts": model_counts,
        "actor_counts_exact": actor_counts_exact,
        "unavailable_actor_remains_explicitly_empty": all(
            int((point_ids == int(actor["rigid_model_index"])).sum()) == 0
            for actor in registry["actors"]
            if actor["availability"] == "unavailable_empty_checkpoint_slice"
        ),
        "heldout_image_count": len(test_indices),
        "heldout_full_image_indices": test_indices,
        "global_metrics": global_metrics,
        "roles": role_metrics,
        "non_target": {"quality": non_target_quality},
        "frozen_mask_file_visits": frozen_mask_files,
        "all_endpoints_complete_and_finite": finite_metric_tree(global_metrics)
        and all(
            region["status"] == "done" and finite_metric_tree(region)
            for role in role_metrics.values()
            for name, region in role.items()
            if name in {"actor_region", "boundary_band"}
        )
        and non_target_quality["status"] == "done"
        and finite_metric_tree(non_target_quality),
        "raw_render_media_written": False,
    }
    del prediction_cache, target_cache
    return result


def historical_replay_rows(
    protocol: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    historical_global = json.loads(
        Path(protocol["baseline_quality"]["d2_evaluation_summary"]["path"]).read_text(
            encoding="utf-8"
        )
    )["heldout_metrics"]
    historical_actor = json.loads(
        Path(protocol["baseline_quality"]["actor_metrics_summary"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    tolerances = protocol["quality_contract"]["baseline_historical_replay_tolerance"]

    def tolerance(name: str) -> float:
        if "lpips" in name:
            return float(tolerances["lpips_absolute"])
        if "ssim" in name:
            return float(tolerances["ssim_absolute"])
        if "psnr" in name:
            return float(tolerances["psnr_absolute"])
        return float(tolerances["mean_absolute_error_absolute"])

    rows = []
    for name, expected in historical_global.items():
        actual = result["global_metrics"].get(name)
        limit = tolerance(name)
        rows.append(
            {
                "endpoint": f"global.{name}",
                "historical": expected,
                "replay": actual,
                "absolute_tolerance": limit,
                "passed": actual is not None
                and math.isfinite(float(actual))
                and abs(float(actual) - float(expected)) <= limit,
            }
        )
    for role in protocol["quality_contract"]["actor_endpoints"]["roles"]:
        for region in protocol["quality_contract"]["actor_endpoints"]["regions"]:
            for name in ("psnr", "ssim", "masked_lpips_alex_tight_crop_256px", "mean_absolute_error"):
                expected = historical_actor["roles"][role][region][name]
                actual = result["roles"][role][region].get(name)
                limit = tolerance(name)
                rows.append(
                    {
                        "endpoint": f"roles.{role}.{region}.{name}",
                        "historical": expected,
                        "replay": actual,
                        "absolute_tolerance": limit,
                        "passed": actual is not None
                        and math.isfinite(float(actual))
                        and abs(float(actual) - float(expected)) <= limit,
                    }
                )
    for name in ("psnr", "ssim", "masked_lpips_alex_tight_crop_256px", "mean_absolute_error"):
        expected = historical_actor["non_target"]["quality"][name]
        actual = result["non_target"]["quality"].get(name)
        limit = tolerance(name)
        rows.append(
            {
                "endpoint": f"non_target.{name}",
                "historical": expected,
                "replay": actual,
                "absolute_tolerance": limit,
                "passed": actual is not None
                and math.isfinite(float(actual))
                and abs(float(actual) - float(expected)) <= limit,
            }
        )
    return rows


def quality_safeguard_rows(
    protocol: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    quality = protocol["quality_contract"]
    rows = [
        {"group": "global", **row}
        for row in compare_metric_group(
            baseline["global_metrics"],
            candidate["global_metrics"],
            quality["global_endpoints"]["metrics"],
        )
    ]
    for role in quality["actor_endpoints"]["roles"]:
        for region in quality["actor_endpoints"]["regions"]:
            rows.extend(
                {"group": f"{role}.{region}", **row}
                for row in compare_metric_group(
                    baseline["roles"][role][region],
                    candidate["roles"][role][region],
                    quality["actor_endpoints"]["metrics"],
                )
            )
    rows.extend(
        {"group": "non_target", **row}
        for row in compare_metric_group(
            baseline["non_target"]["quality"],
            candidate["non_target"]["quality"],
            quality["non_target_endpoints"]["metrics"],
        )
    )
    return rows


def arm_assets(
    run_dir: Path, protocol: Mapping[str, Any], manifest: Mapping[str, Any], arm: str
) -> tuple[Path, Path, dict[str, int]]:
    if arm == "p1-source":
        inventory = protocol["selected_asset"]["inventory"]
        return (
            Path(protocol["selected_asset"]["checkpoint"]["path"]),
            Path(protocol["selected_asset"]["actor_registry"]["path"]),
            {
                "Background": int(inventory["background_gaussians"]),
                "RigidNodes": int(inventory["rigid_gaussians"]),
            },
        )
    stage = load_stage(run_dir, manifest, f"materialize_{arm.replace('-', '_')}")
    return (
        run_dir / stage["candidate_checkpoint"]["path"],
        run_dir / stage["candidate_registry"]["path"],
        {key: int(value) for key, value in stage["source_and_candidate_model_counts"]["candidate"].items()},
    )


def run_evaluate(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: dict[str, Any],
    arms: Sequence[str],
    device: torch.device,
) -> None:
    expected_lists = {
        ("p1-source", "p1-b05"): "evaluate_p1_source_and_b05",
        ("p1-b10",): "evaluate_p1_b10",
        ("p1-b20",): "evaluate_p1_b20",
    }
    key = tuple(arms)
    if key not in expected_lists:
        raise ValueError(f"A4-P1 evaluation arm sequence invalid: {arms}")
    stage_name = expected_lists[key]
    for arm in arms:
        if arm != "p1-source":
            load_stage(run_dir, manifest, f"materialize_{arm.replace('-', '_')}")
    if nvidia_compute_rows():
        raise RuntimeError(f"A4-P1 {stage_name} GPU preflight not idle")
    masks_contract = protocol["baseline_quality"]["actor_masks"]
    masks_before = directory_digest(Path(masks_contract["path"]), masks_contract["file_glob"])
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with ResourceSampler(os.getpid()) as sampler:
        _, dataset, trainer = build_runtime(protocol, device)
        baseline_path = run_dir / "artifacts" / "quality" / "p1-source.json"
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_path.is_file()
            else None
        )
        results = {}
        for arm in arms:
            checkpoint, registry, expected_counts = arm_assets(
                run_dir, protocol, manifest, arm
            )
            result = evaluate_quality_arm(
                protocol=protocol,
                dataset=dataset,
                trainer=trainer,
                arm=arm,
                checkpoint=checkpoint,
                registry_path=registry,
                device=device,
            )
            result["expected_model_counts"] = expected_counts
            result["expected_model_counts_exact"] = result["model_gaussian_counts"] == expected_counts
            if arm == "p1-source":
                replay = historical_replay_rows(protocol, result)
                result["historical_replay_rows"] = replay
                result["baseline_historical_replay_pass"] = all(row["passed"] for row in replay)
                result["quality_safeguard_rows"] = []
                result["all_quality_safeguards_pass"] = result["baseline_historical_replay_pass"]
                baseline = result
            else:
                if baseline is None:
                    raise RuntimeError("A4-P1 source quality baseline artifact missing")
                safeguard_rows = quality_safeguard_rows(protocol, baseline, result)
                result["quality_safeguard_rows"] = safeguard_rows
                result["all_quality_safeguards_pass"] = all(row["passed"] for row in safeguard_rows)
            output_path = run_dir / "artifacts" / "quality" / f"{arm}.json"
            atomic_json(output_path, result)
            results[arm] = {
                "path": str(output_path.relative_to(run_dir)),
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
                "all_quality_safeguards_pass": result["all_quality_safeguards_pass"],
                "all_endpoints_complete_and_finite": result["all_endpoints_complete_and_finite"],
                "checkpoint_reload_exact": result["checkpoint_reload_exact"],
                "expected_model_counts_exact": result["expected_model_counts_exact"],
            }
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset
        torch.cuda.empty_cache()
    resources = stage_resource_summary(device, sampler)
    masks_after = directory_digest(Path(masks_contract["path"]), masks_contract["file_glob"])
    stage = {
        "status": "done",
        "stage": stage_name,
        "duration_seconds": time.perf_counter() - started,
        "arms": list(arms),
        "quality_artifacts": results,
        "frozen_masks_before": masks_before,
        "frozen_masks_after": masks_after,
        "frozen_masks_reused_exactly": masks_before == masks_after
        and masks_after
        == {
            "sha256": masks_contract["sha256"],
            "file_count": int(masks_contract["file_count"]),
            "total_bytes": int(masks_contract["total_bytes"]),
        },
        "candidate_mask_regeneration_performed": False,
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "raw_render_media_written": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "minimum_rerun_unit": f"{stage_name}_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, stage_name, stage)


def render_runtime_view(
    trainer: Any,
    dataset: Any,
    *,
    frame: int,
    camera: int,
    device: torch.device,
    phase: str,
    ordinal: int,
) -> dict[str, Any]:
    image_infos, camera_infos, _, _, _, image_index = get_view_data(
        dataset, frame, camera, device
    )
    try:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = trainer(image_infos, camera_infos)
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        rgb = uint8_rgb(output["rgb"].detach().float().cpu().numpy())
        return {
            "phase": phase,
            "ordinal": ordinal,
            "frame": frame,
            "camera": camera,
            "image_index": image_index,
            "duration_seconds": duration,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "rgb_sha256": rgb_sha256(rgb),
        }
    finally:
        release_trainer_render_info(trainer)


def run_runtime_profile(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any], device: torch.device
) -> None:
    for name in (
        "evaluate_p1_source_and_b05",
        "evaluate_p1_b10",
        "evaluate_p1_b20",
    ):
        load_stage(run_dir, manifest, name)
    if nvidia_compute_rows():
        raise RuntimeError("A4-P1 runtime profile GPU preflight not idle")
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with ResourceSampler(os.getpid()) as sampler:
        _, dataset, trainer = build_runtime(protocol, device)
        contract = protocol["runtime_contract"]
        frames = [int(value) for value in contract["frames"]]
        cameras = [int(value) for value in contract["cameras"]]
        matrix = [(frame, camera) for frame in frames for camera in cameras]
        arm_rows = {}
        for arm in ARM_FRACTIONS:
            checkpoint, registry_path, expected_counts = arm_assets(
                run_dir, protocol, manifest, arm
            )
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            load_started = time.perf_counter()
            load_model_checkpoint_read_only(trainer, checkpoint, device)
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter() - load_started
            trainer.set_eval()
            warmup_rows = [
                render_runtime_view(
                    trainer,
                    dataset,
                    frame=frames[0],
                    camera=cameras[0],
                    device=device,
                    phase="warmup",
                    ordinal=ordinal,
                )
                for ordinal in range(int(contract["warmup_views"]))
            ]
            measured_rows = [
                render_runtime_view(
                    trainer,
                    dataset,
                    frame=frame,
                    camera=camera,
                    device=device,
                    phase="measured",
                    ordinal=ordinal,
                )
                for ordinal, (frame, camera) in enumerate(matrix)
            ]
            durations = [row["duration_seconds"] for row in measured_rows]
            counts = runtime_model_counts(trainer)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            point_ids = runtime_point_ids(trainer)
            actor_counts_exact = all(
                int((point_ids == int(actor["rigid_model_index"])).sum())
                == int(actor["checkpoint_tensor_slice"]["gaussian_count"])
                for actor in registry["actors"]
            )
            arm_rows[arm] = {
                "arm": arm,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_load_seconds": load_seconds,
                "model_gaussian_counts": counts,
                "expected_model_counts": expected_counts,
                "checkpoint_reload_exact": counts == expected_counts and actor_counts_exact,
                "warmup_rows": warmup_rows,
                "measured_rows": measured_rows,
                "render_p50_seconds": nearest_rank(durations, 0.50),
                "render_p95_seconds": nearest_rank(durations, 0.95),
                "aggregate_fps": len(durations) / sum(durations),
                "peak_torch_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
                "peak_torch_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
                "matrix_complete_and_unique": len(measured_rows) == 9
                and len({(row["frame"], row["camera"]) for row in measured_rows}) == 9,
                "resolution_exact": all(
                    [row["width"], row["height"]] == [800, 450]
                    for row in warmup_rows + measured_rows
                ),
                "synchronized_timing_complete": all(value > 0 for value in durations),
                "filesystem_cache": contract["filesystem_cache"],
            }
            print(f"A4-P1 runtime arm={arm} complete", flush=True)
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset
        torch.cuda.empty_cache()
    resources = stage_resource_summary(device, sampler)
    stage = {
        "status": "done",
        "stage": "runtime_profile_all_arms",
        "duration_seconds": time.perf_counter() - started,
        "arm_rows": arm_rows,
        "matrix": {"frames": frames, "cameras": cameras, "sample_count_per_arm": 9},
        "performance_values_report_only": True,
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "raw_render_media_written": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "runtime_profile_all_arms_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "runtime_profile_all_arms", stage)


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--operation",
        required=True,
        choices=("contribution-scan", "materialize", "evaluate", "runtime-profile"),
    )
    parser.add_argument("--arm")
    parser.add_argument("--arms")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256_file(args.protocol):
        raise RuntimeError("A4-P1 protocol hash drift after initialization")
    device = torch.device(args.device)
    if args.operation == "contribution-scan":
        run_contribution_scan(args.run_dir, protocol, manifest, device)
    elif args.operation == "materialize":
        if args.arm is None:
            raise ValueError("A4-P1 materialize requires --arm")
        run_materialize(args.run_dir, protocol, manifest, args.arm)
    elif args.operation == "evaluate":
        if args.arms is None:
            raise ValueError("A4-P1 evaluate requires --arms")
        run_evaluate(args.run_dir, protocol, manifest, args.arms.split(","), device)
    else:
        run_runtime_profile(args.run_dir, protocol, manifest, device)
    print(json.dumps({"status": "done", "operation": args.operation, "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.exists():
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P1_WORKER_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
