#!/usr/bin/env python3
"""在冻结 StreetGS base 上执行 scene0471 的 M1 unary 单变量诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import binary_dilation, binary_erosion
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.resim.drivestudio_adapter import gsplat_first_hit_from_info
from motion_proj.worldsim_v4.evidence_metrics import probability_metrics
from motion_proj.worldsim_v5.bayesian_unary import (
    UNARY_ARM_NAMES,
    accumulate_unary_arm_statistics,
    empty_unary_arm_statistics,
    finalize_unary_arms,
)
from motion_proj.worldsim_v5.evidence_schema import (
    atomic_save_npz,
    sha256_file,
    validate_gaussian_table,
)
from motion_proj.worldsim_v5.geometry_evidence import (
    gaussian_geometry,
    view_angle_cosine,
)
from motion_proj.worldsim_v5.observation_aggregation import (
    aggregate_intersection_observations,
)
from motion_proj.worldsim_v5.observation_builder import (
    build_observation_chunk,
    sparse_contribution_selection,
)
from motion_proj.worldsim_v5.ownership_renderer import (
    rasterize_ownership_probability,
)
from motion_proj.worldsim_v5.renderer_intersections import renderer_intersections
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    prepare_formal_run,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M1-STRUCTURED-OWNERSHIP-01"
SCHEMA_VERSION = "worldsim_v5_m1_unary_diagnostic_v1"


class UnaryDiagnosticError(RuntimeError):
    """冻结输入、renderer 或 unary 诊断合同失败。"""


def _runtime_helpers():
    """延迟导入 DriveStudio bridge，保证 runner 的 help/helper 可独立回归。"""

    from scripts.eval_worldsim_v3_a3_r1_heldout import (
        get_view_data,
        load_model_checkpoint_read_only,
        release_trainer_render_info,
    )

    return get_view_data, load_model_checkpoint_read_only, release_trainer_render_info


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise UnaryDiagnosticError("unary diagnostic config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "structured_unary_mechanism_smoke"
    ):
        raise UnaryDiagnosticError("unary diagnostic task/phase/status 漂移")
    if tuple(payload["unary"]["arms"]) != UNARY_ARM_NAMES:
        raise UnaryDiagnosticError("unary arms 集合或顺序漂移")
    return payload


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _binary_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(predicted, dtype=bool)
    right = np.asarray(target, dtype=bool)
    union = left | right
    if not union.any():
        return 1.0
    return float((left & right).sum() / union.sum())


def _boundary_f1(
    predicted: np.ndarray, target: np.ndarray, *, tolerance: int
) -> float:
    left = np.asarray(predicted, dtype=bool)
    right = np.asarray(target, dtype=bool)
    left_boundary = left & ~binary_erosion(left)
    right_boundary = right & ~binary_erosion(right)
    if not left_boundary.any() and not right_boundary.any():
        return 1.0
    if not left_boundary.any() or not right_boundary.any():
        return 0.0
    left_match = left_boundary & binary_dilation(right_boundary, iterations=tolerance)
    right_match = right_boundary & binary_dilation(left_boundary, iterations=tolerance)
    precision = float(left_match.sum() / left_boundary.sum())
    recall = float(right_match.sum() / right_boundary.sum())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _negative_log_likelihood(probability: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    label = np.asarray(target, dtype=np.float64)
    return float(-(label * np.log(prediction) + (1.0 - label) * np.log(1.0 - prediction)).mean())


def _aggregate_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise UnaryDiagnosticError("没有可聚合的 evaluation rows")
    names = sorted(rows[0])
    return {
        name: float(np.mean([float(row[name]) for row in rows])) for name in names
    }


def _write_terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None = None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m1_unary_status_v1",
            "task_id": TASK_ID,
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def _build_runtime(config: Mapping[str, Any], device: torch.device):
    upstream = Path(config["runtime"]["drivestudio_checkout"])
    if _git(upstream, "rev-parse", "HEAD") != config["runtime"]["drivestudio_commit"]:
        raise UnaryDiagnosticError("DriveStudio source commit 漂移")
    expected_status = str(config["runtime"]["drivestudio_expected_status"]).strip()
    if _git(upstream, "status", "--short") != expected_status:
        raise UnaryDiagnosticError("DriveStudio frozen patch status 漂移")
    sys.path.insert(0, str(upstream))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    source = OmegaConf.load(config["inputs"]["source_config"]["path"])
    dataset = DrivingDataset(data_cfg=source.data)
    trainer = import_str(source.trainer.type)(
        **source.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=source.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    if hasattr(trainer, "optimizer"):
        raise UnaryDiagnosticError("unary diagnostic 必须构造无 optimizer 的只读 trainer")
    _, load_model_checkpoint_read_only, _ = _runtime_helpers()
    load_model_checkpoint_read_only(
        trainer, Path(config["inputs"]["formal_checkpoint"]["path"]), device
    )
    trainer.set_eval()
    return source, dataset, trainer


def _collect_gaussians(
    trainer: Any,
    image_infos: Mapping[str, Any],
    camera_infos: Mapping[str, Any],
) -> tuple[Any, Any]:
    if not hasattr(trainer, "normalized_timestamps") or "normed_time" not in image_infos:
        raise UnaryDiagnosticError("DriveStudio trainer 缺少 frame/timestamp contract")
    normed_time = image_infos["normed_time"].flatten()[0]
    current_frame = torch.argmin(
        torch.abs(trainer.normalized_timestamps - normed_time)
    )
    trainer.cur_frame = current_frame
    for model in trainer.models.values():
        if hasattr(model, "in_test_set"):
            model.in_test_set = trainer.in_test_set
    for class_name in trainer.gaussian_classes:
        model = trainer.models[class_name]
        if hasattr(model, "set_cur_frame"):
            model.set_cur_frame(current_frame)
    processed_camera = trainer.process_camera(
        camera_infos=camera_infos,
        image_ids=image_infos["img_idx"].flatten()[0],
        novel_view=False,
    )
    gaussians = trainer.collect_gaussians(
        cam=processed_camera, image_ids=image_infos["img_idx"].flatten()[0]
    )
    return processed_camera, gaussians


def _collect_render_state(
    trainer: Any,
    image_infos: Mapping[str, Any],
    camera_infos: Mapping[str, Any],
) -> tuple[Any, Any]:
    processed_camera, gaussians = _collect_gaussians(
        trainer, image_infos, camera_infos
    )
    trainer.render_gaussians(
        gs=gaussians,
        cam=processed_camera,
        near_plane=trainer.render_cfg.near_plane,
        far_plane=trainer.render_cfg.far_plane,
        render_mode="RGB+ED",
        radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
    )
    return processed_camera, gaussians


def _global_layout(trainer: Any) -> tuple[np.ndarray, np.ndarray, int, int]:
    background_count = int(trainer.models["Background"]._means.shape[0])
    rigid_count = int(trainer.models["RigidNodes"]._means.shape[0])
    base_model = np.concatenate(
        (
            np.full(background_count, "Background", dtype="<U12"),
            np.full(rigid_count, "RigidNodes", dtype="<U12"),
        )
    )
    base_index = np.concatenate(
        (np.arange(background_count, dtype=np.int64), np.arange(rigid_count, dtype=np.int64))
    )
    return base_model, base_index, background_count, rigid_count


def _gaussian_table(
    *,
    scene: str,
    base_model: np.ndarray,
    base_index: np.ndarray,
    geometry: Mapping[str, np.ndarray],
    prior: np.ndarray,
    unary: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = prior.size
    zeros = np.zeros(count, dtype=np.float32)
    unavailable = np.zeros(count, dtype=np.int8)
    table = {
        "scene": np.asarray(scene),
        "role": np.asarray("moving_rigid_union"),
        "gaussian_id": np.arange(count, dtype=np.int64),
        "base_model": base_model,
        "base_index": base_index,
        "center": np.asarray(geometry["center"], dtype=np.float32),
        "covariance": np.asarray(geometry["covariance"], dtype=np.float32),
        "normal_proxy": np.asarray(geometry["normal_proxy"], dtype=np.float32),
        "normal_available": np.asarray(geometry["normal_available"], dtype=np.int8),
        "prior": prior.astype(np.float32),
        "unary_posterior": np.asarray(unary["unary_posterior"], dtype=np.float32),
        "unary_uncertainty": np.asarray(unary["unary_uncertainty"], dtype=np.float32),
        "effective_evidence_count": np.asarray(unary["effective_evidence_count"], dtype=np.float32),
        "multi_view_disagreement": np.asarray(unary["multi_view_disagreement"], dtype=np.float32),
        "boundary_ambiguity": np.asarray(unary["boundary_ambiguity"], dtype=np.float32),
        "depth_support": np.asarray(unary["depth_support"], dtype=np.float32),
        "lidar_support": zeros,
        "lidar_support_available": unavailable,
        "motion_consistency": zeros.copy(),
        "motion_consistency_available": unavailable.copy(),
    }
    validate_gaussian_table(table)
    return table


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved_record = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        inputs = {
            name: verify_file(value["path"], value["sha256"])
            for name, value in config["inputs"].items()
        }
        formal_summary = json.loads(
            Path(config["inputs"]["formal_summary"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        if (
            formal_summary.get("task_id") != TASK_ID
            or formal_summary.get("status") != "done"
            or formal_summary.get("mode") != "formal"
            or formal_summary.get("scene") != config["scene"]["name"]
            or formal_summary.get("checkpoint", {}).get("sha256")
            != inputs["formal_checkpoint"]["sha256"]
            or formal_summary.get("validation_quality_read") is not False
            or formal_summary.get("test_quality_read") is not False
        ):
            raise UnaryDiagnosticError("formal base summary contract 漂移")
        sam_summary = json.loads(
            Path(config["inputs"]["sam_summary"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        sam_manifest = json.loads(
            Path(config["inputs"]["sam_mask_manifest"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        if (
            sam_summary.get("task_id") != TASK_ID
            or sam_summary.get("status") != "done"
            or sam_summary.get("scene") != config["scene"]["name"]
            or sam_summary.get("mask_manifest_sha256")
            != inputs["sam_mask_manifest"]["sha256"]
            or sam_summary.get("heldout_quality_read") is not False
            or sam_manifest.get("heldout_quality_read") is not False
        ):
            raise UnaryDiagnosticError("SAM manifest heldout contract 漂移")
        checkpoint_path = Path(config["inputs"]["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint_path)
        if not torch.cuda.is_available():
            raise UnaryDiagnosticError("unary diagnostic 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        maximum_start = int(config["resources"]["maximum_gpu_allocated_at_start_mib"])
        if torch.cuda.memory_allocated(device) > maximum_start * 1024**2:
            raise UnaryDiagnosticError("unary diagnostic GPU preflight 非空闲")
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        get_view_data, _, release_trainer_render_info = _runtime_helpers()
        _, dataset, trainer = _build_runtime(config, device)
        base_model, base_index, background_count, rigid_count = _global_layout(trainer)
        gaussian_count = base_model.size
        unassigned = float(config["unary"]["unassigned_probability"])
        prior = np.full(gaussian_count, unassigned, dtype=np.float64)
        prior[background_count:] = 1.0 - unassigned
        statistics = empty_unary_arm_statistics(gaussian_count)
        rows = sam_manifest["views"]
        evidence_rows = [row for row in rows if row["split"] == "evidence"]
        evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
        if not evidence_rows or not evaluation_rows:
            raise UnaryDiagnosticError("SAM evidence/evaluation views 为空")

        reference_row = evidence_rows[0]
        image_infos, camera_infos, *_ = get_view_data(
            dataset,
            int(reference_row["frame"]),
            int(reference_row["camera_id"]),
            device,
        )
        with torch.inference_mode():
            reference_camera, reference_gaussians = _collect_gaussians(
                trainer, image_infos, camera_infos
            )
        reference_center = (
            reference_camera.camtoworlds[..., :3, 3]
            .reshape(-1, 3)[0]
            .detach()
            .cpu()
            .numpy()
        )
        geometry = gaussian_geometry(
            means=reference_gaussians.means.detach().cpu().numpy(),
            scales=reference_gaussians.scales.detach().cpu().numpy(),
            quaternions_wxyz=reference_gaussians.quats.detach().cpu().numpy(),
            reference_camera_center=reference_center,
        )
        del reference_gaussians

        observation_records: list[dict[str, Any]] = []
        dropped_count = 0
        dropped_mass = 0.0
        unary_cfg = config["unary"]
        evidence_cfg = config["evidence"]
        for view_id, row in enumerate(evidence_rows):
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
            sam_run = Path(config["inputs"]["sam_mask_manifest"]["path"]).parents[1]
            mask_path = sam_run / row["mask"]["path"]
            if sha256_file(mask_path) != row["mask"]["sha256"]:
                raise UnaryDiagnosticError(f"SAM mask SHA 漂移: {mask_path}")
            mask = np.load(mask_path, allow_pickle=False)
            image_infos, camera_infos, *_ = get_view_data(
                dataset, frame, camera_id, device
            )
            try:
                with torch.inference_mode():
                    processed_camera, gaussians = _collect_render_state(
                        trainer, image_infos, camera_infos
                    )
                    gids, pids, pixels, weights, depths = renderer_intersections(
                        trainer.info
                    )
                    first_hit, first_valid = gsplat_first_hit_from_info(
                        trainer.info,
                        alpha_threshold=float(evidence_cfg["first_hit_alpha_threshold"]),
                    )
                selected, selection = sparse_contribution_selection(
                    weights,
                    minimum_weight=float(evidence_cfg["minimum_contribution_weight"]),
                )
                dropped_count += int(selection["dropped_count"])
                dropped_mass += float(selection["dropped_contribution_mass"])
                camera_center = (
                    processed_camera.camtoworlds[..., :3, 3]
                    .reshape(-1, 3)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                angles = view_angle_cosine(
                    centers=gaussians.means.detach().cpu().numpy(),
                    normals=geometry["normal_proxy"],
                    camera_center=camera_center,
                )
                chunk = build_observation_chunk(
                    scene=config["scene"]["name"],
                    role="moving_rigid_union",
                    view_id=view_id,
                    frame_id=frame,
                    camera_id=camera_id,
                    gaussian_count=gaussian_count,
                    gaussian_id=gids[selected],
                    pixel_id=pids[selected],
                    projected_pixel=pixels[selected],
                    contribution_weight=weights[selected],
                    projected_depth=depths[selected],
                    first_hit_depth=np.asarray(first_hit),
                    first_hit_valid=np.asarray(first_valid),
                    mask_logits=mask["logits"],
                    mask_binary=mask["binary"],
                    mask_quality_accepted=bool(row["mask_quality_accepted"]),
                    sam_probability_available=bool(row["sam_probability_available"]),
                    view_angle_cosine=angles,
                    lidar_support=None,
                    depth_absolute_tolerance_m=float(evidence_cfg["depth_absolute_tolerance_m"]),
                    depth_relative_tolerance=float(evidence_cfg["depth_relative_tolerance"]),
                    sam_confidence_floor=float(unary_cfg["sam_confidence_floor"]),
                    boundary_distance_scale_px=float(unary_cfg["boundary_distance_scale_px"]),
                    depth_residual_scale_m=float(unary_cfg["depth_residual_scale_m"]),
                )
                aggregated, aggregation = aggregate_intersection_observations(
                    chunk,
                    gaussian_count=gaussian_count,
                    minimum_contribution_mass=float(
                        evidence_cfg["minimum_aggregated_contribution_mass"]
                    ),
                    sam_confidence_floor=float(unary_cfg["sam_confidence_floor"]),
                    boundary_distance_scale_px=float(unary_cfg["boundary_distance_scale_px"]),
                    depth_residual_scale_m=float(unary_cfg["depth_residual_scale_m"]),
                )
                arm_weights = accumulate_unary_arm_statistics(
                    statistics,
                    observations=aggregated,
                    gaussian_count=gaussian_count,
                    sam_confidence_floor=float(unary_cfg["sam_confidence_floor"]),
                    boundary_distance_scale_px=float(unary_cfg["boundary_distance_scale_px"]),
                    depth_residual_scale_m=float(unary_cfg["depth_residual_scale_m"]),
                )
                output = (
                    run_dir
                    / "artifacts/observations"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                atomic_save_npz(output, aggregated)
                observation_records.append(
                    {
                        "frame": frame,
                        "camera_id": camera_id,
                        "raw_intersection_count": int(gids.size),
                        "selected_intersection_count": int(selected.sum()),
                        "aggregated_gaussian_count": int(
                            np.asarray(aggregated["gaussian_id"]).size
                        ),
                        "selection": selection,
                        "aggregation": aggregation,
                        "arm_weight_sum": {
                            arm: float(value.sum()) for arm, value in arm_weights.items()
                        },
                        "path": str(output.relative_to(run_dir)),
                        "sha256": sha256_file(output),
                    }
                )
            finally:
                release_trainer_render_info(trainer)
            print(
                f"M1 unary evidence {view_id + 1}/{len(evidence_rows)} "
                f"frame={frame} camera={camera_id}",
                flush=True,
            )

        unary_outputs = finalize_unary_arms(
            prior_probability=prior,
            prior_strength=float(unary_cfg["prior_strength"]),
            statistics=statistics,
        )
        target_gaussian = (base_model == "RigidNodes").astype(np.float32)
        gaussian_metrics: dict[str, dict[str, float]] = {}
        arm_records: dict[str, dict[str, Any]] = {}
        for arm in UNARY_ARM_NAMES:
            posterior = unary_outputs[arm]["unary_posterior"]
            metrics = probability_metrics(
                posterior,
                target_gaussian,
                bins=int(config["evaluation"]["ece_bins"]),
            )
            metrics.update(
                iou_at_frozen_threshold=_binary_iou(
                    posterior >= float(config["evaluation"]["probability_threshold"]),
                    target_gaussian,
                ),
                nll=_negative_log_likelihood(posterior, target_gaussian),
            )
            gaussian_metrics[arm] = metrics
            table = _gaussian_table(
                scene=config["scene"]["name"],
                base_model=base_model,
                base_index=base_index,
                geometry=geometry,
                prior=prior,
                unary=unary_outputs[arm],
            )
            table_path = run_dir / "artifacts/gaussians" / f"{arm}.npz"
            atomic_save_npz(table_path, table)
            arm_records[arm] = {
                "path": str(table_path.relative_to(run_dir)),
                "sha256": sha256_file(table_path),
                "metrics": metrics,
            }

        evaluation_by_arm: dict[str, list[dict[str, Any]]] = {
            arm: [] for arm in UNARY_ARM_NAMES
        }
        accepted_evaluation_views = 0
        for view_id, row in enumerate(evaluation_rows):
            if not (
                bool(row["sam_probability_available"])
                and bool(row["mask_quality_accepted"])
            ):
                continue
            accepted_evaluation_views += 1
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
            sam_run = Path(config["inputs"]["sam_mask_manifest"]["path"]).parents[1]
            mask_path = sam_run / row["mask"]["path"]
            if sha256_file(mask_path) != row["mask"]["sha256"]:
                raise UnaryDiagnosticError(f"evaluation SAM mask SHA 漂移: {mask_path}")
            target = np.load(mask_path, allow_pickle=False)["binary"].astype(bool)
            image_infos, camera_infos, *_ = get_view_data(
                dataset, frame, camera_id, device
            )
            try:
                with torch.inference_mode():
                    processed_camera, gaussians = _collect_gaussians(
                        trainer, image_infos, camera_infos
                    )
                    for arm in UNARY_ARM_NAMES:
                        alpha, _ = rasterize_ownership_probability(
                            means=gaussians.means,
                            quats=gaussians.quats,
                            scales=gaussians.scales,
                            base_opacities=gaussians.opacities,
                            probability=unary_outputs[arm]["unary_posterior"],
                            viewmats=torch.linalg.inv(processed_camera.camtoworlds)[None, ...],
                            intrinsics=processed_camera.Ks[None, ...],
                            width=int(processed_camera.W),
                            height=int(processed_camera.H),
                            near_plane=float(trainer.render_cfg.near_plane),
                            far_plane=float(trainer.render_cfg.far_plane),
                            packed=bool(trainer.render_cfg.packed),
                            radius_clip=float(trainer.render_cfg.get("radius_clip", 0.0)),
                            antialiased=bool(trainer.render_cfg.antialiased),
                        )
                        probability = alpha.detach().cpu().numpy().astype(np.float32)
                        if probability.shape != target.shape:
                            raise UnaryDiagnosticError("ownership render/SAM shape 漂移")
                        metrics = probability_metrics(
                            probability,
                            target.astype(np.float32),
                            bins=int(config["evaluation"]["ece_bins"]),
                        )
                        metrics.update(
                            iou_at_frozen_threshold=_binary_iou(
                                probability
                                >= float(config["evaluation"]["probability_threshold"]),
                                target,
                            ),
                            boundary_f1=_boundary_f1(
                                probability
                                >= float(config["evaluation"]["probability_threshold"]),
                                target,
                                tolerance=int(
                                    config["evaluation"]["boundary_tolerance_px"]
                                ),
                            ),
                            nll=_negative_log_likelihood(probability, target),
                        )
                        output = (
                            run_dir
                            / "artifacts/evaluation"
                            / arm
                            / f"f{frame:03d}_c{camera_id}.npz"
                        )
                        atomic_save_npz(
                            output,
                            {
                                "probability": probability.astype(np.float16),
                                "target": target.astype(np.int8),
                            },
                        )
                        evaluation_by_arm[arm].append(
                            {
                                "frame": frame,
                                "camera_id": camera_id,
                                **metrics,
                                "path": str(output.relative_to(run_dir)),
                                "sha256": sha256_file(output),
                            }
                        )
            finally:
                release_trainer_render_info(trainer)
            print(
                f"M1 unary evaluation {view_id + 1}/{len(evaluation_rows)} "
                f"frame={frame} camera={camera_id}",
                flush=True,
            )
        if accepted_evaluation_views == 0:
            raise UnaryDiagnosticError("没有通过冻结质量门的 evaluation view")
        evaluation_aggregate = {
            arm: _aggregate_metrics(
                [
                    {key: value for key, value in row.items() if isinstance(value, float)}
                    for row in evaluation_by_arm[arm]
                ]
            )
            for arm in UNARY_ARM_NAMES
        }
        gaussian_delta_vs_b0 = {
            arm: {
                name: float(value - gaussian_metrics["B0"][name])
                for name, value in gaussian_metrics[arm].items()
            }
            for arm in ("B1", "B3")
        }
        evaluation_delta_vs_b0 = {
            arm: {
                name: float(value - evaluation_aggregate["B0"][name])
                for name, value in evaluation_aggregate[arm].items()
            }
            for arm in ("B1", "B3")
        }
        checkpoint_after = sha256_file(checkpoint_path)
        if checkpoint_after != checkpoint_before:
            raise UnaryDiagnosticError("formal checkpoint 在 unary diagnostic 后 mutation")
        duration = time.perf_counter() - started
        diagnostics = {
            "schema_version": "worldsim_v5_m1_unary_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "gaussian_counts": {
                "Background": background_count,
                "RigidNodes": rigid_count,
            },
            "prior": {
                "background": unassigned,
                "rigid_nodes": 1.0 - unassigned,
                "source": "frozen_base_model_membership_proxy",
            },
            "observation_records": observation_records,
            "evaluation_view_count": len(evaluation_rows),
            "accepted_evaluation_view_count": accepted_evaluation_views,
            "abstained_evaluation_view_count": len(evaluation_rows)
            - accepted_evaluation_views,
            "dropped_intersection_count": dropped_count,
            "dropped_contribution_mass": dropped_mass,
            "gaussian_metrics": gaussian_metrics,
            "gaussian_delta_vs_b0": gaussian_delta_vs_b0,
            "evaluation_rows": evaluation_by_arm,
            "evaluation_aggregate": evaluation_aggregate,
            "evaluation_delta_vs_b0": evaluation_delta_vs_b0,
            "graph_inference_started": False,
            "parameter_search_performed": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
        }
        diagnostics_path = run_dir / "artifacts/diagnostics.json"
        atomic_json(diagnostics_path, diagnostics)
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m1_unary_diagnostic.py",
                PROJECT / "scripts/worldsim_v5_forensics_common.py",
                PROJECT / "scripts/eval_worldsim_v3_a3_r1_heldout.py",
                PROJECT / "motion_proj/resim/drivestudio_adapter.py",
                PROJECT / "motion_proj/worldsim_v4/evidence_metrics.py",
                PROJECT / "motion_proj/worldsim_v5/bayesian_unary.py",
                PROJECT / "motion_proj/worldsim_v5/evidence_schema.py",
                PROJECT / "motion_proj/worldsim_v5/geometry_evidence.py",
                PROJECT / "motion_proj/worldsim_v5/observation_builder.py",
                PROJECT / "motion_proj/worldsim_v5/observation_aggregation.py",
                PROJECT / "motion_proj/worldsim_v5/ownership_renderer.py",
                PROJECT / "motion_proj/worldsim_v5/renderer_intersections.py",
                PROJECT / "tests/test_run_worldsim_v5_m1_unary_diagnostic.py",
                PROJECT / "tests/test_worldsim_v5_evidence_schema.py",
                PROJECT / "tests/test_worldsim_v5_observation_aggregation.py",
                PROJECT / "tests/test_worldsim_v5_observation_builder.py",
                PROJECT / "tests/test_worldsim_v5_ownership_renderer.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m1_unary_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "phase": "structured_unary_mechanism_smoke",
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "gaussian_count": gaussian_count,
            "evidence_view_count": len(evidence_rows),
            "evaluation_view_count": len(evaluation_rows),
            "accepted_evaluation_view_count": accepted_evaluation_views,
            "abstained_evaluation_view_count": len(evaluation_rows)
            - accepted_evaluation_views,
            "duration_seconds": duration,
            "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated(device) / 1024**2),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "arm_gaussian_metrics": gaussian_metrics,
            "arm_gaussian_delta_vs_b0": gaussian_delta_vs_b0,
            "arm_evaluation_aggregate": evaluation_aggregate,
            "arm_evaluation_delta_vs_b0": evaluation_delta_vs_b0,
            "graph_inference_started": False,
            "method_inference_started": True,
            "parameter_search_performed": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        fingerprint = {
            "schema_version": "worldsim_v5_m1_unary_fingerprint_v1",
            "task_id": TASK_ID,
            "source_commit": source_head,
            "source_clean": True,
            "resolved_config": resolved_record,
            "inputs": inputs,
            "runtime": {
                "drivestudio_commit": config["runtime"]["drivestudio_commit"],
                "drivestudio_status": config["runtime"]["drivestudio_expected_status"],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "source_snapshot": snapshot,
        }
        fingerprint_path = run_dir / "fingerprint.json"
        atomic_json(fingerprint_path, fingerprint)
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m1_unary_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
            "arm_artifacts": arm_records,
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _write_terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
        )
        return summary
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        write_events(run_dir, events)
        _write_terminal(
            run_dir,
            status="blocked",
            source_head=source_head,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve(), args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
