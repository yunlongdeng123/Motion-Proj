#!/usr/bin/env python
"""只读回填 WorldSim V3 A1 的 E1/E2 诊断端点。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v3.calibration_endpoints import (
    bidirectional_boundary_distances,
    canonicalize_observed_rgb,
    coverage_status,
    cross_camera_residuals,
    inner_boundary,
    static_support_mask,
    summarize_distribution,
    validate_endpoint_contract,
)


CAMERA_NAMES = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
ROLE_NAMES = ("high-support", "boundary-support")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def mask_from_info(image_infos: dict[str, torch.Tensor], key: str, shape: tuple[int, int]) -> np.ndarray:
    value = image_infos.get(key)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    mask = as_numpy(value).squeeze()
    if mask.shape != shape:
        raise RuntimeError(f"{key} shape mismatch: {mask.shape} != {shape}")
    return mask


def affine_matrix(trainer, image_infos: dict[str, torch.Tensor]) -> np.ndarray | None:
    if "Affine" not in trainer.models:
        return None
    with torch.inference_mode():
        value = trainer.models["Affine"](image_infos)
    matrix = as_numpy(value).squeeze()
    if matrix.shape[-2:] == (4, 4):
        matrix = matrix[..., :3, :4]
    if matrix.shape[-2:] != (3, 4):
        raise RuntimeError(f"unexpected affine transform shape: {matrix.shape}")
    return matrix


def render_e1_record(dataset, trainer, position: int, device: torch.device, e1: dict[str, Any]) -> dict[str, Any]:
    raw_image_infos, raw_camera_infos = dataset.test_image_set.get_image(
        position, camera_downscale=1.0
    )
    observed = as_numpy(raw_image_infos["pixels"])
    shape = observed.shape[:2]
    image_infos = to_device(raw_image_infos, device)
    camera_infos = to_device(raw_camera_infos, device)
    with torch.inference_mode():
        output = trainer(image_infos, camera_infos)
        processed_camera = trainer.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
        )
        transform = affine_matrix(trainer, image_infos)
    depth = as_numpy(output["depth"]).squeeze()
    opacity = as_numpy(output["opacity"]).squeeze()
    dynamic_opacity = as_numpy(output["Dynamic_opacity"]).squeeze()
    support = static_support_mask(
        sky_mask=mask_from_info(raw_image_infos, "sky_masks", shape),
        dynamic_mask=mask_from_info(raw_image_infos, "dynamic_masks", shape),
        egocar_mask=mask_from_info(raw_image_infos, "egocar_masks", shape),
        rendered_opacity=opacity,
        dynamic_opacity=dynamic_opacity,
        depth=depth,
        minimum_rendered_opacity=float(e1["minimum_rendered_opacity"]),
        maximum_dynamic_opacity=float(e1["maximum_dynamic_opacity"]),
        maximum_relative_depth_edge=float(e1["maximum_relative_depth_edge"]),
        depth_edge_dilation_pixels=int(e1["depth_edge_dilation_pixels"]),
    )
    canonical_rgb = canonicalize_observed_rgb(observed, transform)
    return {
        "rgb": canonical_rgb,
        "observed_rgb": observed,
        "depth": depth,
        "support": support,
        "camera_to_world": as_numpy(processed_camera.camtoworlds),
        "intrinsics": as_numpy(processed_camera.Ks),
    }


def uint8_rgb(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value) * 255.0, 0, 255).round().astype(np.uint8)


def e1_qa_image(
    source: dict[str, Any],
    target: dict[str, Any],
    target_x: np.ndarray,
    target_y: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    left = uint8_rgb(source["observed_rgb"])
    right = uint8_rgb(target["observed_rgb"])
    overlay = right.copy()
    for x, y, residual in zip(target_x, target_y, residuals):
        if residual <= 0.05:
            color = np.array([40, 230, 80], dtype=np.uint8)
        elif residual <= 0.10:
            color = np.array([255, 220, 40], dtype=np.uint8)
        else:
            color = np.array([255, 50, 50], dtype=np.uint8)
        xi, yi = int(round(x)), int(round(y))
        overlay[max(0, yi - 1) : yi + 2, max(0, xi - 1) : xi + 2] = color
    return np.concatenate((left, right, overlay), axis=1)


def evaluate_e1(
    *,
    records: dict[int, dict[str, Any]],
    frame_positions: dict[int, dict[str, int]],
    e1: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows_path = output_dir / "e1_per_direction.jsonl"
    buckets: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"values": [], "valid": 0, "candidate": 0})
    )
    qa_count = 0
    for frame, cameras in sorted(frame_positions.items()):
        for first, second in e1["camera_pairs"]:
            if first not in cameras or second not in cameras:
                continue
            pair_name = f"{first}<->{second}"
            for source_name, target_name in ((first, second), (second, first)):
                source = records[cameras[source_name]]
                target = records[cameras[target_name]]
                result = cross_camera_residuals(
                    source_rgb=source["rgb"],
                    source_depth=source["depth"],
                    source_support=source["support"],
                    source_camera_to_world=source["camera_to_world"],
                    source_intrinsics=source["intrinsics"],
                    target_rgb=target["rgb"],
                    target_depth=target["depth"],
                    target_support=target["support"],
                    target_camera_to_world=target["camera_to_world"],
                    target_intrinsics=target["intrinsics"],
                    grid_stride_pixels=int(e1["grid_stride_pixels"]),
                    maximum_relative_occlusion_error=float(
                        e1["maximum_relative_occlusion_error"]
                    ),
                )
                residuals = result["residuals"]
                valid_depths = result["depths"]
                candidate_depths = result["candidate_depths"]
                row = {
                    "frame": frame,
                    "pair": pair_name,
                    "source_camera": source_name,
                    "target_camera": target_name,
                    "candidate_supports": int(result["candidate_count"]),
                    "valid_supports": int(residuals.size),
                    "coverage": (
                        float(residuals.size / result["candidate_count"])
                        if result["candidate_count"]
                        else 0.0
                    ),
                    "residual": summarize_distribution(residuals),
                }
                append_jsonl(rows_path, row)
                for stratum, valid_mask, candidate_mask in (
                    (
                        "static_background",
                        np.ones(valid_depths.shape, dtype=bool),
                        np.ones(candidate_depths.shape, dtype=bool),
                    ),
                    (
                        "near",
                        valid_depths < float(e1["near_far_split_meters"]),
                        candidate_depths < float(e1["near_far_split_meters"]),
                    ),
                    (
                        "far",
                        valid_depths >= float(e1["near_far_split_meters"]),
                        candidate_depths >= float(e1["near_far_split_meters"]),
                    ),
                ):
                    bucket = buckets[pair_name][stratum]
                    bucket["values"].extend(residuals[valid_mask].tolist())
                    bucket["valid"] += int(valid_mask.sum())
                    bucket["candidate"] += int(candidate_mask.sum())
                if residuals.size and qa_count < 6:
                    imageio.imwrite(
                        output_dir
                        / "qa_e1"
                        / f"frame_{frame:03d}__{source_name}_to_{target_name}.jpg",
                        e1_qa_image(
                            source,
                            target,
                            result["target_x"],
                            result["target_y"],
                            residuals,
                        ),
                        quality=90,
                    )
                    qa_count += 1

    pairs: dict[str, Any] = {}
    total_values: list[float] = []
    total_valid = 0
    total_candidate = 0
    for pair_name, strata in buckets.items():
        pair_summary: dict[str, Any] = {}
        for stratum, bucket in strata.items():
            status = coverage_status(
                valid_count=bucket["valid"],
                candidate_count=bucket["candidate"],
                minimum_valid_count=int(e1["minimum_valid_supports"]),
                minimum_coverage=float(e1["minimum_coverage"]),
                zero_reason="ZERO_CROSS_CAMERA_SUPPORT",
            )
            pair_summary[stratum] = {
                **status,
                "residual": summarize_distribution(bucket["values"]),
            }
        pairs[pair_name] = pair_summary
        main = strata["static_background"]
        total_values.extend(main["values"])
        total_valid += main["valid"]
        total_candidate += main["candidate"]
    overall = coverage_status(
        valid_count=total_valid,
        candidate_count=total_candidate,
        minimum_valid_count=int(e1["minimum_valid_supports"]),
        minimum_coverage=float(e1["minimum_coverage"]),
        zero_reason="ZERO_CROSS_CAMERA_SUPPORT",
    )
    return {
        **overall,
        "residual": summarize_distribution(total_values),
        "pairs": pairs,
        "per_direction_metrics": str(rows_path),
        "qa_directory": str(output_dir / "qa_e1"),
    }


def prepare_trainer_frame(trainer, image_infos, camera_infos):
    normed_time = image_infos["normed_time"].flatten()[0]
    trainer.cur_frame = torch.argmin(
        torch.abs(trainer.normalized_timestamps - normed_time)
    )
    for model in trainer.models.values():
        if hasattr(model, "in_test_set"):
            model.in_test_set = trainer.in_test_set
    for class_name in trainer.gaussian_classes:
        model = trainer.models[class_name]
        if hasattr(model, "set_cur_frame"):
            model.set_cur_frame(trainer.cur_frame)
    camera = trainer.process_camera(
        camera_infos=camera_infos,
        image_ids=image_infos["img_idx"].flatten()[0],
    )
    gaussians = trainer.collect_gaussians(
        cam=camera, image_ids=image_infos["img_idx"].flatten()[0]
    )
    return camera, gaussians


def render_actor_support(
    dataset,
    trainer,
    position: int,
    model_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    raw_image_infos, raw_camera_infos = dataset.test_image_set.get_image(
        position, camera_downscale=1.0
    )
    observed = as_numpy(raw_image_infos["pixels"])
    image_infos = to_device(raw_image_infos, device)
    camera_infos = to_device(raw_camera_infos, device)
    with torch.inference_mode():
        camera, gaussians = prepare_trainer_frame(
            trainer, image_infos, camera_infos
        )
        _, render_fn = trainer.render_gaussians(
            gs=gaussians,
            cam=camera,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
        )
        rigid_label = trainer.gaussian_classes["RigidNodes"]
        rigid_slots = torch.nonzero(
            trainer.pts_labels == rigid_label, as_tuple=False
        ).flatten()
        rigid = trainer.models["RigidNodes"]
        actor_local = rigid.point_ids[:, 0] == model_index
        if rigid_slots.numel() != actor_local.numel():
            raise RuntimeError(
                "RigidNodes point_ids do not align with collected Gaussian slots"
            )
        full_mask = torch.zeros_like(trainer.pts_labels, dtype=torch.bool)
        full_mask[rigid_slots] = actor_local
        _, _, actor_opacity = render_fn(full_mask)
    return observed, as_numpy(actor_opacity).squeeze()


def e2_qa_image(
    observed: np.ndarray,
    effect: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    original = uint8_rgb(observed)
    overlay = original.copy()
    effect_boundary = inner_boundary(effect)
    support_boundary = inner_boundary(support)
    overlay[effect] = (
        overlay[effect].astype(np.float32) * 0.55
        + np.array([255, 45, 45], dtype=np.float32) * 0.45
    ).round().astype(np.uint8)
    overlay[support] = (
        overlay[support].astype(np.float32) * 0.60
        + np.array([40, 120, 255], dtype=np.float32) * 0.40
    ).round().astype(np.uint8)
    overlay[effect_boundary] = np.array([255, 230, 0], dtype=np.uint8)
    overlay[support_boundary] = np.array([0, 255, 255], dtype=np.uint8)
    masks = np.zeros_like(original)
    masks[effect] = np.array([255, 45, 45], dtype=np.uint8)
    masks[support] = np.maximum(masks[support], np.array([40, 120, 255], dtype=np.uint8))
    return np.concatenate((original, masks, overlay), axis=1)


def evaluate_e2(
    *,
    dataset,
    trainer,
    source: dict[str, Any],
    test_indices: list[int],
    positions_limit: set[int] | None,
    e2: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    actor_metrics = source.get("actor_metrics", {})
    frozen_mask_contract = actor_metrics.get("mask_contract", {})
    if int(frozen_mask_contract.get("effect_threshold_uint8", -1)) != int(
        e2["effect_threshold_uint8"]
    ) or int(frozen_mask_contract.get("effect_dilation_radius_pixels", -1)) != int(
        e2["effect_dilation_radius_pixels"]
    ):
        raise RuntimeError("E2 contract does not match the frozen A0 effect masks")
    mask_directory = Path(actor_metrics["qa_directory"]).parent / "masks"
    if not mask_directory.is_dir():
        raise FileNotFoundError(mask_directory)

    num_cameras = int(dataset.pixel_source.num_cams)
    rigid = trainer.models["RigidNodes"]
    selected = source.get("selected_actors", {})
    rows_path = output_dir / "e2_per_image.jsonl"
    role_results: dict[str, Any] = {}
    for role in ROLE_NAMES:
        actor = selected.get(role)
        if actor is None or actor.get("availability") != "available":
            role_results[role] = {
                "status": "ABSTAIN",
                "reason": "ACTOR_UNAVAILABLE_IN_SOURCE_CONTRACT",
                "actor": actor,
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
        positions = [
            position
            for position, full_index in enumerate(test_indices)
            if full_index // num_cameras in valid_frames
            and (positions_limit is None or position in positions_limit)
        ]
        values: list[float] = []
        direction_values = {"support_to_effect": [], "effect_to_support": []}
        valid_images = 0
        camera_buckets = {
            camera: {"values": [], "valid": 0, "candidate": 0}
            for camera in CAMERA_NAMES
        }
        qa_count = 0
        for ordinal, position in enumerate(positions, start=1):
            full_index = test_indices[position]
            frame, camera_index = divmod(full_index, num_cameras)
            camera_name = CAMERA_NAMES[camera_index]
            camera_buckets[camera_name]["candidate"] += 1
            stem = f"frame_{frame:03d}_camera_{camera_index}"
            effect_path = mask_directory / f"{role}__{stem}.png"
            observed, actor_opacity = render_actor_support(
                dataset, trainer, position, model_index, device
            )
            support = actor_opacity >= float(e2["minimum_actor_support_opacity"])
            row: dict[str, Any] = {
                "role": role,
                "frame": frame,
                "camera": camera_index,
                "camera_name": camera_name,
                "test_position": position,
                "full_image_index": full_index,
                "actor_support_pixels": int(support.sum()),
                "effect_mask": str(effect_path),
            }
            if not effect_path.is_file():
                row.update(
                    status="ABSTAIN",
                    reason="ZERO_COUNTERFACTUAL_EFFECT_BOUNDARY",
                    effect_pixels=0,
                )
            else:
                effect = imageio.imread(effect_path) > 0
                distances = bidirectional_boundary_distances(support, effect)
                if not inner_boundary(support).any():
                    row.update(
                        status="ABSTAIN",
                        reason="ZERO_PROJECTED_ACTOR_SUPPORT_BOUNDARY",
                        effect_pixels=int(effect.sum()),
                    )
                elif not inner_boundary(effect).any():
                    row.update(
                        status="ABSTAIN",
                        reason="ZERO_COUNTERFACTUAL_EFFECT_BOUNDARY",
                        effect_pixels=int(effect.sum()),
                    )
                else:
                    row.update(
                        status="done",
                        reason=None,
                        effect_pixels=int(effect.sum()),
                        normalized_bidirectional_distance=summarize_distribution(
                            distances["combined"]
                        ),
                    )
                    valid_images += 1
                    camera_buckets[camera_name]["valid"] += 1
                    values.extend(distances["combined"].tolist())
                    camera_buckets[camera_name]["values"].extend(
                        distances["combined"].tolist()
                    )
                    for direction in direction_values:
                        direction_values[direction].extend(
                            distances[direction].tolist()
                        )
                    if qa_count < int(e2["qa_images_per_role"]):
                        imageio.imwrite(
                            output_dir / "qa_e2" / f"{role}__{stem}.jpg",
                            e2_qa_image(observed, effect, support),
                            quality=90,
                        )
                        qa_count += 1
            append_jsonl(rows_path, row)
            print(
                f"E2 {role} {ordinal}/{len(positions)} position={position} "
                f"status={row['status']}",
                flush=True,
            )
        status = coverage_status(
            valid_count=valid_images,
            candidate_count=len(positions),
            minimum_valid_count=int(e2["minimum_valid_images"]),
            minimum_coverage=float(e2["minimum_coverage"]),
            zero_reason="ZERO_COUNTERFACTUAL_EFFECT_BOUNDARY",
        )
        by_camera: dict[str, Any] = {}
        for camera, bucket in camera_buckets.items():
            camera_status = coverage_status(
                valid_count=bucket["valid"],
                candidate_count=bucket["candidate"],
                minimum_valid_count=int(e2["minimum_valid_images"]),
                minimum_coverage=float(e2["minimum_coverage"]),
                zero_reason="ZERO_COUNTERFACTUAL_EFFECT_BOUNDARY",
            )
            by_camera[camera] = {
                **camera_status,
                "normalized_bidirectional_distance": summarize_distribution(
                    bucket["values"]
                ),
            }
        role_results[role] = {
            **status,
            "actor": actor,
            "valid_processed_frame_count": len(valid_frames),
            "normalized_bidirectional_distance": summarize_distribution(values),
            "support_to_effect": summarize_distribution(
                direction_values["support_to_effect"]
            ),
            "effect_to_support": summarize_distribution(
                direction_values["effect_to_support"]
            ),
            "by_camera": by_camera,
        }
    return {
        "status": "done",
        "roles": role_results,
        "per_image_metrics": str(rows_path),
        "qa_directory": str(output_dir / "qa_e2"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--endpoint-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drivestudio-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-heldout-frames",
        type=int,
        help="仅用于工程 smoke；正式回填必须省略。",
    )
    args = parser.parse_args()
    if args.max_heldout_frames is not None and args.max_heldout_frames <= 0:
        raise ValueError("max-heldout-frames must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for directory in (args.output_dir, args.output_dir / "qa_e1", args.output_dir / "qa_e2"):
        directory.mkdir(parents=True, exist_ok=True)

    endpoint_contract = yaml.safe_load(args.endpoint_config.read_text(encoding="utf-8"))
    validate_endpoint_contract(endpoint_contract)
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if source.get("status") != "done":
        raise RuntimeError("source A1 summary is not terminal done")
    checkpoint = Path(source["checkpoint"]["checkpoint"])
    expected_checkpoint_sha = source["checkpoint"]["sha256"]
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != expected_checkpoint_sha:
        raise RuntimeError("source checkpoint SHA-256 does not match its summary")

    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    config_path = checkpoint.parent / "config.yaml"
    config = OmegaConf.load(config_path)
    if list(config.data.pixel_source.cameras) != [0, 1, 2]:
        raise RuntimeError("A1-E0 requires the three frozen front cameras")
    device = torch.device(args.device)
    dataset = DrivingDataset(data_cfg=config.data)
    test_indices = [int(value) for value in dataset.test_image_set.split_indices]
    if int(dataset.pixel_source.num_cams) != len(CAMERA_NAMES):
        raise RuntimeError("unexpected camera count")
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

    all_frames = sorted({full_index // len(CAMERA_NAMES) for full_index in test_indices})
    if args.max_heldout_frames is not None:
        all_frames = all_frames[: args.max_heldout_frames]
    selected_frames = set(all_frames)
    positions = [
        position
        for position, full_index in enumerate(test_indices)
        if full_index // len(CAMERA_NAMES) in selected_frames
    ]
    frame_positions: dict[int, dict[str, int]] = defaultdict(dict)
    records: dict[int, dict[str, Any]] = {}
    for ordinal, position in enumerate(positions, start=1):
        full_index = test_indices[position]
        frame, camera_index = divmod(full_index, len(CAMERA_NAMES))
        frame_positions[frame][CAMERA_NAMES[camera_index]] = position
        records[position] = render_e1_record(
            dataset, trainer, position, device, endpoint_contract["e1"]
        )
        print(f"E1 render {ordinal}/{len(positions)} position={position}", flush=True)

    e1_result = evaluate_e1(
        records=records,
        frame_positions=frame_positions,
        e1=endpoint_contract["e1"],
        output_dir=args.output_dir,
    )
    del records
    torch.cuda.empty_cache()
    e2_result = evaluate_e2(
        dataset=dataset,
        trainer=trainer,
        source=source,
        test_indices=test_indices,
        positions_limit=set(positions) if args.max_heldout_frames is not None else None,
        e2=endpoint_contract["e2"],
        output_dir=args.output_dir,
        device=device,
    )
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint changed during read-only A1-E0 evaluation")
    result = {
        "status": "done",
        "task_id": endpoint_contract["task_id"],
        "component": "A1-E0 cross-camera and actor-boundary endpoints",
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "variant": source.get("variant"),
        "source_summary": str(args.source_summary),
        "source_summary_sha256": sha256_file(args.source_summary),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "endpoint_config": str(args.endpoint_config),
        "endpoint_config_sha256": sha256_file(args.endpoint_config),
        "endpoint_version": endpoint_contract["endpoint_version"],
        "heldout_split": {
            "test_image_stride": int(config.data.pixel_source.test_image_stride),
            "test_full_image_count": len(test_indices),
            "evaluated_frames": all_frames,
            "evaluated_image_count": len(positions),
            "max_heldout_frames": args.max_heldout_frames,
            "formal_full_split": args.max_heldout_frames is None,
        },
        "e1": e1_result,
        "e2": e2_result,
    }
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
