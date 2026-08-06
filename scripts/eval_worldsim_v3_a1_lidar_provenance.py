#!/usr/bin/env python
"""复现并审计 WorldSim V3 A1 的最小 LiDAR 初始化 provenance。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from motion_proj.worldsim_v3.lidar_provenance import (
    actor_input_mapping,
    compare_initialization_provenance,
    raw_lidar_block_contract,
    sha256_file,
    sparse_depth_residuals,
    summarize_depth_residual,
    validate_lidar_provenance_contract,
)


CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


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


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def depth_qa_image(
    observed: np.ndarray,
    valid_mask: np.ndarray,
    relative_residual: np.ndarray,
) -> np.ndarray:
    rgb = np.clip(observed * 255.0, 0, 255).round().astype(np.uint8)
    overlay = rgb.copy()
    yy, xx = np.nonzero(valid_mask)
    values = np.asarray(relative_residual)
    if values.size != yy.size:
        raise ValueError("QA residual count does not match valid sparse depth pixels")
    for y, x, residual in zip(yy, xx, values):
        if residual <= 0.10:
            color = np.array([40, 230, 80], dtype=np.uint8)
        elif residual <= 0.25:
            color = np.array([255, 220, 40], dtype=np.uint8)
        else:
            color = np.array([255, 50, 50], dtype=np.uint8)
        overlay[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = color
    return np.concatenate((rgb, overlay), axis=1)


def build_raw_input_contract(
    scene_root: Path,
    num_frames: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    blocks = []
    for frame in range(num_frames):
        blocks.append(
            raw_lidar_block_contract(
                scene_root / "lidar" / f"{frame:03d}.bin",
                scene_root / "lidar_pose" / f"{frame:03d}.txt",
                frame=frame,
                bytes_per_point=int(contract["raw_inputs"]["bytes_per_point"]),
            )
        )
    annotations = []
    for relative in contract["raw_inputs"]["actor_annotations"]:
        path = scene_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        annotations.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "scene_root": str(scene_root),
        "frame_block_count": len(blocks),
        "total_raw_point_count": sum(row["raw_point_count"] for row in blocks),
        "total_scan_bytes": sum(row["scan_bytes"] for row in blocks),
        "blocks": blocks,
        "actor_annotations": annotations,
    }


def evaluate_initial_depth(
    *,
    dataset,
    trainer,
    device: torch.device,
    depth_contract: dict[str, Any],
    output_dir: Path,
    max_heldout_images: int | None,
) -> dict[str, Any]:
    test_indices = [int(value) for value in dataset.test_image_set.split_indices]
    positions = list(range(len(test_indices)))
    if max_heldout_images is not None:
        positions = positions[:max_heldout_images]
    rows_path = output_dir / "initial_depth_per_image.jsonl"
    overall = {"absolute": [], "relative": [], "valid": 0, "candidate": 0}
    by_camera = {
        name: {"absolute": [], "relative": [], "valid": 0, "candidate": 0}
        for name in CAMERA_NAMES
    }
    qa_count = 0
    for ordinal, position in enumerate(positions, start=1):
        raw_image_infos, raw_camera_infos = dataset.test_image_set.get_image(
            position, camera_downscale=1.0
        )
        if "lidar_depth_map" not in raw_image_infos:
            raise RuntimeError("dataset did not expose sparse LiDAR depth maps")
        observed = as_numpy(raw_image_infos["pixels"])
        lidar_depth = as_numpy(raw_image_infos["lidar_depth_map"])
        image_infos = to_device(raw_image_infos, device)
        camera_infos = to_device(raw_camera_infos, device)
        with torch.inference_mode():
            output = trainer(image_infos, camera_infos)
        residual = sparse_depth_residuals(
            rendered_depth=as_numpy(output["depth"]),
            rendered_opacity=as_numpy(output["opacity"]),
            lidar_depth=lidar_depth,
            lidar_valid_minimum_meters=float(
                depth_contract["lidar_valid_minimum_meters"]
            ),
            rendered_valid_minimum_meters=float(
                depth_contract["rendered_valid_minimum_meters"]
            ),
            minimum_rendered_opacity=float(
                depth_contract["minimum_rendered_opacity"]
            ),
        )
        full_index = test_indices[position]
        frame, camera_index = divmod(full_index, len(CAMERA_NAMES))
        camera_name = CAMERA_NAMES[camera_index]
        row = {
            "test_position": position,
            "full_image_index": full_index,
            "frame": frame,
            "camera": camera_index,
            "camera_name": camera_name,
            "candidate_count": residual["candidate_count"],
            "valid_count": residual["valid_count"],
            "coverage": (
                residual["valid_count"] / residual["candidate_count"]
                if residual["candidate_count"]
                else 0.0
            ),
            "absolute_residual_meters": summarize_depth_residual(
                absolute_values=residual["absolute_residual_meters"],
                relative_values=residual["relative_residual"],
                valid_count=residual["valid_count"],
                candidate_count=residual["candidate_count"],
                minimum_valid_points=1,
                minimum_coverage=1e-12,
            )["absolute_residual_meters"],
            "relative_residual": summarize_depth_residual(
                absolute_values=residual["absolute_residual_meters"],
                relative_values=residual["relative_residual"],
                valid_count=residual["valid_count"],
                candidate_count=residual["candidate_count"],
                minimum_valid_points=1,
                minimum_coverage=1e-12,
            )["relative_residual"],
        }
        append_jsonl(rows_path, row)
        for bucket in (overall, by_camera[camera_name]):
            bucket["absolute"].extend(
                residual["absolute_residual_meters"].tolist()
            )
            bucket["relative"].extend(residual["relative_residual"].tolist())
            bucket["valid"] += residual["valid_count"]
            bucket["candidate"] += residual["candidate_count"]
        if residual["valid_count"] and qa_count < int(depth_contract["qa_images"]):
            imageio.imwrite(
                output_dir
                / "qa"
                / f"frame_{frame:03d}_camera_{camera_index}_{camera_name}.jpg",
                depth_qa_image(
                    observed,
                    residual["valid_mask"],
                    residual["relative_residual"],
                ),
                quality=90,
            )
            qa_count += 1
        print(
            f"initial depth {ordinal}/{len(positions)} position={position} "
            f"valid={residual['valid_count']}/{residual['candidate_count']}",
            flush=True,
        )

    def summarize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        return summarize_depth_residual(
            absolute_values=bucket["absolute"],
            relative_values=bucket["relative"],
            valid_count=bucket["valid"],
            candidate_count=bucket["candidate"],
            minimum_valid_points=int(depth_contract["minimum_valid_points"]),
            minimum_coverage=float(depth_contract["minimum_coverage"]),
        )

    return {
        **summarize_bucket(overall),
        "by_camera": {
            name: summarize_bucket(bucket) for name, bucket in by_camera.items()
        },
        "test_full_image_count": len(test_indices),
        "evaluated_image_count": len(positions),
        "formal_full_split": max_heldout_images is None,
        "max_heldout_images": max_heldout_images,
        "per_image_metrics": str(rows_path),
        "qa_directory": str(output_dir / "qa"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drivestudio-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-heldout-images",
        type=int,
        help="仅用于工程 smoke；正式审计必须省略。",
    )
    args = parser.parse_args()
    if args.max_heldout_images is not None and args.max_heldout_images <= 0:
        raise ValueError("max-heldout-images must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    (args.output_dir / "qa").mkdir(parents=True)

    contract = yaml.safe_load(args.audit_config.read_text(encoding="utf-8"))
    validate_lidar_provenance_contract(contract)
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if source.get("status") != "done":
        raise RuntimeError("source A1 run is not terminal done")
    checkpoint = Path(source["checkpoint"]["checkpoint"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != source["checkpoint"]["sha256"]:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    source_provenance_path = Path(source["initialization_provenance"]["path"])
    expected_provenance_sha = source["initialization_provenance"]["sha256"]
    if sha256_file(source_provenance_path) != expected_provenance_sha:
        raise RuntimeError("source initialization provenance SHA-256 mismatch")
    source_provenance = json.loads(
        source_provenance_path.read_text(encoding="utf-8")
    )

    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    config_path = checkpoint.parent / "config.yaml"
    config = OmegaConf.load(config_path)
    if list(config.data.pixel_source.cameras) != [0, 1, 2]:
        raise RuntimeError("A1 LiDAR audit requires camera IDs [0,1,2]")
    initialization = config.model.Background.init
    runtime = contract["runtime_initialization"]
    if int(initialization.from_lidar.num_samples) != int(
        runtime["background_lidar_sample_points"]
    ) or int(initialization.near_randoms) != int(
        runtime["random_near_config_points"]
    ) or int(initialization.far_randoms) != int(
        runtime["random_far_config_points"]
    ):
        raise RuntimeError("source initialization config differs from frozen audit")

    device = torch.device(args.device)
    dataset = DrivingDataset(data_cfg=config.data)
    scene_root = Path(config.data.data_root) / str(int(config.data.scene_idx))
    raw_inputs = build_raw_input_contract(
        scene_root, dataset.num_img_timesteps, contract
    )
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
    reproduced_path = args.output_dir / "reproduced_init_provenance.json"
    old_destination = os.environ.get("WORLDSIM_V3_INIT_PROVENANCE")
    old_seed = os.environ.get("WORLDSIM_V3_INIT_SEED")
    import models.trainers.scene_graph as scene_graph_module

    original_uniform_sample_sphere = scene_graph_module.uniform_sample_sphere
    original_check_visibility = dataset.check_pts_visibility
    random_capture: dict[str, list[dict[str, Any]]] = {
        "uniform_sample_sphere_calls": [],
        "visibility_filter_calls": [],
    }

    def capture_uniform_sample_sphere(*call_args, **call_kwargs):
        value = original_uniform_sample_sphere(*call_args, **call_kwargs)
        inverse = call_kwargs.get("inverse", False)
        if len(call_args) >= 3:
            inverse = call_args[2]
        random_capture["uniform_sample_sphere_calls"].append(
            {
                "point_count": int(value.shape[0]),
                "inverse_radius_sampling": bool(inverse),
                "unit_sphere_points_sha256": tensor_sha256(value),
            }
        )
        return value

    def capture_visibility(points):
        mask = original_check_visibility(points)
        random_capture["visibility_filter_calls"].append(
            {
                "candidate_point_count": int(points.shape[0]),
                "scaled_world_points_sha256": tensor_sha256(points),
                "visibility_mask_sha256": tensor_sha256(mask),
                "visible_point_count": int(mask.sum().item()),
            }
        )
        return mask

    os.environ["WORLDSIM_V3_INIT_PROVENANCE"] = str(reproduced_path)
    os.environ["WORLDSIM_V3_INIT_SEED"] = str(int(runtime["seed"]))
    scene_graph_module.uniform_sample_sphere = capture_uniform_sample_sphere
    dataset.check_pts_visibility = capture_visibility
    try:
        trainer.init_gaussians_from_dataset(dataset=dataset)
    finally:
        scene_graph_module.uniform_sample_sphere = original_uniform_sample_sphere
        dataset.check_pts_visibility = original_check_visibility
        if old_destination is None:
            os.environ.pop("WORLDSIM_V3_INIT_PROVENANCE", None)
        else:
            os.environ["WORLDSIM_V3_INIT_PROVENANCE"] = old_destination
        if old_seed is None:
            os.environ.pop("WORLDSIM_V3_INIT_SEED", None)
        else:
            os.environ["WORLDSIM_V3_INIT_SEED"] = old_seed
    reproduced_sha = sha256_file(reproduced_path)
    reproduced = json.loads(reproduced_path.read_text(encoding="utf-8"))
    provenance_comparison = compare_initialization_provenance(
        source_provenance, reproduced
    )
    if not provenance_comparison["recorded_lidar_actor_inputs_exact"]:
        raise RuntimeError("recorded LiDAR/actor initialization tensors do not match")
    if not provenance_comparison["rigid_initial_gaussian_count_exact"]:
        raise RuntimeError("initial rigid Gaussian count does not match")
    if len(random_capture["uniform_sample_sphere_calls"]) != 2 or len(
        random_capture["visibility_filter_calls"]
    ) != 1:
        raise RuntimeError(f"unexpected random initialization calls: {random_capture}")
    registry_path = Path(source["registry"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    actors = actor_input_mapping(
        reproduced["instance_lidar_samples"],
        registry["actors"],
        source["selected_actors"],
    )
    if sum(row["initial_gaussian_count"] for row in actors) != int(
        reproduced["initialized_gaussians"]["RigidNodes"]
    ):
        raise RuntimeError("actor input blocks do not sum to initial RigidNodes count")

    trainer.set_eval()
    depth_result = evaluate_initial_depth(
        dataset=dataset,
        trainer=trainer,
        device=device,
        depth_contract=contract["initial_depth_residual"],
        output_dir=args.output_dir,
        max_heldout_images=args.max_heldout_images,
    )
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint changed during read-only LiDAR audit")

    random_sources = {
        "seed": int(runtime["seed"]),
        "rng_reset": reproduced["rng_reset"],
        "near_randoms_config_points": int(initialization.near_randoms),
        "far_randoms_config_points": int(initialization.far_randoms),
        "candidate_multiplier": int(
            runtime["random_candidate_multiplier_in_drivestudio"]
        ),
        "scene_graph_source": str(
            args.drivestudio_root / "models/trainers/scene_graph.py"
        ),
        "scene_graph_source_sha256": sha256_file(
            args.drivestudio_root / "models/trainers/scene_graph.py"
        ),
        "geometry_source": str(args.drivestudio_root / "utils/geometry.py"),
        "geometry_source_sha256": sha256_file(
            args.drivestudio_root / "utils/geometry.py"
        ),
        "runtime_capture": random_capture,
    }
    depth_result["truth_tier"] = (
        "exact_source_initialization"
        if reproduced_sha == expected_provenance_sha
        else "seed0_reconstructed_initialization_witness_not_exact_source_initialization"
    )
    result = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A1 minimal LiDAR initialization provenance",
        "audit_version": contract["audit_version"],
        "scene_name": source["scene_name"],
        "scene_index": source["scene_index"],
        "source_summary": str(args.source_summary),
        "source_summary_sha256": sha256_file(args.source_summary),
        "source_checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "audit_config": str(args.audit_config),
        "audit_config_sha256": sha256_file(args.audit_config),
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "source_initialization_provenance": str(source_provenance_path),
        "source_initialization_provenance_sha256": expected_provenance_sha,
        "reproduced_initialization_provenance": str(reproduced_path),
        "reproduced_initialization_provenance_sha256": reproduced_sha,
        "reproduction_exact_sha_match": reproduced_sha == expected_provenance_sha,
        "provenance_comparison": provenance_comparison,
        "raw_inputs": raw_inputs,
        "runtime_sampled_inputs": {
            "background": reproduced["background_lidar_sample"],
            "actors": actors,
            "actor_count": len(actors),
            "actor_input_point_count": sum(
                row["input_point_count"] for row in actors
            ),
        },
        "random_seed_inputs": random_sources,
        "initialized_gaussians": reproduced["initialized_gaussians"],
        "initial_depth_residual": depth_result,
        "scope_boundary": contract["scope_boundary"],
        "limitations": reproduced["limitations"],
    }
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
