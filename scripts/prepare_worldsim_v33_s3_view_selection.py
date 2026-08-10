#!/usr/bin/env python3
"""用 train-only SAM/D2 证据自动选择 Asset Harvester 1/2/4-view 输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.dynamic_editing_v2.pilot_metrics import counterfactual_effect_mask
from motion_proj.worldsim_v33.view_selection import rank_view_candidates, select_view_sets
from scripts.eval_worldsim_v3_a3_r1_heldout import release_trainer_render_info
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.materialize_worldsim_v3_a3_s_b_sidecar import render_variant
from scripts.prepare_worldsim_v32_s3_inputs import (
    array_sha256,
    dilate_binary,
    square_crop,
)


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_file(path: str | Path, expected: str, role: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} 不存在: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(f"{role} SHA 漂移: {actual} != {expected}")
    return source


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"4x4 transform 非法: {path}")
    return matrix


def parse_annotation_matrix(rows: list[Any]) -> np.ndarray:
    parsed = []
    for row in rows:
        values = row.split() if isinstance(row, str) else row
        parsed.append([float(value) for value in values])
    matrix = np.asarray(parsed, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("obj_to_world annotation 非法")
    return matrix


def actor_annotation(
    instances: Mapping[str, Any], dataset_instance_id: int, frame: int
) -> tuple[np.ndarray, list[float]]:
    actor = instances[str(int(dataset_instance_id))]["frame_annotations"]
    frames = [int(value) for value in actor["frame_idx"]]
    if int(frame) not in frames:
        raise RuntimeError(f"actor annotation 缺失: id={dataset_instance_id} frame={frame}")
    index = frames.index(int(frame))
    return (
        parse_annotation_matrix(actor["obj_to_world"][index]),
        [float(value) for value in actor["box_size"][index]],
    )


def actor_view_yaw(
    scene_root: Path,
    instances: Mapping[str, Any],
    dataset_instance_id: int,
    frame: int,
    camera_id: int,
) -> float:
    obj_to_world, _ = actor_annotation(instances, dataset_instance_id, frame)
    camera_to_world = load_matrix(
        scene_root / "extrinsics" / f"{int(frame):03d}_{int(camera_id)}.txt"
    )
    relative = obj_to_world[:3, :3].T @ (
        camera_to_world[:3, 3] - obj_to_world[:3, 3]
    )
    if float(np.linalg.norm(relative[:2])) <= 1e-9:
        raise RuntimeError("actor-camera 方位角退化")
    return float(math.atan2(relative[1], relative[0]))


def mask_confidence(row: Mapping[str, Any]) -> float:
    metrics = row["quality_metrics"]
    prompt_iou = float(metrics.get("prompt_bbox_iou") or 0.0)
    temporal_iou = metrics.get("temporal_iou")
    temporal = 1.0 if temporal_iou is None else float(temporal_iou)
    centroid = float(metrics.get("centroid_to_prompt_diagonal") or 1.0)
    area_ratio = max(float(metrics.get("mask_to_prompt_area_ratio") or 0.0), 1e-6)
    area_consistency = math.exp(-abs(math.log(area_ratio)))
    return float(
        np.clip(
            0.45 * prompt_iou
            + 0.25 * temporal
            + 0.15 * (1.0 - min(1.0, centroid))
            + 0.15 * area_consistency,
            0.0,
            1.0,
        )
    )


def projected_box_mask(shape: tuple[int, int], box: list[float]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = (float(value) for value in box)
    left, top = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    right, bottom = min(width, int(math.ceil(x1))), min(height, int(math.ceil(y1)))
    output = np.zeros(shape, dtype=bool)
    if right > left and bottom > top:
        output[top:bottom, left:right] = True
    return output


def truncation_score(mask: np.ndarray, box: list[float]) -> float:
    height, width = mask.shape
    x0, y0, x1, y1 = (float(value) for value in box)
    box_edges = sum(
        (
            x0 <= 0.5,
            y0 <= 0.5,
            x1 >= width - 0.5,
            y1 >= height - 0.5,
        )
    ) / 4.0
    mask_edges = sum(
        (mask[:, 0].any(), mask[0].any(), mask[:, -1].any(), mask[-1].any())
    ) / 4.0
    return float(max(box_edges, mask_edges))


def laplacian_variance(image: np.ndarray, support: np.ndarray) -> float:
    gray = (
        0.299 * image[..., 0].astype(np.float64)
        + 0.587 * image[..., 1].astype(np.float64)
        + 0.114 * image[..., 2].astype(np.float64)
    )
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    selected = laplacian[support[1:-1, 1:-1]]
    return float(np.var(selected)) if selected.size else 0.0


def load_small_rgb(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize(
            (shape[1], shape[0]), Image.Resampling.LANCZOS
        )
    return np.asarray(image)


def load_small_binary(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("L").resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        )
    return np.asarray(image) > 0


def nearby_dynamic_fraction(
    dynamic_mask: np.ndarray,
    target_mask: np.ndarray,
    box_mask: np.ndarray,
    dilation: int,
) -> float:
    protected = dilate_binary(target_mask, int(dilation))
    other_dynamic = dynamic_mask & ~protected
    denominator = max(1, int(box_mask.sum()))
    return float(np.clip((other_dynamic & box_mask).sum() / denominator, 0.0, 1.0))


def candidate_rows(
    manifest: Mapping[str, Any],
    *,
    role: str,
    reserved_frames: set[int],
    heldout_frames: set[int],
    minimum_sam_pixels: int,
    minimum_projected_box_pixels: int,
) -> list[dict[str, Any]]:
    output = []
    for source in manifest["masks"]:
        frame = int(source["frame"])
        if (
            source["role"] != role
            or not bool(source["accepted"])
            or frame in reserved_frames
            or frame in heldout_frames
            or int(source["positive_pixels"]) < int(minimum_sam_pixels)
            or source.get("projected_box_xyxy") is None
        ):
            continue
        x0, y0, x1, y1 = (float(value) for value in source["projected_box_xyxy"])
        if max(0.0, x1 - x0) * max(0.0, y1 - y0) < float(
            minimum_projected_box_pixels
        ):
            continue
        output.append(dict(source))
    output.sort(key=lambda row: (int(row["frame"]), int(row["camera_id"])))
    return output


def snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, Any]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    sources = {
        "config": config_path,
        "selector": Path(__file__).resolve(),
        "module": PROJECT / "motion_proj" / "worldsim_v33" / "view_selection.py",
        "test": PROJECT / "tests" / "test_worldsim_v33_view_selection.py",
    }
    report = {}
    for role, source in sources.items():
        target = snapshot / source.name
        shutil.copy2(source, target)
        report[role] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", choices=("high_support", "boundary_support"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-selection-sha256")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"S3 run-dir 非空，拒绝覆盖: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir()
    started = time.time()
    atomic_json(args.run_dir / "status.json", {"state": "running", "started_unix": started})

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s3_viewselect_v1":
        raise ValueError("S3 viewselect config schema 漂移")
    inputs = config["inputs"]
    verified = {
        name: str(verify_file(inputs[name], inputs[f"{name}_sha256"], name))
        for name in (
            "s1_config",
            "train_mask_manifest",
            "checkpoint",
            "source_config",
            "instances_info",
            "frame_instances",
        )
    }
    mask_manifest = json.loads(Path(inputs["train_mask_manifest"]).read_text(encoding="utf-8"))
    if not bool(mask_manifest.get("heldout_excluded")):
        raise RuntimeError("train mask manifest 未证明 heldout 排除")
    heldout = {int(value) for value in config["scene"]["heldout_frames"]}
    if heldout != {int(value) for value in mask_manifest["heldout_frames"]}:
        raise RuntimeError("S3 heldout split 与 train mask manifest 不一致")
    reserved = {int(value) for value in config["scene"]["reserved_development_frames"]}
    if reserved & heldout:
        raise RuntimeError("development 与 heldout frame 重叠")
    scene_root = Path(config["scene"]["processed_root"])
    instances = json.loads(Path(inputs["instances_info"]).read_text(encoding="utf-8"))
    actor = config["actors"][args.role]
    instance = instances[str(int(actor["dataset_instance_id"]))]
    if instance["id"] != actor["instance_token"]:
        raise RuntimeError("S3 actor dataset id/token 错配")

    thresholds = config["candidate"]
    sources = candidate_rows(
        mask_manifest,
        role=args.role,
        reserved_frames=reserved,
        heldout_frames=heldout,
        minimum_sam_pixels=int(thresholds["minimum_sam_pixels"]),
        minimum_projected_box_pixels=int(thresholds["minimum_projected_box_pixels"]),
    )
    if not sources:
        raise RuntimeError(f"S3 {args.role} 没有 train-only candidate")

    checkpoint = Path(inputs["checkpoint"])
    checkpoint_before = sha256_file(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("S3 view selection 需要真实 D2 GPU effect")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    runtime_config = {
        "inputs": {"checkpoint": str(checkpoint), "source_config": inputs["source_config"]},
        "runtimes": {"drivestudio_checkout": config["runtimes"]["drivestudio_checkout"]},
    }
    dataset, trainer = build_runtime(runtime_config, device)
    raw_candidates = []
    for index, source in enumerate(sources):
        frame, camera = int(source["frame"]), int(source["camera_id"])
        image_path = verify_file(source["source_image"], source["source_image_sha256"], "source_image")
        mask_path = verify_file(source["mask"], source["mask_sha256"], "source_mask")
        with np.load(mask_path, allow_pickle=False) as arrays:
            sam = arrays["binary"].astype(bool)
        box_mask = projected_box_mask(sam.shape, source["projected_box_xyxy"])
        original = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=frame,
            camera=camera,
            model_index=int(actor["rigid_model_index"]),
            variant="original",
            device=device,
        )["rgb"]
        release_trainer_render_info(trainer)
        deleted = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=frame,
            camera=camera,
            model_index=int(actor["rigid_model_index"]),
            variant="delete",
            device=device,
        )["rgb"]
        release_trainer_render_info(trainer)
        effect = dilate_binary(
            counterfactual_effect_mask(
                original,
                deleted,
                threshold_uint8=int(thresholds["effect_threshold_uint8"]),
                dilation_radius=2,
            ),
            int(thresholds["effect_dilation_radius"]),
        )
        combined = sam & effect
        visible = float(combined.sum() / max(1, int(sam.sum())))
        effect_precision = float(combined.sum() / max(1, int(effect.sum())))
        rgb = load_small_rgb(image_path, sam.shape)
        dynamic_path = scene_root / "dynamic_masks" / "all" / f"{frame:03d}_{camera}.png"
        dynamic = load_small_binary(dynamic_path, sam.shape)
        nearby = nearby_dynamic_fraction(
            dynamic,
            sam,
            box_mask,
            int(thresholds["dynamic_context_dilation_pixels"]),
        )
        mask_box_fraction = float(sam.sum() / max(1, int(box_mask.sum())))
        truncation = truncation_score(sam, source["projected_box_xyxy"])
        occlusion = float(
            np.clip(
                0.35 * (1.0 - min(1.0, mask_box_fraction))
                + 0.45 * (1.0 - visible)
                + 0.20 * nearby,
                0.0,
                1.0,
            )
        )
        eligible = (
            int(combined.sum()) >= int(thresholds["minimum_combined_pixels"])
            and visible >= float(thresholds["minimum_visible_fraction"])
            and truncation <= float(thresholds["maximum_truncation_score"])
        )
        yaw = actor_view_yaw(
            scene_root,
            instances,
            int(actor["dataset_instance_id"]),
            frame,
            camera,
        )
        row = {
            "frame": frame,
            "camera_id": camera,
            "camera_name": source["camera_name"],
            "instance_token": source["instance_token"],
            "source_image": str(image_path),
            "source_image_sha256": source["source_image_sha256"],
            "source_mask": str(mask_path),
            "source_mask_sha256": source["mask_sha256"],
            "dynamic_mask": str(dynamic_path),
            "dynamic_mask_sha256": sha256_file(dynamic_path),
            "prompt_frame": int(source["prompt_frame"]),
            "direct_prompt": int(source["frame"]) == int(source["prompt_frame"]),
            "projected_box_xyxy": [float(value) for value in source["projected_box_xyxy"]],
            "projected_area_fraction": float(box_mask.sum() / sam.size),
            "sam_pixels": int(sam.sum()),
            "d2_effect_pixels": int(effect.sum()),
            "combined_pixels": int(combined.sum()),
            "mask_box_visible_fraction": mask_box_fraction,
            "mask_confidence": mask_confidence(source),
            "sharpness_laplacian_variance": laplacian_variance(rgb, sam),
            "visible_fraction": visible,
            "d2_effect_precision": effect_precision,
            "nearby_dynamic_fraction": nearby,
            "occlusion_score": occlusion,
            "truncation_score": truncation,
            "yaw_radians": yaw,
            "yaw_degrees": math.degrees(yaw),
            "eligible": bool(eligible),
            "heldout": False,
            "reserved_development": False,
        }
        raw_candidates.append(row)
        print(
            f"[{index + 1}/{len(sources)}] {args.role} f{frame:03d} c{camera} "
            f"visible={visible:.3f} eligible={eligible}",
            flush=True,
        )

    ranked = rank_view_candidates(raw_candidates, config["view_score"]["weights"])
    eligible_rows = [row for row in ranked if bool(row["eligible"])]
    set_config = config["set_score"]
    selected = select_view_sets(
        eligible_rows,
        view_counts=set_config["view_counts"],
        yaw_weight=float(set_config["yaw_weight"]),
        temporal_weight=float(set_config["temporal_weight"]),
        camera_weight=float(set_config["camera_weight"]),
        frame_span=int(config["scene"]["frame_count"]) - 1,
        minimum_same_camera_frame_gap=int(set_config["minimum_same_camera_frame_gap"]),
        minimum_pairwise_yaw_degrees=float(set_config["minimum_pairwise_yaw_degrees"]),
        beam_width=int(set_config["beam_width"]),
    )

    selection_manifest = {
        "schema_version": "worldsim_v33_s3_view_selection_v1",
        "task_id": config["task_id"],
        "scene": config["scene"]["name"],
        "role": args.role,
        "actor": actor,
        "config_sha256": sha256_file(args.config),
        "train_mask_manifest_sha256": inputs["train_mask_manifest_sha256"],
        "checkpoint_sha256": checkpoint_before,
        "candidate_protocol": {
            "train_only": True,
            "heldout_read": False,
            "heldout_frames": sorted(heldout),
            "reserved_development_frames": sorted(reserved),
            "thresholds": thresholds,
        },
        "view_score": config["view_score"],
        "set_score": config["set_score"],
        "candidate_count_before_d2": len(sources),
        "eligible_count": len(eligible_rows),
        "candidates": sorted(ranked, key=lambda row: (int(row["frame"]), int(row["camera_id"]))),
        "selected_sets": {str(key): value for key, value in sorted(selected.items())},
    }
    selection_path = artifacts / "selection_manifest.json"
    atomic_json(selection_path, selection_manifest)

    audit_dir = artifacts / "audit"
    samples_dir = artifacts / "samples"
    audit_dir.mkdir()
    samples_dir.mkdir()
    input_samples = []
    for view_count, selected_set in sorted(selected.items()):
        sample_name = f"{args.role}_auto_{view_count}view"
        sample_dir = samples_dir / sample_name
        sample_dir.mkdir()
        views = []
        for view_index, row in enumerate(selected_set["selected_views"]):
            frame, camera = int(row["frame"]), int(row["camera_id"])
            image_path = Path(row["source_image"])
            with Image.open(image_path) as handle:
                source_image = handle.convert("RGB")
            with np.load(row["source_mask"], allow_pickle=False) as arrays:
                sam = arrays["binary"].astype(bool)
            original = render_variant(
                trainer=trainer,
                dataset=dataset,
                checkpoint=checkpoint,
                frame=frame,
                camera=camera,
                model_index=int(actor["rigid_model_index"]),
                variant="original",
                device=device,
            )["rgb"]
            release_trainer_render_info(trainer)
            deleted = render_variant(
                trainer=trainer,
                dataset=dataset,
                checkpoint=checkpoint,
                frame=frame,
                camera=camera,
                model_index=int(actor["rigid_model_index"]),
                variant="delete",
                device=device,
            )["rgb"]
            release_trainer_render_info(trainer)
            effect = dilate_binary(
                counterfactual_effect_mask(
                    original,
                    deleted,
                    threshold_uint8=int(thresholds["effect_threshold_uint8"]),
                    dilation_radius=2,
                ),
                int(thresholds["effect_dilation_radius"]),
            )
            combined_small = sam & effect
            if int(combined_small.sum()) < int(thresholds["minimum_combined_pixels"]):
                raise RuntimeError("selected view 重渲染后 combined mask 漂移")
            stem = f"{sample_name}_f{frame:03d}_c{camera}"
            audits = {}
            for name, image in {
                "d2_original": Image.fromarray(original, mode="RGB"),
                "d2_delete": Image.fromarray(deleted, mode="RGB"),
                "sam": Image.fromarray(sam.astype(np.uint8) * 255, mode="L"),
                "d2_effect": Image.fromarray(effect.astype(np.uint8) * 255, mode="L"),
                "combined": Image.fromarray(combined_small.astype(np.uint8) * 255, mode="L"),
            }.items():
                path = audit_dir / f"{stem}_{name}.png"
                image.save(path)
                audits[name] = {
                    "path": str(path.relative_to(artifacts)),
                    "sha256": sha256_file(path),
                }
            mask_full = np.asarray(
                Image.fromarray(combined_small.astype(np.uint8) * 255, mode="L").resize(
                    source_image.size, Image.Resampling.NEAREST
                )
            ) > 0
            crop_image, crop_mask, crop_xyxy = square_crop(
                source_image,
                mask_full,
                float(config["crop"]["padding_fraction"]),
                int(config["crop"]["output_size"]),
            )
            image_output = sample_dir / f"frame_{view_index}.jpeg"
            mask_output = sample_dir / f"mask_{view_index}.png"
            crop_image.save(image_output, format="JPEG", quality=95, subsampling=0)
            crop_mask.save(mask_output, format="PNG")
            _, target_lwh = actor_annotation(
                instances, int(actor["dataset_instance_id"]), frame
            )
            views.append(
                {
                    **row,
                    "input_mask_provenance": "SAM2_INTERSECT_D2_COUNTERFACTUAL_EFFECT",
                    "d2_original_array_sha256": array_sha256(original),
                    "d2_delete_array_sha256": array_sha256(deleted),
                    "crop_xyxy_source": crop_xyxy,
                    "box_size_source": target_lwh,
                    "audit": audits,
                    "image": str(image_output.relative_to(artifacts)),
                    "image_sha256": sha256_file(image_output),
                    "mask": str(mask_output.relative_to(artifacts)),
                    "mask_sha256": sha256_file(mask_output),
                    "positive_pixels": int((np.asarray(crop_mask) > 0).sum()),
                }
            )
        input_samples.append(
            {
                "sample": sample_name,
                "view_count": int(view_count),
                "set_score": selected_set["set_score"],
                "views": views,
            }
        )

    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("S3 view selection 修改了 D2 checkpoint")
    input_manifest = {
        "schema_version": "worldsim_v33_s3_asset_harvester_input_v1",
        "task_id": config["task_id"],
        "scene": config["scene"]["name"],
        "role": args.role,
        "actor": actor,
        "config_sha256": sha256_file(args.config),
        "selection_manifest": "selection_manifest.json",
        "selection_manifest_sha256": sha256_file(selection_path),
        "samples_dir": "samples",
        "camera_source": "asset_harvester_estimated",
        "provenance": "TRAIN_ONLY_SAM2_WITH_D2_COUNTERFACTUAL_PRIOR",
        "heldout_excluded": True,
        "heldout_read": False,
        "reserved_development_excluded": True,
        "output_size": int(config["crop"]["output_size"]),
        "padding_fraction": float(config["crop"]["padding_fraction"]),
        "samples": input_samples,
    }
    input_path = artifacts / "input_manifest.json"
    atomic_json(input_path, input_manifest)
    selection_sha = sha256_file(selection_path)
    input_sha = sha256_file(input_path)
    if args.expected_selection_sha256 and selection_sha != args.expected_selection_sha256:
        raise RuntimeError("S3 formal selection manifest 与冻结 diagnostic SHA 不一致")
    if args.expected_input_sha256 and input_sha != args.expected_input_sha256:
        raise RuntimeError("S3 formal input manifest 与冻结 diagnostic SHA 不一致")

    snapshot = snapshot_sources(args.run_dir, args.config.resolve())
    elapsed = time.time() - started
    summary = {
        "task_id": config["task_id"],
        "state": "completed",
        "role": args.role,
        "candidate_count": len(sources),
        "eligible_count": len(eligible_rows),
        "selected": {
            str(count): [
                [int(row["frame"]), int(row["camera_id"])]
                for row in value["selected_views"]
            ]
            for count, value in sorted(selected.items())
        },
        "selection_manifest_sha256": selection_sha,
        "input_manifest_sha256": input_sha,
        "heldout_read": False,
        "reserved_development_read": False,
        "checkpoint_immutable": True,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "source_snapshot": snapshot,
        "verified_inputs": verified,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": "completed",
            "started_unix": started,
            "completed_unix": time.time(),
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "selection_manifest_sha256": selection_sha,
            "input_manifest_sha256": input_sha,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
