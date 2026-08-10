#!/usr/bin/env python
"""在冻结 train-only blocks 上运行 SAM2，并写出逐视图 sidecar mask。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file, validate_disjoint_split


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是 mapping: {path}")
    return payload


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection_lo = np.maximum(left[:2], right[:2])
    intersection_hi = np.minimum(left[2:], right[2:])
    intersection_size = np.maximum(intersection_hi - intersection_lo, 0.0)
    intersection = float(np.prod(intersection_size))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def mask_geometry(mask: np.ndarray) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, None, None
    box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
    centroid = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
    return int(xs.size), box, centroid


def quality_gate(
    *,
    binary: np.ndarray,
    projected_box: list[float] | None,
    previous: np.ndarray | None,
    quality: dict[str, Any],
) -> tuple[bool, list[str], dict[str, float | None]]:
    area, mask_box, centroid = mask_geometry(binary)
    reasons: list[str] = []
    metrics: dict[str, float | None] = {
        "prompt_bbox_iou": None,
        "mask_to_prompt_area_ratio": None,
        "centroid_to_prompt_diagonal": None,
        "temporal_iou": None,
        "temporal_centroid_jump_fraction": None,
        "temporal_area_ratio": None,
    }
    if area < int(quality["minimum_positive_pixels"]):
        reasons.append("too_few_positive_pixels")
    if area > 0 and projected_box is not None and mask_box is not None and centroid is not None:
        prompt = np.asarray(projected_box, dtype=np.float64)
        prompt_size = np.maximum(prompt[2:] - prompt[:2], 1.0)
        prompt_area = float(np.prod(prompt_size))
        prompt_center = (prompt[:2] + prompt[2:]) / 2.0
        prompt_diagonal = float(np.linalg.norm(prompt_size))
        metrics["prompt_bbox_iou"] = box_iou(mask_box, prompt)
        metrics["mask_to_prompt_area_ratio"] = area / prompt_area
        metrics["centroid_to_prompt_diagonal"] = float(
            np.linalg.norm(centroid - prompt_center) / prompt_diagonal
        )
        if metrics["prompt_bbox_iou"] < float(quality["minimum_prompt_bbox_iou"]):
            reasons.append("prompt_bbox_iou")
        if metrics["mask_to_prompt_area_ratio"] > float(
            quality["maximum_mask_to_prompt_area_ratio"]
        ):
            reasons.append("prompt_area_ratio")
        if metrics["centroid_to_prompt_diagonal"] > float(
            quality["maximum_centroid_to_prompt_diagonal"]
        ):
            reasons.append("prompt_centroid_distance")
    if area > 0 and previous is not None:
        previous_area, _, previous_centroid = mask_geometry(previous)
        if previous_area > 0 and previous_centroid is not None and centroid is not None:
            intersection = int(np.logical_and(binary, previous).sum())
            union = int(np.logical_or(binary, previous).sum())
            metrics["temporal_iou"] = intersection / union if union else 0.0
            metrics["temporal_centroid_jump_fraction"] = float(
                np.linalg.norm(centroid - previous_centroid)
                / np.linalg.norm(binary.shape)
            )
            metrics["temporal_area_ratio"] = max(
                area / previous_area, previous_area / area
            )
            low_overlap_and_jump = (
                metrics["temporal_iou"] < float(quality["minimum_temporal_iou"])
                and metrics["temporal_centroid_jump_fraction"]
                > float(quality["maximum_temporal_centroid_jump_fraction"])
            )
            if low_overlap_and_jump:
                reasons.append("temporal_centroid_jump")
            if metrics["temporal_area_ratio"] > float(
                quality["maximum_temporal_area_ratio"]
            ):
                reasons.append("temporal_area_ratio")
    return not reasons, reasons, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"S1 SAM output 已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = load_yaml(args.config)
    prompts = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if prompts.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("prompt manifest 与当前 S1 config 不一致")
    if prompts.get("checkpoint_sha256") != config["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("prompt manifest 的 D2 checkpoint 已漂移")
    if prompts.get("actor_registry_sha256") != config["inputs"]["actor_registry_sha256"]:
        raise RuntimeError("prompt manifest 的 actor registry 已漂移")
    validate_disjoint_split(prompts["train_frames"], prompts["heldout_frames"])
    if not prompts.get("heldout_excluded"):
        raise RuntimeError("S1 prompt manifest 未证明 heldout 排除")

    checkpoint = Path(config["sam2"]["checkpoint"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != config["sam2"]["checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint SHA 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("S1 SAM2 需要可见 CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.memory_allocated(device) > 2 * 1024**3:
        raise RuntimeError("S1 SAM2 GPU preflight 非空闲")

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    sam2_root = Path(config["sam2"]["source_checkout"])
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config["sam2"]["model_config"],
        str(checkpoint),
        device=str(device),
        vos_optimized=False,
    )
    target_height = int(config["outputs"]["model_native_height"])
    target_width = int(config["outputs"]["model_native_width"])
    threshold = float(config["sam2"]["mask_logit_threshold"])
    quality = config["sam2"]["quality_gate"]
    role_by_object = {
        int(actor["sam_object_id"]): role for role, actor in config["actors"].items()
    }
    mask_rows: list[dict[str, Any]] = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for block in prompts["blocks"]:
            video_dir = Path(block["video_dir"])
            if not video_dir.is_absolute():
                video_dir = args.prompt_manifest.parent / video_dir
            if not video_dir.is_dir():
                raise FileNotFoundError(f"SAM2 train-only video block 缺失: {video_dir}")
            state = predictor.init_state(
                video_path=str(video_dir), offload_video_to_cpu=True
            )
            predictor.reset_state(state)
            for prompt in block["prompts"]:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(prompt["local_index"]),
                    obj_id=int(prompt["object_id"]),
                    box=np.asarray(prompt["box_xyxy"], dtype=np.float32),
                )
            frames = {int(row["local_index"]): row for row in block["frames"]}
            prompt_by_object = {
                int(prompt["object_id"]): prompt for prompt in block["prompts"]
            }
            previous_by_object: dict[int, tuple[int, np.ndarray]] = {}
            propagated: dict[tuple[int, int], torch.Tensor] = {}
            for reverse in (False, True):
                start_frame_idx = (
                    max(int(prompt["local_index"]) for prompt in block["prompts"])
                    if reverse
                    else None
                )
                outputs = predictor.propagate_in_video(
                    state,
                    start_frame_idx=start_frame_idx,
                    reverse=reverse,
                )
                for local_index, object_ids, logits in outputs:
                    for object_offset, object_id in enumerate(object_ids):
                        numeric_object_id = int(object_id)
                        prompt = prompt_by_object[numeric_object_id]
                        prompt_index = int(prompt["local_index"])
                        eligible = (
                            int(local_index) <= prompt_index
                            if reverse
                            else int(local_index) >= prompt_index
                        )
                        if eligible:
                            propagated.setdefault(
                                (int(local_index), numeric_object_id),
                                logits[object_offset : object_offset + 1].float().cpu(),
                            )

            for (local_index, numeric_object_id), raw_value in sorted(propagated.items()):
                frame_row = frames[int(local_index)]
                frame = int(frame_row["frame"])
                if frame in set(prompts["heldout_frames"]):
                    raise RuntimeError(f"SAM2 输出命中 heldout frame={frame}")
                role = role_by_object[numeric_object_id]
                value = functional.interpolate(
                    raw_value.unsqueeze(0) if raw_value.ndim == 3 else raw_value,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                logits_np = value.numpy().astype(np.float16)
                raw_binary = logits_np > threshold
                previous_row = previous_by_object.get(numeric_object_id)
                previous = (
                    previous_row[1]
                    if previous_row is not None and previous_row[0] + 1 == local_index
                    else None
                )
                projected_box_source = frame_row.get("projected_boxes", {}).get(role)
                projected_box = None
                if projected_box_source is not None:
                    source_width = float(frame_row["width"])
                    source_height = float(frame_row["height"])
                    scale = np.asarray(
                        [
                            target_width / source_width,
                            target_height / source_height,
                            target_width / source_width,
                            target_height / source_height,
                        ],
                        dtype=np.float64,
                    )
                    projected_box = (
                        np.asarray(projected_box_source, dtype=np.float64) * scale
                    ).tolist()
                accepted, rejection_reasons, quality_metrics = quality_gate(
                    binary=raw_binary,
                    projected_box=projected_box,
                    previous=previous,
                    quality=quality,
                )
                binary = raw_binary if accepted else np.zeros_like(raw_binary)
                if accepted:
                    previous_by_object[numeric_object_id] = (local_index, raw_binary)
                else:
                    previous_by_object.pop(numeric_object_id, None)
                output = (
                    args.output_dir
                    / "masks"
                    / role
                    / block["camera_name"]
                    / f"{frame:03d}.npz"
                )
                atomic_npz(
                    output,
                    logits=logits_np,
                    raw_binary=raw_binary,
                    binary=binary,
                )
                prompt = prompt_by_object[numeric_object_id]
                mask_rows.append(
                    {
                        "role": role,
                        "instance_token": config["actors"][role]["instance_token"],
                        "object_id": numeric_object_id,
                        "frame": frame,
                        "timestamp_source": "processed_frame_index_10hz",
                        "camera_id": int(block["camera_id"]),
                        "camera_name": block["camera_name"],
                        "source_image": frame_row["image"],
                        "source_image_sha256": frame_row["image_sha256"],
                        "prompt_frame": int(prompt["frame"]),
                        "prompt_local_index": int(prompt["local_index"]),
                        "prompt_box_xyxy": prompt["box_xyxy"],
                        "prompt_source": prompt["box_source"],
                        "projected_box_xyxy_source": projected_box_source,
                        "projected_box_xyxy": projected_box,
                        "mask": str(output),
                        "mask_sha256": sha256_file(output),
                        "height": target_height,
                        "width": target_width,
                        "accepted": accepted,
                        "rejection_reasons": rejection_reasons,
                        "quality_metrics": quality_metrics,
                        "raw_positive_pixels": int(raw_binary.sum()),
                        "positive_pixels": int(binary.sum()),
                        "logit_min": float(logits_np.min()),
                        "logit_max": float(logits_np.max()),
                    }
                )
            del state
            torch.cuda.empty_cache()

    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("SAM2 checkpoint 在推理后发生 mutation")
    manifest = {
        "schema_version": "worldsim_v32_s1_sam_masks_v1",
        "task_id": config["task_id"],
        "config_sha256": sha256_file(args.config),
        "prompt_manifest": str(args.prompt_manifest),
        "prompt_manifest_sha256": sha256_file(args.prompt_manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "heldout_frames": prompts["heldout_frames"],
        "heldout_excluded": True,
        "propagation": "bidirectional_from_projected_box_keyframe",
        "quality_gate": quality,
        "mask_count": len(mask_rows),
        "accepted_mask_count": sum(bool(row["accepted"]) for row in mask_rows),
        "rejected_mask_count": sum(not bool(row["accepted"]) for row in mask_rows),
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "masks": sorted(
            mask_rows, key=lambda row: (row["role"], row["frame"], row["camera_id"])
        ),
    }
    atomic_json(args.output_dir / "mask_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "done",
                "mask_count": len(mask_rows),
                "manifest": str(args.output_dir / "mask_manifest.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
