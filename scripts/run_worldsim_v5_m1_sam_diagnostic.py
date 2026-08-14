#!/usr/bin/env python3
"""在冻结 V5 development sparse views 上生成 run-local actor-union SAM2 sidecar。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import project_box_prompt
from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from scripts.build_worldsim_v32_sam_masks import quality_gate
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M1-STRUCTURED-OWNERSHIP-01"
SCHEMA_VERSION = "worldsim_v5_m1_sam_diagnostic_v1"
RIGID_CLASS_PREFIXES = ("vehicle.",)
RIGID_CLASS_EXCLUSIONS = {"vehicle.bicycle"}


class SamDiagnosticError(RuntimeError):
    """冻结输入、分区或 SAM sidecar 合同失败。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise SamDiagnosticError("SAM diagnostic config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "structured_unary_mechanism_smoke"
    ):
        raise SamDiagnosticError("SAM diagnostic task/phase/status 漂移")
    return payload


def trajectory_distance(actor: Mapping[str, Any]) -> float:
    poses = np.asarray(
        actor["frame_annotations"]["obj_to_world"], dtype=np.float64
    ).reshape(-1, 4, 4)
    if poses.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def select_prompt_actors(
    instances: Mapping[str, Mapping[str, Any]], *, minimum_trajectory_m: float
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for raw_id, actor in sorted(instances.items(), key=lambda item: int(item[0])):
        class_name = str(actor["class_name"])
        distance = trajectory_distance(actor)
        if (
            class_name.startswith(RIGID_CLASS_PREFIXES)
            and class_name not in RIGID_CLASS_EXCLUSIONS
            and distance > minimum_trajectory_m
        ):
            frames = actor["frame_annotations"]["frame_idx"]
            poses = actor["frame_annotations"]["obj_to_world"]
            sizes = actor["frame_annotations"]["box_size"]
            if not (len(frames) == len(poses) == len(sizes)):
                raise SamDiagnosticError(f"actor {raw_id} annotation 长度漂移")
            selected[int(raw_id)] = {
                "class_name": class_name,
                "trajectory_distance_m": distance,
                "annotations": {
                    int(frame): {
                        "obj_to_world": pose,
                        "box_size": size,
                    }
                    for frame, pose, size in zip(frames, poses, sizes)
                },
            }
    if not selected:
        raise SamDiagnosticError("没有满足冻结规则的 moving rigid actor")
    return selected


def normalize_box_logits(logits: np.ndarray, box_count: int) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim == 3 and box_count == 1:
        values = values[None, ...] if values.shape[0] != 1 else values
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape[0] != box_count or values.ndim != 3:
        raise SamDiagnosticError(
            f"SAM2 box logits shape 漂移: boxes={box_count}, logits={values.shape}"
        )
    if not np.isfinite(values).all():
        raise SamDiagnosticError("SAM2 logits 含非有限值")
    return values


def _resize_logits(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))[None, None]
    resized = functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    return resized.numpy().astype(np.float32)


def validate_frame_contract(config: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    scene = config["scene"]
    frame_count = int(scene["frame_count"])
    split = config["split"]
    evidence = [int(value) for value in split["evidence_frames"]]
    evaluation = [int(value) for value in split["evaluation_frames"]]
    if len(set(evidence)) != len(evidence) or len(set(evaluation)) != len(evaluation):
        raise SamDiagnosticError("evidence/evaluation frame 有重复")
    if set(evidence) & set(evaluation):
        raise SamDiagnosticError("evidence/evaluation frame 不相交合同失败")
    if any(frame < 0 or frame >= frame_count for frame in evidence + evaluation):
        raise SamDiagnosticError("SAM diagnostic frame 越界")
    modulus = int(split["modulus"])
    if any(frame % modulus not in set(split["train_remainders"]) for frame in evidence):
        raise SamDiagnosticError("evidence frame 不是 train remainder")
    if any(frame % modulus != int(split["development_remainder"]) for frame in evaluation):
        raise SamDiagnosticError("evaluation frame 不是 development remainder")
    if any(frame % modulus == int(split["heldout_remainder"]) for frame in evidence + evaluation):
        raise SamDiagnosticError("SAM diagnostic 读取 heldout remainder")
    return evidence, evaluation


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True
    ).strip()


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
            "schema_version": "worldsim_v5_m1_sam_status_v1",
            "task_id": TASK_ID,
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved_record = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        evidence_frames, evaluation_frames = validate_frame_contract(config)
        scene = config["scene"]
        scene_root = Path(scene["processed_scene_dir"])
        instances_path = scene_root / "instances/instances_info.json"
        input_records = {
            "formal_summary": verify_file(
                config["inputs"]["formal_summary"]["path"],
                config["inputs"]["formal_summary"]["sha256"],
            ),
            "formal_checkpoint": verify_file(
                config["inputs"]["formal_checkpoint"]["path"],
                config["inputs"]["formal_checkpoint"]["sha256"],
            ),
            "source_config": verify_file(
                config["inputs"]["source_config"]["path"],
                config["inputs"]["source_config"]["sha256"],
            ),
            "instances_info": verify_file(
                instances_path, scene["instances_info_sha256"]
            ),
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
            or formal_summary.get("scene") != scene["name"]
            or formal_summary.get("checkpoint", {}).get("sha256")
            != input_records["formal_checkpoint"]["sha256"]
            or formal_summary.get("validation_quality_read") is not False
            or formal_summary.get("test_quality_read") is not False
            or formal_summary.get("model_inference_started") is not False
        ):
            raise SamDiagnosticError("formal base summary contract 漂移")
        checkpoint_before = input_records["formal_checkpoint"]["sha256"]
        instances = json.loads(instances_path.read_text(encoding="utf-8"))
        actors = select_prompt_actors(
            instances,
            minimum_trajectory_m=float(config["prompts"]["minimum_trajectory_m"]),
        )

        sam = config["sam2"]
        sam_root = Path(sam["source_checkout"])
        if _git(sam_root, "rev-parse", "HEAD") != sam["source_commit"]:
            raise SamDiagnosticError("SAM2 source commit 漂移")
        if _git(sam_root, "status", "--porcelain"):
            raise SamDiagnosticError("SAM2 source checkout 非 clean")
        sam_checkpoint = verify_file(sam["checkpoint"], sam["checkpoint_sha256"])
        sam_checkpoint_before = sam_checkpoint["sha256"]
        if not torch.cuda.is_available():
            raise SamDiagnosticError("SAM2 diagnostic 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(device) > int(
            config["resources"]["maximum_gpu_allocated_at_start_mib"]
        ) * 1024**2:
            raise SamDiagnosticError("SAM2 diagnostic GPU preflight 非空闲")
        torch.cuda.reset_peak_memory_stats(device)
        if str(sam_root) not in sys.path:
            sys.path.insert(0, str(sam_root))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            sam["model_config"], str(Path(sam["checkpoint"])), device=str(device)
        )
        predictor = SAM2ImagePredictor(model)
        camera_rows = {int(row["id"]): row for row in scene["cameras"]}
        target_height = int(config["outputs"]["height"])
        target_width = int(config["outputs"]["width"])
        output_root = run_dir / "artifacts/masks"
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        all_frames = [("evidence", frame) for frame in evidence_frames] + [
            ("evaluation", frame) for frame in evaluation_frames
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for split_name, frame in all_frames:
                for camera_id in sorted(camera_rows):
                    camera = camera_rows[camera_id]
                    image_path = scene_root / "images" / f"{frame:03d}_{camera_id}.jpg"
                    extrinsics_path = (
                        scene_root / "extrinsics" / f"{frame:03d}_{camera_id}.txt"
                    )
                    intrinsics_path = scene_root / "intrinsics" / f"{camera_id}.txt"
                    for path in (image_path, extrinsics_path, intrinsics_path):
                        if not path.is_file():
                            raise SamDiagnosticError(f"SAM diagnostic view 缺失: {path}")
                    intrinsics_values = np.loadtxt(
                        intrinsics_path, dtype=np.float64
                    ).reshape(-1)
                    intrinsics = np.asarray(
                        [
                            [intrinsics_values[0], 0.0, intrinsics_values[2]],
                            [0.0, intrinsics_values[1], intrinsics_values[3]],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    camera_to_world = np.loadtxt(
                        extrinsics_path, dtype=np.float64
                    ).reshape(4, 4)
                    with Image.open(image_path) as image_handle:
                        image = np.asarray(image_handle.convert("RGB"))
                    height, width = image.shape[:2]
                    boxes: list[list[float]] = []
                    actor_ids: list[int] = []
                    for actor_id, actor in actors.items():
                        annotation = actor["annotations"].get(frame)
                        if annotation is None:
                            continue
                        box = project_box_prompt(
                            obj_to_world=np.asarray(annotation["obj_to_world"]),
                            box_size=np.asarray(annotation["box_size"]),
                            camera_to_world=camera_to_world,
                            intrinsics=intrinsics,
                            image_width=width,
                            image_height=height,
                            minimum_depth_m=float(config["prompts"]["minimum_depth_m"]),
                            padding_fraction=float(config["prompts"]["padding_fraction"]),
                            minimum_side_pixels=float(
                                config["prompts"]["minimum_side_pixels"]
                            ),
                        )
                        if box is not None:
                            boxes.append([float(value) for value in box])
                            actor_ids.append(actor_id)
                    raw_union = np.full((height, width), -20.0, dtype=np.float32)
                    accepted_union = np.full_like(raw_union, -20.0)
                    box_rows: list[dict[str, Any]] = []
                    accepted_count = 0
                    if boxes:
                        predictor.set_image(image)
                        predicted, scores, _ = predictor.predict(
                            box=np.asarray(boxes, dtype=np.float32),
                            multimask_output=False,
                            return_logits=True,
                        )
                        box_logits = normalize_box_logits(predicted, len(boxes))
                        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
                        if score_values.size != len(boxes):
                            raise SamDiagnosticError("SAM2 score/box count 漂移")
                        raw_union = np.max(box_logits, axis=0)
                        for actor_id, box, logits, score in zip(
                            actor_ids, boxes, box_logits, score_values
                        ):
                            raw_binary = logits > float(sam["mask_logit_threshold"])
                            accepted, reasons, metrics = quality_gate(
                                binary=raw_binary,
                                projected_box=box,
                                previous=None,
                                quality=sam["quality_gate"],
                            )
                            if accepted:
                                accepted_union = np.maximum(accepted_union, logits)
                                accepted_count += 1
                            box_rows.append(
                                {
                                    "actor_id": actor_id,
                                    "class_name": actors[actor_id]["class_name"],
                                    "box_xyxy": box,
                                    "predicted_iou": float(score),
                                    "accepted": bool(accepted),
                                    "rejection_reasons": reasons,
                                    "quality_metrics": metrics,
                                    "positive_pixels": int(raw_binary.sum()),
                                }
                            )
                    available = bool(boxes)
                    accepted = accepted_count > 0
                    resized_raw = _resize_logits(raw_union, target_height, target_width)
                    resized_accepted = _resize_logits(
                        accepted_union, target_height, target_width
                    )
                    binary = resized_accepted > float(sam["mask_logit_threshold"])
                    output = output_root / split_name / f"{frame:03d}_{camera_id}.npz"
                    atomic_save_npz(
                        output,
                        {
                            "raw_logits": resized_raw,
                            "logits": resized_accepted,
                            "binary": binary.astype(np.uint8),
                            "sam_probability_available": np.asarray(
                                available, dtype=np.int8
                            ),
                            "mask_quality_accepted": np.asarray(
                                accepted, dtype=np.int8
                            ),
                        },
                    )
                    rows.append(
                        {
                            "split": split_name,
                            "frame": frame,
                            "camera_id": camera_id,
                            "camera_name": camera["name"],
                            "image": {
                                "path": str(image_path),
                                "sha256": sha256_file(image_path),
                            },
                            "extrinsics": {
                                "path": str(extrinsics_path),
                                "sha256": sha256_file(extrinsics_path),
                            },
                            "intrinsics": {
                                "path": str(intrinsics_path),
                                "sha256": sha256_file(intrinsics_path),
                            },
                            "box_count": len(boxes),
                            "accepted_box_count": accepted_count,
                            "sam_probability_available": available,
                            "mask_quality_accepted": accepted,
                            "positive_pixels": int(binary.sum()),
                            "boxes": box_rows,
                            "mask": {
                                "path": str(output.relative_to(run_dir)),
                                "bytes": output.stat().st_size,
                                "sha256": sha256_file(output),
                            },
                        }
                    )
                    predictor.reset_predictor()
                    print(
                        f"SAM diagnostic {split_name} frame={frame} camera={camera_id} "
                        f"boxes={len(boxes)} accepted={accepted_count}",
                        flush=True,
                    )
        duration = time.perf_counter() - started
        sam_checkpoint_after = sha256_file(Path(sam["checkpoint"]))
        checkpoint_after = sha256_file(Path(config["inputs"]["formal_checkpoint"]["path"]))
        if sam_checkpoint_after != sam_checkpoint_before:
            raise SamDiagnosticError("SAM2 checkpoint 在 inference 后 mutation")
        if checkpoint_after != checkpoint_before:
            raise SamDiagnosticError("formal base checkpoint 在 SAM inference 后 mutation")
        mask_manifest = {
            "schema_version": "worldsim_v5_m1_sam_mask_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": scene["name"],
            "scene_index": int(scene["index"]),
            "evidence_frames": evidence_frames,
            "evaluation_frames": evaluation_frames,
            "heldout_quality_read": False,
            "actors": {
                str(actor_id): {
                    "class_name": actor["class_name"],
                    "trajectory_distance_m": actor["trajectory_distance_m"],
                }
                for actor_id, actor in actors.items()
            },
            "temporal_quality_state_used": False,
            "views": rows,
        }
        mask_manifest_path = run_dir / "artifacts/mask_manifest.json"
        atomic_json(mask_manifest_path, mask_manifest)
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m1_sam_diagnostic.py",
                PROJECT / "scripts/worldsim_v5_forensics_common.py",
                PROJECT / "scripts/build_worldsim_v32_sam_masks.py",
                PROJECT / "motion_proj/worldsim_v5/evidence_schema.py",
                PROJECT / "motion_proj/worldsim_v32/semantic_schema.py",
                PROJECT / "tests/test_run_worldsim_v5_m1_sam_diagnostic.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m1_sam_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "phase": "structured_unary_mechanism_smoke",
            "scene": scene["name"],
            "scene_index": int(scene["index"]),
            "view_count": len(rows),
            "evidence_view_count": sum(row["split"] == "evidence" for row in rows),
            "evaluation_view_count": sum(
                row["split"] == "evaluation" for row in rows
            ),
            "available_view_count": sum(
                bool(row["sam_probability_available"]) for row in rows
            ),
            "accepted_view_count": sum(
                bool(row["mask_quality_accepted"]) for row in rows
            ),
            "prompt_actor_count": len(actors),
            "prompt_box_count": sum(int(row["box_count"]) for row in rows),
            "accepted_box_count": sum(
                int(row["accepted_box_count"]) for row in rows
            ),
            "temporal_quality_state_used": False,
            "duration_seconds": duration,
            "peak_gpu_memory_mib": int(
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "source_commit": source_head,
            "formal_checkpoint_sha256_before": checkpoint_before,
            "formal_checkpoint_sha256_after": checkpoint_after,
            "sam_checkpoint_sha256_before": sam_checkpoint_before,
            "sam_checkpoint_sha256_after": sam_checkpoint_after,
            "mask_manifest_sha256": sha256_file(mask_manifest_path),
            "segmentation_inference_started": True,
            "method_inference_started": False,
            "parameter_search_performed": False,
            "network_accessed": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        fingerprint = {
            "schema_version": "worldsim_v5_m1_sam_fingerprint_v1",
            "task_id": TASK_ID,
            "source_commit": source_head,
            "source_clean": True,
            "resolved_config": resolved_record,
            "inputs": input_records,
            "sam2": {
                "source_commit": sam["source_commit"],
                "checkpoint": sam_checkpoint,
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
            "schema_version": "worldsim_v5_m1_sam_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
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
