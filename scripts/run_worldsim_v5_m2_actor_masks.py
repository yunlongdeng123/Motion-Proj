#!/usr/bin/env python3
"""重放冻结 SAM prompts，并物化 M2 one-actor/one-view repair masks。"""

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
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import project_box_prompt
from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from scripts.build_worldsim_v32_sam_masks import quality_gate
from scripts.run_worldsim_v5_m1_sam_diagnostic import (
    _resize_logits,
    normalize_box_logits,
    select_prompt_actors,
)
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


TASK_ID = "WS-V5-M2-GEOMETRY-FIRST-REPAIR-01"
SCHEMA_VERSION = "worldsim_v5_m2_actor_masks_v1"


class M2ActorMaskError(RuntimeError):
    """per-actor SAM replay 或冻结 denominator 漂移。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2ActorMaskError("M2 actor-mask config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "per_actor_mask_materialization"
        or payload["view_protocol"]["request_unit"] != "one_actor_one_view_one_hole"
        or payload["view_protocol"]["union_mask_for_geometry_forbidden"] is not True
    ):
        raise M2ActorMaskError("M2 actor-mask task/request-unit 漂移")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "geometry_quality_read",
    ):
        if payload["scope"].get(name) is not False:
            raise M2ActorMaskError(f"M2 actor-mask restriction 漂移: {name}")
    return payload


def denominator(rows: list[Mapping[str, Any]], view_rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "view_count": len(view_rows),
        "available_view_count": sum(int(row["box_count"]) > 0 for row in view_rows),
        "unavailable_view_count": sum(int(row["box_count"]) == 0 for row in view_rows),
        "actor_mask_count": len(rows),
        "accepted_actor_mask_count": sum(bool(row["accepted"]) for row in rows),
        "rejected_actor_mask_count": sum(not bool(row["accepted"]) for row in rows),
    }


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m2_actor_masks_status_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def _intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    return np.asarray(
        [[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]]
    )


def _original_views(config: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    manifest = json.loads(Path(config["inputs"]["r036_mask_manifest"]["path"]).read_text())
    if manifest.get("scene") != config["scene"]["name"] or manifest.get("heldout_quality_read") is not False:
        raise M2ActorMaskError("r036 manifest scene/provenance 漂移")
    expected = {
        (int(frame), int(camera))
        for frame in config["view_protocol"]["frames"]
        for camera in config["view_protocol"]["cameras"]
    }
    views = {
        (int(row["frame"]), int(row["camera_id"])): row
        for row in manifest["views"]
        if row.get("split") == "evaluation"
        and (int(row["frame"]), int(row["camera_id"])) in expected
    }
    if set(views) != expected:
        raise M2ActorMaskError("r036 selected view denominator 漂移")
    return views


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        inputs = {
            name: verify_file(binding["path"], binding["sha256"])
            for name, binding in config["inputs"].items()
        }
        scene = config["scene"]
        scene_root = Path(scene["processed_scene_dir"])
        instances_path = scene_root / "instances/instances_info.json"
        inputs["instances_info"] = verify_file(instances_path, scene["instances_info_sha256"])
        formal = json.loads(Path(inputs["formal_summary"]["path"]).read_text())
        r036 = json.loads(Path(inputs["r036_summary"]["path"]).read_text())
        if (
            formal.get("status") != "done"
            or formal.get("scene") != scene["name"]
            or formal.get("checkpoint", {}).get("sha256") != inputs["formal_checkpoint"]["sha256"]
            or formal.get("validation_quality_read") is not False
            or formal.get("test_quality_read") is not False
            or r036.get("status") != "done"
            or r036.get("scene") != scene["name"]
            or r036.get("heldout_quality_read") is not False
        ):
            raise M2ActorMaskError("formal/r036 provenance 漂移")
        original_views = _original_views(config)
        actors = select_prompt_actors(
            json.loads(instances_path.read_text()),
            minimum_trajectory_m=float(config["prompts"]["minimum_trajectory_m"]),
        )
        sam = config["sam2"]
        sam_root = Path(sam["source_checkout"])
        if _git(sam_root, "rev-parse", "HEAD") != sam["source_commit"] or _git(
            sam_root, "status", "--porcelain"
        ):
            raise M2ActorMaskError("SAM2 source checkout 漂移")
        sam_checkpoint = verify_file(sam["checkpoint"], sam["checkpoint_sha256"])
        checkpoint_before = sha256_file(inputs["formal_checkpoint"]["path"])
        if not torch.cuda.is_available():
            raise M2ActorMaskError("M2 actor-mask replay 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(device) > int(
            config["resources"]["maximum_gpu_allocated_at_start_mib"]
        ) * 1024**2:
            raise M2ActorMaskError("M2 actor-mask GPU preflight 非空闲")
        torch.cuda.reset_peak_memory_stats(device)
        if str(sam_root) not in sys.path:
            sys.path.insert(0, str(sam_root))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(
            build_sam2(sam["model_config"], str(Path(sam["checkpoint"])), device=str(device))
        )
        target_height, target_width = int(config["outputs"]["height"]), int(config["outputs"]["width"])
        actor_rows: list[dict[str, Any]] = []
        view_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame in config["view_protocol"]["frames"]:
                for camera in config["scene"]["cameras"]:
                    frame, camera_id = int(frame), int(camera["id"])
                    original = original_views[(frame, camera_id)]
                    image_path = scene_root / "images" / f"{frame:03d}_{camera_id}.jpg"
                    extrinsics_path = scene_root / "extrinsics" / f"{frame:03d}_{camera_id}.txt"
                    intrinsics_path = scene_root / "intrinsics" / f"{camera_id}.txt"
                    for path in (image_path, extrinsics_path, intrinsics_path):
                        if not path.is_file():
                            raise M2ActorMaskError(f"actor-mask view 缺失: {path}")
                    with Image.open(image_path) as handle:
                        image = np.asarray(handle.convert("RGB"))
                    height, width = image.shape[:2]
                    intrinsics = _intrinsics(intrinsics_path)
                    camera_to_world = np.loadtxt(extrinsics_path).reshape(4, 4)
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
                            minimum_side_pixels=float(config["prompts"]["minimum_side_pixels"]),
                        )
                        if box is not None:
                            boxes.append([float(value) for value in box])
                            actor_ids.append(actor_id)
                    if len(boxes) != int(original["box_count"]):
                        raise M2ActorMaskError(f"r036 prompt count replay 漂移: f{frame} c{camera_id}")
                    accepted_union = np.full((height, width), -20.0, dtype=np.float32)
                    accepted_count = 0
                    if boxes:
                        predictor.set_image(image)
                        predicted, scores, _ = predictor.predict(
                            box=np.asarray(boxes, dtype=np.float32),
                            multimask_output=False,
                            return_logits=True,
                        )
                        logits_rows = normalize_box_logits(predicted, len(boxes))
                        score_rows = np.asarray(scores, dtype=np.float32).reshape(-1)
                        original_boxes = {int(row["actor_id"]): row for row in original["boxes"]}
                        for actor_id, box, logits, score in zip(actor_ids, boxes, logits_rows, score_rows):
                            raw_binary = logits > float(sam["mask_logit_threshold"])
                            accepted, reasons, metrics = quality_gate(
                                binary=raw_binary,
                                projected_box=box,
                                previous=None,
                                quality=sam["quality_gate"],
                            )
                            old = original_boxes.get(actor_id)
                            if (
                                old is None
                                or not np.allclose(box, old["box_xyxy"], rtol=0.0, atol=1e-5)
                                or bool(accepted) is not bool(old["accepted"])
                                or int(raw_binary.sum()) != int(old["positive_pixels"])
                                or not np.isclose(float(score), float(old["predicted_iou"]), rtol=0.0, atol=1e-6)
                            ):
                                raise M2ActorMaskError(
                                    f"r036 actor replay 漂移: f{frame} c{camera_id} actor={actor_id}"
                                )
                            if accepted:
                                accepted_union = np.maximum(accepted_union, logits)
                                accepted_count += 1
                            resized = _resize_logits(logits, target_height, target_width)
                            binary = resized > float(sam["mask_logit_threshold"])
                            if not accepted:
                                binary[:] = False
                            output = run_dir / "artifacts/masks" / f"f{frame:03d}_c{camera_id}_a{actor_id:03d}.npz"
                            atomic_save_npz(
                                output,
                                {
                                    "actor_id": np.asarray(actor_id, dtype=np.int32),
                                    "raw_logits": resized,
                                    "binary": binary.astype(np.uint8),
                                    "mask_quality_accepted": np.asarray(accepted, dtype=np.int8),
                                },
                            )
                            actor_rows.append(
                                {
                                    "frame": frame,
                                    "camera_id": camera_id,
                                    "actor_id": actor_id,
                                    "class_name": actors[actor_id]["class_name"],
                                    "box_xyxy": box,
                                    "predicted_iou": float(score),
                                    "accepted": bool(accepted),
                                    "rejection_reasons": reasons,
                                    "quality_metrics": metrics,
                                    "raw_positive_pixels_source": int(raw_binary.sum()),
                                    "positive_pixels": int(binary.sum()),
                                    "mask": {
                                        "path": str(output.relative_to(run_dir)),
                                        "bytes": output.stat().st_size,
                                        "sha256": sha256_file(output),
                                    },
                                }
                            )
                    replay_union = _resize_logits(accepted_union, target_height, target_width) > float(
                        sam["mask_logit_threshold"]
                    )
                    old_mask = Path(config["inputs"]["r036_mask_manifest"]["path"]).parent.parent / original["mask"]["path"]
                    verify_file(old_mask, original["mask"]["sha256"])
                    with np.load(old_mask, allow_pickle=False) as payload:
                        old_union = payload["binary"].astype(bool)
                    if not np.array_equal(replay_union, old_union) or accepted_count != int(
                        original["accepted_box_count"]
                    ):
                        raise M2ActorMaskError(f"r036 union replay 非 exact: f{frame} c{camera_id}")
                    view_rows.append(
                        {
                            "frame": frame,
                            "camera_id": camera_id,
                            "box_count": len(boxes),
                            "accepted_box_count": accepted_count,
                            "union_replay_exact": True,
                        }
                    )
                    predictor.reset_predictor()
                    print(
                        f"M2 actor masks frame={frame} camera={camera_id} boxes={len(boxes)} accepted={accepted_count}",
                        flush=True,
                    )
        counts = denominator(actor_rows, view_rows)
        expected = config["view_protocol"]
        frozen = {
            "view_count": int(expected["expected_view_count"]),
            "available_view_count": int(expected["expected_available_view_count"]),
            "unavailable_view_count": int(expected["expected_unavailable_view_count"]),
            "actor_mask_count": int(expected["expected_actor_mask_count"]),
            "accepted_actor_mask_count": int(expected["expected_accepted_actor_mask_count"]),
            "rejected_actor_mask_count": int(expected["expected_rejected_actor_mask_count"]),
        }
        if counts != frozen:
            raise M2ActorMaskError(f"actor-mask denominator 漂移: {counts} != {frozen}")
        checkpoint_after = sha256_file(inputs["formal_checkpoint"]["path"])
        sam_after = sha256_file(sam["checkpoint"])
        if checkpoint_after != checkpoint_before or sam_after != sam_checkpoint["sha256"]:
            raise M2ActorMaskError("checkpoint 在 actor-mask replay 后 mutation")
        mask_manifest = {
            "schema_version": "worldsim_v5_m2_actor_mask_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": scene["name"],
            "request_unit": "one_actor_one_view_one_hole",
            "union_mask_for_geometry_forbidden": True,
            "counts": counts,
            "views": view_rows,
            "masks": sorted(actor_rows, key=lambda row: (row["frame"], row["camera_id"], row["actor_id"])),
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "geometry_quality_read": False,
        }
        mask_manifest_path = run_dir / "artifacts/mask_manifest.json"
        atomic_json(mask_manifest_path, mask_manifest)
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_actor_masks.py",
                PROJECT / "scripts/run_worldsim_v5_m1_sam_diagnostic.py",
                PROJECT / "scripts/build_worldsim_v32_sam_masks.py",
                PROJECT / "tests/test_run_worldsim_v5_m2_actor_masks.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m2_actor_masks_summary_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": "done",
            "phase": config["phase"],
            "scene": scene["name"],
            "source_commit": source_head,
            "conclusion": "per_actor_repair_request_masks_materialized",
            **counts,
            "request_unit": "one_actor_one_view_one_hole",
            "union_replay_exact": True,
            "mask_manifest_sha256": sha256_file(mask_manifest_path),
            "formal_checkpoint_sha256_before": checkpoint_before,
            "formal_checkpoint_sha256_after": checkpoint_after,
            "sam_checkpoint_sha256_before": sam_checkpoint["sha256"],
            "sam_checkpoint_sha256_after": sam_after,
            "duration_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated(device) / 1024**2),
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "geometry_quality_read": False,
            "parameter_search_performed": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v5_m2_actor_masks_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_head,
                "source_clean": True,
                "resolved_config": resolved,
                "inputs": inputs,
                "sam2": {
                    "source_commit": sam["source_commit"],
                    "checkpoint": sam_checkpoint,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                },
                "source_snapshot": snapshot,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now(), **counts})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_actor_masks_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
            reason=None,
        )
        return summary
    except Exception as error:
        events.append({"event": "run_blocked", "at_utc": utc_now(), "reason": f"{type(error).__name__}: {error}"})
        write_events(run_dir, events)
        _terminal(
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
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve(), args.device), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
