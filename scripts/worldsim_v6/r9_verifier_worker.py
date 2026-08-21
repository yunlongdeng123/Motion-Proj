#!/usr/bin/env python3
"""隔离执行 R9 的 geometry 与 semantic verifier，并独立记录 P0–P4。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_depth_model(root: Path) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval().to("cuda")
    return processor, model


def _load_semantic_model(root: Path) -> torch.nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=19,
        aux_loss=False,
    )
    checkpoint = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to("cuda")
    return model


def _predict_depth(processor: Any, model: Any, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    inputs = processor(images=Image.fromarray(image, mode="RGB"), return_tensors="pt")
    inputs = {name: value.to("cuda") for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs).predicted_depth[:, None]
        output = torch_functional.interpolate(
            output, size=(height, width), mode="bicubic", align_corners=False
        )[:, 0]
    return output[0].float().cpu().numpy()


def _predict_semantic(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    tensor = ((tensor - mean) / std).to("cuda")
    with torch.inference_mode():
        logits = model(tensor)["out"]
    return logits.argmax(dim=1)[0].to(torch.int16).cpu().numpy()


def _affine_align(
    predicted: np.ndarray,
    target: np.ndarray,
    calibration_mask: np.ndarray,
    minimum_pixels: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    valid = calibration_mask & np.isfinite(predicted) & np.isfinite(target) & (target > 1.0e-6)
    count = int(np.count_nonzero(valid))
    if count < minimum_pixels:
        return None, {"status": "insufficient_evidence", "pixel_count": count}
    x = predicted[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < 2 or not np.all(np.isfinite(coefficients)):
        return None, {"status": "degenerate_alignment", "pixel_count": count}
    aligned = predicted.astype(np.float64) * coefficients[0] + coefficients[1]
    return aligned.astype(np.float32), {
        "status": "aligned",
        "pixel_count": count,
        "scale": float(coefficients[0]),
        "offset": float(coefficients[1]),
    }


def _decision(accepted: bool) -> str:
    return "ACCEPT" if accepted else "REJECT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-input-dir", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--depth-model-root", required=True, type=Path)
    parser.add_argument("--semantic-model-root", required=True, type=Path)
    parser.add_argument("--photo-max-mae", required=True, type=float)
    parser.add_argument("--photo-truth-pixel-error", required=True, type=float)
    parser.add_argument("--photo-truth-min-fraction", required=True, type=float)
    parser.add_argument("--depth-max-mean-relative-error", required=True, type=float)
    parser.add_argument("--depth-truth-pixel-error", required=True, type=float)
    parser.add_argument("--depth-truth-min-fraction", required=True, type=float)
    parser.add_argument("--minimum-alignment-pixels", required=True, type=int)
    parser.add_argument("--semantic-min-iou", required=True, type=float)
    parser.add_argument("--semantic-truth-min-iou", required=True, type=float)
    parser.add_argument("--dynamic-class-ids", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    dynamic_ids = {int(value) for value in args.dynamic_class_ids.split(",")}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    depth_processor, depth_model = _load_depth_model(args.depth_model_root)
    semantic_model = _load_semantic_model(args.semantic_model_root)
    load_seconds = time.monotonic() - started

    rows: list[dict[str, Any]] = []
    for input_path in sorted(args.verifier_input_dir.glob("*.npz")):
        case_id = input_path.stem
        proposal_path = args.proposal_dir / f"{case_id}__repeat1.npy"
        with np.load(input_path, allow_pickle=False) as archive:
            input_image = np.asarray(archive["input_image"], dtype=np.uint8)
            target_rgb = np.asarray(archive["target_rgb"], dtype=np.uint8)
            target_depth = np.asarray(archive["target_depth"], dtype=np.float32)
            target_depth_valid = np.asarray(archive["target_depth_valid"], dtype=bool)
            target_dynamic = np.asarray(archive["target_dynamic"], dtype=bool)
            mask = np.asarray(archive["mask"], dtype=bool)
            semantic_evidence = bool(np.asarray(archive["semantic_evidence"]).item())
            hole_type = str(np.asarray(archive["hole_type"]).item())
        proposal = np.load(proposal_path, allow_pickle=False).astype(np.uint8)
        if proposal.shape != input_image.shape or mask.shape != input_image.shape[:2]:
            raise RuntimeError(f"proposal 空间合同漂移：{case_id}")
        outside_exact = bool(np.array_equal(proposal[~mask], input_image[~mask]))

        rgb_error = np.mean(
            np.abs(proposal.astype(np.float32) - target_rgb.astype(np.float32)) / 255.0,
            axis=2,
        )
        photo_mae = float(np.mean(rgb_error[mask]))
        photo_usable_fraction = float(
            np.mean(rgb_error[mask] <= args.photo_truth_pixel_error)
        )
        photo_truth_safe = photo_usable_fraction >= args.photo_truth_min_fraction
        photo_accept = photo_mae <= args.photo_max_mae

        predicted_depth = _predict_depth(depth_processor, depth_model, proposal)
        aligned_depth, alignment = _affine_align(
            predicted_depth,
            target_depth,
            (~mask) & target_depth_valid,
            args.minimum_alignment_pixels,
        )
        depth_decision = "ABSTAIN"
        depth_truth_safe = False
        depth_mean_relative_error = None
        depth_usable_fraction = None
        depth_evidence_pixels = int(np.count_nonzero(mask & target_depth_valid))
        if aligned_depth is not None and depth_evidence_pixels:
            valid = mask & target_depth_valid & np.isfinite(aligned_depth)
            relative = np.abs(aligned_depth[valid] - target_depth[valid]) / np.maximum(
                np.abs(target_depth[valid]), 1.0e-3
            )
            depth_mean_relative_error = float(np.mean(relative))
            depth_usable_fraction = float(np.mean(relative <= args.depth_truth_pixel_error))
            depth_truth_safe = depth_usable_fraction >= args.depth_truth_min_fraction
            depth_decision = _decision(
                depth_mean_relative_error <= args.depth_max_mean_relative_error
            )

        semantic_decision = "ABSTAIN"
        semantic_truth_safe = False
        dynamic_iou = None
        if semantic_evidence:
            semantic = _predict_semantic(semantic_model, proposal)
            predicted_dynamic = np.isin(semantic, list(dynamic_ids))
            union = mask & (predicted_dynamic | target_dynamic)
            intersection = mask & predicted_dynamic & target_dynamic
            dynamic_iou = (
                1.0
                if not np.any(union)
                else float(np.count_nonzero(intersection) / np.count_nonzero(union))
            )
            semantic_truth_safe = dynamic_iou >= args.semantic_truth_min_iou
            semantic_decision = _decision(dynamic_iou >= args.semantic_min_iou)

        rows.append(
            {
                "schema_version": "worldsim_v6.r9_case_arms.v1",
                "case_id": case_id,
                "hole_type": hole_type,
                "proposal_sha256": _sha256(proposal_path),
                "outside_mask_exact": outside_exact,
                "P0": {
                    "decision": "ACCEPT",
                    "photo_truth_safe": photo_truth_safe,
                    "geometry_truth_safe": depth_truth_safe,
                    "semantic_truth_safe": semantic_truth_safe if semantic_evidence else None,
                },
                "P1": {
                    "decision": _decision(photo_accept),
                    "masked_rgb_mae": photo_mae,
                    "truth_usable_fraction": photo_usable_fraction,
                    "truth_safe": photo_truth_safe,
                    "false_safe": bool(photo_accept and not photo_truth_safe),
                },
                "P2": {
                    "decision": depth_decision,
                    "alignment": alignment,
                    "masked_evidence_pixels": depth_evidence_pixels,
                    "masked_mean_relative_depth_error": depth_mean_relative_error,
                    "truth_usable_fraction": depth_usable_fraction,
                    "truth_safe": depth_truth_safe,
                    "false_safe": bool(depth_decision == "ACCEPT" and not depth_truth_safe),
                },
                "P3": {
                    "decision": semantic_decision,
                    "semantic_evidence": semantic_evidence,
                    "dynamic_iou": dynamic_iou,
                    "truth_safe": semantic_truth_safe if semantic_evidence else None,
                    "false_safe": bool(
                        semantic_decision == "ACCEPT" and not semantic_truth_safe
                    ),
                },
                "P4": {
                    "decision": "ABSTAIN",
                    "reason": "no_independent_temporal_proposal_or_flow_evidence",
                },
            }
        )

    _write_jsonl(args.output_dir / "PER_CASE_ARMS.jsonl", rows)
    result = {
        "schema_version": "worldsim_v6.r9_verifier_worker.v1",
        "case_count": len(rows),
        "load_seconds": load_seconds,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_reserved() / (1024**2)),
        "all_outside_mask_exact": all(row["outside_mask_exact"] for row in rows),
        "semantic_evidence_case_count": sum(row["P3"]["semantic_evidence"] for row in rows),
        "p4_abstain_count": sum(row["P4"]["decision"] == "ABSTAIN" for row in rows),
        "per_case_arms_sha256": _sha256(args.output_dir / "PER_CASE_ARMS.jsonl"),
    }
    _write_json(args.output_dir / "WORKER_RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
