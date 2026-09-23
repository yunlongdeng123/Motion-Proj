#!/usr/bin/env python3
"""Compute output-side, non-calibrated diagnostics for approved paired videos.

Never turns projected input boxes into output detections or subjective 0–4 scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

try:
    import lpips
    import torch
except ImportError:
    lpips = None
    torch = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if abs(fps - 10.0) > 0.01:
        raise RuntimeError(f"not 10 fps: {path}: {fps}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != 100:
        raise RuntimeError(f"not 100 frames: {path}: {len(frames)}")
    return frames


def allowed_mask(geometry_frame: dict, width: int, height: int, margin_px: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in geometry_frame["allowed_edit_bbox_xyxy"]:
        if box is None:
            continue
        left, top, right, bottom = box
        left = max(0, int(math.floor(left * width / 1600)) - margin_px)
        top = max(0, int(math.floor(top * height / 900)) - margin_px)
        right = min(width, int(math.ceil(right * width / 1600)) + margin_px)
        bottom = min(height, int(math.ceil(bottom * height / 900)) + margin_px)
        if left < right and top < bottom:
            mask[top:bottom, left:right] = True
    return mask


def score_case(case_root: Path, geometry: dict, mode: str) -> dict:
    cid = case_root.name
    result = json.loads((case_root / "result.json").read_text(encoding="utf-8"))
    source_path = case_root / "factual.mp4"
    edited_path = case_root / "counterfactual.mp4"
    source, edited = read_video(source_path), read_video(edited_path)
    width, height = 1024, 576
    passthrough = set(result.get("passthrough_windows", []))
    gated = bool(result.get("first_five_frames_replaced_with_original"))
    native = [mode == "paper_iterative" or (t // 10 not in passthrough and not (gated and t < 5))
              for t in range(100)]
    if mode == "paper_iterative" and result.get("paper_iterative_conditioning") is not True:
        raise RuntimeError("iterative metric mode requires verified iterative result metadata")
    if geometry["target_role"] == "ego":
        raise RuntimeError("ego viewpoint change requires a different geometry protocol")
    if not geometry["geometry_available"] or len(geometry["frames"]) != 100:
        raise RuntimeError("missing approved geometry")

    sum_sq = 0.0
    sum_preserve_abs = 0.0
    preserve_values = 0
    sum_edit_abs = 0.0
    edit_values = 0
    changed_preserve, changed_edit = 0, 0
    preserve_pixels, edit_pixels = 0, 0
    ssim_values = []
    lpips_values = []
    perceptual = None
    if lpips is not None:
        torch.set_num_threads(2)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True).cpu().eval()
    temporal = []
    preceding_residual = None
    preceding_preserve = None
    frame_rows = []
    for t, (src, out) in enumerate(zip(source, edited)):
        if src.shape[:2] != (height, width):
            src = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
        if out.shape[:2] != (height, width):
            raise RuntimeError(f"counterfactual has unexpected dimensions: {out.shape}")
        edit = allowed_mask(geometry["frames"][t], width, height, margin_px=16)
        preserve = ~edit
        residual = out.astype(np.int16) - src.astype(np.int16)
        diff = np.abs(residual).astype(np.float32)
        if native[t]:
            d_preserve = diff[preserve]
            sum_sq += float(np.square(d_preserve).sum())
            sum_preserve_abs += float(d_preserve.sum())
            preserve_values += int(d_preserve.size)
            changed_preserve += int((d_preserve.mean(axis=1) > 12).sum())
            preserve_pixels += int(preserve.sum())
            if edit.any():
                d_edit = diff[edit]
                sum_edit_abs += float(d_edit.sum())
                edit_values += int(d_edit.size)
                changed_edit += int((d_edit.mean(axis=1) > 12).sum())
                edit_pixels += int(edit.sum())
            small_src = cv2.resize(src, (512, 288), interpolation=cv2.INTER_AREA)
            small_out = cv2.resize(out, (512, 288), interpolation=cv2.INTER_AREA)
            small_preserve = cv2.resize(preserve.astype(np.uint8), (512, 288),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
            _, ssim_map = structural_similarity(small_src, small_out, channel_axis=2,
                                                 data_range=255, full=True)
            ssim_values.append(float(ssim_map[small_preserve].mean()))
            if perceptual is not None and t % 5 == 0:
                perceptual_src = cv2.resize(src, (256, 144), interpolation=cv2.INTER_AREA)
                perceptual_out = cv2.resize(out, (256, 144), interpolation=cv2.INTER_AREA)
                perceptual_preserve = cv2.resize(preserve.astype(np.uint8), (256, 144),
                                                  interpolation=cv2.INTER_NEAREST)
                # Erode the preserve region so edit-mask boundaries cannot
                # contribute feature responses to the masked LPIPS average.
                perceptual_preserve = cv2.erode(perceptual_preserve, np.ones((17, 17), np.uint8)) > 0
                if perceptual_preserve.any():
                    rgb_src = cv2.cvtColor(perceptual_src, cv2.COLOR_BGR2RGB)
                    rgb_out = cv2.cvtColor(perceptual_out, cv2.COLOR_BGR2RGB)
                    tensor_src = torch.from_numpy(rgb_src.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
                    tensor_out = torch.from_numpy(rgb_out.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
                    with torch.no_grad():
                        spatial_distance = perceptual(tensor_src, tensor_out)[0, 0].numpy()
                    lpips_values.append(float(spatial_distance[perceptual_preserve].mean()))
            if preceding_residual is not None and native[t - 1]:
                temporal_mask = preserve & preceding_preserve
                value = float(np.abs(residual[temporal_mask] - preceding_residual[temporal_mask]).mean())
                temporal.append({"transition_to_frame": t, "residual_mae_0_255": value})
        preceding_residual = residual
        preceding_preserve = preserve
        frame_rows.append({"frame": t, "native_model_frame": native[t],
                           "edit_area_fraction": round(float(edit.mean()), 6)})
    if not preserve_values:
        raise RuntimeError("no native model frames with preservation pixels")
    mse = sum_sq / preserve_values
    psnr = float("inf") if mse == 0 else 10 * math.log10((255 ** 2) / mse)
    if mode == "legacy_independent":
        boundaries = set(range(10, 100, 10))
    else:
        boundaries = {10 + 9 * k for k in range(10)}
    boundary_vals = [x["residual_mae_0_255"] for x in temporal if x["transition_to_frame"] in boundaries]
    internal_vals = [x["residual_mae_0_255"] for x in temporal if x["transition_to_frame"] not in boundaries]
    change_total = sum_edit_abs + sum_preserve_abs
    metrics = {
        "E": {
            "masked_psnr_db": round(psnr, 3) if math.isfinite(psnr) else "infinite",
            "masked_ssim_512x288": round(float(np.mean(ssim_values)), 5),
            "masked_lpips_squeeze_v0_1_256x144": round(float(np.mean(lpips_values)), 5) if lpips_values else None,
            "masked_lpips_sampled_frames": len(lpips_values),
            "preserve_mean_abs_rgb_0_255": round(sum_preserve_abs / preserve_values, 4),
            "preserve_changed_pixel_fraction_gt12": round(changed_preserve / preserve_pixels, 5),
            "preserve_area_fraction": round(preserve_pixels / (sum(native) * width * height), 5),
        },
        "A": {
            "edit_region_change_energy_fraction": round(sum_edit_abs / change_total, 5) if change_total else None,
            "edit_region_mean_abs_rgb_0_255": round(sum_edit_abs / edit_values, 4) if edit_values else None,
            "non_target_change_energy_fraction": round(sum_preserve_abs / change_total, 5) if change_total else None,
            "limitation": "change localization proxy, not proof that the requested semantic edit occurred",
        },
        "P": {
            "background_residual_temporal_mae_0_255": round(float(np.mean([x["residual_mae_0_255"] for x in temporal])), 4) if temporal else None,
            "segment_boundary_residual_mae_0_255": round(float(np.mean(boundary_vals)), 4) if boundary_vals else None,
            "within_segment_residual_mae_0_255": round(float(np.mean(internal_vals)), 4) if internal_vals else None,
            "segment_boundary_to_internal_ratio": round(float(np.mean(boundary_vals) / np.mean(internal_vals)), 4) if boundary_vals and internal_vals and np.mean(internal_vals) else None,
            "limitation": "2D temporal residual only; not a physical dynamics or collision score",
        },
        "O": {
            "edit_region_changed_pixel_fraction_gt12": round(changed_edit / edit_pixels, 5) if edit_pixels else None,
            "limitation": "edited-region pixel change is not object detection, removal confirmation, or downstream outcome",
        },
        "T": {"pixel_ADE": None, "pixel_FDE": None,
              "reason": "no validated detector/tracker measurement of generated target; planned boxes cannot be substituted"},
        "OP": {"object_identity_similarity": None,
               "background_preservation_reuses_E": True,
               "reason": "no validated generated-object segmentation/identity match"},
    }
    return {
        "schema_version": "cfbench_r9_auto_video_metrics_v1",
        "case_id": cid,
        "method": "DriveEditor",
        "evaluation_mode": mode,
        "calibrated_0_to_4_scores": None,
        "llm_judge": None,
        "human_judge": None,
        "native_model_frames": sum(native),
        "excluded_passthrough_frames": 10 * len(passthrough),
        "excluded_temporal_gate_frames": 5 if gated else 0,
        "source_video_sha256": sha256(source_path),
        "counterfactual_video_sha256": sha256(edited_path),
        "planned_geometry_sha256": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
        "methods": {
            "mask": "union of approved factual and desired projected 3D bounding rectangles after event, 16 px margin at 1024x576; full-frame preservation before event",
            "comparison": "frame-aligned factual RGB versus counterfactual RGB; factual is original observation for DriveEditor",
            "ssim": "skimage structural_similarity full map at 512x288, averaged over preservation pixels",
            "lpips": "pretrained SqueezeNet LPIPS v0.1 spatial map at 256x144, every fifth native frame; preservation mask eroded by 8 feature pixels before averaging",
            "psnr": "pooled RGB MSE only on preservation pixels; 255 range",
            "changed_pixel_threshold": "mean absolute RGB >12/255, descriptive diagnostic only; no score threshold",
            "temporal": "difference of frame-wise counterfactual-minus-factual residuals on intersected preservation masks",
        },
        "metrics": metrics,
        "temporal_transitions": temporal,
        "frames": frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["legacy_independent", "paper_iterative"], required=True)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    selected = set(args.case_id)
    count = 0
    for case_root in sorted(args.cases_root.iterdir()):
        if not case_root.is_dir() or selected and case_root.name not in selected:
            continue
        result_path = case_root / "result.json"
        geometry_path = args.geometry_root / f"{case_root.name}.json"
        if not result_path.is_file() or not geometry_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") not in {"generation_complete", "assembled_with_passthrough"}:
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if not geometry["geometry_available"]:
            continue
        metrics = score_case(case_root, geometry, args.mode)
        (case_root / "auto-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"case_id": case_root.name, "native_frames": metrics["native_model_frames"],
                          "masked_psnr_db": metrics["metrics"]["E"]["masked_psnr_db"]}), flush=True)
        count += 1
    print(json.dumps({"metric_cases": count}))


if __name__ == "__main__":
    main()
