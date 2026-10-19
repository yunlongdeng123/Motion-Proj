"""Foreground/static reconstruction metrics with explicit normalization.

The paper does not specify whether masked PSNR uses selected-pixel or full-image
normalization. Save both. These are reconstruction metrics on supplied masks,
not deletion-hole quality or FID against historical background ground truth.
"""
import json
import math
from pathlib import Path
import torch


def _region_metric(prediction, target, mask):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must be matching CHW tensors")
    if mask.shape != (1, *target.shape[1:]):
        raise ValueError("Mask must be 1HW")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite image values")
    selected = mask.to(device=target.device, dtype=torch.bool)
    count = int(selected.sum().item())
    all_pixels = target.shape[-1] * target.shape[-2]
    if count == 0:
        return {"pixels": 0, "fraction": 0.0, "sse": 0.0,
                "psnr_roi": None, "psnr_zero_padded": None, "perfect": False}
    squared = (prediction.clamp(0, 1).double() - target.clamp(0, 1).double()).square()
    sse = float(squared.masked_select(selected.expand_as(squared)).sum().item())
    channels = target.shape[0]
    def decibels(denominator):
        return -10 * math.log10(sse / denominator) if sse > 0 else None
    return {"pixels": count, "fraction": count / all_pixels, "sse": sse,
            "psnr_roi": decibels(channels * count),
            "psnr_zero_padded": decibels(channels * all_pixels), "perfect": sse == 0}


@torch.no_grad()
def measure_frame(composite, static_only, target, dynamic_mask, frame_id):
    dynamic = dynamic_mask > 0.5
    row = {"frame": str(frame_id),
           "dynamic_psnr_star": _region_metric(composite, target, dynamic),
           "static_psnr_s": _region_metric(composite, target, ~dynamic)}
    if static_only is not None:
        row["static_psnr_sr"] = _region_metric(static_only, target, ~dynamic)
    return row


def save_metrics(rows, directory, split, iteration):
    summary = {}
    for metric in ("dynamic_psnr_star", "static_psnr_s", "static_psnr_sr"):
        selected = [row[metric] for row in rows if metric in row and row[metric]["pixels"] > 0]
        summary[metric] = {"valid_frames": len(selected),
                           "perfect_frames": sum(row["perfect"] for row in selected)}
        for convention in ("psnr_roi", "psnr_zero_padded"):
            values = [row[convention] for row in selected if row[convention] is not None]
            summary[metric][convention + "_mean"] = sum(values) / len(values) if values else None
    report = {
        "split": split, "iteration": iteration, "frame_count": len(rows),
        "aggregation": "Arithmetic mean of per-frame PSNR, later macro-average scenes equally.",
        "perfect_error_policy": "Zero-error frames have infinite mathematical PSNR; saved as null and counted separately, excluded from finite mean.",
        "mask_protocol": "Explicit binary actor masks, union optional separately predicted shadows; nearest-neighbor resize.",
        "interpretation": "PSNR* uses composite RGB on dynamic region. PSNR-S uses composite RGB on static region. PSNR-SR uses static-branch RGB on the same static region after deleting all dynamic Gaussians.",
        "limitations": ["Paper masked-PSNR normalization is unspecified; both ROI and full-image zero-padded conventions are reported.",
                        "No deletion-hole quality is measured: historical unobstructed GT is required for Table V.",
                        "Inferred/prepared masks are not independent manually annotated ground truth."],
        "summary": summary, "frames": rows}
    path = Path(directory) / "paper_region_metrics.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
