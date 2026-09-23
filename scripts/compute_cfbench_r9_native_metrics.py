#!/usr/bin/env python3
"""Uncalibrated paired metrics for a paper-native 10-frame DriveEditor clip."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from compute_cfbench_r9_video_metrics import allowed_mask, sha256

try:
    import lpips
    import torch
except ImportError:
    lpips = None
    torch = None


def read_exact(path: Path, expected: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened() or abs(float(cap.get(cv2.CAP_PROP_FPS)) - 10.0) > .01:
        raise RuntimeError(f"unreadable/non-10Hz video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) != expected:
        raise RuntimeError(f"wrong frame count {len(frames)} for {path}")
    return frames


def psnr(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float | str:
    difference = first.astype(np.float32)[mask] - second.astype(np.float32)[mask]
    mse = float(np.mean(np.square(difference)))
    return round(10 * math.log10(255.0 ** 2 / mse), 4) if mse else "infinite"


def masked_ssim(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = cv2.resize(first, (512, 288), interpolation=cv2.INTER_AREA)
    b = cv2.resize(second, (512, 288), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), (512, 288),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    _, quality = structural_similarity(a, b, channel_axis=2, data_range=255, full=True)
    return float(quality[small_mask].mean())


def masked_lpips(model, first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float | None:
    if model is None:
        return None
    a = cv2.cvtColor(cv2.resize(first, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.resize(second, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    small_mask = cv2.resize(mask.astype(np.uint8), (256, 144), interpolation=cv2.INTER_NEAREST)
    small_mask = cv2.erode(small_mask, np.ones((17, 17), np.uint8)).astype(bool)
    if not small_mask.any():
        return None
    ta = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
    tb = torch.from_numpy(b.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
    with torch.no_grad():
        spatial = model(ta, tb)[0, 0].numpy()
    return float(spatial[small_mask].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--factual-cases", type=Path, required=True)
    parser.add_argument("--native-cases", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.manifest_root / "candidates.json").read_text(encoding="utf-8"))
    approved = next(row for row in manifest["cases"] if row["case_id"] == args.case_id)
    factual_root, native_root = args.factual_cases / args.case_id, args.native_cases / args.case_id
    factual_meta = json.loads((factual_root / "protocol.json").read_text(encoding="utf-8"))
    native_meta = json.loads((native_root / "protocol.json").read_text(encoding="utf-8"))
    if (factual_meta["window_source_frames"] != native_meta["window_source_frames"] or
            factual_meta["reference_frame"] != native_meta["reference_frame"] or
            factual_meta["evaluation_role"] != "factual_identity_reconstruction" or
            native_meta["evaluation_role"] != "counterfactual_edit"):
        raise RuntimeError("factual and counterfactual native clips are not paired")
    geometry = json.loads((args.geometry_root / f"{args.case_id}.json").read_text(encoding="utf-8"))
    if not geometry["geometry_available"]:
        raise RuntimeError("missing approved planned geometry")
    raw = read_exact(args.manifest_root / "cases" / args.case_id / "original-nuscenes.mp4", 100)
    factual_path = factual_root / "factual-reconstruction.mp4"
    native_path = native_root / "counterfactual.mp4"
    factual, edited = read_exact(factual_path, 10), read_exact(native_path, 10)
    source_start = int(approved["source_frame_range"][0])
    offsets = [frame - source_start for frame in factual_meta["window_source_frames"]]
    if offsets != list(range(offsets[0], offsets[0] + 10)):
        raise RuntimeError("native clip frames are not consecutive")
    perceptual = None
    if lpips is not None:
        torch.set_num_threads(2)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True).cpu().eval()
    full = np.ones((576, 1024), dtype=bool)
    rec_psnr, rec_ssim, rec_lpips = [], [], []
    bg_psnr, bg_ssim, bg_lpips = [], [], []
    edit_energy = preserve_energy = edit_changed = edit_pixels = 0
    temporal, prior_residual, prior_preserve = [], None, None
    for index, offset in enumerate(offsets):
        rec_psnr.append(psnr(raw[offset], factual[index], full))
        rec_ssim.append(masked_ssim(raw[offset], factual[index], full))
        lp = masked_lpips(perceptual, raw[offset], factual[index], full)
        if lp is not None:
            rec_lpips.append(lp)
        edit = allowed_mask(geometry["frames"][offset], 1024, 576, margin_px=16)
        preserve = ~edit
        bg_psnr.append(psnr(factual[index], edited[index], preserve))
        bg_ssim.append(masked_ssim(factual[index], edited[index], preserve))
        lp = masked_lpips(perceptual, factual[index], edited[index], preserve)
        if lp is not None:
            bg_lpips.append(lp)
        diff = np.abs(edited[index].astype(np.int16) - factual[index].astype(np.int16))
        edit_energy += int(diff[edit].sum())
        preserve_energy += int(diff[preserve].sum())
        if edit.any():
            edit_changed += int((diff[edit].mean(axis=1) > 12).sum())
            edit_pixels += int(edit.sum())
        residual = edited[index].astype(np.int16) - factual[index].astype(np.int16)
        if prior_residual is not None:
            mask = preserve & prior_preserve
            temporal.append(float(np.abs(residual[mask] - prior_residual[mask]).mean()))
        prior_residual, prior_preserve = residual, preserve
    finite = lambda xs: [float(x) for x in xs if isinstance(x, (float, int))]
    metrics = {
        "factual_reconstruction": {
            "full_frame_psnr_db_mean": round(float(np.mean(finite(rec_psnr))), 4) if finite(rec_psnr) else "infinite",
            "full_frame_ssim_mean": round(float(np.mean(rec_ssim)), 5),
            "full_frame_lpips_squeeze_v0_1_256x144_mean": round(float(np.mean(rec_lpips)), 5) if rec_lpips else None,
            "comparison": "generated factual object reconstruction versus corresponding original 10 source frames; full frame, descriptive only",
        },
        "counterfactual_vs_factual_reconstruction": {
            "masked_background_psnr_db_mean": round(float(np.mean(finite(bg_psnr))), 4) if finite(bg_psnr) else "infinite",
            "masked_background_ssim_mean": round(float(np.mean(bg_ssim)), 5),
            "masked_background_lpips_squeeze_v0_1_256x144_mean": round(float(np.mean(bg_lpips)), 5) if bg_lpips else None,
            "edit_region_change_energy_fraction": round(edit_energy / (edit_energy + preserve_energy), 5) if edit_energy + preserve_energy else None,
            "edit_region_changed_pixel_fraction_gt12": round(edit_changed / edit_pixels, 5) if edit_pixels else None,
            "background_residual_temporal_mae_0_255": round(float(np.mean(temporal)), 5),
            "comparison": "paired generated factual and counterfactual native clips, same source frames/reference/seed; planned union-box mask, not output segmentation",
        },
        "trajectory_ADE_FDE": None,
        "object_identity": None,
    }
    output = {"schema_version": "cfbench_r9_driveeditor_native_paired_metrics_v1",
              "case_id": args.case_id, "native_source_frames": factual_meta["window_source_frames"],
              "native_fps": 10, "native_frames": 10, "llm_judge": None, "human_judge": None,
              "calibrated_0_to_4_scores": None, "metrics": metrics,
              "factual_video_sha256": sha256(factual_path),
              "counterfactual_video_sha256": sha256(native_path),
              "caveat": "10-frame paper-native diagnostics; not a 10-second score; input boxes are not output detections"}
    destination = native_root / "paired-auto-metrics.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": args.case_id, "output": str(destination),
                      "reconstruction_psnr_db": metrics["factual_reconstruction"]["full_frame_psnr_db_mean"],
                      "background_psnr_db": metrics["counterfactual_vs_factual_reconstruction"]["masked_background_psnr_db_mean"]}))


if __name__ == "__main__":
    main()
