#!/usr/bin/env python
"""在真实观测视角评估 S3 生成 actor 的外观、轮廓与回注一致性。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import lpips
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(
            ImageFilter.MaxFilter(radius * 2 + 1)
        )
    ) > 0


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(
            ImageFilter.MinFilter(radius * 2 + 1)
        )
    ) > 0


def boundary_f1(predicted: np.ndarray, target: np.ndarray, tolerance: int = 3) -> float:
    predicted_boundary = predicted & ~erode(predicted, 1)
    target_boundary = target & ~erode(target, 1)
    if not predicted_boundary.any() or not target_boundary.any():
        return 0.0
    precision = float((predicted_boundary & dilate(target_boundary, tolerance)).sum()) / float(
        predicted_boundary.sum()
    )
    recall = float((target_boundary & dilate(predicted_boundary, tolerance)).sum()) / float(
        target_boundary.sum()
    )
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def masked_lpips_inputs(
    source: np.ndarray, generated: np.ndarray, mask: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError("S3 evaluation mask 为空")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = int(round(max(x1 - x0, y1 - y0) * 0.25))
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(mask.shape[1], x1 + padding), min(mask.shape[0], y1 + padding)
    crop_mask = mask[y0:y1, x0:x1]
    pairs = []
    for image in (source, generated):
        crop = image[y0:y1, x0:x1].copy()
        crop[~crop_mask] = 255
        resized = np.asarray(
            Image.fromarray(crop, mode="RGB").resize(
                (256, 256), Image.Resampling.LANCZOS
            )
        )
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float()
        pairs.append((tensor / 127.5 - 1.0).unsqueeze(0))
    return pairs[0], pairs[1]


def make_panel(
    source: np.ndarray,
    original: np.ndarray,
    lateral: np.ndarray,
    deleted: np.ndarray,
    target_mask: np.ndarray,
    effect_mask: np.ndarray,
    label: str,
) -> Image.Image:
    mask_rgb = np.repeat((target_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    effect_rgb = np.repeat((effect_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    arrays = [source, original, lateral, deleted, mask_rgb, effect_rgb]
    names = ["source", "generated", "lateral+1m", "delete", "SAM", "generated effect"]
    width, height = 320, 180
    panel = Image.new("RGB", (width * 3, (height + 28) * 2 + 34), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 8), label, fill="black", font=font)
    for index, (array, name) in enumerate(zip(arrays, names)):
        tile = Image.fromarray(array, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)
        x = (index % 3) * width
        y = 34 + (index // 3) * (height + 28)
        panel.paste(tile, (x, y))
        draw.text((x + 8, y + height + 7), name, fill="black", font=font)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").eval().to(device)
    rows = []
    panels = []
    for render_path in args.render_manifest:
        render = json.loads(render_path.read_text(encoding="utf-8"))
        sample = next(
            row
            for row in inputs["samples"]
            if row["sample"] == render["evaluation_view_sample"]
        )
        view = sample["views"][int(render["source_view_index"])]
        if int(view["frame"]) != int(render["frame"]) or int(view["camera_id"]) != int(
            render["camera_id"]
        ):
            raise RuntimeError("S3 render/input view provenance 错配")
        if sha256_file(Path(view["source_mask"])) != view["source_mask_sha256"]:
            raise RuntimeError("S3 evaluation source mask SHA 漂移")
        if sha256_file(Path(view["source_image"])) != view["source_image_sha256"]:
            raise RuntimeError("S3 evaluation source image SHA 漂移")
        with np.load(view["source_mask"], allow_pickle=False) as arrays:
            target_mask = arrays["binary"].astype(bool)
        images = {}
        for name in ("original", "lateral", "delete"):
            spec = render["images"][name]
            path = Path(spec["path"])
            if sha256_file(path) != spec["sha256"]:
                raise RuntimeError(f"S3 evaluation render SHA 漂移: {path}")
            images[name] = imageio.imread(path)[..., :3]
        height, width = images["original"].shape[:2]
        if target_mask.shape != (height, width):
            raise RuntimeError("S3 evaluation SAM/render shape 错配")
        with Image.open(view["source_image"]) as handle:
            source = np.asarray(
                handle.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            )
        effect_spec = render["effect_masks"]["original"]
        effect_path = Path(effect_spec["path"])
        if sha256_file(effect_path) != effect_spec["sha256"]:
            raise RuntimeError("S3 evaluation effect mask SHA 漂移")
        effect_mask = imageio.imread(effect_path) > 0
        intersection = int((effect_mask & target_mask).sum())
        union = int((effect_mask | target_mask).sum())
        silhouette_iou = float(intersection / union) if union else 0.0
        difference = source.astype(np.float32) - images["original"].astype(np.float32)
        masked_mse = float(np.square(difference[target_mask]).mean())
        masked_psnr = float(10.0 * np.log10((255.0**2) / max(masked_mse, 1e-12)))
        source_tensor, generated_tensor = masked_lpips_inputs(
            source, images["original"], target_mask
        )
        with torch.inference_mode():
            lpips_value = float(
                metric(source_tensor.to(device), generated_tensor.to(device)).item()
            )
        outside = ~dilate(target_mask, 8)
        outside_l1 = float(
            np.abs(
                images["original"].astype(np.int16)
                - images["delete"].astype(np.int16)
            )[outside].mean()
        )
        asset_manifest_path = Path(render["asset_manifest"])
        if sha256_file(asset_manifest_path) != render["asset_manifest_sha256"]:
            raise RuntimeError("S3 evaluation asset manifest SHA 漂移")
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        bounds = np.asarray(asset_manifest["coordinate_contract"]["bounds_upper_m"]) - np.asarray(
            asset_manifest["coordinate_contract"]["bounds_lower_m"]
        )
        target_lwh = np.asarray(asset_manifest["coordinate_contract"]["target_lwh_m"])
        row = {
            "asset_sample": render["asset_sample"],
            "frame": int(render["frame"]),
            "camera_id": int(render["camera_id"]),
            "camera_name": render["camera_name"],
            "silhouette_iou": silhouette_iou,
            "boundary_f1_tolerance_3px": boundary_f1(effect_mask, target_mask, 3),
            "masked_rgb_psnr": masked_psnr,
            "masked_crop_lpips_alex": lpips_value,
            "non_target_original_delete_l1_uint8": outside_l1,
            "generated_effect_pixels": int(effect_mask.sum()),
            "sam_pixels": int(target_mask.sum()),
            "bounds_extent_error_max_m": float(np.max(np.abs(bounds - target_lwh))),
            "gaussian_count": int(asset_manifest["asset"]["gaussian_count"]),
            "render_manifest": str(render_path.resolve()),
            "render_manifest_sha256": sha256_file(render_path),
        }
        rows.append(row)
        panel = make_panel(
            source,
            images["original"],
            images["lateral"],
            images["delete"],
            target_mask,
            effect_mask,
            f"{row['asset_sample']} f{row['frame']:03d} {row['camera_name']}",
        )
        panel_path = args.output_dir / f"{row['asset_sample']}_f{row['frame']:03d}_panel.png"
        panel.save(panel_path)
        panels.append({"path": str(panel_path), "sha256": sha256_file(panel_path)})
    aggregate = {}
    for sample_name in sorted({row["asset_sample"] for row in rows}):
        selected = [row for row in rows if row["asset_sample"] == sample_name]
        aggregate[sample_name] = {
            "view_count_evaluated": len(selected),
            "mean_silhouette_iou": float(np.mean([row["silhouette_iou"] for row in selected])),
            "mean_boundary_f1_tolerance_3px": float(
                np.mean([row["boundary_f1_tolerance_3px"] for row in selected])
            ),
            "mean_masked_rgb_psnr": float(np.mean([row["masked_rgb_psnr"] for row in selected])),
            "mean_masked_crop_lpips_alex": float(
                np.mean([row["masked_crop_lpips_alex"] for row in selected])
            ),
            "max_bounds_extent_error_m": float(
                max(row["bounds_extent_error_max_m"] for row in selected)
            ),
        }
    summary = {
        "schema_version": "worldsim_v32_s3_actor_asset_evaluation_v1",
        "task_id": "WS-V32-S3-ASSET-HARVEST-01",
        "status": "done",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "rows": rows,
        "aggregate": aggregate,
        "panels": panels,
        "claims": {
            "observed_views": "quality metrics against observed RGB/SAM proxy",
            "generated_backside": "completeness/consistency only; no GT correctness claim",
            "provenance": "GENERATED_ACTOR",
        },
    }
    atomic_json(args.output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
