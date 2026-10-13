"""Read local raw20 PNGs only; make native-size contacts and ROI diagnostics.

No source image is overwritten, resized, filtered, or recolored. This script
does not load checkpoints, import Torch, contact a remote host, or make hashes.
ROI statistics concern observed images and are not training data or edited GT.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

WORK = Path(__file__).resolve().parent
OLD = Path("C:/Users/dengyunlong/Documents/Codex/2026-10-11/z/work/vgpu_bugfix_20261012/continuation_diagnostic")
NEW = Path("C:/Users/dengyunlong/Documents/Codex/2026-10-12/w/outputs/drivingeditor_four_arm_30k/assets/C00/observed")
BRANCHES = {
    "B1@32000": "original30k_to32000",
    "B2@32000": "original30k_to32000_noise_reference",
    "S@32000": "observed_static_rgb30k_to32000",
}
CHANNELS = ["gt", "composite", "static", "dynamic", "alpha_s", "alpha_d"]
SOURCES = {
    arm: {c: OLD / directory / "extra_eval/32000" / c / "0017020.png" for c in CHANNELS}
    for arm, directory in BRANCHES.items()
}
SOURCES["C00@30000"] = {c: NEW / f"020_{c}.png" for c in CHANNELS}
ARRAYS = {
    arm: {c: np.asarray(Image.open(p)).copy() for c, p in channels.items()}
    for arm, channels in SOURCES.items()
}
BODY_PATH = NEW / "020_body_mask.png"
OLD_BODY_PATH = OLD.parent / "component_diagnostic/masks/0017020_actor.png"
body_image = np.asarray(Image.open(BODY_PATH))
old_body_image = np.asarray(Image.open(OLD_BODY_PATH))
if body_image.ndim == 3:
    body_image = body_image[..., 0]
if old_body_image.ndim == 3:
    old_body_image = old_body_image[..., 0]
BODY = body_image > 127
height, width = BODY.shape


def rectangle(x0, y0, x1, y1):
    result = np.zeros((height, width), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


band = rectangle(74, 199, 174, 212)
REGIONS = {
    "ordinary_road": (rectangle(200, 240, 360, 280), "x=[200,360), y=[240,280); bare road away from car"),
    "crosswalk_road": (rectangle(190, 184, 330, 203), "x=[190,330), y=[184,203); stripes and road away from car"),
    "undercar_context": (band, "x=[74,174), y=[199,212); visible GT car underside and adjacent projected shadow"),
    "undercar_body_inside": (band & BODY, "same context intersected with existing body mask; static RGB target excludes these pixels"),
    "undercar_body_outside": (band & ~BODY, "same context excluding existing body mask; includes observed undercar ground shadow, not clean edited GT"),
    "strict_lower_shadow_rim": (rectangle(80, 209, 170, 212), "x=[80,170), y=[209,212); all pixels below body mask; a narrow observed GT shadow rim"),
    "body_inside_full": (BODY, "entire existing 3306-pixel body mask; static-versus-observed-GT error is descriptive, not static target failure"),
}
GT = ARRAYS["C00@30000"]["gt"].astype(float) / 255
result = {
    "frame_id": "image_0/0017020",
    "raw_index": 20,
    "new_manifest_timestamp_model": -0.08000000000000002,
    "contract": "8-bit PNG diagnostic only; same C00 observed GT for all RGB errors; manually selected fixed GT-context ROIs; no edits to any training data",
    "limitations": [
        "Not FP32 saved-render measurements; old and new PNG exporters differ by at most one GT byte.",
        "No edited GT or genuine shadow mask; observed shadow fidelity cannot certify correct REMOVE output.",
        "One observed frame does not establish temporal persistence or causality between old S and new C00.",
        "Static RGB within the body mask is excluded from the explicit static RGB objective; its observed-GT error cannot prove a masked-out objective failed.",
    ],
    "sources": {arm: {c: str(p) for c, p in channels.items()} for arm, channels in SOURCES.items()},
    "body_masks": {"old": str(OLD_BODY_PATH), "new": str(BODY_PATH), "exact_equal": bool(np.array_equal(body_image, old_body_image)), "pixel_count": int(BODY.sum())},
    "gt_comparison": {},
    "regions": {},
}
for arm in BRANCHES:
    delta = ARRAYS[arm]["gt"].astype(int) - ARRAYS["C00@30000"]["gt"].astype(int)
    values, counts = np.unique(delta, return_counts=True)
    result["gt_comparison"][arm] = {
        "max_abs_byte_delta": int(abs(delta).max()),
        "mean_abs_byte_delta": float(abs(delta).mean()),
        "byte_delta_counts": {str(v): int(c) for v, c in zip(values, counts)},
    }

for name, (roi, description) in REGIONS.items():
    region = {
        "description": description,
        "pixels": int(roi.sum()),
        "body_inside_pixels": int((roi & BODY).sum()),
        "gt_rgb_mean": float(GT[roi].mean()),
        "arms": {},
    }
    for arm, channels in ARRAYS.items():
        static = channels["static"].astype(float) / 255
        dynamic = channels["dynamic"].astype(float) / 255
        composite = channels["composite"].astype(float) / 255
        alpha = {}
        for c in ["alpha_s", "alpha_d"]:
            values = channels[c]
            if values.ndim == 3:
                assert np.array_equal(values[..., 0], values[..., 1]) and np.array_equal(values[..., 0], values[..., 2])
                values = values[..., 0]
            alpha[c] = values.astype(float) / 255
        region["arms"][arm] = {
            "static_MAE_vs_same_GT": float(abs(static[roi] - GT[roi]).mean()),
            "static_MSE_vs_same_GT": float(((static[roi] - GT[roi]) ** 2).mean()),
            "static_rgb_mean": float(static[roi].mean()),
            "static_signed_bias": float((static[roi] - GT[roi]).mean()),
            "composite_MAE_vs_same_GT": float(abs(composite[roi] - GT[roi]).mean()),
            "dynamic_RGB_support_byte_gt_1_fraction": float((channels["dynamic"][roi].max(-1) > 1).mean()),
            "dynamic_RGB_mean": float(dynamic[roi].mean()),
            "alpha_d_mean": float(alpha["alpha_d"][roi].mean()),
            "alpha_d_gt_0point01_fraction": float((alpha["alpha_d"][roi] > .01).mean()),
            "alpha_d_gt_half_fraction": float((alpha["alpha_d"][roi] > .5).mean()),
            "alpha_s_mean": float(alpha["alpha_s"][roi].mean()),
            "alpha_s_lt_0point1_fraction": float((alpha["alpha_s"][roi] < .1).mean()),
            "composite_static_MAE": float(abs(composite[roi] - static[roi]).mean()),
        }
    result["regions"][name] = region

WORK.mkdir(parents=True, exist_ok=True)
(WORK / "raw20_pixel_diagnostic.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def contact(destination, crop=None, include_body=False):
    x0, y0, x1, y1 = crop or (0, 0, width, height)
    tile_width, tile_height = x1 - x0, y1 - y0
    gutter, top, bottom, gap = 106, 28, 28, 4
    columns = CHANNELS + (["body_mask"] if include_body else [])
    canvas = Image.new("RGB", (gutter + len(columns) * (tile_width + gap), top + len(ARRAYS) * (tile_height + bottom)), (22, 25, 30))
    draw = ImageDraw.Draw(canvas)
    for index, column in enumerate(columns):
        draw.text((gutter + index * (tile_width + gap), 8), column, fill=(240, 240, 240))
    for row_index, (arm, channels) in enumerate(SOURCES.items()):
        y = top + row_index * (tile_height + bottom)
        draw.text((5, y + 8), arm, fill=(240, 240, 240))
        for column_index, column in enumerate(columns):
            source = BODY_PATH if column == "body_mask" else channels[column]
            with Image.open(source) as original:
                panel = original.convert("RGB").crop((x0, y0, x1, y1))
                canvas.paste(panel, (gutter + column_index * (tile_width + gap), y))
        draw.text((gutter, y + tile_height + 7), f"raw20 image_0/0017020; source pixels preserved; crop xyxy={x0},{y0},{x1},{y1}", fill=(190, 195, 200))
    canvas.save(destination)


contact(WORK / "raw20_four_arm_native.png")
contact(WORK / "raw20_undercar_native.png", crop=(55, 155, 190, 230), include_body=True)
print(json.dumps({"frame_id": result["frame_id"], "GT_max_delta_bytes": 1, "body_mask_exact_equal": result["body_masks"]["exact_equal"], "outputs": [str(WORK / name) for name in ["raw20_pixel_diagnostic.json", "raw20_four_arm_native.png", "raw20_undercar_native.png"]]}, ensure_ascii=False))
