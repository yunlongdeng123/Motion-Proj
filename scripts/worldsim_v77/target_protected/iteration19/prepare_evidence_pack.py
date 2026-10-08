"""Build the R001 f05 three-image DELETE evidence pack from observed sources only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent / "evidence_pack"
SOURCE = ROOT / "source"
DELETE_A = "18571d5c431a48eea70624ab9ee09153"
ROI_FACTOR = 2.5
MIN_SIDE = 384
COLORS = {"O": [61, 211, 91], "N": [60, 161, 237], "U": [128, 133, 143]}


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def centered_interval(low: int, high: int, extent: int, limit: int) -> tuple[int, int]:
    extent = min(limit, extent)
    start = round((low + high - extent) / 2)
    start = max(0, min(limit - extent, start))
    return start, start + extent


def context_panel(source: Image.Image, box: list[int]) -> Image.Image:
    """Display source-frame context; this is not a warp into the query frame."""
    source = source.resize((1024, 576), Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = box
    cx0, cx1 = centered_interval(x0, x1, max(256, round((x1 - x0) * 2.5)), 1024)
    cy0, cy1 = centered_interval(y0, y1, max(192, round((y1 - y0) * 2.5)), 576)
    view = source.crop((cx0, cy0, cx1, cy1))
    draw = ImageDraw.Draw(view)
    draw.rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), outline=(255, 180, 55), width=3)
    view.thumbnail((256, 256), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (256, 256), (128, 128, 128))
    panel.paste(view, ((256 - view.width) // 2, (256 - view.height) // 2))
    return panel


def main() -> None:
    case = json.loads((SOURCE / "case.json").read_text(encoding="utf-8"))
    refs = json.loads((SOURCE / "references.json").read_text(encoding="utf-8"))["references"]
    if case["case_id"] != "R001" or case["instance_token"] != DELETE_A:
        raise ValueError("DELETE_A identity does not match original R001")
    protected = {row["instance_token"] for row in case["behind"]}
    eligible = [r for r in refs if r["role"] == "protected_actor_appearance"
                and r["token"] in protected and not r["padding"]
                and r.get("GPU_source_mask_check_pass") is True]
    if not eligible:
        raise ValueError("No validated same-track protected-actor reference")
    ref = max(eligible, key=lambda row: row["quality"])
    b_token, slot = ref["token"], ref["reference_slot"]
    if slot != 0:
        raise ValueError("Cached real source RGB belongs to reference slot 0")
    if b_token == DELETE_A or ref["source_kind"] != "GT_target_envelope_source_exclusion":
        raise ValueError("Reference identity/source policy would confuse DELETE_A and PROTECTED_B")

    query = Image.open(SOURCE / "query_f05.jpg").convert("RGB")
    mask_image = Image.open(SOURCE / "model_mask_f05.png").convert("L")
    mask = np.asarray(mask_image) > 0
    if query.size != (1024, 576) or mask.shape != (576, 1024):
        raise ValueError("Unexpected source RGB/mask geometry")
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Original model mask is empty")
    mask_box = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
    w = max(MIN_SIDE, round((mask_box[2] - mask_box[0]) * ROI_FACTOR))
    h = max(MIN_SIDE, round((mask_box[3] - mask_box[1]) * ROI_FACTOR))
    rx0, rx1 = centered_interval(mask_box[0], mask_box[2], w, query.width)
    ry0, ry1 = centered_interval(mask_box[1], mask_box[3], h, query.height)
    roi_box = [rx0, ry0, rx1, ry1]
    roi_rgb = query.crop(roi_box)
    roi_mask = mask_image.crop(roi_box).point(lambda value: 255 if value > 0 else 0)
    roi_rgb.save(ROOT / "A_original_RGB_ROI.png")
    roi_mask.save(ROOT / "editmask_binary.png")

    with np.load(SOURCE / "condition.npz") as data:
        geometry = data["geometry"][5, :3]
        crop = data["references"][slot].copy()
        valid = data["reference_valid"][slot].astype(bool)
        prior_hole = data["hole"][5]
    if geometry.shape != (3, 144, 256) or not np.allclose(geometry.sum(0), 1):
        raise ValueError("O/N/U control is not a one-hot projected RGB grid")
    if crop.shape != (256, 256, 3) or valid.shape != (256, 256) or not valid.any():
        raise ValueError("Selected same-track reference crop is empty")
    classes = np.argmax(geometry, axis=0).astype("uint8")
    classes = np.asarray(Image.fromarray(classes).resize(query.size, Image.Resampling.NEAREST))
    roi_classes = classes[ry0:ry1, rx0:rx1]
    palette = np.asarray([COLORS[key] for key in ("O", "N", "U")], dtype=np.float32)
    gray = np.asarray(ImageOps.grayscale(roi_rgb).convert("RGB"), dtype=np.float32)
    control = Image.fromarray(np.rint(gray * 0.35 + palette[roi_classes] * 0.65).astype("uint8"))
    edge = ImageChops.difference(roi_mask.filter(ImageFilter.MaxFilter(3)),
                                roi_mask.filter(ImageFilter.MinFilter(3)))
    control.paste((238, 62, 55), mask=edge)
    control.save(ROOT / "B_editmask_ONU_control.png")

    source_frame = Image.open(SOURCE / "reference_source_slot0.jpg").convert("RGB")
    source_box = ref["letterbox"]["roi_xyxy"]
    panel = Image.new("RGB", (512, 256), (128, 128, 128))
    panel.paste(Image.fromarray(crop), (0, 0))
    panel.paste(context_panel(source_frame, source_box), (256, 0))
    panel.save(ROOT / "C_protected_B_same_track.png")

    edit = mask[ry0:ry1, rx0:rx1]
    fractions = {key: float((roi_classes[edit] == index).mean()) for index, key in enumerate(("O", "N", "U"))}
    roi = {"case_id": "R001", "frame_index": 5, "full_image_wh": list(query.size),
           "mask_bbox_full_xyxy_exclusive": mask_box, "roi_full_xyxy_exclusive": roi_box,
           "roi_wh": list(roi_rgb.size), "full_to_roi_xy": [-rx0, -ry0], "roi_to_full_xy": [rx0, ry0],
           "scale": 1, "mask_pixels_full": int(mask.sum()), "mask_pixels_roi": int(edit.sum()),
           "mask_expansion_px": 0, "geometry_grid": [256, 144], "geometry_frame": 5,
           "geometry_already_projected_to_RGB": True, "condition_hole_grid_pixels": int(prior_hole.sum()),
           "ONU_fraction_inside_editmask": fractions,
           "source": {"query_rgb": "source/query_f05.jpg", "original_model_mask": "source/model_mask_f05.png",
                      "projected_condition": "source/condition.npz", "reference_manifest": "source/references.json",
                      "reference_source_rgb": "source/reference_source_slot0.jpg"},
           "roles": {"DELETE_A": DELETE_A, "PROTECTED_B": b_token},
           "uncertainty": {"O": "retained GT cuboid envelope proxy, not visible silhouette",
                           "N": "measured LiDAR background support, not complete background",
                           "U": "unknown; never infer background from mask minus O",
                           "reference_alignment": "same track, different time; no verified warp to query f05"}}
    dump(ROOT / "roi.json", roi)
    delta = (ref["timestamp"] - case["frames"][5]["timestamp"]) / 1_000_000
    prompt = f"""任务：只输出图A同尺寸（{roi_rgb.width}×{roi_rgb.height}）的修复后RGB ROI，不输出标注或整帧。
图A是R001 f05原始观测；红色编辑边界见图B，独立精确二值mask见editmask_binary.png。待删A是银色SUV，实例{DELETE_A}。删除A；mask外原像素保持，最终仅按独立mask硬写回。
图B与A逐像素对齐：绿色O=保留物的3D框投影包络代理；蓝色N=实测LiDAR背景点支持；灰色U=未知；红线=原编辑mask边界。O不是可见轮廓，N未覆盖的区域也不等于背景，U不得推断为无车。
需要保留的B是实例{b_token}。图C左侧深色车辆是同track的B；旁边浅色车辆只是来源上下文，不能拿来替换B。图C右为真实来源帧局部上下文，橙框仅标来源crop范围。参考来自CAM_FRONT、相对f05约{delta:+.2f}秒，未做可靠时序warp，不能把图C当已对齐像素真值。
仅在同track真实参考与当前观测支持的范围保守恢复B的可见部分，维持参考中的身份、相对大小和方向。绿色O还包含其他保留车辆的框，不能单独据O认定是B；不补造无法证实的完整车身，也不把待删A当B。未知区沿图A周围道路/车道线连续恢复，不生成新车或凭空改变场景。保持原始清晰度、曝光和透视。
"""
    (ROOT / "prompt_zh.txt").write_text(prompt, encoding="utf-8")
    dump(ROOT / "manifest.json", {"task": "R001 f05 local DELETE evidence pack",
         "input_images_in_order": ["A_original_RGB_ROI.png", "B_editmask_ONU_control.png", "C_protected_B_same_track.png"],
         "separate_editmask": "editmask_binary.png", "prompt": "prompt_zh.txt", "roi": "roi.json",
         "source_files": {"query_RGB": str(SOURCE / "query_f05.jpg"), "original_model_mask": str(SOURCE / "model_mask_f05.png"),
                          "condition": str(SOURCE / "condition.npz"), "references": str(SOURCE / "references.json"),
                          "case": str(SOURCE / "case.json"), "reference_source_RGB": str(SOURCE / "reference_source_slot0.jpg")},
         "source_reference": {"slot": slot, "token": b_token, "quality": ref["quality"],
                              "source_kind": ref["source_kind"], "source_path_remote": ref["source_path"],
                              "source_roi_xyxy_at_1024x576": source_box,
                              "valid_fraction": float(valid.mean()), "GPU_source_mask_check_pass": True,
                              "time_delta_seconds": delta, "aligned_to_query": False,
                              "visual_note": "dark same-track vehicle is B; adjacent light vehicle is context"},
         "no_generated_or_Y_evidence": True, "model_calls": 0, "output_expected": "RGB ROI only"})
    (ROOT / "README.md").write_text(
        "# R001 f05 证据包\n\n"
        "```text\n原RGB + 原mask ──→ 局部ROI A ───────────────┐\n"
        "已投影O/N/U ────→ 同ROI控制图 B ─────────┼→ 局部RGB补景 → 原mask内硬写回\n"
        "同track真实crop ──→ 来源上下文 C ─────────┘\n```\n\n"
        "图B绿色O是保留车辆cuboid投影代理，蓝色N是实测LiDAR背景支持，灰色U未知；红线仅显示原mask边界。"
        "图C左侧为已验证来源crop，右侧为真实来源上下文；没有可靠warp，不能当当前帧对齐真值。"
        "独立 `editmask_binary.png` 决定允许写回区域，三张输入图按manifest顺序提供。\n",
        encoding="utf-8")
    print(json.dumps({"roi_wh": list(roi_rgb.size), "roi_xyxy": roi_box,
                      "protected_B": b_token, "reference_slot": slot, "valid_fraction": float(valid.mean()),
                      "ONU_inside_mask": fractions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
