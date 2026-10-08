"""Prepare ten R001 local imagegen evidence packs from observed CPU sources.

The projected O/N/U grid is already in query image coordinates.  The B hull is
drawn only when that query frame has an actual B actor annotation in case.json.
No target image, generated image, or model output is read by this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


DELETE_A = "18571d5c431a48eea70624ab9ee09153"
COLORS = np.array([(61, 211, 91), (60, 161, 237), (128, 133, 143)], dtype=np.float32)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def interval(low: int, high: int, extent: int, limit: int) -> tuple[int, int]:
    extent = min(limit, extent)
    start = round((low + high - extent) / 2)
    start = max(0, min(limit - extent, start))
    return start, start + extent


def bbox(binary: np.ndarray) -> list[int]:
    yy, xx = np.where(binary)
    if not len(xx):
        raise ValueError("Empty edit mask")
    return [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]


def source_context(source: Image.Image, box: list[int]) -> Image.Image:
    """Source-frame context only; no temporal warp into the query frame."""
    source = source.resize((1024, 576), Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = box
    cx0, cx1 = interval(x0, x1, max(256, round((x1 - x0) * 2.5)), 1024)
    cy0, cy1 = interval(y0, y1, max(192, round((y1 - y0) * 2.5)), 576)
    view = source.crop((cx0, cy0, cx1, cy1))
    ImageDraw.Draw(view).rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0),
                                   outline=(255, 180, 55), width=3)
    view.thumbnail((256, 256), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (256, 256), (128, 128, 128))
    panel.paste(view, ((256 - view.width) // 2, (256 - view.height) // 2))
    return panel


def make_reference(crop: np.ndarray, source: Image.Image, source_box: list[int]) -> Image.Image:
    panel = Image.new("RGB", (512, 384), (90, 90, 90))
    panel.paste(Image.fromarray(crop.astype("uint8")), (0, 64))
    panel.paste(source_context(source, source_box), (256, 64))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 13), "B: DARK CAR / SAME TRACK / SOURCE TIME", fill="white")
    draw.text((8, 337), "LEFT: B CROP     RIGHT: SOURCE CONTEXT", fill="white")
    draw.text((8, 356), "NOT ALIGNED TO QUERY FRAME", fill="white")
    return panel


def local_path(local_output: str | None, index: int, name: str) -> str | None:
    if local_output is None:
        return None
    return str(PureWindowsPath(local_output) / f"f{index:02d}" / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r50-case", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-output", help="Absolute Windows destination for imagegen task manifest")
    args = parser.parse_args()

    case = json.loads((args.r50_case / "case.json").read_text(encoding="utf-8"))
    refs = json.loads(args.references.read_text(encoding="utf-8"))["references"]
    if case["case_id"] != "R001" or case["instance_token"] != DELETE_A:
        raise ValueError("Unexpected case or DELETE_A identity")
    if len(case["frames"]) != 10:
        raise ValueError("R001 must contain ten actual query frames")
    protected = {row["instance_token"] for row in case["behind"]}
    eligible = [r for r in refs if r["role"] == "protected_actor_appearance"
                and r["token"] in protected and not r["padding"]
                and r.get("GPU_source_mask_check_pass") is True]
    if not eligible:
        raise ValueError("No validated same-track PROTECTED_B source")
    ref = max(eligible, key=lambda row: row["quality"])
    b_token, slot = ref["token"], ref["reference_slot"]
    if slot != 0 or ref["source_kind"] != "GT_target_envelope_source_exclusion":
        raise ValueError("Expected validated reference slot 0 and source-exclusion policy")
    if str(args.reference_source) != ref["source_path"]:
        raise ValueError("Reference RGB path differs from validated reference record")
    source_frame = Image.open(args.reference_source).convert("RGB")
    source_box = ref["letterbox"]["roi_xyxy"]

    with np.load(args.condition) as data:
        geometry = data["geometry"][:, :3].copy()
        hole = data["hole"].copy()
        crop = data["references"][slot].copy()
        valid = data["reference_valid"][slot].astype(bool)
    if geometry.shape != (10, 3, 144, 256) or not np.allclose(geometry.sum(1), 1):
        raise ValueError("Expected ten projected one-hot O/N/U frames")
    if crop.shape != (256, 256, 3) or valid.shape != (256, 256) or not valid.any():
        raise ValueError("Same-track reference crop is empty")
    reference_panel = make_reference(crop, source_frame, source_box)

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing batch: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.condition, args.output / "condition.npz")
    tasks = []
    for index, frame in enumerate(case["frames"]):
        if frame["frame"] != index:
            raise ValueError("Frame index mismatch")
        folder = args.output / f"f{index:02d}"
        folder.mkdir()
        rgb_path = args.r50_case / "rgb" / f"{index:05d}.jpg"
        mask_path = args.r50_case / "model_mask" / f"{index:05d}.png"
        rgb = Image.open(rgb_path).convert("RGB")
        mask_gray = Image.open(mask_path).convert("L")
        mask = np.asarray(mask_gray) > 0
        if rgb.size != (1024, 576) or mask.shape != (576, 1024):
            raise ValueError(f"Unexpected RGB or mask size for f{index:02d}")
        mask_box = bbox(mask)
        width = max(480, min(600, round((mask_box[2] - mask_box[0]) * 2.5)))
        height = max(384, min(480, round((mask_box[3] - mask_box[1]) * 2.5)))
        rx0, rx1 = interval(mask_box[0], mask_box[2], width, rgb.width)
        ry0, ry1 = interval(mask_box[1], mask_box[3], height, rgb.height)
        roi_box = [rx0, ry0, rx1, ry1]
        roi = rgb.crop(roi_box)
        edit_mask = mask_gray.crop(roi_box).point(lambda value: 255 if value > 0 else 0)
        roi.save(folder / "A_original_RGB_ROI.png")
        edit_mask.save(folder / "editmask_binary.png")
        shutil.copyfile(rgb_path, folder / "query_full.jpg")
        shutil.copyfile(mask_path, folder / "model_mask_full.png")

        classes = np.argmax(geometry[index], axis=0).astype("uint8")
        classes = np.asarray(Image.fromarray(classes).resize(rgb.size, Image.Resampling.NEAREST))
        roi_classes = classes[ry0:ry1, rx0:rx1]
        gray = np.asarray(ImageOps.grayscale(roi).convert("RGB"), dtype=np.float32)
        control = Image.fromarray(np.rint(gray * 0.35 + COLORS[roi_classes] * 0.65).astype("uint8"))
        edge = ImageChops.difference(edit_mask.filter(ImageFilter.MaxFilter(3)),
                                     edit_mask.filter(ImageFilter.MinFilter(3)))
        control.paste((238, 62, 55), mask=edge)

        actors_a = [actor for actor in frame["actors"] if actor["instance_token"] == DELETE_A]
        actors_b = [actor for actor in frame["actors"] if actor["instance_token"] == b_token]
        if len(actors_a) != 1 or len(actors_b) > 1:
            raise ValueError(f"Unexpected actor metadata for f{index:02d}")
        b_actor = actors_b[0] if actors_b else None
        b_hull_full = b_actor["hull"] if b_actor else None
        b_hull_roi = [[int(x - rx0), int(y - ry0)] for x, y in b_hull_full] if b_hull_full else None
        if b_hull_roi:
            points = [tuple(point) for point in b_hull_roi]
            if not all(0 <= x < roi.width and 0 <= y < roi.height for x, y in points):
                raise ValueError(f"Annotated B hull outside ROI in f{index:02d}")
            draw = ImageDraw.Draw(control)
            draw.line(points + [points[0]], fill=(255, 217, 51), width=2)
            draw.text((points[0][0] + 4, points[0][1] - 12), "B", fill=(255, 217, 51))
        control.save(folder / "B_editmask_ONU_B_control.png")
        reference_panel.save(folder / "C_protected_B_same_track.png")

        roi_edit = mask[ry0:ry1, rx0:rx1]
        fractions = {name: float((roi_classes[roi_edit] == number).mean())
                     for number, name in enumerate(("O", "N", "U"))}
        delta = (ref["timestamp"] - frame["timestamp"]) / 1_000_000
        roi_record = {
            "case_id": "R001", "frame_index": index, "query_timestamp": frame["timestamp"],
            "full_image_wh": [1024, 576], "mask_bbox_full_xyxy_exclusive": mask_box,
            "roi_full_xyxy_exclusive": roi_box, "roi_wh": list(roi.size),
            "full_to_roi_xy": [-rx0, -ry0], "roi_to_full_xy": [rx0, ry0],
            "scale": 1,
            "mask_bbox_roi_xyxy_exclusive": [mask_box[0]-rx0, mask_box[1]-ry0,
                                              mask_box[2]-rx0, mask_box[3]-ry0],
            "mask_pixels_full": int(mask.sum()), "mask_pixels_roi": int(roi_edit.sum()),
            "mask_expansion_px": 0, "geometry_grid_wh": [256, 144],
            "geometry_frame": index, "geometry_already_projected_to_RGB": True,
            "condition_hole_grid_pixels": int(hole[index].sum()),
            "ONU_fraction_inside_editmask": fractions,
            "roles": {"DELETE_A": DELETE_A, "PROTECTED_B": b_token},
            "DELETE_A_hull_full": actors_a[0]["hull"],
            "B_proxy_hull_full": b_hull_full, "B_proxy_hull_roi": b_hull_roi,
            "B_depth_m": b_actor["depth"] if b_actor else None,
            "B_hull_status": "observed_actor_metadata" if b_actor else "absent_in_query_frame_metadata",
            "uncertainty": {"O": "retained GT cuboid projection proxy; not a visible silhouette",
                            "N": "measured LiDAR background support; not complete background",
                            "U": "unknown; never infer background from mask minus O",
                            "reference_alignment": "same track, different time; no verified query warp"},
            "source": {"query_rgb": str(rgb_path), "original_model_mask": str(mask_path),
                       "condition": str(args.condition), "case": str(args.r50_case / "case.json"),
                       "reference_manifest": str(args.references), "reference_source_rgb": str(args.reference_source),
                       "query_rgb_in_pack": "query_full.jpg", "original_model_mask_in_pack": "model_mask_full.png"},
        }
        write_json(folder / "roi.json", roi_record)
        if b_hull_roi:
            xs = [point[0] for point in b_hull_roi]
            ys = [point[1] for point in b_hull_roi]
            b_instruction = (f"黄色小多边形才是当前 B 的真实标注 3D 框投影代理，ROI 内约 "
                             f"x={min(xs)}..{max(xs)}, y={min(ys)}..{max(ys)}，深度约 {b_actor['depth']:.1f} 米；"
                             "不是精确可见轮廓，不得扩大成近车。")
        else:
            b_instruction = ("当前帧元数据没有 B 的标注 hull；图 B 不画 B 位置。"
                             "不能把其他帧 B 坐标或绿色 O 直接当成本帧 B，证据不足时不凭空造车。")
        prompt = (
            f"R001 f{index:02d} 局部 DELETE 补景。只输出图 A 同一取景的 RGB ROI，尺寸 "
            f"{roi.width}×{roi.height}（可生成 2 倍后按原尺寸缩回）；不输出拼图、边框、文字或控制颜色。\n"
            f"图 A 是当前真实相机帧。删除近处银色 SUV A（实例 {DELETE_A}）及其残影。"
            f"独立 editmask_binary.png 是唯一可写区域；图 B 红线是边界，ROI 内 mask bbox "
            f"x={mask_box[0]-rx0}..{mask_box[2]-rx0-1}, y={mask_box[1]-ry0}..{mask_box[3]-ry0-1}。"
            "mask 外保持图 A 原像素，工程端按独立 mask 硬合成。\n"
            f"保护 B（实例 {b_token}）是图 C 左侧同 track 深色小车；图 C 右侧浅色邻车只是来源上下文。"
            f"参考来自当前时刻 {delta:+.2f} 秒，未可靠对齐，不可复制其像素位置或整排车辆。"
            f"{b_instruction}\n"
            "图 B 与 A 对齐：绿色 O 是多个保留物的 3D 框投影包络代理，蓝色 N 是实测 LiDAR 背景支持，"
            "灰色 U 未知；O 不是可见轮廓，N 以外和 U 都不能判成背景。"
            "图 B 的彩色标记、黄色 hull、文字均不得进入 RGB 输出。\n"
            "优先当前 mask 外真实像素，其次同 track 外观，再次当前投影几何。"
            "只保守恢复有证据的 B 可见部分，保持远近、颜色、姿态、遮挡与相邻真实车一致；"
            "不补造完整车身、具体灯组或新车。其余缺口沿真实道路/标线/建筑边界连续恢复，"
            "保持原车载相机的曝光、模糊、透视和纹理，不美化。输出只是待验收伪标签候选。\n"
        )
        (folder / "prompt_zh.txt").write_text(prompt, encoding="utf-8")
        file_names = ["A_original_RGB_ROI.png", "B_editmask_ONU_B_control.png",
                      "C_protected_B_same_track.png"]
        task = {"task_id": f"R001-f{index:02d}", "frame_index": index,
                "input_images_in_order": [local_path(args.local_output, index, name) or str(folder / name)
                                          for name in file_names],
                "separate_editmask": local_path(args.local_output, index, "editmask_binary.png")
                                     or str(folder / "editmask_binary.png"),
                "prompt_file": local_path(args.local_output, index, "prompt_zh.txt")
                               or str(folder / "prompt_zh.txt"),
                "roi_file": local_path(args.local_output, index, "roi.json")
                            or str(folder / "roi.json"),
                "query_full": local_path(args.local_output, index, "query_full.jpg")
                              or str(folder / "query_full.jpg"),
                "model_mask_full": local_path(args.local_output, index, "model_mask_full.png")
                                   or str(folder / "model_mask_full.png"),
                "output_expected": "RGB ROI only", "generation_output_wh_requested": [roi.width * 2, roi.height * 2],
                "resize_to_roi_wh_before_hard_compose": list(roi.size),
                "B_hull_status": roi_record["B_hull_status"], "B_hull_roi": b_hull_roi,
                "reference_time_delta_seconds": delta, "model_calls": 0}
        write_json(folder / "manifest.json", task)
        tasks.append(task)
        print(json.dumps({"frame": index, "roi": roi_box, "B_hull": b_hull_roi is not None,
                          "mask_pixels": int(mask.sum())}, ensure_ascii=False))

    write_json(args.output / "batch_manifest.json", {
        "batch_id": "R001-10frame-imagegen-local-ROI", "case_id": "R001",
        "scope": "one real training case; ten query times f00..f09",
        "model_calls": 0, "no_generated_or_Y_evidence": True,
        "shared_condition": str(PureWindowsPath(args.local_output) / "condition.npz")
                            if args.local_output else str(args.output / "condition.npz"),
        "reference": {"slot": slot, "token": b_token, "quality": ref["quality"],
                      "source_kind": ref["source_kind"], "source_path": ref["source_path"],
                      "source_roi_xyxy_at_1024x576": source_box,
                      "valid_fraction": float(valid.mean()), "GPU_source_mask_check_pass": True,
                      "aligned_to_query": False},
        "tasks": tasks,
    })
    (args.output / "README.md").write_text(
        "# R001 十帧 imagegen 局部输入包\n\n"
        "```text\n当前 RGB + 原始 mask → 局部 A ─────────────┐\n"
        "逐帧投影 O/N/U + 当前 B hull → 控制 B ──┼→ 局部 RGB 候选 → mask 内硬合成\n"
        "同 track 真实来源帧 → 外观参考 C ────────┘\n```\n\n"
        "每帧按 manifest 顺序提供 A/B/C，另附独立二值 editmask。"
        "f00–f03 无当前 B 的标注 hull，控制图不伪造其位置。"
        "全部都是补景输入，没有生成结果或隐藏真值。\n", encoding="utf-8")


if __name__ == "__main__":
    main()
