"""从真实编辑summary和FP32生成全49帧原生全图/共同ROI接触页，不创建HTML。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FULL_COLUMNS = (("gt", "Observed GT"), ("before", "Before: 300 steps"), ("after", "After edit"),
                ("before_vehicles", "Raw vehicles before"), ("after_vehicles", "Raw vehicles after"))
ROI_COLUMNS = (("gt", "Observed GT"), ("before", "Before: 300 steps"), ("after", "After edit"),
               ("before_target_visible", "Target visible before: 0..1"),
               ("after_target_visible", "Target visible after: 0..1"), ("target_mask", "Real original visible mask"))


def font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def image_pixels(value):
    value = np.asarray(value)
    if value.ndim == 2:
        pixels = np.rint(value.clip(0, 1)*255).astype(np.uint8)
        return Image.fromarray(pixels, "L").convert("RGB")
    if value.shape[0] == 1:
        return image_pixels(value[0])
    return Image.fromarray(np.rint(value.clip(0, 1).transpose(1, 2, 0)*255).astype(np.uint8), "RGB")


def selected_frames(operation, diagnostic_only):
    if not diagnostic_only:
        return list(range(49))
    cases = operation["bad_cases"]
    chosen = [0, 10, 24, 48]
    for key in ("largest_before_mask_MSE", "largest_before_outside_contribution", "largest_edit_RGB_change", "empty_body_frames"):
        if cases[key]:
            chosen.append(cases[key][0])
    return sorted(set(chosen))


def short(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_page(rows, operation, edit_root, prepared, mode, target_index, name, page_index, output, padding):
    columns = FULL_COLUMNS if mode == "full" else ROI_COLUMNS
    loaded = []
    for row in rows:
        path = edit_root / row["asset_relative_path"]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key, _ in columns if key != "target_mask"}
        h, w = arrays["gt"].shape[1:]
        with Image.open(prepared / "instance_masks" / (row["frame_id"]+".png")) as image:
            mask = np.array(image.resize((w, h), Image.Resampling.NEAREST)) == target_index+1
        if int(mask.sum()) != row["target_body_pixels"]:
            raise ValueError("实际mask像素与已验证summary不符: " + row["frame_id"])
        arrays["target_mask"] = mask.astype(np.float32)
        box = row["native_bbox_padded"] if mode == "roi" and row["native_bbox_padded"] is not None else [0, 0, w, h]
        loaded.append((row, arrays, box))
    # 小/远处车辆仍保持1:1像素；只给列标题留空白，不放大源ROI。
    tile_w = max(220, max(box[2]-box[0] for _, _, box in loaded))
    tile_h = max(box[3]-box[1] for _, _, box in loaded)
    margin, gap, title_h, caption_h = 20, 12, 120, 39
    width = 2*margin+len(columns)*tile_w+(len(columns)-1)*gap
    height = title_h+len(rows)*(tile_h+caption_h+gap)+margin
    canvas = Image.new("RGB", (width, height), "#101722")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = font(21), font(14), font(12)
    label = "Native full view" if mode == "full" else f"Native target bbox + {padding}px; shared before/after crop"
    draw.text((margin, 10), f"GMC engineering edit smoke | {name} | {label} | page {page_index+1}", fill="#e8edf5", font=title_font)
    draw.text((margin, 40), "300-step reconstruction is not a converged baseline. No edited GT or photoreal quality verdict.", fill="#e0b77a", font=body_font)
    # 简单模块/箭头：真实输入、检查组件和输出，避免把照片接触页当作质量认证。
    diagram_y, diagram_w = 65, min(300, (width-2*margin-100)//3)
    for i, text in enumerate(("Saved FP32 + reviewed masks", "49-frame pairing / contribution checks", "Native full view + common ROI")):
        x = margin+i*(diagram_w+35)
        draw.rectangle((x, diagram_y, x+diagram_w, diagram_y+28), fill="#1d2b3e", outline="#647a99")
        draw.text((x+7, diagram_y+6), text, fill="#c7d8ef", font=small_font)
        if i < 2:
            draw.text((x+diagram_w+6, diagram_y+5), "->", fill="#9db3d0", font=body_font)
    for column, (_, title) in enumerate(columns):
        draw.text((margin+column*(tile_w+gap), 102), title, fill="#d8e0ed", font=body_font)
    records = []
    for row_index, (row, arrays, box) in enumerate(loaded):
        y = title_h+row_index*(tile_h+caption_h+gap)
        draw.text((margin, y), f"{row['frame']:02d}  {row['frame_id']}  body={row['target_body_pixels']} px  visible(in/out)={short(row['before_visible_body_mean'],3)}/{short(row['before_visible_outside_mean'])}  edit RGB MAE={short(row['edit_full_RGB_MAE'])}", fill="#c4cfde", font=small_font)
        draw.text((margin, y+18), f"native crop xyxy={box}; empty real mask uses full view; no resizing or contrast stretch", fill="#869bb4", font=small_font)
        for column, (key, _) in enumerate(columns):
            image = image_pixels(arrays[key]).crop(box)
            x = margin+column*(tile_w+gap)+(tile_w-image.width)//2
            image_y = y+caption_h+(tile_h-image.height)//2
            canvas.paste(image, (x, image_y))
        records.append({"frame": row["frame"], "frame_id": row["frame_id"], "native_bbox": box,
                        "source_npz": str(edit_root / row["asset_relative_path"]), "resized": False})
    canvas.save(output, optimize=True)
    return {"path": str(output.resolve()), "relative_path": output.name, "mode": mode,
            "native_pixel_scale": 1, "canvas_width": width, "canvas_height": height, "frames": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--edit-root", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--frames-per-page", type=int, default=7)
    parser.add_argument("--diagnostic-only", action="store_true", help="仅首/锚点/中/末帧及实际最差指标帧；默认覆盖全部49帧")
    args = parser.parse_args()
    if not 1 <= args.frames_per_page <= 10:
        parser.error("frames-per-page需1..10")
    summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    if not summary["engineering_contract"]["passed"]:
        raise ValueError("需先通过实际FP32 collector合同")
    if args.output_dir.exists():
        raise FileExistsError("不覆盖旧contacts: " + str(args.output_dir))
    edit_root = (args.edit_root or Path(summary["source"]["edit_root"])).resolve()
    prepared = (args.prepared or Path(summary["source"]["prepared"])).resolve()
    args.output_dir.mkdir(parents=True)
    contacts = []
    for name, operation in summary["operations"].items():
        indices = selected_frames(operation, args.diagnostic_only)
        for offset in range(0, len(indices), args.frames_per_page):
            page_index = offset//args.frames_per_page
            rows = [operation["frames"][index] for index in indices[offset:offset+args.frames_per_page]]
            for mode in ("full", "roi"):
                output = args.output_dir / f"{name}_{mode}_{page_index:02d}.png"
                record = render_page(rows, operation, edit_root, prepared, mode, operation["actor_index"], name, page_index, output, summary["scope"]["roi_padding_pixels"])
                contacts.append(dict(record, operation=name))
                print(json.dumps({"written": output.name, "frames": [row["frame"] for row in rows]}), flush=True)
    receipt = {"schema": "drivingeditor_GMC_native_edit_contacts_v1", "source_summary": str(args.summary.resolve()),
               "all49_covered": not args.diagnostic_only, "GPU_used": False, "image_generation_used": False,
               "FP32_untouched": True, "photoreal_quality_verdict": None, "contact_pages": contacts}
    with (args.output_dir / "contacts.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
