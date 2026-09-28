"""固定每例prompt的一帧组件对照，供助手粗分类，不增加抽帧。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from encode_review import box_image, mask_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    geometry = json.loads((root / "clip_geometry.json").read_text())["clips"]
    report = json.loads((root / "review/review_manifest.json").read_text())
    specs = {r["clip_id"]: r for r in report["clips"]}
    folder = root / "review/diagnostics"
    folder.mkdir(exist_ok=True)
    sheets = []
    for n, clip in enumerate(geometry):
        cid = clip["clip_id"]
        i = clip["prompt_frame"]
        frame = clip["frames"][i]
        src = root / "clips" / cid
        rgb = np.array(Image.open(src / "rgb" / f"{i:05d}.jpg").convert("RGB"))
        model, write, protect = [np.array(Image.open(src / d / f"{i:05d}.png")) > 0
                                  for d in ["model_mask", "write_mask", "protect"]]
        original = box_image(rgb, frame, cid, clip["instance_token"])
        arrays = [original, mask_image(rgb, model, write, protect),
                  np.array(Image.open(src / "native" / f"{i:05d}.png")),
                  np.array(Image.open(src / "background" / f"{i:05d}.png"))]
        x0, y0, x1, y1 = frame["target"]["box_xyxy"]
        w = min(1024, max(256, (x1 - x0) * 2.2, (y1 - y0) * 2.2 * 16 / 9))
        h = min(576, w * 9 / 16)
        left = min(max(0, (x0 + x1 - w) / 2), 1024 - w)
        top = min(max(0, (y0 + y1 - h) / 2), 576 - h)
        region = tuple(map(int, [left, top, left + w, top + h]))
        panel = Image.new("RGB", (1600, 255), (15, 24, 34))
        draw = ImageDraw.Draw(panel)
        for j, (arr, label) in enumerate(zip(arrays, ["ORIGINAL / target", "MASK / amber=write", "NATIVE deletion", "FINAL deletion"])):
            panel.paste(Image.fromarray(arr).crop(region).resize((400, 225), Image.Resampling.BICUBIC), (j * 400, 30))
            draw.text((j * 400 + 8, 8), f"{cid} f{i:02d} | {label}", fill=(255, 224, 75))
        panel.save(root / "review/assets" / cid / "one_frame_components.jpg", quality=94)
        if n % 5 == 0:
            sheet = Image.new("RGB", (1600, 1275), (15, 24, 34))
        sheet.paste(panel, (0, (n % 5) * 255))
        if n % 5 == 4 or n == len(geometry) - 1:
            path = folder / f"review_{n//5+1:02d}.jpg"
            sheet.save(path, quality=94)
            sheets.append(str(path))
    print("REVIEW_SHEETS", len(sheets))


if __name__ == "__main__":
    main()
