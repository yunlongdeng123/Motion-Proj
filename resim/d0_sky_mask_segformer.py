#!/usr/bin/env python
"""Generate sky_masks for DriveStudio-processed scenes using modern SegFormer.

This is the V7 section 5.5 alternative mask path: official mmseg/SegFormer stack is
not installed (incompatible with adapted torch 2.1.2). We use
`nvidia/segformer-b5-finetuned-cityscapes-1024-1024` via transformers, versioned here.

Cityscapes class 10 = sky. Output: PNG uint8 {0,255} matching DriveStudio loader
expectation under `<scene>/sky_masks/{t:03d}_{cam}.png`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


SKY_CLASS = 10
MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scene_ids", type=int, nargs="+", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2],
                    help="camera indices to process (default front-3)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ignore_existing", action="store_true")
    args = ap.parse_args()

    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(args.device)
    model.eval()

    for sid in args.scene_ids:
        scene = Path(args.data_root) / f"{sid:03d}"
        img_dir = scene / "images"
        out_dir = scene / "sky_masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(img_dir.glob("*.jpg"))
        # filter by camera id suffix _{cam}.jpg
        cam_set = set(args.cameras)
        files = [f for f in files if int(f.stem.split("_")[-1]) in cam_set]
        print(f"scene {sid:03d}: {len(files)} images")
        batch_imgs, batch_paths, batch_sizes = [], [], []
        for fpath in tqdm(files, desc=f"sky[{sid:03d}]"):
            out = out_dir / f"{fpath.stem}.png"
            if args.ignore_existing and out.exists():
                continue
            img = Image.open(fpath).convert("RGB")
            batch_imgs.append(img)
            batch_paths.append(out)
            batch_sizes.append(img.size)  # (W,H)
            if len(batch_imgs) < args.batch:
                continue
            _flush(processor, model, batch_imgs, batch_paths, batch_sizes, args.device)
            batch_imgs, batch_paths, batch_sizes = [], [], []
        if batch_imgs:
            _flush(processor, model, batch_imgs, batch_paths, batch_sizes, args.device)


@torch.no_grad()
def _flush(processor, model, imgs, paths, sizes, device):
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits  # (B, C, h, w)
    up = torch.nn.functional.interpolate(
        logits, size=sizes[0][::-1], mode="bilinear", align_corners=False
    )
    # sizes may differ; handle per-image if needed
    if len(set(sizes)) == 1:
        pred = up.argmax(1).cpu().numpy()
        for p, path in zip(pred, paths):
            sky = (p == SKY_CLASS).astype(np.uint8) * 255
            imageio.imwrite(path, sky)
    else:
        for i, (path, (W, H)) in enumerate(zip(paths, sizes)):
            up_i = torch.nn.functional.interpolate(
                logits[i : i + 1], size=(H, W), mode="bilinear", align_corners=False
            )
            p = up_i.argmax(1)[0].cpu().numpy()
            sky = (p == SKY_CLASS).astype(np.uint8) * 255
            imageio.imwrite(path, sky)


if __name__ == "__main__":
    main()
