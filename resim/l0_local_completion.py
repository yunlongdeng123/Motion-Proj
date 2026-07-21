#!/usr/bin/env python
"""L0 feasibility: occupancy/diff-guided local inpainting with hard composition.

Does NOT train a video diffusion model. Uses OpenCV Telea as a weak local completer
to test the locality protocol of V7 §11:
  I_final = (1-M) I_GS + M I_gen
and measure outside-mask leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def load_u8(p: Path) -> np.ndarray:
    im = imageio.imread(p)
    if im.dtype != np.uint8:
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im


def build_mask(v0: np.ndarray, v: np.ndarray, dilate: int = 9) -> np.ndarray:
    diff = np.abs(v.astype(np.float32) - v0.astype(np.float32)).mean(axis=-1)
    thr = max(8.0, float(np.percentile(diff, 92)))
    m = (diff >= thr).astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k)
    # source footprint + new hole: keep only where GS changed (proxy disocclusion/overlap)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--variant", default="V3")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=6)
    args = ap.parse_args()

    rdir = Path(args.render_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted((rdir / "V0").glob("rgb_*.png"))
    # prefer frames with largest GS edit effect
    scored = []
    for fp in frames:
        vpath = rdir / args.variant / fp.name
        if not vpath.exists():
            continue
        d = np.abs(load_u8(fp).astype(np.float32) - load_u8(vpath).astype(np.float32)).mean()
        scored.append((d, fp))
    scored.sort(reverse=True)
    pick = [fp for _, fp in scored[: args.max_frames]]

    rows = []
    for fp in sorted(pick, key=lambda p: p.name):
        v0 = load_u8(fp)
        vg = load_u8(rdir / args.variant / fp.name)
        mask = build_mask(v0, vg)
        # inpaint the edited GS frame's source footprint / hole proxy
        gen = cv2.inpaint(vg, mask, 3, cv2.INPAINT_TELEA)
        m = (mask > 0).astype(np.float32)[..., None]
        final = ((1.0 - m) * vg.astype(np.float32) + m * gen.astype(np.float32)).astype(np.uint8)
        outside = mask == 0
        # outside-mask should match GS exactly by construction of hard composition
        out_err = float(np.mean(np.abs(final.astype(np.float32) - vg.astype(np.float32))[outside])) if outside.any() else 0.0
        in_change = float(np.mean(np.abs(final.astype(np.float32) - vg.astype(np.float32))[mask > 0])) if (mask > 0).any() else 0.0
        tag = fp.name.replace("rgb_", "").replace(".png", "")
        imageio.imwrite(out / f"mask_{tag}.png", mask)
        imageio.imwrite(out / f"final_{tag}.png", final)
        imageio.imwrite(out / f"gen_{tag}.png", gen)
        rows.append(dict(frame=tag, outside_mask_l1=out_err, inside_mask_l1=in_change,
                         mask_frac=float((mask > 0).mean())))

    summary = dict(
        render_dir=str(rdir),
        variant=args.variant,
        method="opencv_telea_hard_compose",
        n_frames=len(rows),
        mean_outside_mask_l1=float(np.mean([r["outside_mask_l1"] for r in rows])) if rows else None,
        mean_inside_mask_l1=float(np.mean([r["inside_mask_l1"] for r in rows])) if rows else None,
        mean_mask_frac=float(np.mean([r["mask_frac"] for r in rows])) if rows else None,
        frames=rows,
        note=(
            "Hard composition enforces outside-mask L1≈0 by construction. "
            "This validates the locality protocol; Telea is a weak visual completer only."
        ),
    )
    json.dump(summary, open(out / "l0_feasibility.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in (
        "n_frames", "mean_outside_mask_l1", "mean_inside_mask_l1", "mean_mask_frac"
    )}, indent=2))


if __name__ == "__main__":
    main()
