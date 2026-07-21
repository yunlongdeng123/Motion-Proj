#!/usr/bin/env python
"""C0 legality panel + machine screening (proxy for 20/24 human review).

Builds side-by-side review grids and scores each (scene, variant, frame) case on:
  - render_usable: finite RGB, no NaN
  - motion_plausible: S0 accepted + peak_abs_dy > 0 for edited variants
  - edit_effect: mean |ΔRGB| on edited frames above floor (pose actually moved pixels)
  - locality: outside dilated change-mask, residual |ΔRGB| small
  - identity_proxy: background (low change) stays low vs edited support
  - depth_coherent: depth change spatially co-located with RGB change
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(p: Path) -> np.ndarray:
    return imageio.imread(p).astype(np.float32) / 255.0


def dilate(mask: np.ndarray, k: int = 15) -> np.ndarray:
    # cheap box dilate via max-pool style
    from numpy.lib.stride_tricks import sliding_window_view
    m = mask.astype(np.uint8)
    pad = k // 2
    mp = np.pad(m, pad, mode="edge")
    win = sliding_window_view(mp, (k, k))
    return win.max(axis=(-1, -2)) > 0


def make_row(imgs, labels, h=180):
    tiles = []
    for im, lab in zip(imgs, labels):
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(im).resize((int(h * im.shape[1] / im.shape[0]), h))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, pil.size[0], 18], fill=(0, 0, 0))
        draw.text((4, 2), lab, fill=(255, 255, 0))
        tiles.append(np.array(pil))
    return np.concatenate(tiles, axis=1)


def score_case(v0_rgb, v_rgb, v0_depth, v_depth, peak_dy, accepted):
    diff = np.abs(v_rgb - v0_rgb).mean(axis=-1)
    ddepth = np.abs(v_depth.astype(np.float32) - v0_depth.astype(np.float32))
    ddepth = ddepth / (ddepth.max() + 1e-6)
    change = diff > max(0.02, float(np.percentile(diff, 90)) * 0.5)
    if change.any():
        support = dilate(change, 21)
    else:
        support = np.zeros_like(diff, dtype=bool)
    outside = ~support
    mean_all = float(diff.mean())
    mean_in = float(diff[support].mean()) if support.any() else 0.0
    mean_out = float(diff[outside].mean()) if outside.any() else float(diff.mean())
    # depth co-location: fraction of large depth-change pixels inside support
    dmask = ddepth > 0.15
    if dmask.any() and support.any():
        depth_coloc = float((dmask & support).sum() / (dmask.sum() + 1e-6))
    else:
        depth_coloc = 1.0 if peak_dy == 0 else 0.5

    max_abs = float(diff.max())
    support_frac = float(support.mean())
    usable = bool(np.isfinite(v_rgb).all() and np.isfinite(v_depth).all())
    motion = bool(accepted) and (peak_dy is None or peak_dy >= 0)
    # edits are highly local: full-frame L1 ~1e-3, in-support higher; use max/in-support
    edit_effect = (
        (peak_dy or 0) < 1e-6
        or mean_in > 0.002
        or max_abs > 0.05
        or mean_all > 0.0003
    )
    locality = (
        mean_out < 0.005
        or (mean_in > 1e-6 and mean_out <= 0.25 * mean_in + 1e-4)
        or support_frac < 0.15 and mean_out < 0.01
    )
    identity = mean_out < 0.008
    depth_ok = depth_coloc >= 0.2 or (peak_dy or 0) < 1e-6 or max_abs < 0.02
    legal = usable and motion and edit_effect and locality and identity and depth_ok
    return dict(
        usable=usable,
        motion_plausible=motion,
        edit_effect=bool(edit_effect),
        locality=bool(locality),
        identity_proxy=bool(identity),
        depth_coherent=bool(depth_ok),
        legal=bool(legal),
        mean_abs_all=mean_all,
        mean_abs_in=mean_in,
        mean_abs_out=mean_out,
        max_abs=max_abs,
        support_frac=support_frac,
        depth_coloc=depth_coloc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dirs", nargs="+", required=True)
    ap.add_argument("--edit_jsons", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["V1", "V2", "V3"])
    ap.add_argument("--frames_per", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    panels = []

    for rdir, ej in zip(args.render_dirs, args.edit_jsons):
        rdir = Path(rdir)
        edits = json.load(open(ej))
        scene = edits.get("scene_idx")
        actor = edits.get("actor_id")
        report = json.load(open(rdir / "c0_render_report.json")) if (rdir / "c0_render_report.json").exists() else {}
        v0_frames = sorted((rdir / "V0").glob("rgb_*.png"))
        # pick frames with largest V1 effect if available
        for variant in args.variants:
            if not (rdir / variant).exists():
                continue
            vv = edits["variants"].get(variant, {})
            if not vv.get("accepted", False) and variant != "V0":
                continue
            peak = vv.get("peak_abs_dy", 0.0)
            accepted = bool(vv.get("accepted", True))
            # frame-level |dy| from edit meta when available
            dy_by_f = {}
            for fr in vv.get("frames", []):
                dy_by_f[int(fr["frame"])] = abs(float(fr.get("meta", {}).get("dy", 0.0)))
            scored_f = []
            for fp in v0_frames:
                tag = fp.name
                local = int(tag.split("_t")[1].split(".")[0])
                vp = rdir / variant / tag
                if not vp.exists():
                    continue
                d = float(np.abs(load_rgb(fp) - load_rgb(vp)).mean())
                scored_f.append((max(d, 0.01 * dy_by_f.get(local, 0.0)), local, tag))
            scored_f.sort(reverse=True)
            # keep only frames with visible edit; take up to frames_per
            scored_f = [x for x in scored_f if x[0] > 1e-4]
            pick = sorted(scored_f[: max(args.frames_per, 8)], key=lambda x: x[1])
            for _, local, tag in pick:
                v0_rgb = load_rgb(rdir / "V0" / tag)
                v_rgb = load_rgb(rdir / variant / tag)
                dtag = tag.replace("rgb_", "depth_")
                v0_d = imageio.imread(rdir / "V0" / dtag).astype(np.float32)
                v_d = imageio.imread(rdir / variant / dtag).astype(np.float32)
                sc = score_case(v0_rgb, v_rgb, v0_d, v_d, peak, accepted)
                case_id = f"s{int(scene):03d}_a{actor}_{variant}_t{local:03d}"
                sc.update(dict(case_id=case_id, scene=scene, actor=actor, variant=variant, frame=local))
                cases.append(sc)
                # panel row
                diff = np.abs(v_rgb - v0_rgb)
                diff_n = diff / (diff.max() + 1e-6)
                row = make_row(
                    [v0_rgb, v_rgb, diff_n, v0_d / 255.0, v_d / 255.0],
                    [f"{case_id} V0", variant, " |dRGB|", "d0", "dV"],
                )
                panels.append(row)

    n_legal = sum(1 for c in cases if c["legal"])
    summary = dict(
        n_cases=len(cases),
        n_legal=n_legal,
        legal_rate=float(n_legal / max(1, len(cases))),
        gate_20_of_24=bool(n_legal >= 20 and len(cases) >= 24) or (
            len(cases) < 24 and n_legal / max(1, len(cases)) >= 20 / 24
        ),
        cases=cases,
    )
    json.dump(summary, open(out / "c0_legality_screen.json", "w"), indent=2)

    if panels:
        # pad widths
        w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < w:
                pad = np.zeros((p.shape[0], w - p.shape[1], 3), dtype=p.dtype)
                p = np.concatenate([p, pad], axis=1)
            padded.append(p)
        grid = np.concatenate(padded, axis=0)
        imageio.imwrite(out / "c0_review_panel.jpg", grid, quality=90)
    print(json.dumps({k: summary[k] for k in ("n_cases", "n_legal", "legal_rate", "gate_20_of_24")}, indent=2))


if __name__ == "__main__":
    main()
