"""scene-065 stage C (r5): full-domain car-attributable mask, inward feather, tone-match A/B.

What changed relative to r4
---------------------------
1. **Mask coverage.**  r4 restricted the mask to `target | dark_defect`, i.e. the instance
   footprint plus a flat-black block.  Probe r5 measured what that leaves untouched: of the
   defect candidates that actually sit in the removed car's *swept footprint*, r4 covers 64%
   and misses 36% (478,106 px over 199 frames).  The mask here is built inside the swept
   footprint (`union of dilate(target_t')` over +-WINDOW frames), so the ghost that leaks
   outside the instance mask is included, and the whole thing is still clipped to `~K` so
   retained vehicles can never be written.

2. **Feather direction.**  r4 built the ramp as `distance-to-core / feather`, which is 1 at
   the *outer* boundary - a hard step by construction (measured seam ratio 1.22-1.59 on the
   frames around raw 087).  The ramp here is `distance-to-outside / feather`, so it is exactly
   0 on the mask boundary and 1 only `FEATHER` px inside.

3. **Tone matching is a switch, not a default.**  r4 force-matched median/IQR per frame.  The
   user asked for the control first: `--tone none` (default) writes the raw prediction,
   `--tone medirq` reproduces r4's behaviour, so the two can be compared instead of assuming.

4. **Acceptance is not "dark pixels went down".**  Every frame is scored on
   coverage / seam gradient ratio / patch-vs-ring statistics / temporal flicker / protection.
   `dark_px` is still logged, labelled as *not* an acceptance criterion.

Modes
-----
  --stage-only     build masks + audit + overlays, no diffusion (works without GPU)
  --compose-only   reuse an existing ProPainter raw output dir, re-composite only
  --full           stage -> run ProPainter -> composite (needs GPU)

Composite is always `final[~mask] = orig[~mask]`; the script exits non-zero if a single pixel
outside the mask changed, or if any pixel inside the retained-vehicle mask changed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

PY = "/root/autodl-tmp/envs/driveeditor/bin/python"
PROPAINTER = "/root/autodl-tmp/third_party/worldsim_v77/ProPainter-hfspace"
SRC = ("/root/autodl-tmp/drivingeditor_repro/campaign/"
       "scene065_closedloop_20261116_r1/handoff_coverage_r2")
OUT = ("/root/autodl-tmp/drivingeditor_repro/campaign/"
       "scene065_propainter_20261126_r5")

WINDOW = 20
FOOTPRINT_DILATE = 6
SWEEP_WINDOW = 8
HOLE_GATE = "near"
NEAR_BAND = 12
STATIC_AT_THR = 0.10
STATIC_DOMINANCE = 0.05
ROAD_ALPHA_HOLE = 0.90
DARK_ABS_THR = 30.0
DARK_REL_THR = 0.40
COLOR_DEV_THR = 18.0
ANCHOR_DILATE = 10        # a residue component must touch the footprint band to be edited
MASK_GROW = 2
FEATHER = 4
MASK_DILATION_ARG = 4
SUBDIVIDEO = 50
IQR_RATIO_CLIP = (0.65, 1.60)
WEBP_QUALITY = 92
EXPECTED_FRAMES = 199


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def disk(r: int) -> np.ndarray:
    return np.ones((r * 2 + 1,) * 2, np.uint8)


def frame_ids() -> list[str]:
    return [n for n in sorted(os.listdir(f"{SRC}/rendered")) if n.isdigit()]


def retained_components(seed: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Keep only 8-connected components of `seed` that intersect `anchor`."""
    n, lab, _stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[anchor & (lab > 0)])] = True
    keep[0] = False
    return keep[lab]


MASK_CORE = ("target | dark | static_ghost | residue")


def build_mask(fid: str, swept: np.ndarray, near: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    arrays = np.load(f"{SRC}/rendered/{fid}/arrays.npz")
    dom = np.load(f"{SRC}/domains/{fid}.npz")
    img = cv2.imread(f"{SRC}/rendered/{fid}/new_delete.png", cv2.IMREAD_COLOR)
    static_at = cv2.imread(f"{SRC}/rendered/{fid}/static_contribution.png",
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    road_at = cv2.imread(f"{SRC}/rendered/{fid}/road_contribution.png",
                         cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if img is None:
        raise RuntimeError(f"missing render inputs for frame {fid}")

    target = dom["target"].astype(bool)
    K = dom["K"].astype(bool)
    road_alpha = arrays["road_alpha"][0].astype(np.float32)
    alpha = arrays["alpha"][0].astype(np.float32)
    luma = img.astype(np.float32).mean(axis=2)
    med = cv2.medianBlur(img, 15).astype(np.float32)
    color_dev = np.abs(img.astype(np.float32) - med).max(axis=2)

    ring = ((cv2.dilate(target.astype(np.uint8), disk(20)) > 0)
            & ~(cv2.dilate(target.astype(np.uint8), disk(4)) > 0) & ~K)
    ring_med = float(np.median(luma[ring])) if ring.any() else 90.0
    dark_thr = min(DARK_ABS_THR, DARK_REL_THR * ring_med)

    # `hole_gate` decides how far the "no road raster support" family may reach:
    #   swept - anywhere in the swept footprint (aggressive, covers the whole vacated road)
    #   near  - only within NEAR_BAND px of the current footprint (tight)
    #   none  - not used as a defect trigger at all (only structure/colour residue)
    if p["hole_gate"] == "swept":
        hole = swept & (road_alpha < ROAD_ALPHA_HOLE)
    elif p["hole_gate"] == "near":
        hole = near & (road_alpha < ROAD_ALPHA_HOLE)
    else:
        hole = np.zeros_like(target)

    static_dom = (static_at > STATIC_AT_THR) & (static_at > road_at + STATIC_DOMINANCE)
    static_ghost = swept & static_dom & ((color_dev > p["color_dev"]) | (luma < dark_thr))
    dark = swept & (luma < dark_thr)
    residue = swept & (color_dev > p["color_dev"]) & ~static_dom & (alpha < 0.95)
    seed = target | dark | static_ghost | residue | hole
    anchor = cv2.dilate(target.astype(np.uint8), disk(ANCHOR_DILATE)) > 0
    seed = retained_components(seed, anchor)
    mask = (cv2.dilate(seed.astype(np.uint8), disk(MASK_GROW)) > 0) & ~K
    mask &= cv2.dilate(swept.astype(np.uint8), disk(MASK_GROW)) > 0
    mask |= target & ~K

    stats = {
        "frame": fid,
        "mask_px": int(mask.sum()),
        "target_px": int(target.sum()),
        "static_ghost_px": int((static_ghost & mask).sum()),
        "hole_px": int((hole & mask).sum()),
        "dark_px": int((dark & mask).sum()),
        "residue_px": int((residue & mask).sum()),
        "ghost_outside_target_px": int((mask & ~target).sum()),
        "ring_median_luma": round(ring_med, 2),
        "dark_threshold": round(dark_thr, 2),
        "mask_in_K_px": int((mask & K).sum()),
        "static_dom_in_swept_px": int((static_dom & swept).sum()),
        "static_dom_covered_px": int((static_dom & swept & mask).sum()),
    }
    return mask, stats


def swept_map(ids: list[str], targets: dict[str, np.ndarray], Ks: dict[str, np.ndarray],
              window: int) -> dict[str, np.ndarray]:
    out = {}
    for idx, fid in enumerate(ids):
        lo, hi = max(0, idx - window), min(len(ids) - 1, idx + window)
        acc = np.zeros_like(targets[fid])
        for j in range(lo, hi + 1):
            acc |= targets[ids[j]]
        out[fid] = (cv2.dilate(acc, disk(FOOTPRINT_DILATE)) > 0) & ~Ks[fid]
    return out


def near_map(ids: list[str], targets: dict[str, np.ndarray], Ks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {fid: (cv2.dilate(targets[fid], disk(NEAR_BAND)) > 0) & ~Ks[fid] for fid in ids}


def feather_ramp(mask: np.ndarray) -> np.ndarray:
    """1 deep inside the mask, exactly 0 on its boundary.  r4 had this backwards."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip((dt - 1.0) / max(FEATHER, 1), 0.0, 1.0).astype(np.float32)


def tone_medirq(orig: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    outer = cv2.dilate(mask.astype(np.uint8), disk(18)) > 0
    ring = outer & ~(cv2.dilate(mask.astype(np.uint8), disk(3)) > 0)
    if ring.sum() < 150 or not mask.any():
        return pred, {"applied": False}
    ref = orig[ring].astype(np.float32)
    src = pred[mask].astype(np.float32)
    out = pred.copy()
    tuned = np.empty_like(src)
    ratios = []
    for c in range(3):
        r_med = np.median(ref[:, c])
        r_iqr = np.percentile(ref[:, c], 75) - np.percentile(ref[:, c], 25)
        s_med = np.median(src[:, c])
        s_iqr = np.percentile(src[:, c], 75) - np.percentile(src[:, c], 25)
        ratio = float(np.clip(r_iqr / max(s_iqr, 1e-6), *IQR_RATIO_CLIP))
        ratios.append(round(ratio, 4))
        tuned[:, c] = (src[:, c] - s_med) * ratio + r_med
    out[mask] = np.clip(tuned, 0, 255).astype(np.uint8)
    return out, {"applied": True, "iqr_ratios_bgr": ratios, "ring_px": int(ring.sum())}


def region_stats(img: np.ndarray, mask: np.ndarray) -> dict:
    """Structure/colour agreement between the edited region and the road immediately around it."""
    ring_in = cv2.dilate(mask.astype(np.uint8), disk(3))
    ring = (cv2.dilate(mask.astype(np.uint8), disk(16)) > 0) & (ring_in == 0)
    if not mask.any() or ring.sum() < 100:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_32F)
    luma = img.astype(np.float32).mean(axis=2)

    def hist(sel, ch):
        h = cv2.calcHist([img], [ch], sel.astype(np.uint8), [32], [0, 256])
        return h / max(h.sum(), 1e-6)

    chi2 = 0.0
    for c in range(3):
        a, b = hist(mask, c), hist(ring, c)
        chi2 += float(np.sum((a - b) ** 2 / (a + b + 1e-6)))
    inner = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), disk(3))
    outside = cv2.dilate(mask.astype(np.uint8), disk(3)) - mask.astype(np.uint8)
    far = (cv2.dilate(mask.astype(np.uint8), disk(16)) > 0) & (cv2.dilate(mask.astype(np.uint8), disk(6)) == 0)

    def e(sel):
        sel = sel.astype(bool)
        return float(np.abs(grad[sel]).mean()) if sel.any() else 0.0

    e_far = max(e(far), 1e-6)
    return {
        "luma_patch": round(float(luma[mask].mean()), 2),
        "luma_ring": round(float(luma[ring].mean()), 2),
        "luma_delta": round(float(luma[mask].mean() - luma[ring].mean()), 2),
        "grad_ratio_patch_vs_ring": round(e(mask) / max(e(ring), 1e-6), 3),
        "hist_chi2_bgr": round(chi2, 4),
        "seam_inner_over_far": round(e(inner) / e_far, 3),
        "seam_outer_over_far": round(e(outside) / e_far, 3),
    }


def stage(ids, masks_dir, frames_dir, report_path, targets, Ks, p):
    swept = swept_map(ids, targets, Ks, p["window"])
    near = near_map(ids, targets, Ks)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    per_frame = []
    for idx, fid in enumerate(ids):
        img = cv2.imread(f"{SRC}/rendered/{fid}/new_delete.png", cv2.IMREAD_COLOR)
        cv2.imwrite(f"{frames_dir}/{idx:05d}.png", img)
        mask, stats = build_mask(fid, swept[fid], near[fid], p)
        cv2.imwrite(f"{masks_dir}/{idx:05d}.png", mask.astype(np.uint8) * 255)
        per_frame.append(stats)
        if idx % 25 == 0:
            log(f"staged {idx + 1}/{len(ids)} {fid} mask={stats['mask_px']} "
                f"ghost_outside_target={stats['ghost_outside_target_px']}")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"policy": p, "per_frame": per_frame}, fh, indent=2, ensure_ascii=False)
    return per_frame


def find_propainter_frames(root: str) -> str | None:
    for base, _ds, files in os.walk(root):
        if sum(1 for f in files if f.lower().endswith(".png")) >= EXPECTED_FRAMES:
            return base
    return None


def run_propainter(frames_dir: str, masks_dir: str, raw_out: str) -> str:
    cached = find_propainter_frames(raw_out)
    if cached:
        log(f"reusing cached ProPainter frames at {cached}")
        return cached
    os.makedirs(raw_out, exist_ok=True)
    cmd = [PY, "inference_propainter.py", "-i", frames_dir, "-m", masks_dir, "-o", raw_out,
           "--fp16", "--save_frames", "--mask_dilation", str(MASK_DILATION_ARG),
           "--neighbor_length", "10", "--ref_stride", "10",
           "--subvideo_length", str(SUBDIVIDEO), "--save_fps", "5"]
    log("propainter: " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROPAINTER, capture_output=True, text=True)
    log(f"propainter rc={proc.returncode} in {time.time() - t0:.1f}s")
    with open(f"{OUT}/propainter_stdout.log", "w", encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
        fh.write("\n--- STDERR ---\n")
        fh.write(proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"ProPainter failed rc={proc.returncode}:\n"
                           + "\n".join((proc.stderr or "").splitlines()[-30:]))
    found = find_propainter_frames(raw_out)
    if not found:
        raise RuntimeError("ProPainter frames not found")
    return found


def composite(ids, produced, masks_dir, tone, out_dir):
    os.makedirs(f"{out_dir}/final", exist_ok=True)
    os.makedirs(f"{out_dir}/preview_webp", exist_ok=True)
    raw_frames = sorted(f for f in os.listdir(produced) if f.lower().endswith(".png"))
    audit, prev = [], None
    for idx, fid in enumerate(ids):
        orig = cv2.imread(f"{SRC}/rendered/{fid}/new_delete.png", cv2.IMREAD_COLOR)
        pred = cv2.imread(f"{produced}/{raw_frames[idx]}", cv2.IMREAD_COLOR)
        mask = cv2.imread(f"{masks_dir}/{idx:05d}.png", cv2.IMREAD_GRAYSCALE) > 0
        if pred.shape != orig.shape:
            pred = cv2.resize(pred, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

        tone_info = {"applied": False}
        if tone == "medirq":
            pred, tone_info = tone_medirq(orig, pred, mask)

        ramp = feather_ramp(mask)[..., None]
        blended = orig.astype(np.float32) * (1 - ramp) + pred.astype(np.float32) * ramp
        final = np.clip(blended, 0, 255).astype(np.uint8)
        final[~mask] = orig[~mask]
        cv2.imwrite(f"{out_dir}/final/{idx:05d}.png", final)
        cv2.imwrite(f"{out_dir}/preview_webp/{idx:05d}.webp", final,
                    [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])

        dom = np.load(f"{SRC}/domains/{fid}.npz")
        K = dom["K"].astype(bool)
        d = cv2.absdiff(final, orig).max(axis=2)
        rec = {
            "frame": fid,
            "mask_px": int(mask.sum()),
            "outside_mask_changed_px": int((d[~mask] > 0).sum()),
            "retained_vehicle_changed_px": int((d[K] > 0).sum()) if K.any() else 0,
            "edited_px_share_of_mask": round(float((d[mask] > 0).mean()) if mask.any() else 0.0, 4),
            "tone": tone_info,
            "stats": region_stats(final, mask),
            "orig_stats": region_stats(orig, mask),
        }
        if prev is not None:
            ov = prev[1] & mask
            ring_prev = prev[2]
            ring_now = cv2.dilate(mask.astype(np.uint8), disk(16)) > 0
            ring_now &= ~(cv2.dilate(mask.astype(np.uint8), disk(3)) > 0)
            ring = ring_now & ring_prev
            if ov.any():
                rec["temporal_patch_delta"] = round(float(
                    np.abs(final.astype(np.float32) - prev[0].astype(np.float32)).mean(axis=2)[ov].mean()), 3)
            if ring.any():
                rec["temporal_ring_delta"] = round(float(
                    np.abs(final.astype(np.float32) - prev[0].astype(np.float32)).mean(axis=2)[ring].mean()), 3)
        prev = (final, mask, (cv2.dilate(mask.astype(np.uint8), disk(16)) > 0)
                & ~(cv2.dilate(mask.astype(np.uint8), disk(3)) > 0))
        audit.append(rec)
        if idx % 25 == 0:
            log(f"composited {idx + 1}/{len(ids)} tone={tone}")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--compose-only", default=None,
                    help="existing ProPainter frame dir to re-composite")
    ap.add_argument("--tone", default="none", choices=["none", "medirq"])
    ap.add_argument("--window", type=int, default=None,
                    help="+-frames for the swept footprint (default 8)")
    ap.add_argument("--hole-gate", default=None, choices=["swept", "near", "none"],
                    help="how far the 'no road support' family may reach")
    ap.add_argument("--color-dev", type=float, default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--tag", default=None, help="label for this mask policy, used in paths")
    args = ap.parse_args()

    policy = {
        "window": args.window if args.window is not None else SWEEP_WINDOW,
        "hole_gate": args.hole_gate or HOLE_GATE,
        "color_dev": args.color_dev if args.color_dev is not None else COLOR_DEV_THR,
    }
    tag = args.tag or f"w{policy['window']}_{policy['hole_gate']}_cd{int(policy['color_dev'])}"

    os.makedirs(OUT, exist_ok=True)
    ids = frame_ids()
    if len(ids) != EXPECTED_FRAMES:
        raise RuntimeError(f"expected {EXPECTED_FRAMES} frames, got {len(ids)}")

    targets, Ks = {}, {}
    for fid in ids:
        dom = np.load(f"{SRC}/domains/{fid}.npz")
        targets[fid] = dom["target"].astype(np.uint8)
        Ks[fid] = dom["K"].astype(bool)

    masks_dir = f"{OUT}/masks_{tag}"
    frames_dir = f"{OUT}/input_frames"
    staged = f"{OUT}/stage_report_{tag}.json"
    reuse = os.path.exists(staged) and os.path.exists(f"{masks_dir}/00198.png")
    if reuse:
        log(f"reusing staged masks [{tag}]")
        per_frame = json.load(open(staged, encoding="utf-8"))["per_frame"]
    else:
        per_frame = stage(ids, masks_dir, frames_dir, staged, targets, Ks, policy)

    mask_px = [s["mask_px"] for s in per_frame]
    sd = sum(s["static_dom_in_swept_px"] for s in per_frame)
    sdc = sum(s["static_dom_covered_px"] for s in per_frame)
    log(f"[{tag}] mask_px mean={np.mean(mask_px):.0f} min={min(mask_px)} max={max(mask_px)}")
    log(f"[{tag}] static-dominant ghost covered {sdc}/{sd} "
        f"({(sdc / max(1, sd)):.1%})  mask_in_K={sum(s['mask_in_K_px'] for s in per_frame)}")
    if args.stage_only:
        return 0

    produced = args.compose_only or run_propainter(frames_dir, masks_dir, f"{OUT}/raw_out_{tag}")
    suffix = args.out_suffix or f"_{tag}_{args.tone}"
    out_dir = f"{OUT}/variants{suffix}"
    audit = composite(ids, produced, masks_dir, args.tone, out_dir)

    def agg(key, sub=None):
        vals = [a[sub][key] for a in audit if sub and a.get(sub) and key in a[sub]] if sub else \
               [a[key] for a in audit if key in a]
        return {"mean": round(float(np.mean(vals)), 4) if vals else None,
                "median": round(float(np.median(vals)), 4) if vals else None,
                "max": round(float(np.max(vals)), 4) if vals else None,
                "min": round(float(np.min(vals)), 4) if vals else None}

    summary = {
        "schema": "scene065_propainter_stage_c.v5",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_campaign": SRC,
        "variant": {"tone": args.tone, "feather_px": FEATHER, "out_dir": out_dir},
        "mask_policy": {
            "swept_window": policy["window"], "footprint_dilate": FOOTPRINT_DILATE,
            "hole_gate": policy["hole_gate"], "near_band": NEAR_BAND,
            "static_at_thr": STATIC_AT_THR, "static_dominance": STATIC_DOMINANCE,
            "road_alpha_hole": ROAD_ALPHA_HOLE, "dark_abs": DARK_ABS_THR,
            "dark_rel": DARK_REL_THR, "color_dev": policy["color_dev"],
            "anchor_dilate": ANCHOR_DILATE, "mask_grow": MASK_GROW,
            "protect": "mask &= ~K  (retained vehicles are never written)",
        },
        "feather": "ramp = clip((distanceTransform(mask) - 1) / FEATHER, 0, 1); 0 on the boundary",
        "acceptance_metrics": ("coverage / seam_outer_over_far / grad_ratio_patch_vs_ring / "
                              "hist_chi2 / luma_delta / temporal_patch_delta. "
                              "dark_px is logged but is NOT an acceptance criterion"),
        "frame_count": len(ids),
        "mask_px_mean": round(float(np.mean(mask_px)), 1),
        "mask_in_K_px_total": int(sum(s["mask_in_K_px"] for s in per_frame)),
        "ghost_outside_target_px_total": int(sum(s["ghost_outside_target_px"] for s in per_frame)),
        "outside_mask_changed_px_total": int(sum(a["outside_mask_changed_px"] for a in audit)),
        "retained_vehicle_changed_px_total": int(sum(a["retained_vehicle_changed_px"] for a in audit)),
        "metrics": {
            "final": {
                "seam_outer_over_far": agg("seam_outer_over_far", "stats"),
                "seam_inner_over_far": agg("seam_inner_over_far", "stats"),
                "grad_ratio_patch_vs_ring": agg("grad_ratio_patch_vs_ring", "stats"),
                "hist_chi2_bgr": agg("hist_chi2_bgr", "stats"),
                "luma_delta": agg("luma_delta", "stats"),
            },
            "orig_same_mask": {
                "seam_outer_over_far": agg("seam_outer_over_far", "orig_stats"),
                "grad_ratio_patch_vs_ring": agg("grad_ratio_patch_vs_ring", "orig_stats"),
                "hist_chi2_bgr": agg("hist_chi2_bgr", "orig_stats"),
                "luma_delta": agg("luma_delta", "orig_stats"),
            },
            "temporal_patch_delta": agg("temporal_patch_delta"),
            "temporal_ring_delta": agg("temporal_ring_delta"),
        },
        "per_frame_mask": per_frame,
        "per_frame_audit": audit,
    }
    with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    strict = (summary["outside_mask_changed_px_total"] == 0
              and summary["retained_vehicle_changed_px_total"] == 0
              and summary["mask_in_K_px_total"] == 0)
    summary["strict_local_audit"] = "PASS" if strict else "FAIL"
    log("=== r5 SUMMARY tone=%s ===" % args.tone)
    log(f"mask_px_mean={summary['mask_px_mean']} ghost_outside_target_total="
        f"{summary['ghost_outside_target_px_total']}")
    log(f"outside_mask_changed={summary['outside_mask_changed_px_total']} "
        f"retained_vehicle_changed={summary['retained_vehicle_changed_px_total']}")
    for group in ("final", "orig_same_mask"):
        log(f"{group}: " + json.dumps(summary["metrics"][group], ensure_ascii=False))
    log(f"temporal flicker patch/ring: {summary['metrics']['temporal_patch_delta']['mean']} / "
        f"{summary['metrics']['temporal_ring_delta']['mean']}")
    log(f"STRICT_LOCAL_AUDIT={summary['strict_local_audit']}")
    return 0 if strict else 2


if __name__ == "__main__":
    sys.exit(main())
