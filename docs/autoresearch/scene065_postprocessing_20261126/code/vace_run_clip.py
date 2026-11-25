"""Frozen inference of the official Wan2.1-VACE-1.3B on the two short scene-065 clips.

Nothing is trained, nothing is fine-tuned: `generate.py --task vace-1.3B` is called exactly as
shipped, with `src_video` = the r5 staged frames, `src_mask` = the r5 mask (white = regenerate),
No extra reference image is used. Official inpainting inputs are neutralized inside the mask.
Output is composited back with the same
inward feather and the same strict-local rule as ProPainter, then scored with the same metric
suite, so the only variable is the backend.

Records wall time and peak VRAM (nvidia-smi sampled in a thread + torch's own high-water mark),
because "128 GB/s, needs 20 GB" style claims need to be replaced by numbers from this card.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/_wsjob")
from scene065_propainter_20261126_r5 import (  # noqa: E402
    SRC, feather_ramp, region_stats, tone_medirq)

REPO = "/root/autodl-tmp/third_party/Wan2.1_vace_official"
CKPT = "/root/autodl-tmp/models/Wan2.1-VACE-1.3B"
SITE = "/root/autodl-tmp/envs/vace13b_site"
PYBIN = "/root/miniconda3/bin/python3.10"
OUT = "/root/autodl-tmp/drivingeditor_repro/campaign/scene065_vace13b_20261126_r2_official_inpainting"
R5 = "/root/autodl-tmp/drivingeditor_repro/campaign/scene065_propainter_20261126_r5"
CLIPS = {"087": (78, 98), "141": (132, 152)}
PROMPT = "a clean empty asphalt road at a traffic light intersection, lane markings, daylight"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class VramSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0.0
        self.samples: list[float] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0:
                    val = float(out.stdout.strip().splitlines()[0])
                    self.samples.append(val)
                    self.peak = max(self.peak, val)
            except Exception:
                pass
            self._stop_event.wait(0.5)

    def stop(self) -> None:
        self._stop_event.set()


def build_cmd(clip: str, steps: int, low_vram: bool = False) -> list[str]:
    lo, hi = CLIPS[clip]
    n = hi - lo + 1
    cmd = [
        PYBIN, "generate.py", "--task", "vace-1.3B",
        "--size", "480*320",
        "--ckpt_dir", CKPT,
        "--src_video", f"{OUT}/{clip}/src_video",
        "--src_mask", f"{OUT}/{clip}/src_mask",
        "--prompt", PROMPT,
        "--frame_num", str(n),
        "--sample_steps", str(steps),
        "--base_seed", "2026",
        "--save_file", f"{OUT}/{clip}/vace_out_steps{steps}_{steps}.mp4",
    ]
    if low_vram:
        # 1.3B dit (7.1 GB) + umt5-xxl (11.4 GB) do not fit together with activations on a
        # 24 GB card; on the 32 GB 4080S the default path is tried first and this is the fallback.
        cmd += ["--offload_model", "True", "--t5_cpu"]
    return cmd


def read_mp4(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, choices=list(CLIPS))
    ap.add_argument("--mask-tag", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--tone", default="none", choices=["none", "medirq"])
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--low-vram", action="store_true")
    args = ap.parse_args()

    lo, hi = CLIPS[args.clip]
    tag = f"steps{args.steps}"
    mp4 = f"{OUT}/{args.clip}/vace_out_{tag}_{args.steps}.mp4"

    if not args.skip_generate:
        cmd = build_cmd(args.clip, args.steps, args.low_vram)
        log("vace cmd: " + " ".join(cmd))
        env = dict(os.environ, PYTHONPATH=SITE)
        sampler = VramSampler()
        sampler.start()
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        elapsed = time.time() - t0
        sampler.stop()
        sampler.join(timeout=6)
        log(f"generate rc={proc.returncode} wall={elapsed:.1f}s peakVram={sampler.peak:.0f} MB "
            f"({sampler.peak / 1024:.1f} GB)")
        with open(f"{OUT}/{args.clip}/vace_stdout_{tag}.log", "w", encoding="utf-8") as fh:
            fh.write(proc.stdout or "")
            fh.write("\n--- STDERR ---\n")
            fh.write(proc.stderr or "")
        run_info = {"wall_s": round(elapsed, 1), "peak_vram_mb": round(sampler.peak, 1),
                    "nvidia_smi_samples": len(sampler.samples),
                    "returncode": proc.returncode, "steps": args.steps}
        if proc.returncode != 0:
            log((proc.stderr or "")[-3000:])
            with open(f"{OUT}/{args.clip}/run_info_{tag}.json", "w", encoding="utf-8") as fh:
                json.dump(run_info, fh, indent=2)
            return 1
    else:
        run_info = {"skipped_generate": True, "steps": args.steps}

    preds = read_mp4(mp4)
    n = hi - lo + 1
    log(f"decoded {len(preds)} predicted frames (expected {n})")
    assert len(preds) == n, (len(preds), n)

    masks_dir = f"{R5}/masks_{args.mask_tag}"
    os.makedirs(f"{OUT}/{args.clip}/composite", exist_ok=True)
    os.makedirs(f"{OUT}/{args.clip}/preview_webp", exist_ok=True)
    audit = []
    for k in range(n):
        idx = lo + k
        orig = cv2.imread(f"{R5}/input_frames/{idx:05d}.png", cv2.IMREAD_COLOR)
        mask = cv2.imread(f"{masks_dir}/{idx:05d}.png", cv2.IMREAD_GRAYSCALE) > 0
        pred = preds[k]
        if pred.shape != orig.shape:
            pred = cv2.resize(pred, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        tone_info = {"applied": False}
        if args.tone == "medirq":
            pred, tone_info = tone_medirq(orig, pred, mask)

        ramp = feather_ramp(mask)[..., None]
        blended = orig.astype(np.float32) * (1 - ramp) + pred.astype(np.float32) * ramp
        final = np.clip(blended, 0, 255).astype(np.uint8)
        final[~mask] = orig[~mask]
        cv2.imwrite(f"{OUT}/{args.clip}/composite/{idx:05d}.png", final)
        cv2.imwrite(f"{OUT}/{args.clip}/preview_webp/{idx:05d}.webp", final,
                    [cv2.IMWRITE_WEBP_QUALITY, 92])

        dom = np.load(f"{SRC}/domains/{idx:03d}.npz")
        K = dom["K"].astype(bool)
        d = cv2.absdiff(final, orig).max(axis=2)
        audit.append({
            "raw_index": idx,
            "mask_px": int(mask.sum()),
            "outside_mask_changed_px": int((d[~mask] > 0).sum()),
            "retained_vehicle_changed_px": int((d[K] > 0).sum()) if K.any() else 0,
            "tone": tone_info,
            "stats": region_stats(final, mask),
            "orig_stats": region_stats(orig, mask),
        })

    def agg(key, sub):
        vals = [a[sub][key] for a in audit if a.get(sub) and key in a[sub]]
        return {"mean": round(float(np.mean(vals)), 4) if vals else None,
                "median": round(float(np.median(vals)), 4) if vals else None}

    summary = {
        "schema": "scene065_vace13b_frozen.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clip": args.clip, "raw_range": [lo, hi], "mask_tag": args.mask_tag,
        "tone": args.tone, "prompt": PROMPT,
        "reference_images_used": False,
        "reference_policy": "Same RGB+mask as ProPainter; no unmasked ghost-bearing reference image",
        "run": run_info,
        "frames_composited": len(audit),
        "outside_mask_changed_px_total": int(sum(a["outside_mask_changed_px"] for a in audit)),
        "retained_vehicle_changed_px_total": int(sum(a["retained_vehicle_changed_px"] for a in audit)),
        "metrics": {
            "vace": {
                "seam_outer_over_far": agg("seam_outer_over_far", "stats"),
                "grad_ratio_patch_vs_ring": agg("grad_ratio_patch_vs_ring", "stats"),
                "hist_chi2_bgr": agg("hist_chi2_bgr", "stats"),
                "luma_delta": agg("luma_delta", "stats"),
            },
            "orig_same_mask": {
                "seam_outer_over_far": agg("seam_outer_over_far", "orig_stats"),
                "hist_chi2_bgr": agg("hist_chi2_bgr", "orig_stats"),
                "luma_delta": agg("luma_delta", "orig_stats"),
            },
        },
        "per_frame": audit,
    }
    with open(f"{OUT}/{args.clip}/summary_{tag}_{args.tone}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log(json.dumps({"run": run_info, "metrics": summary["metrics"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
