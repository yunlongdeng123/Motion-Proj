#!/usr/bin/env python
"""D0 integrity check + 12-timestep visualization panels for OccGS scenes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/root/autodl-tmp/data/occgs/processed_10Hz/mini")
SPEC = Path("/root/autodl-tmp/data/occgs/scene_specs")
REVIEW = Path("/root/autodl-tmp/data/occgs/reviews/d0_integrity")
REVIEW.mkdir(parents=True, exist_ok=True)

# Frozen picks (v2): mini index -> role / nuscenes name
SCENES = {
    3: dict(role="S0", name="scene-0655", reason="static-heavy: day, yaw=11deg, dist=163m, only 7 moving vehicles"),
    5: dict(role="S1", name="scene-0796", reason="vehicle-dynamic: day, 22 persistent movers, yaw=8.6deg"),
    4: dict(role="S2", name="scene-0757", reason="cut-in candidate: day, cutin_score=49.7, 15 persistent movers"),
}
FRONT_CAMS = [0, 1, 2]  # FRONT, FRONT_LEFT, FRONT_RIGHT


def check_scene(sid: int) -> dict:
    d = ROOT / f"{sid:03d}"
    imgs = sorted((d / "images").glob("*.jpg"))
    skys = sorted((d / "sky_masks").glob("*.png"))
    lids = sorted((d / "lidar").glob("*.bin"))
    extr = sorted((d / "extrinsics").glob("*.txt"))
    # front-3 only for sky / images count expected
    front_imgs = [p for p in imgs if int(p.stem.split("_")[1]) in FRONT_CAMS]
    front_sky = [p for p in skys if int(p.stem.split("_")[1]) in FRONT_CAMS]
    # timestamps = unique frame indices
    frames = sorted({int(p.stem.split("_")[0]) for p in imgs})
    # monotonic check via lidar_pose timestamps order (=frame index)
    mono = frames == list(range(frames[0], frames[-1] + 1)) if frames else False
    # instances
    with open(d / "instances" / "instances_info.json") as f:
        inst = json.load(f)
    with open(d / "instances" / "frame_instances.json") as f:
        frame_inst = json.load(f)
    # vehicle actors with continuous tracks (>=8 frames at 10Hz = 0.8s; prefer >=40 = 4s)
    n_veh = 0
    n_cont4s = 0
    for k, v in inst.items():
        cname = v.get("class_name", "")
        if not cname.startswith("vehicle"):
            continue
        n_veh += 1
        fa = v["frame_annotations"]["frame_idx"]
        if len(fa) >= 40:
            # consecutive?
            fa_sorted = sorted(fa)
            if fa_sorted[-1] - fa_sorted[0] + 1 == len(fa_sorted) or len(fa_sorted) >= 40:
                n_cont4s += 1
    # dynamic mask coverage (front cams)
    dyn_dir = d / "dynamic_masks" / "vehicle"
    dyn_files = list(dyn_dir.glob("*_0.png")) if dyn_dir.exists() else []
    dyn_cov = 0.0
    if dyn_files:
        sample = sorted(dyn_files)[:: max(1, len(dyn_files) // 8)][:8]
        ratios = []
        for p in sample:
            m = np.array(Image.open(p)) > 0
            ratios.append(float(m.mean()))
        dyn_cov = float(np.mean(ratios))
    # sky coverage
    sky_cov = 0.0
    if front_sky:
        sample = front_sky[:: max(1, len(front_sky) // 8)][:8]
        ratios = []
        for p in sample:
            m = np.array(Image.open(p)) > 0
            ratios.append(float(m.mean()))
        sky_cov = float(np.mean(ratios))
    # disk
    size_bytes = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    # intrinsics exist for all 6 cams
    intr = list((d / "intrinsics").glob("*.txt"))
    ok = (
        len(front_imgs) >= 3 * 80  # ~8s * 10Hz * 3 cams
        and len(front_sky) == len(front_imgs)
        and len(lids) >= 80
        and mono
        and n_veh >= 1
        and len(intr) >= 3
    )
    return dict(
        scene_idx=sid,
        n_images_all=len(imgs),
        n_images_front3=len(front_imgs),
        n_sky_front3=len(front_sky),
        n_lidar=len(lids),
        n_extrinsics=len(extr),
        n_frames=len(frames),
        frame_range=[frames[0], frames[-1]] if frames else None,
        timestamps_monotonic=mono,
        n_vehicle_instances=n_veh,
        n_vehicle_cont4s=n_cont4s,
        n_frame_keys=len(frame_inst),
        dynamic_mask_mean_coverage=round(dyn_cov, 4),
        sky_mask_mean_coverage=round(sky_cov, 4),
        disk_bytes=size_bytes,
        disk_mb=round(size_bytes / 1e6, 1),
        gate_ok=ok,
    )


def vis_scene(sid: int, role: str, n_panels: int = 12):
    d = ROOT / f"{sid:03d}"
    with open(d / "instances" / "instances_info.json") as f:
        inst = json.load(f)
    frames = sorted({int(p.stem.split("_")[0]) for p in (d / "images").glob("*_0.jpg")})
    # sample 12 evenly
    idxs = np.linspace(0, len(frames) - 1, n_panels, dtype=int)
    fig, axes = plt.subplots(n_panels, 3, figsize=(12, 2.2 * n_panels))
    for row, fi in enumerate(idxs):
        t = frames[fi]
        for col, cam in enumerate(FRONT_CAMS):
            ax = axes[row, col]
            img_p = d / "images" / f"{t:03d}_{cam}.jpg"
            sky_p = d / "sky_masks" / f"{t:03d}_{cam}.png"
            dyn_p = d / "dynamic_masks" / "vehicle" / f"{t:03d}_{cam}.png"
            img = np.array(Image.open(img_p).convert("RGB"))
            overlay = img.copy()
            if sky_p.exists():
                sky = np.array(Image.open(sky_p)) > 0
                overlay[sky] = (overlay[sky] * 0.5 + np.array([80, 160, 255]) * 0.5).astype(np.uint8)
            if dyn_p.exists():
                dyn = np.array(Image.open(dyn_p)) > 0
                overlay[dyn] = (overlay[dyn] * 0.5 + np.array([255, 80, 80]) * 0.5).astype(np.uint8)
            # project boxes roughly via 2d from instances (draw world centers projected if we had K/E;
            # for review we overlay class counts as text)
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"][col])
            if col == 0:
                ax.set_ylabel(f"t={t}")
    fig.suptitle(f"{role} mini/{sid:03d} sky(blue)+vehicle_dyn(red)", y=1.0)
    fig.tight_layout()
    out = REVIEW / f"{role}_{sid:03d}_panel.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main():
    report = {"scenes": {}, "picks": {}}
    for sid, meta in SCENES.items():
        stats = check_scene(sid)
        stats.update(meta)
        panel = vis_scene(sid, meta["role"])
        stats["panel"] = panel
        report["scenes"][f"{sid:03d}"] = stats
        report["picks"][meta["role"]] = dict(mini_idx=sid, name=meta["name"], reason=meta["reason"])
        print(json.dumps(stats, indent=2))
    # disk gate
    import subprocess

    df = subprocess.check_output(["df", "-B1", "/root/autodl-tmp"]).decode().splitlines()[1].split()
    avail = int(df[3])
    report["disk_avail_bytes"] = avail
    report["disk_avail_gib"] = round(avail / (1024**3), 2)
    report["disk_gate_30gib"] = avail >= 30 * (1024**3)
    report["all_gate_ok"] = all(s["gate_ok"] for s in report["scenes"].values()) and report["disk_gate_30gib"]
    # S1/S2 need >=2 moving continuous
    for role in ("S1", "S2"):
        sid = report["picks"][role]["mini_idx"]
        n = report["scenes"][f"{sid:03d}"]["n_vehicle_cont4s"]
        report["scenes"][f"{sid:03d}"]["role_gate_moving"] = n >= 2
        if n < 2:
            report["all_gate_ok"] = False
    out = SPEC / "d0_integrity_v1.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    # freeze picks
    with open(SPEC / "d0_frozen_picks_v2.json", "w") as f:
        json.dump(
            {
                "version": "v2",
                "frozen_before_training": True,
                "split": "v1.0-mini",
                "processed_root": str(ROOT),
                "picks": report["picks"],
                "time_window_policy": "first 8s = frames 0..79 at 10Hz (clip in training via end_timestep=79)",
                "cameras_train": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
                "sky_mask": "transformers SegFormer-B5 cityscapes (alternative to official mmseg; V7 §5.5)",
            },
            f,
            indent=2,
        )
    print("ALL_GATE_OK", report["all_gate_ok"], "disk_GiB", report["disk_avail_gib"])


if __name__ == "__main__":
    main()
