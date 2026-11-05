"""Replot the existing fixed-ROI295 ledger only; no observation recollection."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project(xyz, K, W):
    camera = xyz @ W[:3, :3].T + W[:3, 3]
    homogeneous = camera @ K.T
    return homogeneous[:, :2] / np.maximum(camera[:, 2:], 1e-8) - .5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--full", type=Path, default=Path(
        "/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026"))
    args = parser.parse_args()
    output = args.run / "observations/patch137_fixed295_rgb_witnesses.png"
    if output.exists():
        raise FileExistsError("Preserve existing figures; use a new output")
    with np.load(args.run / "observations/patch137_observation_ledger.npz") as ledger:
        member = ledger["member_fixed_roi"].astype(bool)
        xyz = ledger["local_xyz"][member]
        valid = ledger["geometric_RGB_qualified"][:, member]
    assert len(xyz) == 295
    shown = [137, 162, 163, 165]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=180)
    counts = {}
    for axis, raw in zip(axes, shown):
        image = np.asarray(Image.open(args.full / "rgb" / f"{raw:03d}.png"))
        with np.load(args.full / "frames" / f"{raw:03d}.npz") as frame:
            uv = project(xyz, frame["K"], frame["w2l"])
        good = valid[raw]
        counts[str(raw)] = int(good.sum())
        axis.imshow(image)
        axis.scatter(uv[:, 0], uv[:, 1], s=4, c="#f0bc45", alpha=.35, linewidths=0)
        axis.scatter(uv[good, 0], uv[good, 1], s=4, c="#1be591", alpha=.8, linewidths=0)
        axis.set_xlim(max(0, uv[:, 0].min()-20), min(479, uv[:, 0].max()+20))
        axis.set_ylim(min(319, uv[:, 1].max()+20), max(0, uv[:, 1].min()-20))
        axis.set_title(f"FRONT raw{raw}: {counts[str(raw)]}/295 witnesses", fontsize=9)
        axis.axis("off")
    fig.suptitle("Same 295 road-surface query points across real source images", fontsize=12)
    fig.text(.5, .027,
        "Independent 720-pixel audit ROI: 295 supported query points; the user's 570-pixel patch was unavailable.\n"
        "Geometry is inferred from measured support. Yellow: projected queries. Green: geometric RGB witnesses.\n"
        "A stable RGB reference is a proxy, not clean ground truth.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.85, bottom=.24, wspace=.035)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(dict(output=str(output), query_points=295,
        fixed_ROI_pixels=720, source_frame_witness_counts=counts,
        original_observation_collection_rerun=False, CPU_only=True), indent=2))


if __name__ == "__main__":
    main()
