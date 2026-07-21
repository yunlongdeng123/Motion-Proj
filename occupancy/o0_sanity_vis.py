#!/usr/bin/env python
"""O0 occupancy sanity + BEV visualization + unknown/free ratio report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OCC = Path("/root/autodl-tmp/data/occgs/occupancy")
REV = Path("/root/autodl-tmp/data/occgs/reviews/o0_occupancy")
REV.mkdir(parents=True, exist_ok=True)

UNK, FREE, STATIC, DYNAMIC = 0, 1, 2, 3


def summarize(sid: int):
    d = OCC / f"{sid:03d}"
    summary = json.load(open(d / "summary.json"))
    # aggregate
    keys = ["unk", "free", "static", "dyn"]
    means = {k: float(np.mean([s[k] for s in summary])) for k in keys}
    total = sum(means.values())
    ratios = {k: means[k] / total for k in keys}
    # unknown must dominate early; free should be non-trivial; never treat all unknown as free
    assert ratios["unk"] > 0.3, "too little unknown — possible over-carving"
    assert ratios["free"] > 0.005, "no free space carved"
    assert ratios["static"] > 0.0005, "no static occupied"
    # BEV at mid frame
    mid = summary[len(summary) // 2]["frame"]
    z = np.load(d / f"frame_{mid:03d}.npz")
    sem = z["semantics"]
    # collapse z by priority: dyn > static > free > unk
    bev = np.zeros(sem.shape[:2], dtype=np.uint8)
    for label in (UNK, FREE, STATIC, DYNAMIC):
        bev[np.any(sem == label, axis=2)] = label
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = np.array([[0.15, 0.15, 0.15], [0.7, 0.9, 0.7], [0.5, 0.5, 0.55], [0.95, 0.3, 0.25]])
    rgb = cmap[bev.T]  # transpose so x forward is up-ish; y left → plot x
    ax.imshow(rgb, origin="lower", extent=[-40, 40, -40, 40])
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"scene {sid:03d} t={mid} BEV (gray=unk green=free slate=static red=dyn)")
    out = REV / f"bev_{sid:03d}_t{mid:03d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return dict(scene=sid, mean_counts=means, ratios=ratios, bev=str(out), mid_frame=mid,
                unknown_preserved=True, n_frames=len(summary))


def main():
    report = {}
    for sid in (3, 4, 5):
        if not (OCC / f"{sid:03d}" / "summary.json").exists():
            report[sid] = dict(status="pending")
            continue
        report[sid] = summarize(sid)
        print(sid, report[sid]["ratios"])
    with open(OCC / "o0_sanity_v1.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
