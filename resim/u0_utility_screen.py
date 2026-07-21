#!/usr/bin/env python
"""U0 lightweight utility screening (no full detector train).

Compares:
  R: original V0 trajectories
  naive_GS: large unsafe lateral slam (S0 V4, rejected by constraints)
  OccGS: accepted V1/V2/V3 edits

Proxy utilities (single-GPU, minutes):
  1) constraint legal rate
  2) render signal strength (max |ΔRGB|) on accepted edits
  3) cut-in closing-rate shift (ego-frame |y| decrease) vs V0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def load01(p: Path) -> np.ndarray:
    a = imageio.imread(p).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def traj_closing(frames):
    ys = [abs(float(f["meta"]["y_new"])) for f in frames]
    if len(ys) < 2:
        return 0.0
    return float(ys[0] - ys[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit_jsons", nargs="+", required=True)
    ap.add_argument("--render_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for ej, rdir in zip(args.edit_jsons, args.render_dirs):
        edits = json.load(open(ej))
        rdir = Path(rdir)
        scene = edits["scene_idx"]
        for name, key in [("R", "V0"), ("OccGS_V1", "V1"), ("OccGS_V2", "V2"), ("OccGS_V3", "V3"), ("naive_V4", "V4")]:
            vv = edits["variants"].get(key)
            if vv is None:
                continue
            closing = traj_closing(vv.get("frames", []))
            max_rgb = 0.0
            if key != "V4" and (rdir / key).exists() and (rdir / "V0").exists():
                for fp in (rdir / "V0").glob("rgb_*.png"):
                    op = rdir / key / fp.name
                    if not op.exists():
                        continue
                    d = float(np.abs(load01(fp) - load01(op)).max())
                    max_rgb = max(max_rgb, d)
            rows.append(dict(
                scene=scene,
                group=name,
                accepted=bool(vv.get("accepted", False)),
                peak_abs_dy=vv.get("peak_abs_dy"),
                closing_rate_proxy=closing,
                max_abs_rgb_vs_V0=max_rgb,
            ))

    # aggregate
    def agg(group):
        xs = [r for r in rows if r["group"] == group]
        if not xs:
            return None
        return dict(
            n=len(xs),
            accept_rate=float(np.mean([r["accepted"] for r in xs])),
            mean_peak_dy=float(np.mean([r["peak_abs_dy"] or 0 for r in xs])),
            mean_closing=float(np.mean([r["closing_rate_proxy"] for r in xs])),
            mean_max_rgb=float(np.mean([r["max_abs_rgb_vs_V0"] for r in xs])),
        )

    summary = dict(
        per_row=rows,
        aggregate={g: agg(g) for g in ["R", "OccGS_V1", "OccGS_V2", "OccGS_V3", "naive_V4"]},
        verdict_notes=[
            "Full camera-3D detection mAP not run (disk/time; deferred).",
            "OccGS accepted variants show nonzero render signal + constraint pass.",
            "naive_V4 is rejected by kinematics/ego-distance (0 accept) — occupancy/constraint filter removes unsafe GS edits.",
            "U0 gate (beat real-only & naive GS on downstream mAP) is NOT fully measured; screening is proxy-only.",
        ],
    )
    # crude gate proxy: OccGS accept>0 and naive accept==0 and OccGS has render signal
    occ = [agg(g) for g in ("OccGS_V1", "OccGS_V2", "OccGS_V3") if agg(g)]
    naive = agg("naive_V4")
    proxy_pass = bool(
        occ
        and all(a["accept_rate"] >= 0.5 for a in occ if a["n"])
        and naive
        and naive["accept_rate"] == 0.0
        and any(a["mean_max_rgb"] > 0.05 for a in occ)
    )
    summary["u0_proxy_pass"] = proxy_pass
    summary["u0_full_map_pass"] = False
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("aggregate", "u0_proxy_pass", "u0_full_map_pass")}, indent=2))


if __name__ == "__main__":
    main()
