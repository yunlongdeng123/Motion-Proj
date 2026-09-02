"""Render the frozen P20/P21 literal first-return boundary for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "proxy": "#94a3b8",
    "literal": "#dc2626",
    "p17": "#2563eb",
    "p17r": "#f59e0b",
    "p19": "#059669",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(
    p20_summary: Path,
    p21_summary: Path,
    output_stem: Path,
    legacy_total_percent: float,
    legacy_hazard_percent: float,
) -> None:
    p20 = _read_json(p20_summary)
    p21 = _read_json(p21_summary)
    baseline = p20["policies"]["p17"]
    literal_total = 100.0 * float(baseline["baseline"]["new_early_rate"])
    literal_hazard = 100.0 * float(baseline["hazard"]["baseline"]["new_early_rate"])

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.05))
    figure.patch.set_facecolor("white")

    groups = np.arange(2)
    width = 0.32
    proxy = np.asarray([legacy_total_percent, legacy_hazard_percent])
    literal = np.asarray([literal_total, literal_hazard])
    left = axes[0]
    left.bar(groups - width / 2, proxy, width, color=COLORS["proxy"], label="Target-nearest proxy")
    left.bar(groups + width / 2, literal, width, color=COLORS["literal"], label="Literal first return")
    for x, value in zip(groups - width / 2, proxy):
        left.text(x, value + 0.20, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for x, value in zip(groups + width / 2, literal):
        left.text(x, value + 0.20, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for x, old, new in zip(groups, proxy, literal):
        left.text(
            x,
            new + 1.05,
            f"{new / old:.1f}×",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#991b1b",
        )
    left.set_xticks(groups, ["All Actor rays", "Hazardous-Actor rays"])
    left.set_ylabel("New-early returns (%)")
    left.set_ylim(0.0, max(literal) * 1.28)
    left.set_title("(a) Proxy underestimates exposure", loc="left", fontweight="bold")
    left.grid(axis="y", alpha=0.22, linewidth=0.7)
    left.legend(frameon=False, fontsize=7.5, loc="upper left")

    frontier = p21["policy_frontier"]
    right = axes[1]
    all_x = np.asarray([float(row["new_hits_lost"]) for row in frontier])
    all_y = np.asarray([float(row["hazard_events_removed"]) for row in frontier])
    low = min(all_x.min(), all_y.min()) * 0.70
    high = max(all_x.max(), all_y.max()) * 1.35
    right.plot([low, high], [low, high], "--", color="#64748b", linewidth=1.0)
    right.text(130, 150, "1 event / hit", color="#475569", fontsize=7.5, rotation=38)

    label_offsets = {
        "p17": (-8, -20),
        "p17r": (8, -4),
        "p19": (8, 8),
    }
    display = {"p17": "P17", "p17r": "P17R", "p19": "P19"}
    for row in frontier:
        name = str(row["policy"])
        x = float(row["new_hits_lost"])
        y = float(row["hazard_events_removed"])
        penalty_mm = 1000.0 * float(row["mean_chamfer_penalty_m"])
        size = 65.0 + 95.0 * np.sqrt(penalty_mm)
        right.scatter(
            x,
            y,
            s=size,
            color=COLORS[name],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        dx, dy = label_offsets[name]
        right.annotate(
            f"{display[name]}  +{penalty_mm:.3f} mm CD",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=COLORS[name],
        )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(low, high)
    right.set_ylim(low, high)
    right.set_xlabel("New matched hits lost (log)")
    right.set_ylabel("Hazard early events removed (log)")
    right.set_title("(b) Certified direction, nonzero utility cost", loc="left", fontweight="bold")
    right.grid(which="both", alpha=0.20, linewidth=0.7)
    right.text(
        0.02,
        0.02,
        "Bubble area scales with Chamfer penalty; every policy has ΔCD > 0.",
        transform=right.transAxes,
        fontsize=7.2,
        color="#475569",
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)

    figure.tight_layout(w_pad=2.4)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p20-summary", type=Path, required=True)
    parser.add_argument("--p21-summary", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--legacy-total-percent", type=float, required=True)
    parser.add_argument("--legacy-hazard-percent", type=float, required=True)
    args = parser.parse_args()
    render(
        args.p20_summary,
        args.p21_summary,
        args.output_stem,
        args.legacy_total_percent,
        args.legacy_hazard_percent,
    )


if __name__ == "__main__":
    main()
