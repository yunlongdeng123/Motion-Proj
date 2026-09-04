"""Plot the frozen M49 attenuation sign boundary from its summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _stacked(
    axis: plt.Axes,
    labels: list[str],
    first: list[float],
    second: list[float],
    first_label: str,
    second_label: str,
) -> None:
    neutral = [max(0.0, 1.0 - a - b) for a, b in zip(first, second)]
    y = np.arange(len(labels))
    axis.barh(y, first, color="#B2182B", label=first_label)
    axis.barh(y, second, left=first, color="#2166AC", label=second_label)
    axis.barh(y, neutral, left=np.asarray(first) + np.asarray(second), color="#D9D9D9", label="near-zero")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    axis.set_xlabel("rays (\%)")
    axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    axis.set_axisbelow(True)
    for row, (a, b) in enumerate(zip(first, second)):
        axis.text(a / 2, row, f"{100*a:.1f}", ha="center", va="center", color="white", fontsize=7)
        axis.text(a + b / 2, row, f"{100*b:.1f}", ha="center", va="center", color="white", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.summary.read_text(encoding="utf-8"))["metrics"]
    strata = ["all", "hazard", "clear"]
    labels = ["All", "Hazard", "Clear"]

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)
    _stacked(
        axes[0],
        labels,
        [metrics[name]["uniform_child_attenuation_adverse_fraction"] for name in strata],
        [metrics[name]["uniform_child_attenuation_safe_fraction"] for name in strata],
        "attenuation adverse",
        "attenuation safe",
    )
    axes[0].set_title(r"Analytic sign: $r_j(C_j-F)$", fontsize=9)
    _stacked(
        axes[1],
        labels,
        [metrics[name]["cdf_increase_fraction"] for name in strata],
        [metrics[name]["cdf_decrease_fraction"] for name in strata],
        "adverse / CDF increase",
        "safe / CDF decrease",
    )
    axes[1].set_title("Observed finite attenuation", fontsize=9)
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(-0.08, -0.42),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8, length=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
