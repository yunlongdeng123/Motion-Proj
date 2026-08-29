"""Fit a conditional rational-quadratic spline density for visited-state cost."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml

import scripts.run_worldsim_v67_p182_log_cost_mixture_density as p182


_TAIL_BOUND = 6.0


class ConditionalSplineLogCostDensity(nn.Module):
    """Condition an exact one-dimensional monotone spline on the P182 features."""

    def __init__(self, bin_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 3
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, 3 * bin_count - 1))
        self.network = nn.Sequential(*layers)
        self.bin_count = bin_count

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(condition)
        widths = output[:, :self.bin_count]
        heights = output[:, self.bin_count:2 * self.bin_count]
        derivatives = output[:, 2 * self.bin_count:]
        return widths, heights, derivatives


def _spline_nll(
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    latent, logabsdet = unconstrained_rational_quadratic_spline(
        target, widths, heights, derivatives, inverse=False,
        tails="linear", tail_bound=_TAIL_BOUND,
    )
    base_log_density = -0.5 * latent.square() - 0.5 * math.log(2.0 * math.pi)
    return -(base_log_density + logabsdet).mean()


@torch.no_grad()
def _predict_spline_cdf(
    model: ConditionalSplineLogCostDensity,
    score: np.ndarray,
    horizon: np.ndarray,
    clearance: np.ndarray,
    budgets: np.ndarray,
    norms: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    conditions = np.stack((
        (score - norms[0]) / norms[1],
        (horizon - norms[2]) / norms[3],
        (clearance - norms[4]) / norms[5],
    ), axis=1).astype(np.float32)
    log_budgets = torch.from_numpy(np.log1p(budgets).astype(np.float32)).cuda()
    outputs = []
    for start in range(0, len(conditions), 131072):
        condition = torch.from_numpy(conditions[start:start + 131072]).cuda()
        widths, heights, derivatives = model(condition)
        probabilities = []
        for budget in log_budgets:
            latent, _ = unconstrained_rational_quadratic_spline(
                torch.full((len(condition),), budget, device="cuda"),
                widths, heights, derivatives, inverse=False,
                tails="linear", tail_bound=_TAIL_BOUND,
            )
            probabilities.append(0.5 * (1.0 + torch.erf(latent / math.sqrt(2.0))))
        outputs.append(torch.stack(probabilities, dim=1).cpu().numpy())
    return np.concatenate(outputs)


def _rewrite_summary(config_path: Path, runs_root: Path, run_id: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    control_path = runs_root / config["frozen_p182_summary"]["run"] / "summary.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    reductions = []
    for name, row in summary["consumed_development_evaluations"].items():
        control_row = control["consumed_development_evaluations"][name]
        brier = float(row["log_cost_mixture_integrated_brier"])
        control_brier = float(control_row["log_cost_mixture_integrated_brier"])
        error = float(row["log_cost_mixture_mean_absolute_reliability_error"])
        control_error = float(control_row["log_cost_mixture_mean_absolute_reliability_error"])
        reduction = (control_error - error) / max(control_error, 1e-12)
        row.update({
            "spline_integrated_brier": brier,
            "p182_integrated_brier": control_brier,
            "brier_change_vs_p182": (brier - control_brier) / max(control_brier, 1e-12),
            "spline_mean_absolute_reliability_error": error,
            "p182_mean_absolute_reliability_error": control_error,
            "calibration_error_reduction_vs_p182": reduction,
        })
        reductions.append(reduction)
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            row["spline_integrated_brier"] <= row["p182_integrated_brier"]
            for row in summary["consumed_development_evaluations"].values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p182": float(np.mean(reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"]),
    }
    summary["schema_version"] = config["output_schema_version"]
    summary["verdict"] = (
        config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    )
    summary["role"] = config["role"]
    summary["training"]["bin_count"] = int(config["model"]["component_count"])
    summary["training"]["tail_bound"] = _TAIL_BOUND
    summary["decision_checks"] = checks
    summary["mean_calibration_error_reduction_vs_p182"] = float(np.mean(reductions))
    summary["claim_boundary"] = config["claim_boundary"]
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "verdict": summary["verdict"]}, indent=2))


def main() -> None:
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    runs_root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    p182.LogCostMixtureDensity = ConditionalSplineLogCostDensity
    p182._mixture_nll = _spline_nll
    p182._predict_cdf = _predict_spline_cdf
    p182.main()
    _rewrite_summary(config_path, runs_root, run_id)


if __name__ == "__main__":
    main()
