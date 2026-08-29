"""Test a fixed heavy-tailed conditional density for continuous visited-state cost."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import stdtr
from torch import nn
import torch.nn.functional as functional
import yaml

import scripts.run_worldsim_v67_p182_log_cost_mixture_density as p182


class StudentTLogCostMixtureDensity(nn.Module):
    """P182-compatible network with fixed-degree Student-t components."""

    def __init__(self, component_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 3
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, 3 * component_count))
        self.network = nn.Sequential(*layers)
        self.component_count = component_count

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(condition)
        logits, means, raw_scales = output.chunk(3, dim=1)
        scales = 0.05 + functional.softplus(raw_scales)
        return logits, means, scales


_DEGREES_OF_FREEDOM = 3.0


def _student_t_mixture_nll(
    logits: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    standardized = (target[:, None] - means) / scales
    degrees = _DEGREES_OF_FREEDOM
    log_normalizer = (
        math.lgamma((degrees + 1.0) / 2.0)
        - math.lgamma(degrees / 2.0)
        - 0.5 * math.log(degrees * math.pi)
    )
    log_density = (
        log_normalizer - torch.log(scales)
        - 0.5 * (degrees + 1.0) * torch.log1p(standardized.square() / degrees)
    )
    return -torch.logsumexp(
        functional.log_softmax(logits, dim=1) + log_density, dim=1,
    ).mean()


@torch.no_grad()
def _predict_student_t_cdf(
    model: StudentTLogCostMixtureDensity,
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
    log_budgets = np.log1p(budgets).astype(np.float32)
    outputs = []
    for start in range(0, len(conditions), 131072):
        condition = torch.from_numpy(conditions[start:start + 131072]).cuda()
        logits, means, scales = model(condition)
        standardized = (
            log_budgets[None, :, None]
            - means.detach().cpu().numpy()[:, None]
        ) / scales.detach().cpu().numpy()[:, None]
        component_cdf = stdtr(_DEGREES_OF_FREEDOM, standardized)
        weights = functional.softmax(logits, dim=1).cpu().numpy()
        outputs.append(np.sum(weights[:, None] * component_cdf, axis=2))
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
            "student_t_integrated_brier": brier,
            "p182_integrated_brier": control_brier,
            "brier_change_vs_p182": (brier - control_brier) / max(control_brier, 1e-12),
            "student_t_mean_absolute_reliability_error": error,
            "p182_mean_absolute_reliability_error": control_error,
            "calibration_error_reduction_vs_p182": reduction,
        })
        reductions.append(reduction)
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            row["student_t_integrated_brier"] <= row["p182_integrated_brier"]
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
    summary["training"]["degrees_of_freedom"] = _DEGREES_OF_FREEDOM
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
    p182.LogCostMixtureDensity = StudentTLogCostMixtureDensity
    p182._mixture_nll = _student_t_mixture_nll
    p182._predict_cdf = _predict_student_t_cdf
    p182.main()
    _rewrite_summary(config_path, runs_root, run_id)


if __name__ == "__main__":
    main()
