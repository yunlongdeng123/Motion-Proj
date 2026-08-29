"""Fine-tune P182 with norm-balanced PCGrad over NLL and budget Brier."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml

import scripts.run_worldsim_v67_p186_noise_regularized_log_cost_density as p186
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import (
    LogCostMixtureDensity as P182Density,
    _mixture_nll,
)


_LOG_BUDGETS = torch.tensor([0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]).log1p()
_INITIAL_STATE: dict | None = None


class P182InitializedDensity(P182Density):
    def __init__(self, component_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__(component_count, hidden_dimensions)
        if _INITIAL_STATE is not None:
            self.load_state_dict(_INITIAL_STATE)


def _budget_brier(
    logits: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    budgets = _LOG_BUDGETS.to(device=target.device, dtype=target.dtype)
    standardized = (budgets[None, :, None] - means[:, None]) / scales[:, None]
    component_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    predicted = torch.sum(
        functional.softmax(logits, dim=1)[:, None] * component_cdf, dim=2,
    )
    observed = (target[:, None] <= budgets[None]).to(dtype=predicted.dtype)
    return torch.mean(torch.square(predicted - observed))


def _pcgrad_loss(
    logits: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    outputs = (logits, means, scales)
    nll = _mixture_nll(logits, means, scales, target)
    brier = _budget_brier(logits, means, scales, target)
    nll_grads = torch.autograd.grad(nll, outputs, retain_graph=True)
    brier_grads = torch.autograd.grad(brier, outputs, retain_graph=True)
    nll_norm = torch.sqrt(sum(torch.sum(grad.square()) for grad in nll_grads)).clamp_min(1e-12)
    brier_norm = torch.sqrt(sum(torch.sum(grad.square()) for grad in brier_grads)).clamp_min(1e-12)
    common_norm = 0.5 * (nll_norm + brier_norm)
    first = tuple(grad * (common_norm / nll_norm) for grad in nll_grads)
    second = tuple(grad * (common_norm / brier_norm) for grad in brier_grads)
    dot = sum(torch.sum(left * right) for left, right in zip(first, second))
    if float(dot.detach()) < 0.0:
        first_norm_sq = sum(torch.sum(grad.square()) for grad in first).clamp_min(1e-12)
        second_norm_sq = sum(torch.sum(grad.square()) for grad in second).clamp_min(1e-12)
        projected_first = tuple(
            left - dot / second_norm_sq * right for left, right in zip(first, second)
        )
        projected_second = tuple(
            right - dot / first_norm_sq * left for left, right in zip(first, second)
        )
    else:
        projected_first, projected_second = first, second
    combined = tuple(
        0.5 * (left + right) for left, right in zip(projected_first, projected_second)
    )
    surrogate = sum(torch.sum(output * gradient.detach()) for output, gradient in zip(outputs, combined))
    return surrogate - surrogate.detach() + (nll + brier).detach()


def _rewrite_summary(config_path: Path, runs_root: Path, run_id: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["consumed_development_evaluations"].values():
        row["pcgrad_integrated_brier"] = row.pop("noise_regularized_integrated_brier")
        row["pcgrad_mean_absolute_reliability_error"] = row.pop(
            "noise_regularized_mean_absolute_reliability_error"
        )
    summary["schema_version"] = config["output_schema_version"]
    summary["role"] = config["role"]
    summary["verdict"] = (
        config["verdict_on_pass"]
        if all(summary["decision_checks"].values()) else config["verdict_on_failure"]
    )
    summary["training"]["objective"] = "norm_balanced_PCGrad_of_NLL_and_seven_budget_Brier"
    summary["claim_boundary"] = config["claim_boundary"]
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "verdict": summary["verdict"]}, indent=2))


def main() -> None:
    global _INITIAL_STATE
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    runs_root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = torch.load(
        runs_root / config["frozen_p182"]["run"] / config["frozen_p182"]["artifact"],
        map_location="cpu",
    )
    _INITIAL_STATE = frozen["model_state_dict"]
    p186.LogCostMixtureDensity = P182InitializedDensity
    p186._mixture_nll = _pcgrad_loss
    p186.main()
    _rewrite_summary(config_path, runs_root, run_id)


if __name__ == "__main__":
    main()
