"""Train the P182 conditional CDF directly with the seven fixed budget Brier scores."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml

import scripts.run_worldsim_v67_p186_noise_regularized_log_cost_density as p186


_LOG_BUDGETS = torch.tensor([0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]).log1p()


def _budget_brier_loss(
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    budgets = _LOG_BUDGETS.to(device=target.device, dtype=target.dtype)
    standardized = (budgets[None, :, None] - means[:, None]) / scales[:, None]
    component_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    predicted = torch.sum(
        functional.softmax(logits, dim=1)[:, None] * component_cdf, dim=2,
    )
    observed = (target[:, None] <= budgets[None]).to(dtype=predicted.dtype)
    return torch.mean(torch.square(predicted - observed))


def _rewrite_summary(config_path: Path, runs_root: Path, run_id: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["consumed_development_evaluations"].values():
        row["budget_brier_integrated_brier"] = row.pop("noise_regularized_integrated_brier")
        row["budget_brier_mean_absolute_reliability_error"] = row.pop(
            "noise_regularized_mean_absolute_reliability_error"
        )
    summary["schema_version"] = config["output_schema_version"]
    summary["role"] = config["role"]
    summary["verdict"] = (
        config["verdict_on_pass"]
        if all(summary["decision_checks"].values()) else config["verdict_on_failure"]
    )
    summary["training"]["objective"] = "mean_Brier_over_seven_fixed_reliability_budgets"
    summary["claim_boundary"] = config["claim_boundary"]
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "verdict": summary["verdict"]}, indent=2))


def main() -> None:
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    runs_root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    p186._mixture_nll = _budget_brier_loss
    p186.main()
    _rewrite_summary(config_path, runs_root, run_id)


if __name__ == "__main__":
    main()
