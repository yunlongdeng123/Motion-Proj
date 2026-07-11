"""7–9 小时 Optuna 搜索的不可变空间、淘汰与晋级规则。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass


STAGES = ((16, 100), (4, 300), (2, 800))


@dataclass
class SearchBudget:
    started_at: float
    hard_limit_hours: float = 9.0

    @classmethod
    def start(cls, hard_limit_hours: float = 9.0) -> "SearchBudget":
        return cls(time.time(), hard_limit_hours)

    def may_start_trial(self, now: float | None = None) -> bool:
        return (now or time.time()) < self.started_at + self.hard_limit_hours * 3600


def suggest_params(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "lambda_proj": trial.suggest_float("lambda_proj", 0.03, 0.3),
        "beta_anchor": trial.suggest_float("beta_anchor", 0.1, 1.0),
        "bound_B": trial.suggest_categorical("bound_B", [3, 4, 6, 8]),
        "tube_upper": trial.suggest_categorical("tube_upper", [0.25, 0.35, 0.45]),
        "lora_rank": 16,
    }


def prune_reason(metrics: dict, base_lpips: float) -> str | None:
    values = [metrics.get(key) for key in ("static_drift", "track_acceleration", "lpips")]
    if any(value is None or value != value for value in values):
        return "nan"
    if float(metrics.get("projection_eligible_fraction", 0.0)) < 0.70:
        return "eligible_fraction_below_70pct"
    if float(metrics["lpips"]) > float(base_lpips) * 1.05:
        return "lpips_worse_than_base_by_5pct"
    return None


def objective_score(metrics: dict) -> float:
    return 0.5 * (float(metrics["normalized_static_drift_improvement"])
                  + float(metrics["normalized_track_acceleration_improvement"]))


def derived_run_id(parent_run_id: str, params: dict, target_steps: int) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    return f"{parent_run_id}-s{target_steps}-{digest}"
