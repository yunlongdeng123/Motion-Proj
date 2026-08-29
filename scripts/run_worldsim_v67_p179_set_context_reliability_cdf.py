"""Condition the frozen P173 reliability CDF on a top Actor-query set context."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p144_trajectory_set_rank_compiler import (
    _build_sets,
    _p126_row_score,
    _row_features,
)
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p173_monotone_visit_reliability_cdf import (
    HorizonOnlyReliabilityCDF,
    MonotoneReliabilityCDF,
    _predict_surface,
)


class SetContextResidual(nn.Module):
    def __init__(self, feature_count: int, residual_bound: float) -> None:
        super().__init__()
        self.element_encoder = nn.Sequential(
            nn.Linear(feature_count, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(65, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.residual_bound = float(residual_bound)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, horizon: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.element_encoder(features)
        expanded = mask.unsqueeze(-1)
        mean = (encoded * expanded).sum(1) / expanded.sum(1).clamp(min=1)
        maximum = encoded.masked_fill(~expanded, -torch.inf).max(1).values
        pooled = torch.cat((mean, maximum, horizon), dim=1)
        return self.residual_bound * torch.tanh(self.decoder(pooled).squeeze(1))


@torch.no_grad()
def _predict_residual(
    model: SetContextResidual,
    sets: np.ndarray,
    mask: np.ndarray,
    horizon: np.ndarray,
    horizon_mean: float,
    horizon_scale: float,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(sets), 4096):
        outputs.append(model(
            torch.from_numpy(sets[start:start + 4096]).cuda(),
            torch.from_numpy(mask[start:start + 4096]).cuda(),
            torch.from_numpy(
                ((horizon[start:start + 4096] - horizon_mean) / horizon_scale).astype(np.float32)
            ).cuda()[:, None],
        ).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    arrays: dict[str, np.ndarray],
    models: list[DirectionalActorGaussian],
    ensemble: dict,
    residual_model: SetContextResidual,
    p173_model: MonotoneReliabilityCDF,
    p173_baseline: HorizonOnlyReliabilityCDF,
    frozen_p173: dict,
    token_mean: np.ndarray,
    token_scale: np.ndarray,
    maximum_rows: int,
    config: dict,
) -> dict:
    row_score = _p126_row_score(
        arrays, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    grouped = _build_sets(arrays, row_score, maximum_rows)
    normalized = (grouped["sets"] - token_mean) / token_scale
    normalized[~grouped["mask"]] = 0.0
    score, scenes = _ensemble_trajectory_score(
        arrays, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not (
        np.array_equal(scenes, cost_scenes)
        and np.array_equal(scenes, grouped["scene_index"])
    ):
        raise RuntimeError("P179 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    p173, horizon_only = _predict_surface(
        p173_model, p173_baseline, score, horizon, budgets, tuple(frozen_p173["norms"]),
    )
    residual = _predict_residual(
        residual_model, normalized.astype(np.float32), grouped["mask"], horizon,
        float(frozen_p173["norms"][2]), float(frozen_p173["norms"][3]),
    )
    p173_logit = np.log(np.clip(p173, 1e-6, 1.0 - 1e-6)) - np.log1p(
        -np.clip(p173, 1e-6, 1.0 - 1e-6)
    )
    predicted = 1.0 / (1.0 + np.exp(-(p173_logit + residual[:, None])))
    target = actual_cost[:, None] <= budgets[None]
    brier = float(np.mean(np.square(predicted - target)))
    p173_brier = float(np.mean(np.square(p173 - target)))
    error = float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0))))
    p173_error = float(np.mean(np.abs(p173.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "set_context_integrated_brier": brier,
        "p173_integrated_brier": p173_brier,
        "horizon_only_integrated_brier": float(np.mean(np.square(horizon_only - target))),
        "brier_change_vs_p173": float((brier - p173_brier) / max(p173_brier, 1e-12)),
        "set_context_mean_absolute_reliability_error": error,
        "p173_mean_absolute_reliability_error": p173_error,
        "calibration_error_reduction_vs_p173": float(
            (p173_error - error) / max(p173_error, 1e-12)
        ),
        "mean_absolute_logit_residual": float(np.mean(np.abs(residual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()

    ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    models = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        models.append(member.eval())
    frozen_p173 = torch.load(
        args.runs_root / config["frozen_p173"]["run"] / config["frozen_p173"]["artifact"],
        map_location="cuda",
    )
    p173_model = MonotoneReliabilityCDF(
        list(frozen_p173["score_knots"]), list(frozen_p173["budget_knots"]),
    ).cuda()
    p173_model.load_state_dict(frozen_p173["model_state_dict"])
    p173_model.eval()
    p173_baseline = HorizonOnlyReliabilityCDF(list(frozen_p173["budget_knots"])).cuda()
    p173_baseline.load_state_dict(frozen_p173["baseline_state_dict"])
    p173_baseline.eval()

    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    row_score = _p126_row_score(
        source, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    model_config = config["model"]
    maximum_rows = int(model_config["maximum_actor_query_rows"])
    grouped = _build_sets(source, row_score, maximum_rows)
    source_cost, cost_scenes = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(grouped["scene_index"], cost_scenes):
        raise RuntimeError("P179 source grouping is not aligned")
    all_tokens = _row_features(source, row_score)
    token_mean = all_tokens.mean(axis=0)
    token_scale = all_tokens.std(axis=0).clip(min=1e-4)
    normalized = (grouped["sets"] - token_mean) / token_scale
    normalized[~grouped["mask"]] = 0.0
    trajectory_score, score_scenes = _ensemble_trajectory_score(
        source, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    if not np.array_equal(score_scenes, cost_scenes):
        raise RuntimeError("P179 source score grouping is not aligned")
    horizon_values = _trajectory_horizon(source)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    p173_norms = tuple(frozen_p173["norms"])
    tensors = {
        "features": torch.from_numpy(normalized.astype(np.float32)).cuda(),
        "mask": torch.from_numpy(grouped["mask"]).cuda(),
        "score": torch.from_numpy(
            ((trajectory_score - p173_norms[0]) / p173_norms[1]).astype(np.float32)
        ).cuda(),
        "horizon": torch.from_numpy(
            ((horizon_values - p173_norms[2]) / p173_norms[3]).astype(np.float32)
        ).cuda()[:, None],
        "cost": torch.from_numpy(source_cost.astype(np.float32)).cuda(),
        "budget": torch.from_numpy(
            ((np.log1p(budgets) - p173_norms[4]) / p173_norms[5]).astype(np.float32)
        ).cuda(),
        "raw_budget": torch.from_numpy(budgets).cuda(),
    }

    torch.manual_seed(int(config["seed"]))
    model = SetContextResidual(normalized.shape[2], float(model_config["residual_bound"])).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(
            len(source_cost), (int(model_config["batch_size"]),), device="cuda",
        )
        budget_index = torch.randint(len(budgets), (len(index),), device="cuda")
        with torch.no_grad():
            base_logit = p173_model(
                tensors["score"][index], tensors["horizon"][index], tensors["budget"][budget_index],
            )
        residual = model(
            tensors["features"][index], tensors["mask"][index], tensors["horizon"][index],
        )
        target = (tensors["cost"][index] <= tensors["raw_budget"][budget_index]).float()
        loss = functional.binary_cross_entropy_with_logits(base_logit + residual, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P179 set-context step={step + 1} bce={final_loss:.6f}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(), "token_mean": token_mean, "token_scale": token_scale,
        "maximum_actor_query_rows": maximum_rows, "residual_bound": model_config["residual_bound"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, models, ensemble, model.eval(), p173_model, p173_baseline, frozen_p173,
            token_mean, token_scale, maximum_rows, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_p173"]) for row in evaluations.values()
    ]
    checks = {
        "brier_noninferior_to_p173_every_cohort": all(
            float(row["set_context_integrated_brier"]) <= float(row["p173_integrated_brier"])
            for row in evaluations.values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p173": float(np.mean(calibration_reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p173"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {
            "source_trajectory_count": int(len(source_cost)), "source_actor_query_rows": int(len(row_score)),
            "final_set_context_bce": final_loss,
        },
        "consumed_development_evaluations": evaluations,
        "decision_checks": checks,
        "mean_calibration_error_reduction_vs_p173": float(np.mean(calibration_reductions)),
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
