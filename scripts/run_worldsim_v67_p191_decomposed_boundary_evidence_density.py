"""Condition continuous cost density on decomposed frozen boundary evidence."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml

from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian,
    _actor_entries,
    _predict,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import (
    LogCostMixtureDensity,
    _mixture_nll,
    _predict_cdf as _predict_p182_cdf,
)


class DecomposedBoundaryEvidenceDensity(nn.Module):
    def __init__(self, component_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 6
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, 3 * component_count))
        self.network = nn.Sequential(*layers)
        self.component_count = component_count

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, means, raw_scales = self.network(condition).chunk(3, dim=1)
        return logits, means, 0.05 + functional.softplus(raw_scales)


def _boundary_evidence(
    arrays: dict[str, np.ndarray],
    models: list[DirectionalActorGaussian],
    ensemble: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict(
            model.eval(), actor_features,
            np.asarray(ensemble["feature_mean"], dtype=np.float32),
            np.asarray(ensemble["feature_scale"], dtype=np.float32),
            np.asarray(ensemble["target_mean"], dtype=np.float32),
            np.asarray(ensemble["target_scale"], dtype=np.float32),
        )
        member_means.append(mean.reshape(-1, point_count, 2)[inverse])
        member_scales.append(scale.reshape(-1, point_count, 2)[inverse])
    means = np.stack(member_means, axis=0)
    scales = np.stack(member_scales, axis=0)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    aleatoric_variance = np.mean(np.sum(np.square(normal[None] * scales), axis=3), axis=0)
    epistemic_variance = projected_member_mean.var(axis=0)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    total_variance = np.maximum(aleatoric_variance + epistemic_variance, 1e-8)
    row_score = np.max(
        -(np.abs(signed) + np.sign(signed) * projected_mean) / np.sqrt(total_variance),
        axis=1,
    )
    row_context = np.stack((
        np.log1p(np.max(np.sqrt(np.maximum(aleatoric_variance, 0.0)), axis=1)),
        np.log1p(np.max(np.sqrt(np.maximum(epistemic_variance, 0.0)), axis=1)),
        np.log1p(np.max(np.abs(projected_mean), axis=1)),
    ), axis=1).astype(np.float32)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    score = _aligned_group_max(keys, row_score, identities)
    context = np.stack(
        [_aligned_group_max(keys, row_context[:, index], identities) for index in range(3)],
        axis=1,
    )
    return score.astype(np.float32), identities[:, 0].astype(np.int32), context


def _normalize(
    score: np.ndarray,
    horizon: np.ndarray,
    clearance: np.ndarray,
    context: np.ndarray,
    norms: tuple[float, ...],
) -> np.ndarray:
    raw = np.column_stack((score, horizon, clearance, context)).astype(np.float32)
    means = np.asarray(norms[0::2], dtype=np.float32)
    scales = np.asarray(norms[1::2], dtype=np.float32)
    return (raw - means) / scales


@torch.no_grad()
def _predict_cdf(
    model: DecomposedBoundaryEvidenceDensity,
    conditions: np.ndarray,
    budgets: np.ndarray,
) -> np.ndarray:
    budget_tensor = torch.from_numpy(np.log1p(budgets).astype(np.float32)).cuda()
    outputs = []
    for start in range(0, len(conditions), 131072):
        logits, means, scales = model(torch.from_numpy(conditions[start:start + 131072]).cuda())
        standardized = (budget_tensor[None, :, None] - means[:, None]) / scales[:, None]
        component_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        outputs.append(torch.sum(
            functional.softmax(logits, dim=1)[:, None] * component_cdf, dim=2,
        ).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    arrays: dict[str, np.ndarray],
    models: list[DirectionalActorGaussian],
    ensemble: dict,
    model: DecomposedBoundaryEvidenceDensity,
    p182_model: LogCostMixtureDensity,
    frozen_p182: dict,
    norms: tuple[float, ...],
    config: dict,
) -> dict:
    score, scenes, context = _boundary_evidence(arrays, models, ensemble)
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P191 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    clearance = _trajectory_clearance(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    predicted = _predict_cdf(model, _normalize(score, horizon, clearance, context, norms), budgets)
    p182 = _predict_p182_cdf(
        p182_model, score, horizon, clearance, budgets, tuple(frozen_p182["norms"]),
    )
    target = actual_cost[:, None] <= budgets[None]
    brier = float(np.mean(np.square(predicted - target)))
    p182_brier = float(np.mean(np.square(p182 - target)))
    error = float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0))))
    p182_error = float(np.mean(np.abs(p182.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "decomposed_evidence_integrated_brier": brier, "p182_integrated_brier": p182_brier,
        "brier_change_vs_p182": float((brier - p182_brier) / max(p182_brier, 1e-12)),
        "decomposed_evidence_mean_absolute_reliability_error": error,
        "p182_mean_absolute_reliability_error": p182_error,
        "calibration_error_reduction_vs_p182": float(
            (p182_error - error) / max(p182_error, 1e-12)
        ),
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
    frozen_p182 = torch.load(
        args.runs_root / config["frozen_p182"]["run"] / config["frozen_p182"]["artifact"],
        map_location="cuda",
    )
    p182_model = LogCostMixtureDensity(
        int(frozen_p182["component_count"]), list(frozen_p182["hidden_dimensions"]),
    ).cuda()
    p182_model.load_state_dict(frozen_p182["model_state_dict"])
    p182_model.eval()
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    score, scenes, context = _boundary_evidence(source, models, ensemble)
    source_cost, cost_scenes = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P191 source grouping is not aligned")
    horizon, clearance = _trajectory_horizon(source), _trajectory_clearance(source)
    raw = np.column_stack((score, horizon, clearance, context)).astype(np.float32)
    norms = tuple(
        value for column in range(raw.shape[1])
        for value in (float(raw[:, column].mean()), float(max(raw[:, column].std(), 1e-4)))
    )
    condition = torch.from_numpy(_normalize(score, horizon, clearance, context, norms)).cuda()
    target = torch.from_numpy(np.log1p(source_cost).astype(np.float32)).cuda()
    model_config = config["model"]
    torch.manual_seed(int(config["seed"]))
    model = DecomposedBoundaryEvidenceDensity(
        int(model_config["component_count"]), list(model_config["hidden_dimensions"]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_nll = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(len(condition), (int(model_config["batch_size"]),), device="cuda")
        logits, means, scales = model(condition[index])
        loss = _mixture_nll(logits, means, scales, target[index])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        final_nll = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P191 decomposed-evidence density step={step + 1} nll={final_nll:.6f}", flush=True)
    torch.save({
        "model_state_dict": model.state_dict(), "norms": norms,
        "component_count": model_config["component_count"],
        "hidden_dimensions": model_config["hidden_dimensions"],
        "condition_names": ["score", "horizon", "clearance", "aleatoric", "epistemic", "mean_magnitude"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, models, ensemble, model.eval(), p182_model, frozen_p182, norms, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    reductions = [row["calibration_error_reduction_vs_p182"] for row in evaluations.values()]
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            row["decomposed_evidence_integrated_brier"] <= row["p182_integrated_brier"]
            for row in evaluations.values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p182": float(np.mean(reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"trajectory_count": int(len(score)), "source_scene_count": int(len(np.unique(scenes))), "final_nll": final_nll},
        "consumed_development_evaluations": evaluations, "decision_checks": checks,
        "mean_calibration_error_reduction_vs_p182": float(np.mean(reductions)),
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
