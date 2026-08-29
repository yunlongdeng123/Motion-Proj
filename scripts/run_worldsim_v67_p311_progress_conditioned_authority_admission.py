"""Learn a monotone progress-preference axis on top of frozen P309 action admission."""

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
import torch.nn.functional as F
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula, _align, _load_density, _trajectory_payload
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p279_epistemic_tail_cvar_allocator import EpistemicTailCVaRAllocator
from scripts.run_worldsim_v67_p297_direct_variable_set_authority_compiler import SharedContextPiecewiseAnchorAuthorityCompiler, _load_base_state
from scripts.run_worldsim_v67_p309_authority_residual_topk_admission import (
    AuthorityResidualTopK,
    _action_groups,
    _authority_curves,
    _geometry,
    _soft_topk_weights,
)


class ProgressConditionedAdmission(nn.Module):
    """Positive per-action progress penalty rate; lambda=0 exactly recovers P309."""

    def __init__(self, input_width: int, hidden_width: int, maximum_rate_adjustment: float) -> None:
        super().__init__()
        self.rate = nn.Sequential(
            nn.Linear(input_width, hidden_width), nn.SiLU(),
            nn.Linear(hidden_width, hidden_width), nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.rate[-1].weight)
        nn.init.zeros_(self.rate[-1].bias)
        self.maximum_rate_adjustment = float(maximum_rate_adjustment)

    def forward(self, descriptors, base_score, progress_deficit, preference):
        multiplier = 1 + self.maximum_rate_adjustment * torch.tanh(self.rate(descriptors).squeeze(2))
        return base_score + preference[:, None] * progress_deficit * multiplier


def _evaluate(model, descriptors, base_score, costs, queries, scenes, preferences, selected_count, target_scale, gap):
    x = torch.from_numpy(descriptors).cuda()
    base = torch.from_numpy(base_score).cuda()
    deficit = torch.from_numpy(np.where(queries < 3, 0.5, 0.0).astype(np.float32)).cuda()
    log_cost = np.log1p(costs)
    by_preference = {}
    all_deltas = []
    all_score_deltas = []
    full_fractions = []
    reductions = []
    regrets = []
    left, right = np.triu_indices(costs.shape[1], 1)
    with torch.no_grad():
        for preference in preferences:
            local = torch.full((len(x),), float(preference), device="cuda")
            score = model(x, base, deficit, local).cpu().numpy()
            selected = np.argsort(score, axis=1)[:, :selected_count]
            objective = log_cost + float(preference) * float(target_scale) * deficit.cpu().numpy()
            selected_objective = np.take_along_axis(objective, selected, axis=1).mean(1)
            all_objective = objective.mean(1)
            oracle = np.sort(objective, axis=1)[:, :selected_count].mean(1)
            selected_queries = np.take_along_axis(queries, selected, axis=1)
            full_fraction = float(np.mean(selected_queries >= 3))
            reduction = float(1 - selected_objective.mean() / max(all_objective.mean(), 1e-8))
            regret = float(np.mean(selected_objective - oracle))
            delta = objective[:, left] - objective[:, right]
            score_delta = score[:, left] - score[:, right]
            mask = np.abs(delta) >= float(gap)
            all_deltas.append(delta[mask])
            all_score_deltas.append(score_delta[mask])
            full_fractions.append(full_fraction)
            reductions.append(reduction)
            regrets.append(regret)
            by_preference[str(float(preference))] = {
                "selected_composite_objective": float(selected_objective.mean()),
                "all_action_composite_objective": float(all_objective.mean()),
                "oracle_topk_composite_objective": float(oracle.mean()),
                "relative_composite_reduction_vs_all_actions": reduction,
                "topk_composite_regret_vs_oracle": regret,
                "selected_full_progress_fraction": full_fraction,
            }
    delta = np.concatenate(all_deltas)
    score_delta = np.concatenate(all_score_deltas)
    monotonicity_violations = int(np.sum(np.diff(np.asarray(full_fractions)) < -1e-8))
    lower = equal = higher = 0
    highest = str(float(preferences[-1]))
    with torch.no_grad():
        score = model(x, base, deficit, torch.full((len(x),), float(preferences[-1]), device="cuda")).cpu().numpy()
    selected = np.argsort(score, axis=1)[:, :selected_count]
    objective = log_cost + float(preferences[-1]) * float(target_scale) * deficit.cpu().numpy()
    selected_objective = np.take_along_axis(objective, selected, axis=1).mean(1)
    all_objective = objective.mean(1)
    for scene in np.unique(scenes):
        local = scenes == scene
        candidate = float(selected_objective[local].mean())
        baseline = float(all_objective[local].mean())
        if candidate < baseline - 1e-8:
            lower += 1
        elif candidate > baseline + 1e-8:
            higher += 1
        else:
            equal += 1
    return {
        "group_count": int(len(costs)),
        "heldout_progress_preferences": [float(value) for value in preferences],
        "mean_relative_composite_reduction_vs_all_actions": float(np.mean(reductions)),
        "mean_topk_composite_regret_vs_oracle": float(np.mean(regrets)),
        "pooled_pairwise_composite_concordance": float(np.mean(delta * score_delta > 0)),
        "pooled_pairwise_evaluable_count": int(len(delta)),
        "selected_full_progress_fractions": [float(value) for value in full_fractions],
        "progress_preference_full_progress_monotonicity_violations": monotonicity_violations,
        "highest_preference_scene_lower_equal_higher_vs_all": [int(lower), int(equal), int(higher)],
        "by_progress_preference": by_preference,
        "highest_preference_key": highest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.monotonic()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    ensemble = torch.load(args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda")
    members = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        members.append(member.eval())
    density, density_metadata = _load_density(args.runs_root / config["frozen_p182"]["run"] / config["frozen_p182"]["artifact"])
    p199 = torch.load(args.runs_root / config["frozen_p199"]["run"] / config["frozen_p199"]["artifact"], map_location="cuda")
    copula = JointHorizonCopula(8, p199["hidden_dimensions"], 4).cuda()
    copula.load_state_dict(p199["model_state_dict"])
    copula.eval()
    p203 = torch.load(args.runs_root / config["frozen_p203"]["run"] / config["frozen_p203"]["artifact"], map_location="cuda")
    calibrator = MonotoneBetaCalibration().cuda()
    calibrator.load_state_dict(p203["model_state_dict"])
    calibrator.eval()
    horizons = np.asarray(config["horizons_seconds"], np.float32)
    anchors = np.asarray(config["feature_anchor_budgets"], np.float32)
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    common = (
        members, ensemble, density, density_metadata, p199, copula, calibrator, horizons, anchors, floor,
        int(config["teacher"]["monte_carlo_samples"]), seed,
        float(config["teacher"]["ignored_future_marginal_probability"]),
    )

    def materialize(reference):
        path = args.runs_root / reference["run"] / reference["artifact"]
        feature, _, _, _, _ = _dataset(path, *common)
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        _, _, costs, _ = _align(_trajectory_payload(arrays, members, ensemble, floor), horizons)
        return _action_groups(feature, costs, path, horizons, int(config["action_group_size"]))

    source_groups, source_costs, source_scenes, source_queries = materialize(config["source_rows"])
    p201_groups, p201_costs, p201_scenes, p201_queries = materialize(config["p201_rows"])
    policy_artifact = torch.load(args.runs_root / config["frozen_policy"]["run"] / config["frozen_policy"]["artifact"], map_location="cuda")
    base = EpistemicTailCVaRAllocator(
        int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
    ).cuda()
    authority_artifact = torch.load(args.runs_root / config["frozen_authority"]["run"] / config["frozen_authority"]["artifact"], map_location="cuda")
    _load_base_state(base, authority_artifact)
    authority = SharedContextPiecewiseAnchorAuthorityCompiler(base, authority_artifact["shared_context_anchor_fractions"]).cuda().eval()
    condition = config["authority_condition"]
    tolerance_domain = np.asarray(policy_artifact["risk_tolerance_domain"], np.float32)
    tolerance_z = 2 * (float(condition["risk_tolerance"]) - tolerance_domain[0]) / (tolerance_domain[1] - tolerance_domain[0]) - 1
    floor_domain = np.asarray(policy_artifact["floor_domain"], np.float32)
    floor_z = 2 * (float(condition["final_reliability_floor"]) - floor_domain[0]) / (floor_domain[1] - floor_domain[0]) - 1
    fractions = np.asarray(condition["attainable_budget_fractions"], np.float32)
    source_authority = _authority_curves(authority, source_groups, fractions, float(condition["alpha_fairness"]), tolerance_z, floor_z, float(condition["tail_mass"]))
    p201_authority = _authority_curves(authority, p201_groups, fractions, float(condition["alpha_fairness"]), tolerance_z, floor_z, float(condition["tail_mass"]))
    source_descriptor = np.concatenate((source_groups, source_authority, _geometry(source_queries)), 2)
    p201_descriptor = np.concatenate((p201_groups, p201_authority, _geometry(p201_queries)), 2)
    source_authority_base = source_authority.mean(2)
    p201_authority_base = p201_authority.mean(2)

    selector_artifact = torch.load(args.runs_root / config["frozen_selector"]["run"] / config["frozen_selector"]["artifact"], map_location="cuda")
    mean = np.asarray(selector_artifact["input_mean"], np.float32)
    scale = np.asarray(selector_artifact["input_scale"], np.float32)
    source_descriptor = ((source_descriptor - mean) / scale).astype(np.float32)
    p201_descriptor = ((p201_descriptor - mean) / scale).astype(np.float32)
    selector = AuthorityResidualTopK(
        int(selector_artifact["input_width"]), int(selector_artifact["element_width"]),
        int(selector_artifact["context_width"]), float(selector_artifact["maximum_authority_residual"]),
    ).cuda()
    selector.load_state_dict(selector_artifact["model_state_dict"])
    selector.eval()
    with torch.no_grad():
        source_score = selector(torch.from_numpy(source_descriptor).cuda(), torch.from_numpy(source_authority_base).cuda()).cpu().numpy()
        p201_score = selector(torch.from_numpy(p201_descriptor).cuda(), torch.from_numpy(p201_authority_base).cuda()).cpu().numpy()

    development = source_scenes % int(config["split"]["development_scene_modulus"]) == int(config["split"]["development_scene_remainder"])
    train = ~development
    target_mean = float(selector_artifact["target_log1p_mean"])
    target_scale = float(selector_artifact["target_log1p_scale"])
    source_target = ((np.log1p(source_costs) - target_mean) / target_scale).astype(np.float32)
    source_deficit = np.where(source_queries < 3, 0.5, 0.0).astype(np.float32)
    train_preferences = np.asarray(config["training_progress_preferences"], np.float32)
    heldout_preferences = np.asarray(config["heldout_progress_preferences"], np.float32)
    model_config = config["model"]
    confirmation_only = bool(config.get("confirmation_only", False))
    progress_artifact = None
    if confirmation_only:
        progress_artifact = torch.load(
            args.runs_root / config["frozen_progress_compiler"]["run"] / config["frozen_progress_compiler"]["artifact"],
            map_location="cuda",
        )
    hidden_width = int(progress_artifact["hidden_width"]) if progress_artifact else int(model_config["hidden_width"])
    rate_adjustment = float(progress_artifact["maximum_rate_adjustment"]) if progress_artifact else float(model_config["maximum_rate_adjustment"])
    model = ProgressConditionedAdmission(
        source_descriptor.shape[2], hidden_width, rate_adjustment,
    ).cuda()
    last = 0.0
    if confirmation_only:
        model.load_state_dict(progress_artifact["model_state_dict"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        x = torch.from_numpy(source_descriptor).cuda()
        base_score = torch.from_numpy(source_score).cuda()
        target = torch.from_numpy(source_target).cuda()
        deficit = torch.from_numpy(source_deficit).cuda()
        train_index = torch.from_numpy(np.flatnonzero(train)).cuda()
        preference_tensor = torch.from_numpy(train_preferences).cuda()
        left, right = np.triu_indices(int(config["action_group_size"]), 1)
        left = torch.from_numpy(left).cuda()
        right = torch.from_numpy(right).cuda()
        for step in range(int(model_config["steps"])):
            row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
            preference = preference_tensor[torch.randint(len(preference_tensor), (len(row),), device="cuda")]
            truth = target[row] + preference[:, None] * deficit[row]
            score = model(x[row], base_score[row], deficit[row], preference)
            weights = _soft_topk_weights(score, int(config["selected_action_count"]), float(model_config["rank_temperature"]), float(model_config["membership_temperature"]))
            decision_loss = torch.mean(torch.sum(weights * truth, 1) / float(config["selected_action_count"]))
            target_probability = F.softmax(-truth / float(model_config["listwise_temperature"]), 1)
            listwise_loss = torch.mean(torch.sum(-target_probability * F.log_softmax(-score / float(model_config["listwise_temperature"]), 1), 1))
            truth_delta = truth[:, left] - truth[:, right]
            score_delta = score[:, left] - score[:, right]
            pairwise_loss = F.softplus(-torch.sign(truth_delta) * score_delta / float(model_config["pairwise_temperature"]))
            pairwise_loss = pairwise_loss[torch.abs(truth_delta) >= float(model_config["pairwise_minimum_target_gap_z"])].mean()
            loss = decision_loss + float(model_config["listwise_weight"]) * listwise_loss + float(model_config["pairwise_weight"]) * pairwise_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
            if step % 500 == 0:
                print(f"P311 progress-conditioned admission step={step + 1} objective={last:.7f}", flush=True)

    evaluation = config["evaluation"]
    source_metrics = None if confirmation_only else _evaluate(
        model, source_descriptor[development], source_score[development], source_costs[development], source_queries[development],
        source_scenes[development], heldout_preferences, int(config["selected_action_count"]), target_scale,
        float(evaluation["pairwise_minimum_composite_gap"]),
    )
    p201_metrics = _evaluate(
        model, p201_descriptor, p201_score, p201_costs, p201_queries, p201_scenes, heldout_preferences,
        int(config["selected_action_count"]), target_scale, float(evaluation["pairwise_minimum_composite_gap"]),
    )
    decision = config["decision"]
    checks = {
        "P201_task_conditioned_composite_reduction": p201_metrics["mean_relative_composite_reduction_vs_all_actions"] >= float(decision["minimum_P201_mean_relative_composite_reduction_vs_all_actions"]),
        "P201_task_conditioned_pairwise_ranking": p201_metrics["pooled_pairwise_composite_concordance"] >= float(decision["minimum_P201_pooled_pairwise_composite_concordance"]),
        "P201_progress_preference_monotonicity": p201_metrics["progress_preference_full_progress_monotonicity_violations"] <= int(decision["maximum_P201_progress_preference_monotonicity_violations"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_width": source_descriptor.shape[2],
        "hidden_width": hidden_width,
        "maximum_rate_adjustment": rate_adjustment,
        "frozen_selector": config["frozen_selector"],
        "progress_preference_units": "P309_normalized_log1p_future_visited_state_cost",
    }, run_dir / config["model_artifact"])
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "training": {"group_count": 0 if confirmation_only else int(train.sum()), "steps": 0 if confirmation_only else int(model_config["steps"]), "final_objective": last},
        "source_development": source_metrics,
        "P201_post_hoc_development": p201_metrics,
        "decision_checks": checks,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
