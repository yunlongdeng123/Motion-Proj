"""Compile frozen task-conditioned authority budgets into differentiable top-k action admission."""

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
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import (
    JointHorizonCopula,
    _align,
    _load_density,
    _trajectory_payload,
)
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p279_epistemic_tail_cvar_allocator import EpistemicTailCVaRAllocator
from scripts.run_worldsim_v67_p297_direct_variable_set_authority_compiler import (
    SharedContextPiecewiseAnchorAuthorityCompiler,
    _load_base_state,
)


class AuthorityResidualTopK(nn.Module):
    """A bounded equivariant correction around the frozen authority-curve score."""

    def __init__(self, input_width: int, element_width: int, context_width: int, residual_bound: float) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(input_width, element_width), nn.SiLU(),
            nn.Linear(element_width, element_width), nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(4 * element_width, context_width), nn.SiLU(),
            nn.Linear(context_width, context_width), nn.SiLU(),
        )
        self.residual = nn.Linear(context_width, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.log_base_scale = nn.Parameter(torch.tensor(0.0))
        self.base_bias = nn.Parameter(torch.tensor(0.0))
        self.residual_bound = float(residual_bound)

    def forward(self, descriptors: torch.Tensor, authority_base: torch.Tensor) -> torch.Tensor:
        encoded = self.element(descriptors)
        size = encoded.shape[1]
        mean = encoded.mean(1)
        std = torch.sqrt(encoded.var(1, unbiased=False) + 1e-6)
        maximum = encoded.amax(1)
        context = self.context(torch.cat((
            encoded,
            mean[:, None].expand(-1, size, -1),
            std[:, None].expand(-1, size, -1),
            maximum[:, None].expand(-1, size, -1),
        ), 2))
        correction = self.residual_bound * torch.tanh(self.residual(context).squeeze(2))
        return self.base_bias + F.softplus(self.log_base_scale) * authority_base + correction


def _action_groups(feature, costs, rows_path: Path, horizons, expected_size: int):
    with np.load(rows_path, allow_pickle=False) as loaded:
        identities = np.unique(np.stack((
            loaded["scene_index"],
            np.rint(loaded["horizon_seconds"] * 10).astype(np.int32),
            loaded["anchor_frame"],
            loaded["query_id"],
        ), 1), axis=0)
    key_sets = []
    for horizon in horizons:
        code = int(round(float(horizon) * 10))
        key_sets.append({(int(row[0]), int(row[2]), int(row[3])) for row in identities if int(row[1]) == code})
    keys = sorted(set.intersection(*key_sets))
    if len(keys) != len(feature) or len(costs) != len(feature):
        raise RuntimeError(f"action feature/cost alignment failed: {len(keys)}, {len(feature)}, {len(costs)}")
    buckets = {}
    for index, (scene, anchor, query) in enumerate(keys):
        buckets.setdefault((scene, anchor), []).append((index, query))
    valid = [(key, rows) for key, rows in buckets.items() if len(rows) == int(expected_size)]
    return (
        np.stack([feature[[row[0] for row in rows]] for _, rows in valid]).astype(np.float32),
        np.stack([costs[[row[0] for row in rows]].max(1) for _, rows in valid]).astype(np.float32),
        np.asarray([key[0] for key, _ in valid], np.int64),
        np.stack([[row[1] for row in rows] for _, rows in valid]).astype(np.int64),
    )


@torch.no_grad()
def _authority_curves(model, groups, fractions, alpha, tolerance_z, floor_z, tail_mass):
    x = torch.from_numpy(groups).cuda()
    outputs = []
    for fraction in fractions:
        count = len(x)
        outputs.append(model(
            x,
            torch.full((count,), float(1 - 2 * fraction), device="cuda"),
            torch.full((count,), float(alpha), device="cuda"),
            torch.full((count,), float(tolerance_z), device="cuda"),
            torch.full((count,), float(floor_z), device="cuda"),
            torch.full((count,), float(tail_mass), device="cuda"),
        ).cpu().numpy())
    return np.stack(outputs, 2).astype(np.float32)


def _geometry(query_ids):
    progress = np.where(query_ids < 3, -1.0, 1.0)
    lateral = np.take(np.asarray([-1.0, 0.0, 1.0], np.float32), query_ids % 3)
    return np.stack((progress, lateral), 2).astype(np.float32)


def _soft_topk_weights(scores, count: int, rank_temperature: float, membership_temperature: float):
    pairwise = (scores[:, :, None] - scores[:, None, :]) / rank_temperature
    ranks = 0.5 + torch.sigmoid(pairwise).sum(2)
    membership = torch.sigmoid((float(count) + 0.5 - ranks) / membership_temperature)
    return float(count) * membership / membership.sum(1, keepdim=True).clamp_min(1e-6)


def _metrics(model, descriptors, authority_base, costs, scenes, selected_count, gap):
    with torch.no_grad():
        scores = model(
            torch.from_numpy(descriptors).cuda(), torch.from_numpy(authority_base).cuda()
        ).cpu().numpy()
    selected = np.argsort(scores, axis=1)[:, :selected_count]
    selected_cost = np.take_along_axis(costs, selected, axis=1).mean(1)
    authority_selected = np.argsort(authority_base, axis=1)[:, :selected_count]
    authority_cost = np.take_along_axis(costs, authority_selected, axis=1).mean(1)
    all_cost = costs.mean(1)
    nominal_cost = costs[:, 4]
    oracle_cost = np.sort(costs, axis=1)[:, :selected_count].mean(1)
    left, right = np.triu_indices(costs.shape[1], 1)
    delta_cost = costs[:, left] - costs[:, right]
    delta_score = scores[:, left] - scores[:, right]
    mask = np.abs(delta_cost) >= float(gap)
    concordance = float(np.mean(delta_cost[mask] * delta_score[mask] > 0)) if np.any(mask) else 0.0
    lower = equal = higher = 0
    for scene in np.unique(scenes):
        local = scenes == scene
        candidate = float(selected_cost[local].mean())
        baseline = float(all_cost[local].mean())
        if candidate < baseline - 1e-8:
            lower += 1
        elif candidate > baseline + 1e-8:
            higher += 1
        else:
            equal += 1
    return {
        "group_count": int(len(costs)),
        "selected_action_count_per_group": int(selected_count),
        "all_action_mean_actual_future_visited_state_cost": float(all_cost.mean()),
        "selected_mean_actual_future_visited_state_cost": float(selected_cost.mean()),
        "nominal_full_progress_straight_actual_cost": float(nominal_cost.mean()),
        "authority_only_selected_actual_cost": float(authority_cost.mean()),
        "oracle_topk_actual_cost": float(oracle_cost.mean()),
        "relative_cost_reduction_vs_all_actions": float(1 - selected_cost.mean() / max(all_cost.mean(), 1e-8)),
        "relative_cost_reduction_vs_nominal": float(1 - selected_cost.mean() / max(nominal_cost.mean(), 1e-8)),
        "relative_cost_reduction_vs_authority_only": float(1 - selected_cost.mean() / max(authority_cost.mean(), 1e-8)),
        "pairwise_actual_cost_concordance": concordance,
        "pairwise_evaluable_count": int(mask.sum()),
        "scene_lower_equal_higher_vs_all_actions": [int(lower), int(equal), int(higher)],
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
    authority = SharedContextPiecewiseAnchorAuthorityCompiler(
        base, authority_artifact["shared_context_anchor_fractions"]
    ).cuda().eval()
    condition = config["authority_condition"]
    tolerance_domain = np.asarray(policy_artifact["risk_tolerance_domain"], np.float32)
    tolerance_z = 2 * (float(condition["risk_tolerance"]) - tolerance_domain[0]) / (tolerance_domain[1] - tolerance_domain[0]) - 1
    floor_domain = np.asarray(policy_artifact["floor_domain"], np.float32)
    floor_z = 2 * (float(condition["final_reliability_floor"]) - floor_domain[0]) / (floor_domain[1] - floor_domain[0]) - 1
    fractions = np.asarray(condition["attainable_budget_fractions"], np.float32)
    source_authority = _authority_curves(authority, source_groups, fractions, float(condition["alpha_fairness"]), tolerance_z, floor_z, float(condition["tail_mass"]))
    p201_authority = _authority_curves(authority, p201_groups, fractions, float(condition["alpha_fairness"]), tolerance_z, floor_z, float(condition["tail_mass"]))
    source_descriptors = np.concatenate((source_groups, source_authority, _geometry(source_queries)), 2)
    p201_descriptors = np.concatenate((p201_groups, p201_authority, _geometry(p201_queries)), 2)
    source_base = source_authority.mean(2)
    p201_base = p201_authority.mean(2)

    confirmation_only = bool(config.get("confirmation_only", False))
    development = source_scenes % int(config["split"]["development_scene_modulus"]) == int(config["split"]["development_scene_remainder"])
    train = ~development
    selector_artifact = None
    if confirmation_only:
        selector_artifact = torch.load(
            args.runs_root / config["frozen_selector"]["run"] / config["frozen_selector"]["artifact"], map_location="cuda"
        )
        mean = np.asarray(selector_artifact["input_mean"], np.float32)
        scale = np.asarray(selector_artifact["input_scale"], np.float32)
        target_mean = float(selector_artifact["target_log1p_mean"])
        target_scale = float(selector_artifact["target_log1p_scale"])
    else:
        mean = source_descriptors[train].reshape(-1, source_descriptors.shape[2]).mean(0)
        scale = source_descriptors[train].reshape(-1, source_descriptors.shape[2]).std(0).clip(min=1e-5)
        target = np.log1p(source_costs).astype(np.float32)
        target_mean = float(target[train].mean())
        target_scale = float(target[train].std().clip(min=1e-5))
    source_descriptors = ((source_descriptors - mean) / scale).astype(np.float32)
    p201_descriptors = ((p201_descriptors - mean) / scale).astype(np.float32)
    target = np.log1p(source_costs).astype(np.float32)
    target_z = (target - target_mean) / target_scale

    model_config = config["model"]
    element_width = int(selector_artifact["element_width"]) if selector_artifact else int(model_config["element_width"])
    context_width = int(selector_artifact["context_width"]) if selector_artifact else int(model_config["context_width"])
    residual_bound = float(selector_artifact["maximum_authority_residual"]) if selector_artifact else float(model_config["maximum_authority_residual"])
    model = AuthorityResidualTopK(
        source_descriptors.shape[2], element_width, context_width, residual_bound,
    ).cuda()
    last = 0.0
    if confirmation_only:
        model.load_state_dict(selector_artifact["model_state_dict"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        x = torch.from_numpy(source_descriptors).cuda()
        base_tensor = torch.from_numpy(source_base).cuda()
        y = torch.from_numpy(target_z).cuda()
        train_index = torch.from_numpy(np.flatnonzero(train)).cuda()
        left, right = np.triu_indices(int(config["action_group_size"]), 1)
        left = torch.from_numpy(left).cuda()
        right = torch.from_numpy(right).cuda()
        for step in range(int(model_config["steps"])):
            row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
            score = model(x[row], base_tensor[row])
            truth = y[row]
            weights = _soft_topk_weights(score, int(config["selected_action_count"]), float(model_config["rank_temperature"]), float(model_config["membership_temperature"]))
            decision_loss = torch.mean(torch.sum(weights * truth, 1) / float(config["selected_action_count"]))
            target_probability = F.softmax(-truth / float(model_config["listwise_temperature"]), 1)
            listwise_loss = torch.mean(torch.sum(-target_probability * F.log_softmax(-score / float(model_config["listwise_temperature"]), 1), 1))
            truth_delta = truth[:, left] - truth[:, right]
            score_delta = score[:, left] - score[:, right]
            pairwise_loss = F.softplus(-torch.sign(truth_delta) * score_delta / float(model_config["pairwise_temperature"]))
            pairwise_loss = pairwise_loss[torch.abs(truth_delta) >= float(model_config["pairwise_minimum_target_gap_z"])].mean()
            regression_loss = F.smooth_l1_loss(score, truth, beta=float(model_config["huber_beta_z"]))
            loss = (
                decision_loss
                + float(model_config["listwise_weight"]) * listwise_loss
                + float(model_config["pairwise_weight"]) * pairwise_loss
                + float(model_config["regression_weight"]) * regression_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
            if step % 500 == 0:
                print(f"P309 authority top-k step={step + 1} objective={last:.7f}", flush=True)

    source_metrics = None if confirmation_only else _metrics(
        model, source_descriptors[development], source_base[development], source_costs[development], source_scenes[development],
        int(config["selected_action_count"]), float(config["evaluation"]["pairwise_minimum_actual_cost_gap"]),
    )
    p201_metrics = _metrics(
        model, p201_descriptors, p201_base, p201_costs, p201_scenes,
        int(config["selected_action_count"]), float(config["evaluation"]["pairwise_minimum_actual_cost_gap"]),
    )
    decision = config["decision"]
    checks = {
        "P201_topk_actual_cost_reduction": p201_metrics["relative_cost_reduction_vs_all_actions"] >= float(decision["minimum_P201_relative_cost_reduction_vs_all_actions"]),
        "P201_pairwise_actual_cost_ranking": p201_metrics["pairwise_actual_cost_concordance"] >= float(decision["minimum_P201_pairwise_actual_cost_concordance"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_mean": mean,
        "input_scale": scale,
        "target_log1p_mean": target_mean,
        "target_log1p_scale": target_scale,
        "input_width": source_descriptors.shape[2],
        "element_width": element_width,
        "context_width": context_width,
        "maximum_authority_residual": residual_bound,
        "frozen_authority": config["frozen_authority"],
        "authority_condition": condition,
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
