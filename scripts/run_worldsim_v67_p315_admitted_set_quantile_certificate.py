"""Predict calibrated cost quantiles for the task-conditioned top-2 admitted action set."""

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
from scripts.run_worldsim_v67_p309_authority_residual_topk_admission import AuthorityResidualTopK, _action_groups, _authority_curves, _geometry
from scripts.run_worldsim_v67_p311_progress_conditioned_authority_admission import ProgressConditionedAdmission
from scripts.run_worldsim_v67_p313_maneuver_conditioned_authority_admission import ManeuverConditionedAdmission


class AdmittedSetQuantileHead(nn.Module):
    """Monotone q50/q80/q95 log-cost head."""

    def __init__(self, input_width: int, hidden_dimensions) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        raw = self.network(features)
        median = F.softplus(raw[:, :1])
        increments = F.softplus(raw[:, 1:])
        return torch.cat((median, median + torch.cumsum(increments, 1)), 1)


class CeilingConditionedSelectiveAuthority(nn.Module):
    """Bounded scene/task correction around a frozen conservative set-cost score."""

    def __init__(self, input_width: int, hidden_dimensions, maximum_residual: float) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.maximum_residual = float(maximum_residual)

    def forward(self, features, frozen_score):
        residual = self.maximum_residual * torch.tanh(self.network(features).squeeze(1))
        return frozen_score + residual, residual


@torch.no_grad()
def _task_examples(descriptors, selector_score, costs, queries, scenes, progress_model, maneuver_model,
                   progress_values, commands, lateral_weight, selected_count):
    x = torch.from_numpy(descriptors).cuda()
    base = torch.from_numpy(selector_score).cuda()
    progress_deficit = torch.from_numpy(np.where(queries < 3, 0.5, 0.0).astype(np.float32)).cuda()
    lateral = torch.from_numpy(np.take(np.asarray([-1.0, 0.0, 1.0], np.float32), queries % 3)).cuda()
    features, targets, example_scenes, conditions = [], [], [], []
    for progress in progress_values:
        progress_tensor = torch.full((len(x),), float(progress), device="cuda")
        progress_score = progress_model(x, base, progress_deficit, progress_tensor)
        for command in commands:
            command_tensor = torch.full((len(x),), float(command), device="cuda")
            distance = torch.abs(lateral - command_tensor[:, None])
            score = maneuver_model(x, progress_score, distance, command_tensor, float(lateral_weight)).cpu().numpy()
            selected = np.argsort(score, axis=1)[:, :selected_count]
            row = np.arange(len(descriptors))[:, None]
            chosen = descriptors[row, selected]
            chosen_score = score[row, selected]
            condition = np.broadcast_to(
                np.asarray([progress, command], np.float32)[None], (len(descriptors), 2)
            )
            feature = np.concatenate((
                chosen.mean(1), chosen.std(1), chosen.max(1),
                chosen_score.mean(1, keepdims=True), chosen_score.min(1, keepdims=True), chosen_score.max(1, keepdims=True),
                condition,
            ), 1).astype(np.float32)
            selected_cost = costs[row, selected]
            target = np.log1p(selected_cost.max(1)).astype(np.float32)
            features.append(feature)
            targets.append(target)
            example_scenes.append(scenes)
            conditions.append(condition)
    return (
        np.concatenate(features), np.concatenate(targets), np.concatenate(example_scenes), np.concatenate(conditions)
    )


def _evaluate(model, features, target, conditions, quantiles, offsets):
    with torch.no_grad():
        predictions = model(torch.from_numpy(features).cuda()).cpu().numpy()
    predictions = np.maximum.accumulate(predictions + offsets[None], axis=1)
    coverage = np.mean(target[:, None] <= predictions, axis=0)
    error = target[:, None] - predictions
    pinball = np.maximum(quantiles[None] * error, (quantiles[None] - 1) * error)
    by_condition = {}
    for progress, command in np.unique(conditions, axis=0):
        local = np.isclose(conditions[:, 0], progress) & np.isclose(conditions[:, 1], command)
        by_condition[f"progress={float(progress)},command={float(command)}"] = {
            "example_count": int(local.sum()),
            "empirical_coverages": [float(value) for value in np.mean(target[local, None] <= predictions[local], axis=0)],
            "median_absolute_log_cost_error": float(np.mean(np.abs(predictions[local, 0] - target[local]))),
        }
    return {
        "example_count": int(len(target)),
        "quantile_levels": [float(value) for value in quantiles],
        "empirical_coverages": [float(value) for value in coverage],
        "maximum_quantile_undercoverage": float(np.max(quantiles - coverage)),
        "median_absolute_log_cost_error": float(np.mean(np.abs(predictions[:, 0] - target))),
        "mean_pinball_loss": float(np.mean(pinball)),
        "mean_q95_minus_q50_width": float(np.mean(predictions[:, 2] - predictions[:, 0])),
        "quantile_order_violations": int(np.sum(np.diff(predictions, axis=1) < -1e-8)),
        "by_task_condition": by_condition,
    }


def _selective_metrics(score, target, conditions, ceiling_values):
    actual_cost = np.expm1(target)
    by_ceiling = {}
    coverages, unsafe_rates = [], []
    previous_admitted = None
    monotonicity_violations = 0
    for ceiling in ceiling_values:
        admitted = score <= float(ceiling)
        if previous_admitted is not None:
            monotonicity_violations += int(np.sum(previous_admitted & ~admitted))
        previous_admitted = admitted
        count = int(admitted.sum())
        unsafe_rate = float(np.mean(target[admitted] > ceiling)) if count else 0.0
        all_unsafe_rate = float(np.mean(target > ceiling))
        admitted_cost = float(np.mean(actual_cost[admitted])) if count else None
        by_condition = {}
        for progress, command in np.unique(conditions, axis=0):
            local = np.isclose(conditions[:, 0], progress) & np.isclose(conditions[:, 1], command)
            local_admitted = local & admitted
            local_count = int(local_admitted.sum())
            by_condition[f"progress={float(progress)},command={float(command)}"] = {
                "admission_coverage": float(local_count / max(int(local.sum()), 1)),
                "unsafe_admission_rate": float(np.mean(target[local_admitted] > ceiling)) if local_count else 0.0,
            }
        by_ceiling[str(float(ceiling))] = {
            "log_cost_ceiling": float(ceiling),
            "actual_cost_ceiling": float(np.expm1(ceiling)),
            "admitted_count": count,
            "admission_coverage": float(np.mean(admitted)),
            "unsafe_admission_rate": unsafe_rate,
            "unselective_unsafe_rate": all_unsafe_rate,
            "unsafe_rate_relative_reduction": float((all_unsafe_rate - unsafe_rate) / max(all_unsafe_rate, 1e-8)),
            "mean_admitted_actual_cost": admitted_cost,
            "mean_all_actual_cost": float(np.mean(actual_cost)),
            "by_task_condition": by_condition,
        }
        coverages.append(float(np.mean(admitted)))
        unsafe_rates.append(unsafe_rate)
    return {
        "example_count": int(len(target)),
        "mean_admission_coverage": float(np.mean(coverages)),
        "highest_ceiling_admission_coverage": float(coverages[-1]),
        "maximum_unsafe_admission_rate": float(np.max(unsafe_rates)),
        "mean_unsafe_admission_rate": float(np.mean(unsafe_rates)),
        "ceiling_admission_monotonicity_violations": monotonicity_violations,
        "by_ceiling": by_ceiling,
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
    confirmation_only = bool(config.get("confirmation_only", False))
    selective_authority_training = bool(config.get("selective_authority_training", False))
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
    if config["p201_rows"] == config["source_rows"]:
        p201_groups, p201_costs, p201_scenes, p201_queries = source_groups, source_costs, source_scenes, source_queries
    else:
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
    input_mean = np.asarray(selector_artifact["input_mean"], np.float32)
    input_scale = np.asarray(selector_artifact["input_scale"], np.float32)
    source_descriptor = ((source_descriptor - input_mean) / input_scale).astype(np.float32)
    p201_descriptor = ((p201_descriptor - input_mean) / input_scale).astype(np.float32)
    selector = AuthorityResidualTopK(
        int(selector_artifact["input_width"]), int(selector_artifact["element_width"]),
        int(selector_artifact["context_width"]), float(selector_artifact["maximum_authority_residual"]),
    ).cuda()
    selector.load_state_dict(selector_artifact["model_state_dict"])
    selector.eval()
    with torch.no_grad():
        source_selector_score = selector(torch.from_numpy(source_descriptor).cuda(), torch.from_numpy(source_authority_base).cuda()).cpu().numpy()
        p201_selector_score = selector(torch.from_numpy(p201_descriptor).cuda(), torch.from_numpy(p201_authority_base).cuda()).cpu().numpy()
    progress_artifact = torch.load(args.runs_root / config["frozen_progress_compiler"]["run"] / config["frozen_progress_compiler"]["artifact"], map_location="cuda")
    progress_model = ProgressConditionedAdmission(
        int(progress_artifact["input_width"]), int(progress_artifact["hidden_width"]), float(progress_artifact["maximum_rate_adjustment"])
    ).cuda()
    progress_model.load_state_dict(progress_artifact["model_state_dict"])
    progress_model.eval()
    maneuver_artifact = torch.load(args.runs_root / config["frozen_maneuver_compiler"]["run"] / config["frozen_maneuver_compiler"]["artifact"], map_location="cuda")
    maneuver_model = ManeuverConditionedAdmission(
        int(maneuver_artifact["input_width"]), int(maneuver_artifact["hidden_width"]), float(maneuver_artifact["maximum_rate_adjustment"])
    ).cuda()
    maneuver_model.load_state_dict(maneuver_artifact["model_state_dict"])
    maneuver_model.eval()

    source_feature, source_target, source_example_scenes, source_conditions = _task_examples(
        source_descriptor, source_selector_score, source_costs, source_queries, source_scenes, progress_model, maneuver_model,
        config["training_progress_preferences"], config["training_lateral_commands"],
        float(config["lateral_preference_weight"]), int(config["selected_action_count"]),
    )
    p201_feature, p201_target, _, p201_conditions = _task_examples(
        p201_descriptor, p201_selector_score, p201_costs, p201_queries, p201_scenes, progress_model, maneuver_model,
        config["heldout_progress_preferences"], config["heldout_lateral_commands"],
        float(config["lateral_preference_weight"]), int(config["selected_action_count"]),
    )
    if selective_authority_training:
        certificate = torch.load(
            args.runs_root / config["frozen_certificate"]["run"] / config["frozen_certificate"]["artifact"],
            map_location="cuda",
        )
        feature_mean = np.asarray(certificate["input_mean"], np.float32)
        feature_scale = np.asarray(certificate["input_scale"], np.float32)
        offsets = np.asarray(certificate["calibration_offsets"], np.float32)
        source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
        p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
        quantile_model = AdmittedSetQuantileHead(
            int(certificate["input_width"]), certificate["hidden_dimensions"]
        ).cuda()
        quantile_model.load_state_dict(certificate["model_state_dict"])
        quantile_model.eval()
        with torch.no_grad():
            source_quantiles = quantile_model(torch.from_numpy(source_feature).cuda()).cpu().numpy()
            p201_quantiles = quantile_model(torch.from_numpy(p201_feature).cuda()).cpu().numpy()
        source_frozen_score = np.maximum.accumulate(source_quantiles + offsets[None], axis=1)[:, 2].astype(np.float32)
        p201_frozen_score = np.maximum.accumulate(p201_quantiles + offsets[None], axis=1)[:, 2].astype(np.float32)

        split = config["split"]
        modulus = int(split["scene_modulus"])
        calibration = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
        development = source_example_scenes % modulus == int(split["development_scene_remainder"])
        train = ~(calibration | development)
        model_config = config["selective_model"]
        anchor_levels = np.asarray(config["training_ceiling_quantile_levels"], np.float32)
        evaluation_levels = np.asarray(config["heldout_ceiling_quantile_levels"], np.float32)
        anchor_ceilings = np.quantile(source_target[train], anchor_levels).astype(np.float32)
        evaluation_ceilings = np.quantile(source_target[train], evaluation_levels).astype(np.float32)
        model = CeilingConditionedSelectiveAuthority(
            source_feature.shape[1], model_config["hidden_dimensions"], float(model_config["maximum_residual"])
        ).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
        )
        x = torch.from_numpy(source_feature).cuda()
        y = torch.from_numpy(source_target).cuda()
        frozen_score = torch.from_numpy(source_frozen_score).cuda()
        ceiling_tensor = torch.from_numpy(anchor_ceilings).cuda()
        train_index = torch.from_numpy(np.flatnonzero(train)).cuda()
        temperature = float(model_config["admission_temperature"])
        last = 0.0
        for step in range(int(model_config["steps"])):
            row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
            ceiling = ceiling_tensor[torch.randint(len(ceiling_tensor), (len(row),), device="cuda")]
            risk_score, residual = model(x[row], frozen_score[row])
            safe = (y[row] <= ceiling).float()
            logits = (ceiling - risk_score) / temperature
            loss = (
                F.binary_cross_entropy_with_logits(logits, safe)
                + float(model_config["cost_regression_weight"]) * F.smooth_l1_loss(risk_score, y[row])
                + float(model_config["residual_l2_weight"]) * residual.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
            if step % 500 == 0:
                print(f"P317 selective authority step={step + 1} loss={last:.7f}", flush=True)

        model.eval()
        with torch.no_grad():
            source_score = model(x, frozen_score)[0].cpu().numpy()
            p201_score = model(
                torch.from_numpy(p201_feature).cuda(), torch.from_numpy(p201_frozen_score).cuda()
            )[0].cpu().numpy()
        calibration_residual = source_target[calibration] - source_score[calibration]
        calibration_margin = max(
            0.0, float(np.quantile(calibration_residual, float(config["calibration_score_quantile"])))
        )
        source_score = source_score + calibration_margin
        p201_score = p201_score + calibration_margin
        source_metrics = _selective_metrics(
            source_score[development], source_target[development], source_conditions[development], evaluation_ceilings
        )
        p201_metrics = _selective_metrics(p201_score, p201_target, p201_conditions, evaluation_ceilings)
        frozen_source_metrics = _selective_metrics(
            source_frozen_score[development], source_target[development], source_conditions[development], evaluation_ceilings
        )
        frozen_p201_metrics = _selective_metrics(
            p201_frozen_score, p201_target, p201_conditions, evaluation_ceilings
        )
        decision = config["decision"]
        checks = {
            "P201_selective_authority_risk": p201_metrics["maximum_unsafe_admission_rate"] <= float(decision["maximum_P201_unsafe_admission_rate"]),
            "P201_selective_authority_coverage": p201_metrics["mean_admission_coverage"] >= float(decision["minimum_P201_mean_admission_coverage"]),
            "P201_ceiling_admission_monotonicity": p201_metrics["ceiling_admission_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_admission_monotonicity_violations"]),
        }
        verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_width": source_feature.shape[1],
            "hidden_dimensions": model_config["hidden_dimensions"],
            "maximum_residual": float(model_config["maximum_residual"]),
            "calibration_margin": calibration_margin,
            "training_ceiling_quantile_levels": anchor_levels,
            "training_log_cost_ceilings": anchor_ceilings,
            "heldout_ceiling_quantile_levels": evaluation_levels,
            "heldout_log_cost_ceilings": evaluation_ceilings,
            "frozen_certificate": config["frozen_certificate"],
        }, run_dir / config["model_artifact"])
        summary = {
            "schema_version": config["output_schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "role": config["role"],
            "training": {
                "example_count": int(train.sum()),
                "steps": int(model_config["steps"]),
                "final_loss": last,
                "training_ceiling_quantile_levels": [float(value) for value in anchor_levels],
                "training_log_cost_ceilings": [float(value) for value in anchor_ceilings],
            },
            "calibration": {
                "example_count": int(calibration.sum()),
                "score_quantile": float(config["calibration_score_quantile"]),
                "nonnegative_log_cost_margin": calibration_margin,
            },
            "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
            "source_development": source_metrics,
            "source_frozen_q95_baseline": frozen_source_metrics,
            "P201_post_hoc_development": p201_metrics,
            "P201_frozen_q95_baseline": frozen_p201_metrics,
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
        return
    if confirmation_only:
        certificate = torch.load(
            args.runs_root / config["frozen_certificate"]["run"] / config["frozen_certificate"]["artifact"],
            map_location="cuda",
        )
        feature_mean = np.asarray(certificate["input_mean"], np.float32)
        feature_scale = np.asarray(certificate["input_scale"], np.float32)
        quantiles = np.asarray(certificate["quantile_levels"], np.float32)
        offsets = np.asarray(certificate["calibration_offsets"], np.float32)
        hidden_dimensions = certificate["hidden_dimensions"]
        model = AdmittedSetQuantileHead(int(certificate["input_width"]), hidden_dimensions).cuda()
        model.load_state_dict(certificate["model_state_dict"])
        model.eval()
        p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
        source_metrics = None
        p201_metrics = _evaluate(model, p201_feature, p201_target, p201_conditions, quantiles, offsets)
        training_summary = {"example_count": 0, "steps": 0, "final_pinball_loss": None}
        calibration_summary = {
            "example_count": 0,
            "additive_offsets": [float(value) for value in offsets],
            "reused_from": config["frozen_certificate"],
        }
    else:
        split = config["split"]
        modulus = int(split["scene_modulus"])
        calibration = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
        development = source_example_scenes % modulus == int(split["development_scene_remainder"])
        train = ~(calibration | development)
        feature_mean = source_feature[train].mean(0)
        feature_scale = source_feature[train].std(0).clip(min=1e-5)
        source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
        p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
        model_config = config["model"]
        quantiles = np.asarray(config["quantile_levels"], np.float32)
        hidden_dimensions = model_config["hidden_dimensions"]
        model = AdmittedSetQuantileHead(source_feature.shape[1], hidden_dimensions).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        x = torch.from_numpy(source_feature).cuda()
        y = torch.from_numpy(source_target).cuda()
        q = torch.from_numpy(quantiles).cuda()
        train_index = torch.from_numpy(np.flatnonzero(train)).cuda()
        last = 0.0
        for step in range(int(model_config["steps"])):
            row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
            prediction = model(x[row])
            error = y[row, None] - prediction
            loss = torch.maximum(q[None] * error, (q[None] - 1) * error).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
            if step % 500 == 0:
                print(f"P315 admitted-set quantiles step={step + 1} pinball={last:.7f}", flush=True)

        with torch.no_grad():
            calibration_prediction = model(x[torch.from_numpy(np.flatnonzero(calibration)).cuda()]).cpu().numpy()
        residual = source_target[calibration, None] - calibration_prediction
        offsets = np.asarray([np.quantile(residual[:, index], float(level)) for index, level in enumerate(quantiles)], np.float32)
        source_metrics = _evaluate(model, source_feature[development], source_target[development], source_conditions[development], quantiles, offsets)
        p201_metrics = _evaluate(model, p201_feature, p201_target, p201_conditions, quantiles, offsets)
        training_summary = {"example_count": int(train.sum()), "steps": int(model_config["steps"]), "final_pinball_loss": last}
        calibration_summary = {"example_count": int(calibration.sum()), "additive_offsets": [float(value) for value in offsets]}
    decision = config["decision"]
    checks = {
        "P201_admitted_set_quantile_coverage": p201_metrics["maximum_quantile_undercoverage"] <= float(decision["maximum_P201_quantile_undercoverage"]),
        "P201_admitted_set_median_fidelity": p201_metrics["median_absolute_log_cost_error"] <= float(decision["maximum_P201_median_absolute_log_cost_error"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_width": source_feature.shape[1],
        "hidden_dimensions": hidden_dimensions,
        "input_mean": feature_mean,
        "input_scale": feature_scale,
        "quantile_levels": quantiles,
        "calibration_offsets": offsets,
        "frozen_maneuver_compiler": config["frozen_maneuver_compiler"],
    }, run_dir / config["model_artifact"])
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "training": training_summary,
        "calibration": calibration_summary,
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
