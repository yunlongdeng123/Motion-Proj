"""Learn a context-adaptive residual quantile and conformalize it once by scene split."""

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
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula, _load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface, _paired_dataset
from scripts.run_worldsim_v67_p275_epistemic_lcb_surface import _ensemble
from scripts.run_worldsim_v67_p283_conformalized_epistemic_lcb_surface import ConformalizedLCBSurface


class ContextAdaptiveResidualQuantile(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(38, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.constant_(self.network[-1].bias, -1.2)

    def forward(self, feature, budget, tolerance):
        value = self.network(torch.cat((feature, budget[:, None], tolerance[:, None]), 1)).squeeze(1)
        return 0.15 * torch.sigmoid(value)


def _finite_quantile(values, delta):
    flat = np.sort(np.asarray(values).reshape(-1))
    rank = min(len(flat) - 1, max(0, int(np.ceil((len(flat) + 1) * (1 - float(delta)))) - 1))
    return float(flat[rank])


def _quantile_grid(model, feature, budgets, tolerances, budget_mean, budget_scale, tol_domain, chunk):
    result = np.empty((len(feature), len(tolerances), len(budgets)), np.float32)
    with torch.no_grad():
        for di, delta in enumerate(tolerances):
            tz = 2 * (float(delta) - tol_domain[0]) / (tol_domain[1] - tol_domain[0]) - 1
            for bi, budget in enumerate(budgets):
                bz = (np.log(float(budget)) - budget_mean) / budget_scale
                values = []
                for start in range(0, len(feature), chunk):
                    x = torch.from_numpy(feature[start:start + chunk]).cuda()
                    values.append(model(
                        x,
                        torch.full((len(x),), bz, device="cuda"),
                        torch.full((len(x),), tz, device="cuda"),
                    ).cpu().numpy())
                result[:, di, bi] = np.concatenate(values)
    return result


def _student_grid(model, feature, budgets, tolerances, budget_mean, budget_scale, tol_domain, chunk):
    result = []
    with torch.no_grad():
        for delta in tolerances:
            by_budget = []
            tz = 2 * (float(delta) - tol_domain[0]) / (tol_domain[1] - tol_domain[0]) - 1
            for budget in budgets:
                values = []
                bz = (np.log(float(budget)) - budget_mean) / budget_scale
                for start in range(0, len(feature), chunk):
                    x = torch.from_numpy(feature[start:start + chunk]).cuda()
                    values.append(model(
                        x,
                        torch.full((len(x),), bz, device="cuda"),
                        torch.full((len(x),), tz, device="cuda"),
                    ).cpu().numpy())
                by_budget.append(np.concatenate(values))
            result.append(np.stack(by_budget, 2))
    return np.stack(result, 1)


def _evaluate(student, quantile_model, feature, mean, truth, budgets, tolerances, corrections,
              budget_mean, budget_scale, tol_domain, chunk):
    predicted_quantiles = _quantile_grid(
        quantile_model, feature, budgets, tolerances, budget_mean, budget_scale, tol_domain, chunk
    )
    offsets = np.maximum(predicted_quantiles + corrections[None, :, None], 0)
    teacher = np.clip(mean[:, None] - offsets[:, :, None, :], 0, 1)
    candidate = _student_grid(
        student, feature, budgets, tolerances, budget_mean, budget_scale, tol_domain, chunk
    )
    coverage = np.asarray([
        np.mean(np.all(candidate[:, di] <= truth + 1e-7, axis=1)) for di in range(len(tolerances))
    ])
    teacher_coverage = np.asarray([
        np.mean(np.all(teacher[:, di] <= truth + 1e-7, axis=1)) for di in range(len(tolerances))
    ])
    desired = 1 - np.asarray(tolerances)
    return {
        "trajectory_count": int(len(feature)),
        "risk_tolerances": [float(v) for v in tolerances],
        "frozen_global_corrections": [float(v) for v in corrections],
        "mean_context_adaptive_offset": float(np.mean(offsets)),
        "surface_adaptive_teacher_probability_MAE": float(np.mean(np.abs(candidate - teacher))),
        "final_curve_adaptive_teacher_probability_MAE": float(np.mean(np.abs(candidate[:, :, -1] - teacher[:, :, -1]))),
        "simultaneous_horizon_empirical_coverages": [float(v) for v in coverage],
        "adaptive_teacher_empirical_coverages": [float(v) for v in teacher_coverage],
        "desired_coverages": [float(v) for v in desired],
        "maximum_simultaneous_horizon_undercoverage": float(np.max(np.maximum(desired - coverage, 0))),
        "mean_conservatism_vs_teacher": float(np.mean(truth[:, None] - candidate)),
        "budget_monotonicity_violations": int(np.sum(np.diff(candidate, axis=3) < -1e-7)),
        "horizon_monotonicity_violations": int(np.sum(np.diff(candidate, axis=2) > 1e-7)),
        "tolerance_monotonicity_violations": int(np.sum(np.diff(candidate, axis=1) < -1e-7)),
    }


def main():
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
    heldout_anchors = np.asarray(config["heldout_anchor_budgets"], np.float32)
    heldout_budgets = np.sqrt(heldout_anchors[:-1] * heldout_anchors[1:])
    domain = config["training_budget_domain"]
    fractions = (np.arange(int(config["training_budget_count"]), dtype=np.float32) + float(config["training_budget_offset"])) / int(config["training_budget_count"])
    training_budgets = np.exp(np.log(domain[0]) + fractions * (np.log(domain[1]) - np.log(domain[0]))).astype(np.float32)
    common = (
        members, ensemble, density, density_metadata, p199, copula, calibrator, horizons,
        float(config["boundary_state_cost"]["clearance_floor_m"]),
        int(config["teacher"]["monte_carlo_samples"]), seed,
        float(config["teacher"]["ignored_future_marginal_probability"]),
    )
    source_feature, source_truth, _, _, _, scenes, _ = _paired_dataset(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], common, anchors, training_budgets, heldout_budgets
    )
    p201_feature, _, _, p201_truth, _, _, _ = _paired_dataset(
        args.runs_root / config["p201_rows"]["run"] / config["p201_rows"]["artifact"], common, anchors, training_budgets, heldout_budgets
    )
    fresh_feature, _, _, fresh_truth, _, _, _ = _paired_dataset(
        args.runs_root / config["fresh_rows"]["run"] / config["fresh_rows"]["artifact"], common, anchors, training_budgets, heldout_budgets
    )
    base = torch.load(args.runs_root / config["frozen_p274"]["run"] / config["frozen_p274"]["artifact"], map_location="cuda")
    base_models = []
    for state in base["member_state_dicts"]:
        member = MonotoneRateSplineSurface(int(base["context_width"]), int(base["rate_knot_count"])).cuda()
        member.load_state_dict(state)
        base_models.append(member.eval())
    budget_mean = float(base["budget_log_mean"])
    budget_scale = float(base["budget_log_scale"])
    chunk = int(config["inference_chunk_size"])
    source_mean, _ = _ensemble(base_models, source_feature, training_budgets, budget_mean, budget_scale, chunk)
    p201_mean, _ = _ensemble(base_models, p201_feature, heldout_budgets, budget_mean, budget_scale, chunk)
    fresh_mean, _ = _ensemble(base_models, fresh_feature, heldout_budgets, budget_mean, budget_scale, chunk)

    calibration = scenes % int(config["split"]["calibration_scene_modulus"]) == int(config["split"]["calibration_scene_remainder"])
    fit = ~calibration
    fit_scores = np.max(source_mean[fit] - source_truth[fit], axis=1).astype(np.float32)
    calibration_scores = np.max(source_mean[calibration] - source_truth[calibration], axis=1).astype(np.float32)
    tolerances = np.asarray(config["training_risk_tolerances"], np.float32)
    heldout_tolerances = np.asarray(config["heldout_risk_tolerances"], np.float32)
    tolerance_domain = np.asarray(config["risk_tolerance_domain"], np.float32)
    budget_z = ((np.log(training_budgets) - budget_mean) / budget_scale).astype(np.float32)
    fit_x = torch.from_numpy(source_feature[fit]).cuda()
    fit_y = torch.from_numpy(fit_scores).cuda()
    budget_tensor = torch.from_numpy(budget_z).cuda()
    tolerance_z = torch.from_numpy((2 * (tolerances - tolerance_domain[0]) / (tolerance_domain[1] - tolerance_domain[0]) - 1).astype(np.float32)).cuda()
    quantile_config = config["quantile_model"]
    quantile_model = ContextAdaptiveResidualQuantile(int(quantile_config["width"])).cuda()
    quantile_optimizer = torch.optim.AdamW(
        quantile_model.parameters(), lr=float(quantile_config["learning_rate"]), weight_decay=float(quantile_config["weight_decay"])
    )
    quantile_loss = 0.0
    for step in range(int(quantile_config["steps"])):
        row = torch.randint(len(fit_x), (int(quantile_config["batch_size"]),), device="cuda")
        budget_index = torch.randint(len(training_budgets), (len(row),), device="cuda")
        tolerance_index = torch.randint(len(tolerances), (len(row),), device="cuda")
        prediction = quantile_model(fit_x[row], budget_tensor[budget_index], tolerance_z[tolerance_index])
        error = fit_y[row, budget_index] - prediction
        level = 1 - torch.from_numpy(tolerances).cuda()[tolerance_index]
        loss = torch.maximum(level * error, (level - 1) * error).mean()
        quantile_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        quantile_optimizer.step()
        quantile_loss = float(loss.detach())
        if step % 500 == 0:
            print(f"P291 adaptive residual quantile step={step + 1} pinball={quantile_loss:.7f}", flush=True)

    calibration_prediction_train = _quantile_grid(
        quantile_model, source_feature[calibration], training_budgets, tolerances,
        budget_mean, budget_scale, tolerance_domain, chunk,
    )
    training_corrections = np.asarray([
        _finite_quantile(calibration_scores - calibration_prediction_train[:, di], delta)
        for di, delta in enumerate(tolerances)
    ], np.float32)
    calibration_prediction_heldout = _quantile_grid(
        quantile_model, source_feature[calibration], training_budgets, heldout_tolerances,
        budget_mean, budget_scale, tolerance_domain, chunk,
    )
    heldout_corrections = np.asarray([
        _finite_quantile(calibration_scores - calibration_prediction_heldout[:, di], delta)
        for di, delta in enumerate(heldout_tolerances)
    ], np.float32)
    source_quantiles = _quantile_grid(
        quantile_model, source_feature, training_budgets, tolerances,
        budget_mean, budget_scale, tolerance_domain, chunk,
    )
    source_offsets = np.maximum(source_quantiles + training_corrections[None, :, None], 0)
    target = np.clip(source_mean[:, None] - source_offsets[:, :, None, :], 0, 1).astype(np.float32)

    student_config = config["student"]
    student = ConformalizedLCBSurface(
        int(student_config["context_width"]), int(student_config["budget_rate_knot_count"]), int(student_config["tolerance_rate_knot_count"])
    ).cuda()
    warm = torch.load(args.runs_root / config["frozen_p284"]["run"] / config["frozen_p284"]["artifact"], map_location="cuda")
    student.load_state_dict(warm["model_state_dict"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(student_config["learning_rate"]), weight_decay=float(student_config["weight_decay"]))
    x = torch.from_numpy(source_feature).cuda()
    y = torch.from_numpy(target).cuda()
    fit_index = torch.from_numpy(np.flatnonzero(fit)).cuda()
    student_loss = 0.0
    for step in range(int(student_config["steps"])):
        row = fit_index[torch.randint(len(fit_index), (int(student_config["batch_size"]),), device="cuda")]
        budget_index = torch.randint(len(training_budgets), (len(row),), device="cuda")
        tolerance_index = torch.randint(len(tolerances), (len(row),), device="cuda")
        loss = F.l1_loss(student(x[row], budget_tensor[budget_index], tolerance_z[tolerance_index]), y[row, tolerance_index, :, budget_index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        student_loss = float(loss.detach())
        if step % 500 == 0:
            print(f"P291 adaptive additive surface step={step + 1} mae={student_loss:.7f}", flush=True)

    p201 = _evaluate(
        student, quantile_model, p201_feature, p201_mean, p201_truth, heldout_budgets, heldout_tolerances,
        heldout_corrections, budget_mean, budget_scale, tolerance_domain, chunk,
    )
    fresh = _evaluate(
        student, quantile_model, fresh_feature, fresh_mean, fresh_truth, heldout_budgets, heldout_tolerances,
        heldout_corrections, budget_mean, budget_scale, tolerance_domain, chunk,
    )
    decision = config["decision"]
    checks = {
        "P201_adaptive_surface_fidelity": p201["surface_adaptive_teacher_probability_MAE"] <= float(decision["maximum_P201_surface_adaptive_teacher_MAE"]),
        "P201_empirical_simultaneous_horizon_coverage": p201["maximum_simultaneous_horizon_undercoverage"] <= float(decision["maximum_P201_simultaneous_horizon_undercoverage"]),
        "P201_reduces_mean_conservatism_vs_P284": p201["mean_conservatism_vs_teacher"] < float(decision["P284_P201_mean_conservatism"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    torch.save({
        "model_state_dict": student.state_dict(),
        "context_width": student_config["context_width"],
        "budget_rate_knot_count": student_config["budget_rate_knot_count"],
        "tolerance_rate_knot_count": student_config["tolerance_rate_knot_count"],
        "quantile_model_state_dict": quantile_model.state_dict(),
        "quantile_model_width": quantile_config["width"],
        "budget_log_mean": budget_mean,
        "budget_log_scale": budget_scale,
        "risk_tolerance_domain": tolerance_domain,
        "training_corrections": training_corrections,
        "heldout_corrections": heldout_corrections,
        "base_model": config["frozen_p274"],
    }, run_dir / config["model_artifact"])
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "method": "context_quantile_regression_plus_scene_split_global_residual_correction",
        "calibration": {
            "trajectory_count": int(calibration.sum()),
            "score_count": int(calibration_scores.size),
            "training_risk_tolerances": tolerances.tolist(),
            "training_global_corrections": training_corrections.tolist(),
            "heldout_risk_tolerances": heldout_tolerances.tolist(),
            "heldout_global_corrections": heldout_corrections.tolist(),
        },
        "training": {
            "trajectory_count": int(fit.sum()),
            "final_quantile_pinball_loss": quantile_loss,
            "final_adaptive_teacher_MAE": student_loss,
        },
        "P201_post_hoc_development": p201,
        "P243_consumed_descriptive": fresh,
        "P284_P201_mean_conservatism_baseline": float(decision["P284_P201_mean_conservatism"]),
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
