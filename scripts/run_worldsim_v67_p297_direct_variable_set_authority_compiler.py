"""Directly compile attainable fractions to variable-set budgets, bypassing a dual at inference."""

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
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p256_group_budget_dual_compiler import _groups
from scripts.run_worldsim_v67_p279_epistemic_tail_cvar_allocator import EpistemicTailCVaRAllocator
from scripts.run_worldsim_v67_p280_epistemic_tail_cvar_group_dual import _allocate, _risk, _target_prices
from scripts.run_worldsim_v67_p283_conformalized_epistemic_lcb_surface import ConformalizedLCBSurface


class AttentiveDirectAuthorityCompiler(EpistemicTailCVaRAllocator):
    """P295 allocator plus a zero-gated equivariant self-attention interaction block."""

    def __init__(self, element_width, context_width, knot_count, attention_heads):
        super().__init__(element_width, context_width, knot_count)
        self.attention_heads = int(attention_heads)
        self.attention = nn.MultiheadAttention(element_width, self.attention_heads, batch_first=True)
        self.attention_gain = nn.Parameter(torch.zeros(()))

    def forward(self, groups, price, alpha, beta, floor, tail_mass):
        encoded = self.element(groups)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        encoded = encoded + torch.tanh(self.attention_gain) * attended
        mean = encoded.mean(1)
        std = torch.sqrt(encoded.var(1, unbiased=False) + 1e-6)
        maximum = encoded.amax(1)
        size = groups.shape[1]
        context = self.context(torch.cat((
            encoded,
            mean[:, None].expand(-1, size, -1),
            std[:, None].expand(-1, size, -1),
            maximum[:, None].expand(-1, size, -1),
            alpha[:, None, None].expand(-1, size, 1),
            beta[:, None, None].expand(-1, size, 1),
            tail_mass[:, None, None].expand(-1, size, 1),
        ), 2))
        return torch.tanh(
            self.intercept(context).squeeze(2)
            - self._integral(self.price_rates(context), price[:, None].expand(-1, size))
            + self._integral(self.floor_rates(context), floor[:, None].expand(-1, size))
        )


def _predict_direct(model, groups, alphas, tolerance_z, floor_z, tails, fractions):
    x = torch.from_numpy(groups).cuda()
    by_alpha = []
    with torch.no_grad():
        for alpha in alphas:
            by_tolerance = []
            for tolerance in tolerance_z:
                by_floor = []
                for floor in floor_z:
                    by_tail = []
                    for tail in tails:
                        by_tail.append(np.stack([
                            model(
                                x,
                                torch.full((len(x),), float(1 - 2 * fraction), device="cuda"),
                                torch.full((len(x),), float(alpha), device="cuda"),
                                torch.full((len(x),), float(tolerance), device="cuda"),
                                torch.full((len(x),), float(floor), device="cuda"),
                                torch.full((len(x),), float(tail), device="cuda"),
                            ).cpu().numpy()
                            for fraction in fractions
                        ], 1))
                    by_floor.append(np.stack(by_tail, 1))
                by_tolerance.append(np.stack(by_floor, 1))
            by_alpha.append(np.stack(by_tolerance, 1))
    return np.stack(by_alpha, 1)


def _evaluate(model, policy, teacher, groups, target_price, low, high, alphas, tolerance_z, floors,
              floor_z, tails, fractions, price_mean, price_scale, epsilon, penalty, chunk):
    torch.cuda.synchronize()
    started = time.monotonic()
    candidate_budget = _predict_direct(model, groups, alphas, tolerance_z, floor_z, tails, fractions)
    torch.cuda.synchronize()
    forward = time.monotonic() - started
    teacher_budget = _allocate(policy, groups, target_price.astype(np.float32), alphas, tolerance_z, floor_z, tails, chunk)
    candidate_cost = np.mean((candidate_budget + 1) / 2, 6)
    teacher_cost = np.mean((teacher_budget + 1) / 2, 6)
    attained = (candidate_cost - low[:, :, :, :, :, None]) / np.clip(high - low, 1e-6, None)[:, :, :, :, :, None]
    candidate_value = _risk(teacher, groups, candidate_budget.astype(np.float32), alphas, tolerance_z, floors, tails, epsilon, penalty, chunk)
    teacher_value = _risk(teacher, groups, teacher_budget.astype(np.float32), alphas, tolerance_z, floors, tails, epsilon, penalty, chunk)
    price = np.exp(price_mean + price_scale * target_price)
    regret = (teacher_value - price * teacher_cost) - (candidate_value - price * candidate_cost)
    return {
        "group_count": int(len(groups)),
        "group_size": int(groups.shape[1]),
        "normalized_budget_MAE_vs_bisection_teacher": float(np.mean(np.abs(candidate_budget - teacher_budget))),
        "attained_budget_fraction_MAE": float(np.mean(np.abs(attained - fractions[None, None, None, None, None, :]))),
        "mean_frozen_composite_Lagrangian_group_utility_regret": float(np.mean(regret)),
        "fraction_budget_monotonicity_violations": int(np.sum(np.diff(candidate_budget, axis=5) < -1e-7)),
        "direct_student_forward_seconds": forward,
    }


def _aggregate(rows):
    keys = (
        "normalized_budget_MAE_vs_bisection_teacher",
        "attained_budget_fraction_MAE",
        "mean_frozen_composite_Lagrangian_group_utility_regret",
        "direct_student_forward_seconds",
    )
    return {
        "evaluated_group_sizes": [int(key) for key in rows],
        **{key: float(np.mean([row[key] for row in rows.values()])) for key in keys},
        "fraction_budget_monotonicity_violations": int(sum(row["fraction_budget_monotonicity_violations"] for row in rows.values())),
        "by_group_size": rows,
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
    torch.manual_seed(int(config["seed"]))
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
    common = (members, ensemble, density, density_metadata, p199, copula, calibrator, horizons)
    tail = (
        float(config["boundary_state_cost"]["clearance_floor_m"]),
        int(config["teacher"]["monte_carlo_samples"]),
        int(config["seed"]),
        float(config["teacher"]["ignored_future_marginal_probability"]),
    )
    source_feature, _, _, source_scenes, _ = _dataset(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], *common, anchors, *tail
    )
    p201_feature, _, _, p201_scenes, _ = _dataset(
        args.runs_root / config["p201_rows"]["run"] / config["p201_rows"]["artifact"], *common, anchors, *tail
    )
    surface_artifact = torch.load(args.runs_root / config["frozen_surface"]["run"] / config["frozen_surface"]["artifact"], map_location="cuda")
    teacher = ConformalizedLCBSurface(
        int(surface_artifact["context_width"]), int(surface_artifact["budget_rate_knot_count"]), int(surface_artifact["tolerance_rate_knot_count"])
    ).cuda()
    teacher.load_state_dict(surface_artifact["model_state_dict"])
    teacher.eval()
    policy_artifact = torch.load(args.runs_root / config["frozen_policy"]["run"] / config["frozen_policy"]["artifact"], map_location="cuda")
    policy = EpistemicTailCVaRAllocator(
        int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
    ).cuda()
    policy.load_state_dict(policy_artifact["model_state_dict"])
    policy.eval()

    alphas = np.asarray(config["training_alpha_fairness"], np.float32)
    heldout_alphas = np.asarray(config["heldout_alpha_fairness"], np.float32)
    tolerances = np.asarray(config["training_risk_tolerances"], np.float32)
    heldout_tolerances = np.asarray(config["heldout_risk_tolerances"], np.float32)
    tolerance_domain = np.asarray(policy_artifact["risk_tolerance_domain"], np.float32)
    tolerance_z = (2 * (tolerances - tolerance_domain[0]) / (tolerance_domain[1] - tolerance_domain[0]) - 1).astype(np.float32)
    heldout_tolerance_z = (2 * (heldout_tolerances - tolerance_domain[0]) / (tolerance_domain[1] - tolerance_domain[0]) - 1).astype(np.float32)
    floors = np.asarray(config["training_final_reliability_floors"], np.float32)
    heldout_floors = np.asarray(config["heldout_final_reliability_floors"], np.float32)
    floor_domain = np.asarray(policy_artifact["floor_domain"], np.float32)
    floor_z = (2 * (floors - floor_domain[0]) / (floor_domain[1] - floor_domain[0]) - 1).astype(np.float32)
    heldout_floor_z = (2 * (heldout_floors - floor_domain[0]) / (floor_domain[1] - floor_domain[0]) - 1).astype(np.float32)
    tails = np.asarray(config["training_tail_masses"], np.float32)
    heldout_tails = np.asarray(config["heldout_tail_masses"], np.float32)
    fractions = np.asarray(config["training_attainable_budget_fractions"], np.float32)
    heldout_fractions = np.asarray(config["heldout_attainable_budget_fractions"], np.float32)
    chunk = int(config["inference_chunk_size"])
    bisection_steps = int(config["teacher_bisection_steps"])
    train = {}
    for size in map(int, config["training_group_sizes"]):
        groups, scenes = _groups(source_feature, source_scenes, size)
        target_price, _, _ = _target_prices(policy, groups, alphas, tolerance_z, floor_z, tails, fractions, bisection_steps, chunk)
        development = scenes % int(config["split"]["development_scene_modulus"]) == int(config["split"]["development_scene_remainder"])
        train[size] = (
            torch.from_numpy(groups).cuda(),
            torch.from_numpy(target_price).cuda(),
            torch.from_numpy(np.flatnonzero(~development)).cuda(),
        )

    fraction_input = torch.from_numpy(1 - 2 * fractions).cuda()
    alpha_tensor = torch.from_numpy(alphas).cuda()
    tolerance_tensor = torch.from_numpy(tolerance_z).cuda()
    floor_tensor = torch.from_numpy(floor_z).cuda()
    tail_tensor = torch.from_numpy(tails).cuda()
    model_config = config["student"]
    if "attention_heads" in model_config:
        student = AttentiveDirectAuthorityCompiler(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]),
            int(policy_artifact["rate_knot_count"]), int(model_config["attention_heads"]),
        ).cuda()
        direct_reference = config["frozen_direct"]
        direct_artifact = torch.load(
            args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
        )
        student.load_state_dict(direct_artifact["model_state_dict"], strict=False)
    else:
        student = EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
        ).cuda()
        student.load_state_dict(policy_artifact["model_state_dict"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    sizes = list(train)
    last = 0.0
    for step in range(int(model_config["steps"])):
        size = sizes[step % len(sizes)]
        groups, target_price, index = train[size]
        row = index[torch.randint(len(index), (int(model_config["batch_size"]),), device="cuda")]
        alpha_index = torch.randint(len(alphas), (len(row),), device="cuda")
        tolerance_index = torch.randint(len(tolerances), (len(row),), device="cuda")
        floor_index = torch.randint(len(floors), (len(row),), device="cuda")
        tail_index = torch.randint(len(tails), (len(row),), device="cuda")
        fraction_index = torch.randint(len(fractions), (len(row),), device="cuda")
        with torch.no_grad():
            target_budget = policy(
                groups[row], target_price[row, alpha_index, tolerance_index, floor_index, tail_index, fraction_index],
                alpha_tensor[alpha_index], tolerance_tensor[tolerance_index], floor_tensor[floor_index], tail_tensor[tail_index],
            )
        prediction = student(
            groups[row], fraction_input[fraction_index], alpha_tensor[alpha_index], tolerance_tensor[tolerance_index],
            floor_tensor[floor_index], tail_tensor[tail_index],
        )
        loss = F.l1_loss(prediction, target_budget)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        if step % 500 == 0:
            print(f"P297 direct authority step={step + 1} size={size} budget_mae={last:.7f}", flush=True)

    price_mean = float(policy_artifact["shadow_price_log_mean"])
    price_scale = float(policy_artifact["shadow_price_log_scale"])
    epsilon = float(policy_artifact["alpha_utility_epsilon"])
    penalty = float(policy_artifact["tail_CVaR_shortfall_penalty"])
    source = {}
    p201 = {}
    for size in map(int, config["heldout_group_sizes"]):
        groups, scenes = _groups(source_feature, source_scenes, size)
        development = scenes % int(config["split"]["development_scene_modulus"]) == int(config["split"]["development_scene_remainder"])
        target_price, low, high = _target_prices(
            policy, groups[development], heldout_alphas, heldout_tolerance_z, heldout_floor_z,
            heldout_tails, heldout_fractions, bisection_steps, chunk,
        )
        source[str(size)] = _evaluate(
            student, policy, teacher, groups[development], target_price, low, high, heldout_alphas,
            heldout_tolerance_z, heldout_floors, heldout_floor_z, heldout_tails, heldout_fractions,
            price_mean, price_scale, epsilon, penalty, chunk,
        )
        groups, _ = _groups(p201_feature, p201_scenes, size)
        target_price, low, high = _target_prices(
            policy, groups, heldout_alphas, heldout_tolerance_z, heldout_floor_z,
            heldout_tails, heldout_fractions, bisection_steps, chunk,
        )
        p201[str(size)] = _evaluate(
            student, policy, teacher, groups, target_price, low, high, heldout_alphas,
            heldout_tolerance_z, heldout_floors, heldout_floor_z, heldout_tails, heldout_fractions,
            price_mean, price_scale, epsilon, penalty, chunk,
        )
    source = _aggregate(source)
    p201 = _aggregate(p201)
    decision = config["decision"]
    checks = {
        "P201_direct_attained_budget_fraction_fidelity": p201["attained_budget_fraction_MAE"] <= float(decision["maximum_P201_attained_budget_fraction_MAE"]),
        "P201_direct_composite_regret": p201["mean_frozen_composite_Lagrangian_group_utility_regret"] <= float(decision["maximum_P201_mean_frozen_composite_Lagrangian_regret"]),
    }
    if "P297_P201_attained_budget_fraction_MAE" in decision:
        checks["P201_attention_improves_attained_fraction_over_P297"] = (
            p201["attained_budget_fraction_MAE"] < float(decision["P297_P201_attained_budget_fraction_MAE"])
        )
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    torch.save({
        "model_state_dict": student.state_dict(),
        "element_width": policy_artifact["element_width"],
        "context_width": policy_artifact["context_width"],
        "rate_knot_count": policy_artifact["rate_knot_count"],
        "risk_tolerance_domain": tolerance_domain,
        "floor_domain": floor_domain,
        "fraction_input_is_one_minus_two_fraction": True,
        "attention_heads": model_config.get("attention_heads"),
        "base_model": config["frozen_policy"],
    }, run_dir / config["model_artifact"])
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "architecture": config.get("architecture", "monotone_deep_sets_direct_allocator"),
        "training": {"group_sizes": sizes, "final_normalized_budget_mae": last},
        "heldout_group_sizes": list(map(int, config["heldout_group_sizes"])),
        "source_development": source,
        "P201_post_hoc_development": p201,
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
