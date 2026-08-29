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


class SelfConsistentProjectedAuthorityCompiler(nn.Module):
    """Shift direct budgets to the mean interpolated between their own endpoint predictions."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, groups, pseudo_price, alpha, beta, floor, tail_mass):
        raw = self.base(groups, pseudo_price, alpha, beta, floor, tail_mass)
        low = self.base(groups, torch.ones_like(pseudo_price), alpha, beta, floor, tail_mass)
        high = self.base(groups, -torch.ones_like(pseudo_price), alpha, beta, floor, tail_mass)
        fraction = (1 - pseudo_price) / 2
        raw_cost = (raw + 1) / 2
        low_mean = ((low + 1) / 2).mean(1)
        high_mean = ((high + 1) / 2).mean(1)
        desired_mean = low_mean + fraction * (high_mean - low_mean)
        shift = desired_mean - raw_cost.mean(1)
        return 2 * (raw_cost + shift[:, None]).clamp(0, 1) - 1


class ConvexEndpointAuthorityCompiler(nn.Module):
    """Interpolate elementwise between monotone fraction-zero and fraction-one budgets."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, groups, pseudo_price, alpha, beta, floor, tail_mass):
        low = self.base(groups, torch.ones_like(pseudo_price), alpha, beta, floor, tail_mass)
        high = self.base(groups, -torch.ones_like(pseudo_price), alpha, beta, floor, tail_mass)
        fraction = (1 - pseudo_price) / 2
        return low + fraction[:, None] * (high - low)


class PiecewiseAnchorAuthorityCompiler(nn.Module):
    """Interpolate between fixed monotone fraction anchors without a runtime dual."""

    def __init__(self, base, anchor_fractions):
        super().__init__()
        anchors = torch.as_tensor(anchor_fractions, dtype=torch.float32)
        if anchors.ndim != 1 or len(anchors) < 3 or not torch.all(anchors[1:] > anchors[:-1]):
            raise ValueError("piecewise anchor fractions must be a strictly increasing 1D sequence")
        self.base = base
        self.anchor_values = tuple(float(value) for value in anchors.tolist())
        self.register_buffer("anchor_fractions", anchors)

    def forward(self, groups, pseudo_price, alpha, beta, floor, tail_mass):
        predictions = torch.stack([
            self.base(
                groups, torch.full_like(pseudo_price, float(1 - 2 * anchor)),
                alpha, beta, floor, tail_mass,
            )
            for anchor in self.anchor_values
        ], 1)
        fraction = ((1 - pseudo_price) / 2).clamp(
            float(self.anchor_fractions[0]), float(self.anchor_fractions[-1])
        )
        segment = torch.bucketize(fraction.contiguous(), self.anchor_fractions[1:-1])
        row = torch.arange(len(fraction), device=fraction.device)
        low = predictions[row, segment]
        high = predictions[row, segment + 1]
        low_fraction = self.anchor_fractions[segment]
        high_fraction = self.anchor_fractions[segment + 1]
        weight = (fraction - low_fraction) / (high_fraction - low_fraction)
        return low + weight[:, None] * (high - low)


class SharedContextPiecewiseAnchorAuthorityCompiler(PiecewiseAnchorAuthorityCompiler):
    """Encode the Actor set once and evaluate several monotone spline anchors."""

    def forward(self, groups, pseudo_price, alpha, beta, floor, tail_mass):
        encoded = self.base.element(groups)
        mean = encoded.mean(1)
        std = torch.sqrt(encoded.var(1, unbiased=False) + 1e-6)
        maximum = encoded.amax(1)
        size = groups.shape[1]
        context = self.base.context(torch.cat((
            encoded,
            mean[:, None].expand(-1, size, -1),
            std[:, None].expand(-1, size, -1),
            maximum[:, None].expand(-1, size, -1),
            alpha[:, None, None].expand(-1, size, 1),
            beta[:, None, None].expand(-1, size, 1),
            tail_mass[:, None, None].expand(-1, size, 1),
        ), 2))
        intercept = self.base.intercept(context).squeeze(2)
        price_rates = self.base.price_rates(context)
        floor_term = self.base._integral(
            self.base.floor_rates(context), floor[:, None].expand(-1, size)
        )
        predictions = torch.stack([
            torch.tanh(
                intercept
                - self.base._integral(price_rates, torch.full_like(intercept, float(1 - 2 * anchor)))
                + floor_term
            )
            for anchor in self.anchor_values
        ], 1)
        fraction = ((1 - pseudo_price) / 2).clamp(
            float(self.anchor_fractions[0]), float(self.anchor_fractions[-1])
        )
        segment = torch.bucketize(fraction.contiguous(), self.anchor_fractions[1:-1])
        row = torch.arange(len(fraction), device=fraction.device)
        low = predictions[row, segment]
        high = predictions[row, segment + 1]
        low_fraction = self.anchor_fractions[segment]
        high_fraction = self.anchor_fractions[segment + 1]
        weight = (fraction - low_fraction) / (high_fraction - low_fraction)
        return low + weight[:, None] * (high - low)


class NormalizedMonotoneWarpAuthorityCompiler(nn.Module):
    """Learn a group-conditioned monotone fraction warp, then call the base once."""

    def __init__(self, base, hidden_width, knot_count):
        super().__init__()
        self.base = base
        self.warp_rates = nn.Sequential(
            nn.Linear(3 * 36 + 4, int(hidden_width)),
            nn.SiLU(),
            nn.Linear(int(hidden_width), int(knot_count)),
        )
        nn.init.zeros_(self.warp_rates[-1].weight)
        nn.init.zeros_(self.warp_rates[-1].bias)

    @staticmethod
    def _integral(rates, value):
        rates = F.softplus(rates)
        knot_count = rates.shape[1]
        width = 2.0 / (knot_count - 1)
        areas = 0.5 * (rates[:, :-1] + rates[:, 1:]) * width
        cumulative = torch.cat((torch.zeros_like(rates[:, :1]), torch.cumsum(areas, 1)), 1)
        position = ((value + 1) / width).clamp(0, knot_count - 1)
        index = torch.floor(position).long().clamp(max=knot_count - 2)
        row = torch.arange(len(value), device=value.device)
        fraction = position - index
        r0 = rates[row, index]
        r1 = rates[row, index + 1]
        base = cumulative[row, index]
        return base + width * (r0 * fraction + 0.5 * (r1 - r0) * fraction.square())

    def forward(self, groups, pseudo_price, alpha, beta, floor, tail_mass):
        mean = groups.mean(1)
        std = torch.sqrt(groups.var(1, unbiased=False) + 1e-6)
        maximum = groups.amax(1)
        rates = self.warp_rates(torch.cat((mean, std, maximum, alpha[:, None], beta[:, None], floor[:, None], tail_mass[:, None]), 1))
        fraction = ((1 - pseudo_price) / 2).clamp(0, 1)
        warped_area = self._integral(rates, 2 * fraction - 1)
        total_area = self._integral(rates, torch.ones_like(fraction))
        warped_fraction = warped_area / total_area.clamp_min(1e-6)
        return self.base(groups, 1 - 2 * warped_fraction, alpha, beta, floor, tail_mass)


def _load_base_state(module, artifact):
    state = artifact["model_state_dict"]
    prefixed = {key.removeprefix("base."): value for key, value in state.items() if key.startswith("base.")}
    module.load_state_dict(prefixed if prefixed else state)


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


def _action_query_groups(feature, rows_path, horizons, expected_size):
    with np.load(rows_path, allow_pickle=False) as loaded:
        identities = np.unique(np.stack((
            loaded["scene_index"],
            np.rint(loaded["horizon_seconds"] * 10).astype(np.int32),
            loaded["anchor_frame"],
            loaded["query_id"],
        ), 1), axis=0)
    key_sets = []
    for horizon in horizons:
        horizon_code = int(round(float(horizon) * 10))
        key_sets.append({(int(row[0]), int(row[2]), int(row[3])) for row in identities if int(row[1]) == horizon_code})
    keys = sorted(set.intersection(*key_sets))
    if len(keys) != len(feature):
        raise RuntimeError(f"action keys/features are not aligned: {len(keys)} != {len(feature)}")
    buckets = {}
    for index, (scene, anchor, _) in enumerate(keys):
        buckets.setdefault((scene, anchor), []).append(index)
    groups = [indices for indices in buckets.values() if len(indices) == int(expected_size)]
    return (
        np.stack([feature[indices] for indices in groups]).astype(np.float32),
        np.asarray([key[0] for key, indices in buckets.items() if len(indices) == int(expected_size)], np.int64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    confirmation_only = bool(config.get("confirmation_only", False))
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
    p201_rows_path = args.runs_root / config["p201_rows"]["run"] / config["p201_rows"]["artifact"]
    source_rows_path = args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"]
    p201_feature, _, _, p201_scenes, _ = _dataset(p201_rows_path, *common, anchors, *tail)
    if confirmation_only:
        source_feature, source_scenes = p201_feature, p201_scenes
    else:
        source_feature, _, _, source_scenes, _ = _dataset(source_rows_path, *common, anchors, *tail)
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
    def grouped(feature, scenes, size, rows_path):
        if config.get("grouping") == "task_conditioned_action_query_set":
            return _action_query_groups(feature, rows_path, horizons, size)
        return _groups(feature, scenes, size)
    train = {}
    if not confirmation_only:
        for size in map(int, config["training_group_sizes"]):
            groups, scenes = grouped(source_feature, source_scenes, size, source_rows_path)
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
    if "shared_context_anchor_fractions" in model_config:
        base_student = EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
        ).cuda()
        direct_reference = config["frozen_direct"]
        direct_artifact = torch.load(
            args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
        )
        _load_base_state(base_student, direct_artifact)
        student = SharedContextPiecewiseAnchorAuthorityCompiler(
            base_student, model_config["shared_context_anchor_fractions"]
        ).cuda()
    elif "normalized_monotone_warp_knots" in model_config:
        base_student = EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
        ).cuda()
        direct_reference = config["frozen_direct"]
        direct_artifact = torch.load(
            args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
        )
        _load_base_state(base_student, direct_artifact)
        student = NormalizedMonotoneWarpAuthorityCompiler(
            base_student, model_config["normalized_monotone_warp_hidden_width"],
            model_config["normalized_monotone_warp_knots"],
        ).cuda()
    elif "piecewise_anchor_fractions" in model_config:
        base_student = EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
        ).cuda()
        direct_reference = config["frozen_direct"]
        direct_artifact = torch.load(
            args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
        )
        _load_base_state(base_student, direct_artifact)
        student = PiecewiseAnchorAuthorityCompiler(base_student, model_config["piecewise_anchor_fractions"]).cuda()
    elif model_config.get("convex_endpoint_rule", False):
        base_student = EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
        ).cuda()
        direct_reference = config["frozen_direct"]
        direct_artifact = torch.load(
            args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
        )
        _load_base_state(base_student, direct_artifact)
        student = ConvexEndpointAuthorityCompiler(base_student).cuda()
    elif model_config.get("self_consistent_projection", False):
        base_student = EpistemicTailCVaRAllocator(
            int(policy_artifact["element_width"]), int(policy_artifact["context_width"]), int(policy_artifact["rate_knot_count"])
        ).cuda()
        direct_reference = config["frozen_direct"]
        direct_artifact = torch.load(
            args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
        )
        _load_base_state(base_student, direct_artifact)
        student = SelfConsistentProjectedAuthorityCompiler(base_student).cuda()
    elif "attention_heads" in model_config:
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
        if "frozen_direct" in config:
            direct_reference = config["frozen_direct"]
            direct_artifact = torch.load(
                args.runs_root / direct_reference["run"] / direct_reference["artifact"], map_location="cuda"
            )
            _load_base_state(student, direct_artifact)
        else:
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
        element_loss = F.l1_loss(prediction, target_budget)
        if "group_mean_loss_weight" in model_config:
            group_mean_loss = F.l1_loss(prediction.mean(1), target_budget.mean(1))
            loss = element_loss + float(model_config["group_mean_loss_weight"]) * group_mean_loss
        else:
            loss = element_loss
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
        if not confirmation_only:
            groups, scenes = grouped(source_feature, source_scenes, size, source_rows_path)
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
        groups, _ = grouped(p201_feature, p201_scenes, size, p201_rows_path)
        target_price, low, high = _target_prices(
            policy, groups, heldout_alphas, heldout_tolerance_z, heldout_floor_z,
            heldout_tails, heldout_fractions, bisection_steps, chunk,
        )
        p201[str(size)] = _evaluate(
            student, policy, teacher, groups, target_price, low, high, heldout_alphas,
            heldout_tolerance_z, heldout_floors, heldout_floor_z, heldout_tails, heldout_fractions,
            price_mean, price_scale, epsilon, penalty, chunk,
        )
    source = _aggregate(source) if source else None
    p201 = _aggregate(p201)
    decision = config["decision"]
    checks = {
        "P201_direct_attained_budget_fraction_fidelity": p201["attained_budget_fraction_MAE"] <= float(decision["maximum_P201_attained_budget_fraction_MAE"]),
        "P201_direct_composite_regret": p201["mean_frozen_composite_Lagrangian_group_utility_regret"] <= float(decision["maximum_P201_mean_frozen_composite_Lagrangian_regret"]),
    }
    if "P297_P201_attained_budget_fraction_MAE" in decision:
        checks["P201_direct_improves_attained_fraction_over_P297"] = (
            p201["attained_budget_fraction_MAE"] < float(decision["P297_P201_attained_budget_fraction_MAE"])
        )
    if "baseline_P201_attained_budget_fraction_MAE" in decision:
        checks["P201_direct_improves_attained_fraction_over_frozen_baseline"] = (
            p201["attained_budget_fraction_MAE"] < float(decision["baseline_P201_attained_budget_fraction_MAE"])
        )
    if "maximum_P201_fraction_budget_monotonicity_violations" in decision:
        checks["P201_fraction_budget_monotonicity"] = (
            p201["fraction_budget_monotonicity_violations"]
            <= int(decision["maximum_P201_fraction_budget_monotonicity_violations"])
        )
    if "maximum_P201_direct_student_forward_seconds" in decision:
        checks["P201_direct_forward_latency"] = (
            p201["direct_student_forward_seconds"]
            <= float(decision["maximum_P201_direct_student_forward_seconds"])
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
        "self_consistent_projection": model_config.get("self_consistent_projection", False),
        "convex_endpoint_rule": model_config.get("convex_endpoint_rule", False),
        "piecewise_anchor_fractions": model_config.get("piecewise_anchor_fractions"),
        "normalized_monotone_warp_knots": model_config.get("normalized_monotone_warp_knots"),
        "shared_context_anchor_fractions": model_config.get("shared_context_anchor_fractions"),
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
        "training": {
            "group_sizes": sizes,
            "final_training_objective": last,
            "group_mean_loss_weight": model_config.get("group_mean_loss_weight", 0.0),
        },
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
