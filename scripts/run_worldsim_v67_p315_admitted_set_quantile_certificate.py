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


class HorizonNonconformityScale(nn.Module):
    """Positive task/horizon-dependent scale for a frozen raw authority score."""

    def __init__(self, input_width: int, hidden_dimensions, minimum_scale: float) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.minimum_scale = float(minimum_scale)

    def forward(self, features):
        return self.minimum_scale + F.softplus(self.network(features).squeeze(1))


class RiskMatchedHorizonQuantileHead(nn.Module):
    """One risk-aligned quantile boundary with positive continuous-horizon slope."""

    def __init__(self, input_width: int, hidden_dimensions) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, features, normalized_horizon):
        raw = self.network(features)
        return F.softplus(raw[:, 0]) + normalized_horizon * F.softplus(raw[:, 1])


class RiskConditionedHorizonQuantileSurface(nn.Module):
    """Continuous cost quantile surface monotone in horizon and quantile level."""

    def __init__(self, input_width: int, hidden_dimensions, quantile_floor: float) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 4))
        self.network = nn.Sequential(*layers)
        self.quantile_floor = float(quantile_floor)

    def forward(self, features, normalized_horizon, quantile_level):
        coefficient = F.softplus(self.network(features))
        normalized_quantile = (quantile_level - self.quantile_floor) / (1.0 - self.quantile_floor)
        return (
            coefficient[:, 0]
            + normalized_horizon * coefficient[:, 1]
            + normalized_quantile * coefficient[:, 2]
            + normalized_horizon * normalized_quantile * coefficient[:, 3]
        )


class MonotoneSplineRiskHorizonSurface(nn.Module):
    """Piecewise-linear quantile surface with positive base and horizon increments."""

    def __init__(self, input_width: int, hidden_dimensions, quantile_knots) -> None:
        super().__init__()
        knots = torch.as_tensor(quantile_knots, dtype=torch.float32)
        self.register_buffer("quantile_knots", knots)
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 2 * len(knots)))
        self.network = nn.Sequential(*layers)

    def _interpolate(self, knot_values, quantile_level):
        segment = torch.bucketize(quantile_level.contiguous(), self.quantile_knots[1:-1])
        left_q = self.quantile_knots[segment]
        right_q = self.quantile_knots[segment + 1]
        weight = (quantile_level - left_q) / (right_q - left_q)
        left_value = knot_values.gather(1, segment[:, None]).squeeze(1)
        right_value = knot_values.gather(1, (segment + 1)[:, None]).squeeze(1)
        return left_value + weight * (right_value - left_value)

    def forward(self, features, normalized_horizon, quantile_level):
        knot_count = len(self.quantile_knots)
        coefficient = F.softplus(self.network(features)).reshape(-1, 2, knot_count)
        base_knots = torch.cumsum(coefficient[:, 0], 1)
        slope_knots = torch.cumsum(coefficient[:, 1], 1)
        return self._interpolate(base_knots, quantile_level) + normalized_horizon * self._interpolate(
            slope_knots, quantile_level
        )


class SizeConditionedHorizonQuantileHead(nn.Module):
    """Nested-set quantile boundaries monotone in authority size and horizon."""

    def __init__(self, input_width: int, hidden_dimensions, set_size_count: int) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 2 * int(set_size_count)))
        self.network = nn.Sequential(*layers)
        self.set_size_count = int(set_size_count)

    def forward(self, features, normalized_horizon):
        raw = self.network(features).reshape(-1, 2, self.set_size_count)
        base = torch.cumsum(F.softplus(raw[:, 0]), 1)
        slope = torch.cumsum(F.softplus(raw[:, 1]), 1)
        return base + normalized_horizon[:, None] * slope


class RiskSizeConditionedHorizonQuantileSurface(nn.Module):
    """Nested-set surface monotone in set size, horizon, and requested quantile."""

    def __init__(self, input_width: int, hidden_dimensions, set_size_count: int, quantile_floor: float) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 4 * int(set_size_count)))
        self.network = nn.Sequential(*layers)
        self.set_size_count = int(set_size_count)
        self.quantile_floor = float(quantile_floor)

    def forward(self, features, normalized_horizon, quantile_level):
        raw = self.network(features).reshape(-1, 4, self.set_size_count)
        coefficient = torch.cumsum(F.softplus(raw), 2)
        normalized_quantile = (quantile_level - self.quantile_floor) / (1.0 - self.quantile_floor)
        return (
            coefficient[:, 0]
            + normalized_horizon[:, None] * coefficient[:, 1]
            + normalized_quantile[:, None] * coefficient[:, 2]
            + normalized_horizon[:, None] * normalized_quantile[:, None] * coefficient[:, 3]
        )


class ContextPositiveRiskSlope(nn.Module):
    """Positive context-only derivative that preserves a frozen quantile anchor and all other orders."""

    def __init__(self, input_width: int, hidden_dimensions) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return F.softplus(self.network(features)).squeeze(1)


class ContextPositiveRiskCurvature(nn.Module):
    """Positive slope and two-sided curvature around an exactly preserved quantile anchor."""

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
        return F.softplus(self.network(features))


class LatticeRiskSizeHorizonAuthority(nn.Module):
    """Context-conditioned partial-monotone lattice over horizon, quantile, and set size."""

    def __init__(self, input_width: int, hidden_dimensions, horizon_knots, quantile_knots, set_size_count: int) -> None:
        super().__init__()
        self.register_buffer("horizon_knots", torch.as_tensor(horizon_knots, dtype=torch.float32))
        self.register_buffer("quantile_knots", torch.as_tensor(quantile_knots, dtype=torch.float32))
        self.set_size_count = int(set_size_count)
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, len(horizon_knots) * len(quantile_knots) * self.set_size_count))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def _segment(value, knots):
        segment = torch.bucketize(value.contiguous(), knots[1:-1])
        left = knots[segment]
        right = knots[segment + 1]
        return segment, (value - left) / (right - left)

    def forward(self, features, normalized_horizon, quantile_level):
        batch = len(features)
        vertex = F.softplus(self.network(features)).reshape(
            batch, len(self.horizon_knots), len(self.quantile_knots), self.set_size_count
        )
        vertex = torch.cummax(vertex, dim=1)[0]
        vertex = torch.cummax(vertex, dim=2)[0]
        vertex = torch.cummax(vertex, dim=3)[0]
        horizon_segment, horizon_weight = self._segment(normalized_horizon, self.horizon_knots)
        quantile_segment, quantile_weight = self._segment(quantile_level, self.quantile_knots)
        row = torch.arange(batch, device=features.device)
        lower_left = vertex[row, horizon_segment, quantile_segment]
        lower_right = vertex[row, horizon_segment, quantile_segment + 1]
        upper_left = vertex[row, horizon_segment + 1, quantile_segment]
        upper_right = vertex[row, horizon_segment + 1, quantile_segment + 1]
        lower = lower_left + quantile_weight[:, None] * (lower_right - lower_left)
        upper = upper_left + quantile_weight[:, None] * (upper_right - upper_left)
        return lower + horizon_weight[:, None] * (upper - lower)


class GroupwisePairSelector(nn.Module):
    """Permutation-equivariant selector over the complete 15-pair candidate set."""

    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, hidden_width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1))

    def forward(self, features):
        element = self.encoder(features)
        context = element.mean(1, keepdim=True).expand_as(element)
        return self.head(torch.cat((element, context), 2)).squeeze(2)


class SelectedPairQuantileHead(nn.Module):
    """Group-aware q90 log-cost head applied only after one pair has been selected."""

    def __init__(self, input_width: int, hidden_dimensions) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return F.softplus(self.network(features).squeeze(1))


class HorizonMonotoneQuantileHead(nn.Module):
    """Continuous horizon surface monotone in both horizon and quantile level."""

    def __init__(self, input_width: int, hidden_dimensions) -> None:
        super().__init__()
        layers = []
        width = int(input_width)
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 6))
        self.network = nn.Sequential(*layers)

    def forward(self, features, normalized_horizon):
        raw = self.network(features).reshape(-1, 2, 3)
        base = torch.cumsum(F.softplus(raw[:, 0]), 1)
        slope = torch.cumsum(F.softplus(raw[:, 1]), 1)
        return base + normalized_horizon[:, None] * slope


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


@torch.no_grad()
def _pair_examples(descriptors, selector_score, costs, queries, scenes, progress_model, maneuver_model,
                   progress_values, commands, lateral_weight):
    pair_indices = np.asarray([(left, right) for left in range(6) for right in range(left + 1, 6)], np.int64)
    pair_lookup = {tuple(pair): index for index, pair in enumerate(pair_indices.tolist())}
    x = torch.from_numpy(descriptors).cuda()
    base = torch.from_numpy(selector_score).cuda()
    progress_deficit = torch.from_numpy(np.where(queries < 3, 0.5, 0.0).astype(np.float32)).cuda()
    lateral = torch.from_numpy(np.take(np.asarray([-1.0, 0.0, 1.0], np.float32), queries % 3)).cuda()
    features, targets, example_scenes, conditions, nominal_pairs, pair_utilities = [], [], [], [], [], []
    for progress in progress_values:
        progress_tensor = torch.full((len(x),), float(progress), device="cuda")
        progress_score = progress_model(x, base, progress_deficit, progress_tensor)
        for command in commands:
            command_tensor = torch.full((len(x),), float(command), device="cuda")
            distance = torch.abs(lateral - command_tensor[:, None])
            score = maneuver_model(x, progress_score, distance, command_tensor, float(lateral_weight)).cpu().numpy()
            chosen = descriptors[:, pair_indices]
            chosen_score = score[:, pair_indices]
            condition = np.broadcast_to(
                np.asarray([progress, command], np.float32)[None, None], (len(descriptors), len(pair_indices), 2)
            )
            feature = np.concatenate((
                chosen.mean(2), chosen.std(2), chosen.max(2),
                chosen_score.mean(2, keepdims=True), chosen_score.min(2, keepdims=True), chosen_score.max(2, keepdims=True),
                condition,
            ), 2).astype(np.float32)
            target = np.log1p(costs[:, pair_indices].max(2)).astype(np.float32)
            nominal = np.sort(np.argsort(score, axis=1)[:, :2], axis=1)
            nominal = np.asarray([pair_lookup[tuple(pair)] for pair in nominal.tolist()], np.int64)
            features.append(feature)
            targets.append(target)
            example_scenes.append(np.broadcast_to(scenes[:, None], target.shape))
            conditions.append(condition)
            nominal_pairs.append(nominal)
            pair_utilities.append(chosen_score.mean(2).astype(np.float32))
    return (
        np.concatenate(features), np.concatenate(targets), np.concatenate(example_scenes),
        np.concatenate(conditions), np.concatenate(nominal_pairs), np.concatenate(pair_utilities), pair_indices,
    )


def _action_groups_by_horizon(feature, costs, rows_path: Path, horizons, expected_size: int):
    with np.load(rows_path, allow_pickle=False) as loaded:
        identities = np.unique(np.stack((
            loaded["scene_index"], np.rint(loaded["horizon_seconds"] * 10).astype(np.int32),
            loaded["anchor_frame"], loaded["query_id"],
        ), 1), axis=0)
    key_sets = []
    for horizon in horizons:
        code = int(round(float(horizon) * 10))
        key_sets.append({(int(row[0]), int(row[2]), int(row[3])) for row in identities if int(row[1]) == code})
    keys = sorted(set.intersection(*key_sets))
    if len(keys) != len(feature) or len(costs) != len(feature):
        raise RuntimeError(f"horizon action alignment failed: {len(keys)}, {len(feature)}, {len(costs)}")
    buckets = {}
    for index, (scene, anchor, query) in enumerate(keys):
        buckets.setdefault((scene, anchor), []).append((index, query))
    valid = [(key, rows) for key, rows in buckets.items() if len(rows) == int(expected_size)]
    return (
        np.stack([feature[[row[0] for row in rows]] for _, rows in valid]).astype(np.float32),
        np.stack([costs[[row[0] for row in rows]] for _, rows in valid]).astype(np.float32),
        np.asarray([key[0] for key, _ in valid], np.int64),
        np.stack([[row[1] for row in rows] for _, rows in valid]).astype(np.int64),
    )


@torch.no_grad()
def _horizon_task_examples(descriptors, selector_score, costs_by_horizon, queries, scenes,
                           progress_model, maneuver_model, progress_values, commands,
                           lateral_weight, selected_count):
    x = torch.from_numpy(descriptors).cuda()
    base = torch.from_numpy(selector_score).cuda()
    progress_deficit = torch.from_numpy(np.where(queries < 3, 0.5, 0.0).astype(np.float32)).cuda()
    lateral = torch.from_numpy(np.take(np.asarray([-1.0, 0.0, 1.0], np.float32), queries % 3)).cuda()
    prefix_cost = np.maximum.accumulate(costs_by_horizon, axis=2)
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
            condition = np.broadcast_to(np.asarray([progress, command], np.float32)[None], (len(descriptors), 2))
            feature = np.concatenate((
                chosen.mean(1), chosen.std(1), chosen.max(1),
                chosen_score.mean(1, keepdims=True), chosen_score.min(1, keepdims=True), chosen_score.max(1, keepdims=True),
                condition,
            ), 1).astype(np.float32)
            target = np.log1p(prefix_cost[row, selected].max(1)).astype(np.float32)
            features.append(feature)
            targets.append(target)
            example_scenes.append(scenes)
            conditions.append(condition)
    return np.concatenate(features), np.concatenate(targets), np.concatenate(example_scenes), np.concatenate(conditions)


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


def _variable_set_metrics(scores, targets, conditions, ceiling_values, set_sizes):
    set_sizes = np.asarray(set_sizes, np.int64)
    score_size_violations = int(np.sum(np.diff(scores, axis=1) < -1e-8))
    by_ceiling = {}
    coverages, unsafe_rates, mean_sizes = [], [], []
    previous_selected_size = None
    ceiling_size_violations = 0
    row_index = np.arange(len(scores))
    for ceiling in ceiling_values:
        feasible = scores <= float(ceiling)
        selected_position = feasible.sum(1) - 1
        admitted = selected_position >= 0
        selected_size = np.where(admitted, set_sizes[np.maximum(selected_position, 0)], 0)
        if previous_selected_size is not None:
            ceiling_size_violations += int(np.sum(selected_size < previous_selected_size))
        previous_selected_size = selected_size
        selected_target = np.zeros(len(scores), np.float32)
        selected_target[admitted] = targets[row_index[admitted], selected_position[admitted]]
        count = int(admitted.sum())
        unsafe_rate = float(np.mean(selected_target[admitted] > ceiling)) if count else 0.0
        by_condition = {}
        for progress, command in np.unique(conditions, axis=0):
            local = np.isclose(conditions[:, 0], progress) & np.isclose(conditions[:, 1], command)
            local_admitted = local & admitted
            local_count = int(local_admitted.sum())
            by_condition[f"progress={float(progress)},command={float(command)}"] = {
                "any_authority_coverage": float(local_count / max(int(local.sum()), 1)),
                "unsafe_selected_set_rate": float(np.mean(selected_target[local_admitted] > ceiling)) if local_count else 0.0,
                "mean_selected_set_size": float(np.mean(selected_size[local_admitted])) if local_count else 0.0,
            }
        mean_size = float(np.mean(selected_size[admitted])) if count else 0.0
        by_ceiling[str(float(ceiling))] = {
            "log_cost_ceiling": float(ceiling),
            "actual_cost_ceiling": float(np.expm1(ceiling)),
            "admitted_count": count,
            "any_authority_coverage": float(np.mean(admitted)),
            "unsafe_selected_set_rate": unsafe_rate,
            "mean_selected_set_size": mean_size,
            "mean_normalized_authority_size": float(np.mean(selected_size) / float(set_sizes[-1])),
            "by_task_condition": by_condition,
        }
        coverages.append(float(np.mean(admitted)))
        unsafe_rates.append(unsafe_rate)
        mean_sizes.append(mean_size)
    return {
        "example_count": int(len(scores)),
        "set_sizes": [int(value) for value in set_sizes],
        "mean_any_authority_coverage": float(np.mean(coverages)),
        "highest_ceiling_any_authority_coverage": float(coverages[-1]),
        "maximum_unsafe_selected_set_rate": float(np.max(unsafe_rates)),
        "mean_unsafe_selected_set_rate": float(np.mean(unsafe_rates)),
        "mean_selected_set_size_by_ceiling": mean_sizes,
        "score_set_size_monotonicity_violations": score_size_violations,
        "ceiling_selected_size_monotonicity_violations": ceiling_size_violations,
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
    selective_authority_confirmation = bool(config.get("selective_authority_confirmation", False))
    task_projection_evaluation = bool(config.get("task_projection_evaluation", False))
    action_pair_editor_training = bool(config.get("action_pair_editor_training", False))
    groupwise_pair_editor_training = bool(config.get("groupwise_pair_editor_training", False))
    horizon_quantile_training = bool(config.get("horizon_quantile_training", False))
    horizon_quantile_confirmation = bool(config.get("horizon_quantile_confirmation", False))
    horizon_selective_authority_training = bool(config.get("horizon_selective_authority_training", False))
    horizon_temporal_calibration_training = bool(config.get("horizon_temporal_calibration_training", False))
    horizon_risk_matched_quantile_training = bool(config.get("horizon_risk_matched_quantile_training", False))
    horizon_risk_conditioned_surface_training = bool(config.get("horizon_risk_conditioned_surface_training", False))
    horizon_implicit_quantile_surface_training = bool(config.get("horizon_implicit_quantile_surface_training", False))
    horizon_spline_quantile_surface_training = bool(config.get("horizon_spline_quantile_surface_training", False))
    horizon_size_conditioned_authority_training = bool(config.get("horizon_size_conditioned_authority_training", False))
    horizon_risk_size_authority_training = bool(config.get("horizon_risk_size_authority_training", False))
    horizon_lattice_risk_size_authority_training = bool(config.get("horizon_lattice_risk_size_authority_training", False))
    horizon_locally_adaptive_lattice_calibration_training = bool(config.get("horizon_locally_adaptive_lattice_calibration_training", False))
    horizon_risk_locally_adaptive_lattice_calibration_training = bool(config.get("horizon_risk_locally_adaptive_lattice_calibration_training", False))
    horizon_anchor_preserving_continuous_risk_training = bool(config.get("horizon_anchor_preserving_continuous_risk_training", False))
    horizon_anchor_preserving_risk_curvature_training = bool(config.get("horizon_anchor_preserving_risk_curvature_training", False))
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
        if horizon_quantile_training or horizon_quantile_confirmation or horizon_selective_authority_training or horizon_temporal_calibration_training or horizon_risk_matched_quantile_training or horizon_risk_conditioned_surface_training or horizon_implicit_quantile_surface_training or horizon_spline_quantile_surface_training or horizon_size_conditioned_authority_training or horizon_risk_size_authority_training or horizon_lattice_risk_size_authority_training or horizon_locally_adaptive_lattice_calibration_training or horizon_risk_locally_adaptive_lattice_calibration_training or horizon_anchor_preserving_continuous_risk_training or horizon_anchor_preserving_risk_curvature_training:
            return _action_groups_by_horizon(feature, costs, path, horizons, int(config["action_group_size"]))
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

    if horizon_quantile_training or horizon_quantile_confirmation or horizon_selective_authority_training or horizon_temporal_calibration_training or horizon_risk_matched_quantile_training or horizon_risk_conditioned_surface_training or horizon_implicit_quantile_surface_training or horizon_spline_quantile_surface_training or horizon_size_conditioned_authority_training or horizon_risk_size_authority_training or horizon_lattice_risk_size_authority_training or horizon_locally_adaptive_lattice_calibration_training or horizon_risk_locally_adaptive_lattice_calibration_training or horizon_anchor_preserving_continuous_risk_training or horizon_anchor_preserving_risk_curvature_training:
        if horizon_size_conditioned_authority_training or horizon_risk_size_authority_training or horizon_lattice_risk_size_authority_training or horizon_locally_adaptive_lattice_calibration_training or horizon_risk_locally_adaptive_lattice_calibration_training or horizon_anchor_preserving_continuous_risk_training or horizon_anchor_preserving_risk_curvature_training:
            source_prefix, source_prefix_target = [], []
            p201_prefix, p201_prefix_target = [], []
            for set_size in config["authority_set_sizes"]:
                local_source = _horizon_task_examples(
                    source_descriptor, source_selector_score, source_costs, source_queries, source_scenes,
                    progress_model, maneuver_model, config["training_progress_preferences"],
                    config["training_lateral_commands"], float(config["lateral_preference_weight"]), int(set_size),
                )
                local_p201 = _horizon_task_examples(
                    p201_descriptor, p201_selector_score, p201_costs, p201_queries, p201_scenes,
                    progress_model, maneuver_model, config["heldout_progress_preferences"],
                    config["heldout_lateral_commands"], float(config["lateral_preference_weight"]), int(set_size),
                )
                source_prefix.append(local_source[0])
                source_prefix_target.append(local_source[1])
                p201_prefix.append(local_p201[0])
                p201_prefix_target.append(local_p201[1])
                source_example_scenes, source_conditions = local_source[2], local_source[3]
                p201_conditions = local_p201[3]
            source_feature = np.concatenate(source_prefix, 1)
            source_target = np.stack(source_prefix_target, 1)
            p201_feature = np.concatenate(p201_prefix, 1)
            p201_target = np.stack(p201_prefix_target, 1)
        else:
            source_feature, source_target, source_example_scenes, source_conditions = _horizon_task_examples(
                source_descriptor, source_selector_score, source_costs, source_queries, source_scenes,
                progress_model, maneuver_model, config["training_progress_preferences"],
                config["training_lateral_commands"], float(config["lateral_preference_weight"]),
                int(config["selected_action_count"]),
            )
            p201_feature, p201_target, _, p201_conditions = _horizon_task_examples(
                p201_descriptor, p201_selector_score, p201_costs, p201_queries, p201_scenes,
                progress_model, maneuver_model, config["heldout_progress_preferences"],
                config["heldout_lateral_commands"], float(config["lateral_preference_weight"]),
                int(config["selected_action_count"]),
            )
        if horizon_quantile_confirmation:
            frozen_horizon = torch.load(
                args.runs_root / config["frozen_horizon_certificate"]["run"] / config["frozen_horizon_certificate"]["artifact"],
                map_location="cuda",
            )
            feature_mean = np.asarray(frozen_horizon["input_mean"], np.float32)
            feature_scale = np.asarray(frozen_horizon["input_scale"], np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            quantiles = np.asarray(frozen_horizon["quantile_levels"], np.float32)
            horizon_values = np.asarray(frozen_horizon["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            offsets = np.asarray(frozen_horizon["calibration_offsets"], np.float32)
            model = HorizonMonotoneQuantileHead(
                int(frozen_horizon["input_width"]), frozen_horizon["hidden_dimensions"]
            ).cuda()
            model.load_state_dict(frozen_horizon["model_state_dict"])
            model.eval()
            heldout_horizon_index = int(config["heldout_horizon_index"])
            with torch.no_grad():
                p201_x = torch.from_numpy(p201_feature).cuda()
                h = torch.full((len(p201_x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                prediction = model(p201_x, h).cpu().numpy()
            prediction = np.maximum.accumulate(prediction + offsets[None], axis=1)
            local_target = p201_target[:, heldout_horizon_index]
            coverage = np.mean(local_target[:, None] <= prediction, axis=0)
            error = local_target[:, None] - prediction
            pinball = np.maximum(quantiles[None] * error, (quantiles[None] - 1) * error)
            by_condition = {}
            for progress, command in np.unique(p201_conditions, axis=0):
                local = np.isclose(p201_conditions[:, 0], progress) & np.isclose(p201_conditions[:, 1], command)
                by_condition[f"progress={float(progress)},command={float(command)}"] = {
                    "example_count": int(local.sum()),
                    "empirical_coverages": [float(value) for value in np.mean(local_target[local, None] <= prediction[local], axis=0)],
                    "median_absolute_log_cost_error": float(np.mean(np.abs(prediction[local, 0] - local_target[local]))),
                }
            p201_metrics = {
                "example_count": int(len(local_target)),
                "horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "quantile_levels": [float(value) for value in quantiles],
                "empirical_coverages": [float(value) for value in coverage],
                "maximum_quantile_undercoverage": float(np.max(quantiles - coverage)),
                "median_absolute_log_cost_error": float(np.mean(np.abs(prediction[:, 0] - local_target))),
                "mean_pinball_loss": float(np.mean(pinball)),
                "mean_q95_minus_q50_width": float(np.mean(prediction[:, 2] - prediction[:, 0])),
                "by_task_condition": by_condition,
            }
            with torch.no_grad():
                all_predictions = []
                for horizon_value in normalized_horizons:
                    h = torch.full((len(p201_x),), float(horizon_value), device="cuda")
                    local_prediction = model(p201_x, h).cpu().numpy() + offsets[None]
                    all_predictions.append(np.maximum.accumulate(local_prediction, axis=1))
            all_predictions = np.stack(all_predictions, axis=1)
            horizon_violations = int(np.sum(np.diff(all_predictions, axis=1) < -1e-8))
            quantile_violations = int(np.sum(np.diff(all_predictions, axis=2) < -1e-8))
            p201_metrics["horizon_monotonicity_violations"] = horizon_violations
            p201_metrics["quantile_order_violations"] = quantile_violations
            decision = config["decision"]
            checks = {
                "P201_heldout_horizon_quantile_coverage": p201_metrics["maximum_quantile_undercoverage"] <= float(decision["maximum_P201_quantile_undercoverage"]),
                "P201_heldout_horizon_median_fidelity": p201_metrics["median_absolute_log_cost_error"] <= float(decision["maximum_P201_median_absolute_log_cost_error"]),
                "P201_horizon_quantile_monotonicity": horizon_violations <= int(decision["maximum_P201_horizon_monotonicity_violations"]) and quantile_violations == 0,
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                **frozen_horizon,
                "frozen_horizon_certificate": config["frozen_horizon_certificate"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {"row_count": 0, "horizon_example_count": 0, "steps": 0},
                "calibration": {"horizon_example_count": 0, "additive_offsets": [float(value) for value in offsets], "reused_from": config["frozen_horizon_certificate"]},
                "source_heldout_horizon_development": None,
                "P201_heldout_task_and_horizon_development": p201_metrics,
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
        if horizon_anchor_preserving_continuous_risk_training or horizon_anchor_preserving_risk_curvature_training:
            risk_curvature = horizon_anchor_preserving_risk_curvature_training
            frozen_lattice = torch.load(
                args.runs_root / config["frozen_lattice_authority"]["run"] / config["frozen_lattice_authority"]["artifact"],
                map_location="cuda",
            )
            frozen_anchor = torch.load(
                args.runs_root / config["frozen_local_anchor"]["run"] / config["frozen_local_anchor"]["artifact"],
                map_location="cuda",
            )
            feature_mean = np.asarray(frozen_lattice["input_mean"], np.float32)
            feature_scale = np.asarray(frozen_lattice["input_scale"], np.float32)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(frozen_lattice["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            training_horizon_indices = np.asarray(frozen_lattice["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(frozen_lattice["heldout_horizon_index"])
            set_sizes = np.asarray(frozen_lattice["authority_set_sizes"], np.int64)
            anchor_q = float(frozen_anchor["risk_quantile_level"])
            anchor_threshold = float(frozen_anchor["normalized_calibration_threshold"])
            lattice_model = LatticeRiskSizeHorizonAuthority(
                int(frozen_lattice["input_width"]), frozen_lattice["hidden_dimensions"],
                normalized_horizons[training_horizon_indices], frozen_lattice["quantile_knots"], len(set_sizes),
            ).cuda()
            lattice_model.load_state_dict(frozen_lattice["model_state_dict"])
            lattice_model.eval()
            anchor_model = SizeConditionedHorizonQuantileHead(
                int(frozen_anchor["input_width"]), frozen_anchor["hidden_dimensions"], len(set_sizes)
            ).cuda()
            anchor_model.load_state_dict(frozen_anchor["scale_model_state_dict"])
            anchor_model.eval()
            source_x = torch.from_numpy(source_feature).cuda()
            p201_x = torch.from_numpy(p201_feature).cuda()
            horizon_tensor = torch.from_numpy(normalized_horizons).cuda()
            with torch.no_grad():
                source_anchor_scale = torch.stack(
                    [anchor_model(source_x, torch.full((len(source_x),), float(value), device="cuda")) for value in normalized_horizons],
                    2,
                )
                p201_anchor_scale = torch.stack(
                    [anchor_model(p201_x, torch.full((len(p201_x),), float(value), device="cuda")) for value in normalized_horizons],
                    2,
                )
            split = config["adaptive_split"]
            modulus = int(split["scene_modulus"])
            scale_fit_rows = source_example_scenes % modulus == int(split["scale_fit_scene_remainder"])
            normalized_calibration_rows = source_example_scenes % modulus == int(split["normalized_calibration_scene_remainder"])
            slope_config = config["risk_slope_model"]
            risk_slope = (
                ContextPositiveRiskCurvature(source_feature.shape[1], slope_config["hidden_dimensions"])
                if risk_curvature else ContextPositiveRiskSlope(source_feature.shape[1], slope_config["hidden_dimensions"])
            ).cuda()
            frozen_linear_slope = None
            if risk_curvature:
                frozen_linear = torch.load(
                    args.runs_root / config["frozen_linear_risk"]["run"] / config["frozen_linear_risk"]["artifact"],
                    map_location="cuda",
                )
                frozen_linear_slope = ContextPositiveRiskSlope(
                    int(frozen_linear["input_width"]), frozen_linear["hidden_dimensions"]
                ).cuda()
                frozen_linear_slope.load_state_dict(frozen_linear["risk_slope_state_dict"])
                frozen_linear_slope.eval()
            optimizer = torch.optim.AdamW(
                risk_slope.parameters(), lr=float(slope_config["learning_rate"]), weight_decay=float(slope_config["weight_decay"])
            )
            y = torch.from_numpy(source_target).cuda()
            fit_index = torch.from_numpy(np.flatnonzero(scale_fit_rows)).cuda()
            training_horizon_tensor = torch.from_numpy(training_horizon_indices).cuda()
            training_quantile_range = np.asarray(slope_config["training_quantile_range"], np.float32)
            last = 0.0
            for step in range(int(slope_config["steps"])):
                row = fit_index[torch.randint(len(fit_index), (int(slope_config["batch_size"]),), device="cuda")]
                local_horizon_index = training_horizon_tensor[
                    torch.randint(len(training_horizon_tensor), (len(row),), device="cuda")
                ]
                local_size_index = torch.randint(len(set_sizes), (len(row),), device="cuda")
                local_q = torch.empty(len(row), device="cuda").uniform_(
                    float(training_quantile_range[0]), float(training_quantile_range[1])
                )
                with torch.no_grad():
                    raw_prediction = lattice_model(source_x[row], horizon_tensor[local_horizon_index], local_q)[
                        torch.arange(len(row), device="cuda"), local_size_index
                    ]
                    anchor_margin = anchor_threshold * source_anchor_scale[
                        row, local_size_index, local_horizon_index
                    ]
                delta = local_q - anchor_q
                if risk_curvature:
                    local_shape = risk_slope(source_x[row])
                    deformation = delta * local_shape[:, 0] + torch.where(
                        delta < 0, -(delta ** 2) * local_shape[:, 1], (delta ** 2) * local_shape[:, 2]
                    )
                else:
                    deformation = delta * risk_slope(source_x[row])
                prediction = raw_prediction + anchor_margin + deformation
                error = y[row, local_size_index, local_horizon_index] - prediction
                loss = torch.maximum(local_q * error, (local_q - 1.0) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                if step % 500 == 0:
                    print(f"{config['task_id']} slope step={step + 1} loss={last:.7f}", flush=True)
            risk_slope.eval()
            evaluation_quantiles = np.asarray(config["evaluation_quantiles"], np.float32)
            source_frontier, p201_frontier = {}, {}
            source_score_stack, p201_score_stack = [], []
            with torch.no_grad():
                source_shape = risk_slope(source_x)
                p201_shape = risk_slope(p201_x)
                if risk_curvature:
                    source_linear_shape = frozen_linear_slope(source_x)[:, None]
                else:
                    source_shape = source_shape[:, None]
                    p201_shape = p201_shape[:, None]
                heldout_h = torch.full((len(source_x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                p201_heldout_h = torch.full((len(p201_x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                for q_value in evaluation_quantiles:
                    source_q = torch.full((len(source_x),), float(q_value), device="cuda")
                    p201_q = torch.full((len(p201_x),), float(q_value), device="cuda")
                    delta = float(q_value) - anchor_q
                    if risk_curvature:
                        source_deformation = delta * source_shape[:, :1] + (
                            -(delta ** 2) * source_shape[:, 1:2] if delta < 0 else (delta ** 2) * source_shape[:, 2:3]
                        )
                        p201_deformation = delta * p201_shape[:, :1] + (
                            -(delta ** 2) * p201_shape[:, 1:2] if delta < 0 else (delta ** 2) * p201_shape[:, 2:3]
                        )
                    else:
                        source_deformation = delta * source_shape
                        p201_deformation = delta * p201_shape
                    source_score = lattice_model(source_x, heldout_h, source_q) + anchor_threshold * source_anchor_scale[:, :, heldout_horizon_index] + source_deformation
                    p201_score = lattice_model(p201_x, p201_heldout_h, p201_q) + anchor_threshold * p201_anchor_scale[:, :, heldout_horizon_index] + p201_deformation
                    source_score_np = source_score.cpu().numpy()
                    p201_score_np = p201_score.cpu().numpy()
                    source_score_stack.append(source_score_np)
                    p201_score_stack.append(p201_score_np)
                    source_frontier[f"{float(q_value):.2f}"] = _variable_set_metrics(
                        source_score_np[normalized_calibration_rows], source_target[normalized_calibration_rows, :, heldout_horizon_index],
                        source_conditions[normalized_calibration_rows], np.asarray(frozen_lattice["heldout_log_cost_ceilings"], np.float32), set_sizes,
                    )
                    p201_frontier[f"{float(q_value):.2f}"] = _variable_set_metrics(
                        p201_score_np, p201_target[:, :, heldout_horizon_index], p201_conditions,
                        np.asarray(frozen_lattice["heldout_log_cost_ceilings"], np.float32), set_sizes,
                    )
            quantile_order_violations = int(np.sum(np.diff(np.stack(p201_score_stack, 0), axis=0) < -1e-6))
            source_pinball = []
            linear_source_pinball = []
            source_local_target = source_target[normalized_calibration_rows, :, heldout_horizon_index]
            for quantile_index, q_value in enumerate(evaluation_quantiles):
                error = source_local_target - source_score_stack[quantile_index][normalized_calibration_rows]
                source_pinball.append(float(np.mean(np.maximum(float(q_value) * error, (float(q_value) - 1.0) * error))))
                if risk_curvature:
                    with torch.no_grad():
                        local_q = torch.full((len(source_x),), float(q_value), device="cuda")
                        raw_linear = lattice_model(source_x, heldout_h, local_q)
                        score_linear = raw_linear + anchor_threshold * source_anchor_scale[:, :, heldout_horizon_index] + (float(q_value) - anchor_q) * source_linear_shape
                    linear_error = source_local_target - score_linear.cpu().numpy()[normalized_calibration_rows]
                    linear_source_pinball.append(float(np.mean(np.maximum(float(q_value) * linear_error, (float(q_value) - 1.0) * linear_error))))
            mean_source_pinball = float(np.mean(source_pinball))
            mean_linear_source_pinball = None if not risk_curvature else float(np.mean(linear_source_pinball))
            anchor_key = f"{anchor_q:.2f}"
            p201_metrics = p201_frontier[anchor_key]
            decision = config["decision"]
            checks = {
                "P201_anchor_risk": p201_metrics["maximum_unsafe_selected_set_rate"] <= float(decision["maximum_P201_unsafe_selected_set_rate"]),
                "P201_anchor_coverage": p201_metrics["mean_any_authority_coverage"] >= float(decision["minimum_P201_mean_any_authority_coverage"]),
                "P201_anchor_size_and_ceiling_monotonicity": p201_metrics["score_set_size_monotonicity_violations"] == 0 and p201_metrics["ceiling_selected_size_monotonicity_violations"] == 0,
            }
            if risk_curvature:
                checks["source_continuous_risk_pinball_not_worse_than_linear"] = mean_source_pinball <= mean_linear_source_pinball
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "risk_slope_state_dict": risk_slope.state_dict(), "input_width": source_feature.shape[1],
                "hidden_dimensions": slope_config["hidden_dimensions"], "anchor_quantile_level": anchor_q,
                "risk_deformation_type": "two_sided_positive_quadratic_curvature" if risk_curvature else "positive_linear_slope",
                "training_quantile_range": training_quantile_range, "frozen_local_anchor": config["frozen_local_anchor"],
                "frozen_lattice_authority": config["frozen_lattice_authority"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"], "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
                "training": {"example_count": int(scale_fit_rows.sum()), "steps": int(slope_config["steps"]),
                             "final_pinball_loss": last, "training_quantile_range": [float(value) for value in training_quantile_range]},
                "anchor": {"quantile_level": anchor_q, "normalized_threshold": anchor_threshold,
                           "frozen_local_anchor": config["frozen_local_anchor"]},
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "source_continuous_risk_frontier": source_frontier,
                "P201_continuous_risk_frontier": p201_frontier,
                "P201_quantile_order_violations": quantile_order_violations,
                "source_heldout_quantile_pinball_by_level": {f"{float(q):.2f}": value for q, value in zip(evaluation_quantiles, source_pinball)},
                "source_heldout_mean_quantile_pinball": mean_source_pinball,
                "P335_linear_source_heldout_mean_quantile_pinball": mean_linear_source_pinball,
                "decision_checks": checks,
                "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                              "wall_seconds": time.monotonic() - started},
                "claim_boundary": config["claim_boundary"],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
            return
        if horizon_locally_adaptive_lattice_calibration_training or horizon_risk_locally_adaptive_lattice_calibration_training:
            risk_conditioned_scale = horizon_risk_locally_adaptive_lattice_calibration_training
            frozen_lattice = torch.load(
                args.runs_root / config["frozen_lattice_authority"]["run"] / config["frozen_lattice_authority"]["artifact"],
                map_location="cuda",
            )
            feature_mean = np.asarray(frozen_lattice["input_mean"], np.float32)
            feature_scale = np.asarray(frozen_lattice["input_scale"], np.float32)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(frozen_lattice["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            training_horizon_indices = np.asarray(frozen_lattice["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(frozen_lattice["heldout_horizon_index"])
            set_sizes = np.asarray(frozen_lattice["authority_set_sizes"], np.int64)
            q = float(frozen_lattice["risk_quantile_level"])
            lattice_model = LatticeRiskSizeHorizonAuthority(
                int(frozen_lattice["input_width"]), frozen_lattice["hidden_dimensions"],
                normalized_horizons[training_horizon_indices], frozen_lattice["quantile_knots"], len(set_sizes),
            ).cuda()
            lattice_model.load_state_dict(frozen_lattice["model_state_dict"])
            lattice_model.eval()
            source_x = torch.from_numpy(source_feature).cuda()
            p201_x = torch.from_numpy(p201_feature).cuda()
            with torch.no_grad():
                source_raw_by_horizon, p201_raw_by_horizon = [], []
                for horizon_value in normalized_horizons:
                    source_h = torch.full((len(source_x),), float(horizon_value), device="cuda")
                    source_q = torch.full((len(source_x),), q, device="cuda")
                    p201_h = torch.full((len(p201_x),), float(horizon_value), device="cuda")
                    p201_q = torch.full((len(p201_x),), q, device="cuda")
                    source_raw_by_horizon.append(lattice_model(source_x, source_h, source_q).cpu().numpy())
                    p201_raw_by_horizon.append(lattice_model(p201_x, p201_h, p201_q).cpu().numpy())
            source_raw = np.stack(source_raw_by_horizon, 2)
            p201_raw = np.stack(p201_raw_by_horizon, 2)
            split = config["adaptive_split"]
            modulus = int(split["scene_modulus"])
            scale_fit_rows = source_example_scenes % modulus == int(split["scale_fit_scene_remainder"])
            normalized_calibration_rows = source_example_scenes % modulus == int(split["normalized_calibration_scene_remainder"])
            scale_config = config["scale_model"]
            if risk_conditioned_scale:
                scale_model = RiskSizeConditionedHorizonQuantileSurface(
                    source_feature.shape[1], scale_config["hidden_dimensions"], len(set_sizes),
                    float(scale_config["quantile_floor"]),
                ).cuda()
                training_quantile_range = np.asarray(scale_config["training_quantile_range"], np.float32)
                minimum_scale = float(scale_config["minimum_scale"])
            else:
                scale_model = SizeConditionedHorizonQuantileHead(
                    source_feature.shape[1], scale_config["hidden_dimensions"], len(set_sizes)
                ).cuda()
                training_quantile_range = None
                minimum_scale = 0.0
            optimizer = torch.optim.AdamW(
                scale_model.parameters(), lr=float(scale_config["learning_rate"]), weight_decay=float(scale_config["weight_decay"])
            )
            y = torch.from_numpy(source_target).cuda()
            raw = torch.from_numpy(source_raw).cuda()
            fit_index = torch.from_numpy(np.flatnonzero(scale_fit_rows)).cuda()
            training_horizon_tensor = torch.from_numpy(training_horizon_indices).cuda()
            horizon_tensor = torch.from_numpy(normalized_horizons).cuda()
            last = 0.0
            for step in range(int(scale_config["steps"])):
                row = fit_index[torch.randint(len(fit_index), (int(scale_config["batch_size"]),), device="cuda")]
                local_horizon_index = training_horizon_tensor[
                    torch.randint(len(training_horizon_tensor), (len(row),), device="cuda")
                ]
                local_size_index = torch.randint(len(set_sizes), (len(row),), device="cuda")
                if risk_conditioned_scale:
                    local_q = torch.empty(len(row), device="cuda").uniform_(
                        float(training_quantile_range[0]), float(training_quantile_range[1])
                    )
                    raw_prediction = lattice_model(source_x[row], horizon_tensor[local_horizon_index], local_q).detach()[
                        torch.arange(len(row), device="cuda"), local_size_index
                    ]
                    prediction = minimum_scale + scale_model(
                        source_x[row], horizon_tensor[local_horizon_index], local_q
                    )[torch.arange(len(row), device="cuda"), local_size_index]
                    local_loss_quantile = local_q
                else:
                    raw_prediction = raw[row, local_size_index, local_horizon_index]
                    prediction = scale_model(source_x[row], horizon_tensor[local_horizon_index])[
                        torch.arange(len(row), device="cuda"), local_size_index
                    ]
                    local_loss_quantile = q
                target_gap = torch.clamp(y[row, local_size_index, local_horizon_index] - raw_prediction, min=0.0)
                error = target_gap - prediction
                loss = torch.maximum(local_loss_quantile * error, (local_loss_quantile - 1.0) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                if step % 500 == 0:
                    print(f"{config['task_id']} scale step={step + 1} loss={last:.7f}", flush=True)
            scale_model.eval()
            with torch.no_grad():
                source_scale_by_horizon, p201_scale_by_horizon = [], []
                for horizon_value in normalized_horizons:
                    source_h = torch.full((len(source_x),), float(horizon_value), device="cuda")
                    p201_h = torch.full((len(p201_x),), float(horizon_value), device="cuda")
                    if risk_conditioned_scale:
                        source_q = torch.full((len(source_x),), q, device="cuda")
                        p201_q = torch.full((len(p201_x),), q, device="cuda")
                        source_scale_by_horizon.append((minimum_scale + scale_model(source_x, source_h, source_q)).cpu().numpy())
                        p201_scale_by_horizon.append((minimum_scale + scale_model(p201_x, p201_h, p201_q)).cpu().numpy())
                    else:
                        source_scale_by_horizon.append(scale_model(source_x, source_h).cpu().numpy())
                        p201_scale_by_horizon.append(scale_model(p201_x, p201_h).cpu().numpy())
            source_scale = np.stack(source_scale_by_horizon, 2)
            p201_scale = np.stack(p201_scale_by_horizon, 2)
            normalized_residual = []
            for horizon_index in training_horizon_indices:
                normalized_residual.append(
                    (source_target[normalized_calibration_rows, :, horizon_index] - source_raw[normalized_calibration_rows, :, horizon_index])
                    / np.maximum(source_scale[normalized_calibration_rows, :, horizon_index], 1e-4)
                )
            normalized_threshold = max(0.0, float(np.quantile(np.concatenate(normalized_residual).reshape(-1), q)))
            source_score = source_raw[:, :, heldout_horizon_index] + normalized_threshold * source_scale[:, :, heldout_horizon_index]
            p201_score = p201_raw[:, :, heldout_horizon_index] + normalized_threshold * p201_scale[:, :, heldout_horizon_index]
            source_local_target = source_target[:, :, heldout_horizon_index]
            p201_local_target = p201_target[:, :, heldout_horizon_index]
            evaluation_ceilings = np.asarray(frozen_lattice["heldout_log_cost_ceilings"], np.float32)
            source_metrics = _variable_set_metrics(
                source_score[normalized_calibration_rows], source_local_target[normalized_calibration_rows],
                source_conditions[normalized_calibration_rows], evaluation_ceilings, set_sizes,
            )
            p201_metrics = _variable_set_metrics(
                p201_score, p201_local_target, p201_conditions, evaluation_ceilings, set_sizes
            )
            p332_baseline_metrics = _variable_set_metrics(
                p201_raw[:, :, heldout_horizon_index] + float(frozen_lattice["calibration_offset"]),
                p201_local_target, p201_conditions, evaluation_ceilings, set_sizes,
            )
            decision = config["decision"]
            checks = {
                "P201_locally_adaptive_lattice_risk": p201_metrics["maximum_unsafe_selected_set_rate"] <= float(decision["maximum_P201_unsafe_selected_set_rate"]),
                "P201_locally_adaptive_lattice_coverage": p201_metrics["mean_any_authority_coverage"] >= float(decision["minimum_P201_mean_any_authority_coverage"]),
                "P201_size_and_ceiling_monotonicity": p201_metrics["score_set_size_monotonicity_violations"] == 0 and p201_metrics["ceiling_selected_size_monotonicity_violations"] == 0,
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "scale_model_state_dict": scale_model.state_dict(),
                "input_width": source_feature.shape[1],
                "hidden_dimensions": scale_config["hidden_dimensions"],
                "normalized_calibration_threshold": normalized_threshold,
                "risk_quantile_level": q,
                "scale_type": "risk_size_horizon_positive_surface" if risk_conditioned_scale else "size_horizon_positive_head",
                "training_quantile_range": None if training_quantile_range is None else training_quantile_range,
                "minimum_scale": minimum_scale,
                "frozen_lattice_authority": config["frozen_lattice_authority"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done", "verdict": verdict, "role": config["role"],
                "frozen_lattice_authority": config["frozen_lattice_authority"],
                "training": {
                    "size_horizon_example_count": int(scale_fit_rows.sum() * len(set_sizes) * len(training_horizon_indices)),
                    "steps": int(scale_config["steps"]), "final_pinball_loss": last,
                    "training_quantile_range": None if training_quantile_range is None else [float(value) for value in training_quantile_range],
                },
                "calibration": {
                    "size_horizon_example_count": int(normalized_calibration_rows.sum() * len(set_sizes) * len(training_horizon_indices)),
                    "risk_quantile_level": q, "normalized_nonconformity_threshold": normalized_threshold,
                    "minimum_scale": minimum_scale,
                    "source_scale_mean": float(np.mean(source_scale[normalized_calibration_rows])),
                    "source_scale_range": [float(np.min(source_scale[normalized_calibration_rows])), float(np.max(source_scale[normalized_calibration_rows]))],
                },
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "source_normalized_calibration_development": source_metrics,
                "P201_heldout_task_and_horizon_development": p201_metrics,
                "P201_frozen_P332_global_offset_baseline": p332_baseline_metrics,
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
        if horizon_size_conditioned_authority_training or horizon_risk_size_authority_training or horizon_lattice_risk_size_authority_training:
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration_rows = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
            development_rows = source_example_scenes % modulus == int(split["development_scene_remainder"])
            train_rows = ~(calibration_rows | development_rows)
            feature_mean = source_feature[train_rows].mean(0)
            feature_scale = source_feature[train_rows].std(0).clip(min=1e-5)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(config["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            training_horizon_indices = np.asarray(config["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(config["heldout_horizon_index"])
            set_sizes = np.asarray(config["authority_set_sizes"], np.int64)
            model_config = config["size_model"]
            risk_size_axis = horizon_risk_size_authority_training or horizon_lattice_risk_size_authority_training
            if horizon_lattice_risk_size_authority_training:
                model = LatticeRiskSizeHorizonAuthority(
                    source_feature.shape[1], model_config["hidden_dimensions"],
                    normalized_horizons[training_horizon_indices], config["quantile_knots"], len(set_sizes),
                ).cuda()
                training_quantile_range = np.asarray(config["training_quantile_range"], np.float32)
                q = float(config["heldout_quantile_level"])
            elif horizon_risk_size_authority_training:
                model = RiskSizeConditionedHorizonQuantileSurface(
                    source_feature.shape[1], model_config["hidden_dimensions"], len(set_sizes),
                    float(config["quantile_floor"]),
                ).cuda()
                training_quantile_range = np.asarray(config["training_quantile_range"], np.float32)
                q = float(config["heldout_quantile_level"])
            else:
                model = SizeConditionedHorizonQuantileHead(
                    source_feature.shape[1], model_config["hidden_dimensions"], len(set_sizes)
                ).cuda()
                training_quantile_range = None
                q = float(config["risk_quantile_level"])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            x = torch.from_numpy(source_feature).cuda()
            y = torch.from_numpy(source_target).cuda()
            train_index = torch.from_numpy(np.flatnonzero(train_rows)).cuda()
            training_horizon_tensor = torch.from_numpy(training_horizon_indices).cuda()
            horizon_tensor = torch.from_numpy(normalized_horizons).cuda()
            last = 0.0
            for step in range(int(model_config["steps"])):
                row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
                local_horizon_index = training_horizon_tensor[
                    torch.randint(len(training_horizon_tensor), (len(row),), device="cuda")
                ]
                local_size_index = torch.randint(len(set_sizes), (len(row),), device="cuda")
                if risk_size_axis:
                    local_quantile = (
                        float(training_quantile_range[0])
                        + (float(training_quantile_range[1]) - float(training_quantile_range[0]))
                        * torch.rand(len(row), device="cuda")
                    )
                    all_predictions = model(x[row], horizon_tensor[local_horizon_index], local_quantile)
                    loss_quantile = local_quantile
                else:
                    all_predictions = model(x[row], horizon_tensor[local_horizon_index])
                    loss_quantile = q
                prediction = all_predictions[
                    torch.arange(len(row), device="cuda"), local_size_index
                ]
                target = y[row, local_size_index, local_horizon_index]
                error = target - prediction
                loss = torch.maximum(loss_quantile * error, (loss_quantile - 1.0) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                if step % 500 == 0:
                    if horizon_lattice_risk_size_authority_training:
                        card = "P332 lattice-risk-size"
                    else:
                        card = "P331 risk-size" if horizon_risk_size_authority_training else "P330 size-conditioned"
                    print(f"{card} qxH step={step + 1} loss={last:.7f}", flush=True)
            model.eval()
            calibration_index = np.flatnonzero(calibration_rows)
            with torch.no_grad():
                calibration_x = x[torch.from_numpy(calibration_index).cuda()]
                calibration_residuals = []
                for horizon_index in training_horizon_indices:
                    local_h = torch.full((len(calibration_x),), float(normalized_horizons[horizon_index]), device="cuda")
                    if risk_size_axis:
                        local_q = torch.full((len(calibration_x),), q, device="cuda")
                        prediction = model(calibration_x, local_h, local_q).cpu().numpy()
                    else:
                        prediction = model(calibration_x, local_h).cpu().numpy()
                    calibration_residuals.append(source_target[calibration_rows, :, horizon_index] - prediction)
                calibration_offset = float(np.quantile(np.concatenate(calibration_residuals).reshape(-1), q))
                source_h = torch.full((len(x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                p201_x = torch.from_numpy(p201_feature).cuda()
                p201_h = torch.full((len(p201_x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                if risk_size_axis:
                    source_q = torch.full((len(x),), q, device="cuda")
                    p201_q = torch.full((len(p201_x),), q, device="cuda")
                    source_score = model(x, source_h, source_q).cpu().numpy()
                    p201_score = model(p201_x, p201_h, p201_q).cpu().numpy()
                else:
                    source_score = model(x, source_h).cpu().numpy()
                    p201_score = model(p201_x, p201_h).cpu().numpy()
            source_score = np.maximum(source_score + calibration_offset, 0.0)
            p201_score = np.maximum(p201_score + calibration_offset, 0.0)
            source_local_target = source_target[:, :, heldout_horizon_index]
            p201_local_target = p201_target[:, :, heldout_horizon_index]
            frozen_selective = torch.load(
                args.runs_root / config["frozen_horizon_selective_authority"]["run"] / config["frozen_horizon_selective_authority"]["artifact"],
                map_location="cpu",
            )
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            source_metrics = _variable_set_metrics(
                source_score[development_rows], source_local_target[development_rows],
                source_conditions[development_rows], evaluation_ceilings, set_sizes,
            )
            p201_metrics = _variable_set_metrics(
                p201_score, p201_local_target, p201_conditions, evaluation_ceilings, set_sizes
            )
            decision = config["decision"]
            checks = {
                "P201_size_conditioned_authority_risk": p201_metrics["maximum_unsafe_selected_set_rate"] <= float(decision["maximum_P201_unsafe_selected_set_rate"]),
                "P201_size_conditioned_authority_coverage": p201_metrics["mean_any_authority_coverage"] >= float(decision["minimum_P201_mean_any_authority_coverage"]),
                "P201_size_and_ceiling_monotonicity": p201_metrics["score_set_size_monotonicity_violations"] == 0 and p201_metrics["ceiling_selected_size_monotonicity_violations"] == 0,
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_width": source_feature.shape[1],
                "hidden_dimensions": model_config["hidden_dimensions"],
                "input_mean": feature_mean,
                "input_scale": feature_scale,
                "authority_set_sizes": set_sizes,
                "horizons_seconds": horizon_values,
                "training_horizon_indices": training_horizon_indices,
                "heldout_horizon_index": heldout_horizon_index,
                "risk_quantile_level": q,
                "training_quantile_range": training_quantile_range,
                "quantile_floor": None if not horizon_risk_size_authority_training else float(config["quantile_floor"]),
                "quantile_knots": None if not horizon_lattice_risk_size_authority_training else np.asarray(config["quantile_knots"], np.float32),
                "surface_type": "partial_monotone_lattice" if horizon_lattice_risk_size_authority_training else ("positive_interaction" if horizon_risk_size_authority_training else "fixed_quantile"),
                "calibration_offset": calibration_offset,
                "heldout_log_cost_ceilings": evaluation_ceilings,
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {
                    "size_horizon_example_count": int(train_rows.sum() * len(set_sizes) * len(training_horizon_indices)),
                    "authority_set_sizes": [int(value) for value in set_sizes], "risk_quantile_level": q,
                    "training_quantile_range": None if training_quantile_range is None else [float(value) for value in training_quantile_range],
                    "steps": int(model_config["steps"]), "final_pinball_loss": last,
                },
                "calibration": {
                    "size_horizon_example_count": int(calibration_rows.sum() * len(set_sizes) * len(training_horizon_indices)),
                    "risk_quantile_level": q, "signed_log_cost_offset": calibration_offset,
                },
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
                "source_heldout_horizon_development": source_metrics,
                "P201_heldout_task_and_horizon_development": p201_metrics,
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
        if horizon_risk_conditioned_surface_training or horizon_implicit_quantile_surface_training or horizon_spline_quantile_surface_training:
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration_rows = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
            development_rows = source_example_scenes % modulus == int(split["development_scene_remainder"])
            train_rows = ~(calibration_rows | development_rows)
            feature_mean = source_feature[train_rows].mean(0)
            feature_scale = source_feature[train_rows].std(0).clip(min=1e-5)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(config["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            training_horizon_indices = np.asarray(config["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(config["heldout_horizon_index"])
            implicit_quantile_training = horizon_implicit_quantile_surface_training or horizon_spline_quantile_surface_training
            if implicit_quantile_training:
                training_quantile_range = np.asarray(config["training_quantile_range"], np.float32)
                training_quantiles = None
            else:
                training_quantile_range = None
                training_quantiles = np.asarray(config["training_quantile_levels"], np.float32)
            heldout_quantile = float(config["heldout_quantile_level"])
            model_config = config["risk_surface_model"]
            if horizon_spline_quantile_surface_training:
                model = MonotoneSplineRiskHorizonSurface(
                    source_feature.shape[1], model_config["hidden_dimensions"], config["quantile_knots"]
                ).cuda()
            else:
                model = RiskConditionedHorizonQuantileSurface(
                    source_feature.shape[1], model_config["hidden_dimensions"], float(config["quantile_floor"])
                ).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            x = torch.from_numpy(source_feature).cuda()
            y = torch.from_numpy(source_target).cuda()
            train_index = torch.from_numpy(np.flatnonzero(train_rows)).cuda()
            training_horizon_tensor = torch.from_numpy(training_horizon_indices).cuda()
            horizon_tensor = torch.from_numpy(normalized_horizons).cuda()
            quantile_tensor = None if training_quantiles is None else torch.from_numpy(training_quantiles).cuda()
            last = 0.0
            for step in range(int(model_config["steps"])):
                row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
                local_horizon_index = training_horizon_tensor[
                    torch.randint(len(training_horizon_tensor), (len(row),), device="cuda")
                ]
                if implicit_quantile_training:
                    local_quantile = (
                        float(training_quantile_range[0])
                        + (float(training_quantile_range[1]) - float(training_quantile_range[0]))
                        * torch.rand(len(row), device="cuda")
                    )
                else:
                    local_quantile = quantile_tensor[
                        torch.randint(len(quantile_tensor), (len(row),), device="cuda")
                    ]
                prediction = model(x[row], horizon_tensor[local_horizon_index], local_quantile)
                error = y[row, local_horizon_index] - prediction
                loss = torch.maximum(local_quantile * error, (local_quantile - 1.0) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                if step % 500 == 0:
                    if horizon_spline_quantile_surface_training:
                        card = "P329 spline-risk"
                    else:
                        card = "P328 implicit-risk" if implicit_quantile_training else "P327 risk-conditioned"
                    print(f"{card} Hxq surface step={step + 1} loss={last:.7f}", flush=True)
            model.eval()
            calibration_index = np.flatnonzero(calibration_rows)
            with torch.no_grad():
                calibration_x = x[torch.from_numpy(calibration_index).cuda()]
                calibration_residuals = []
                for horizon_index in training_horizon_indices:
                    local_h = torch.full((len(calibration_x),), float(normalized_horizons[horizon_index]), device="cuda")
                    local_q = torch.full((len(calibration_x),), heldout_quantile, device="cuda")
                    prediction = model(calibration_x, local_h, local_q).cpu().numpy()
                    calibration_residuals.append(source_target[calibration_rows, horizon_index] - prediction)
                calibration_offset = float(np.quantile(np.concatenate(calibration_residuals), heldout_quantile))
                source_h = torch.full((len(x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                source_q = torch.full((len(x),), heldout_quantile, device="cuda")
                source_score = model(x, source_h, source_q).cpu().numpy()
                p201_x = torch.from_numpy(p201_feature).cuda()
                p201_h = torch.full((len(p201_x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                p201_q = torch.full((len(p201_x),), heldout_quantile, device="cuda")
                p201_score = model(p201_x, p201_h, p201_q).cpu().numpy()
            source_score = np.maximum(source_score + calibration_offset, 0.0)
            p201_score = np.maximum(p201_score + calibration_offset, 0.0)
            source_local_target = source_target[:, heldout_horizon_index]
            p201_local_target = p201_target[:, heldout_horizon_index]
            frozen_selective = torch.load(
                args.runs_root / config["frozen_horizon_selective_authority"]["run"] / config["frozen_horizon_selective_authority"]["artifact"],
                map_location="cpu",
            )
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            source_metrics = _selective_metrics(
                source_score[development_rows], source_local_target[development_rows],
                source_conditions[development_rows], evaluation_ceilings,
            )
            p201_metrics = _selective_metrics(p201_score, p201_local_target, p201_conditions, evaluation_ceilings)
            decision = config["decision"]
            checks = {
                "P201_risk_conditioned_horizon_authority_risk": p201_metrics["maximum_unsafe_admission_rate"] <= float(decision["maximum_P201_unsafe_admission_rate"]),
                "P201_risk_conditioned_horizon_authority_coverage": p201_metrics["mean_admission_coverage"] >= float(decision["minimum_P201_mean_admission_coverage"]),
                "P201_ceiling_admission_monotonicity": p201_metrics["ceiling_admission_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_admission_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_width": source_feature.shape[1],
                "hidden_dimensions": model_config["hidden_dimensions"],
                "input_mean": feature_mean,
                "input_scale": feature_scale,
                "quantile_floor": float(config["quantile_floor"]),
                "quantile_knots": None if not horizon_spline_quantile_surface_training else np.asarray(config["quantile_knots"], np.float32),
                "training_quantile_levels": training_quantiles,
                "training_quantile_range": training_quantile_range,
                "heldout_quantile_level": heldout_quantile,
                "horizons_seconds": horizon_values,
                "training_horizon_indices": training_horizon_indices,
                "heldout_horizon_index": heldout_horizon_index,
                "calibration_offset": calibration_offset,
                "heldout_log_cost_ceilings": evaluation_ceilings,
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {
                    "horizon_quantile_example_count": int(
                        train_rows.sum() * len(training_horizon_indices)
                        * (1 if training_quantiles is None else len(training_quantiles))
                    ),
                    "training_quantile_levels": None if training_quantiles is None else [float(value) for value in training_quantiles],
                    "training_quantile_range": None if training_quantile_range is None else [float(value) for value in training_quantile_range],
                    "surface_type": "monotone_piecewise_linear_spline" if horizon_spline_quantile_surface_training else "positive_bilinear",
                    "steps": int(model_config["steps"]), "final_pinball_loss": last,
                },
                "calibration": {
                    "horizon_example_count": int(calibration_rows.sum() * len(training_horizon_indices)),
                    "heldout_quantile_level": heldout_quantile, "signed_log_cost_offset": calibration_offset,
                },
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
                "source_heldout_horizon_and_quantile_development": source_metrics,
                "P201_heldout_task_horizon_and_quantile_development": p201_metrics,
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
        if horizon_risk_matched_quantile_training:
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration_rows = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
            development_rows = source_example_scenes % modulus == int(split["development_scene_remainder"])
            train_rows = ~(calibration_rows | development_rows)
            feature_mean = source_feature[train_rows].mean(0)
            feature_scale = source_feature[train_rows].std(0).clip(min=1e-5)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(config["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            training_horizon_indices = np.asarray(config["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(config["heldout_horizon_index"])
            model_config = config["risk_quantile_model"]
            model = RiskMatchedHorizonQuantileHead(
                source_feature.shape[1], model_config["hidden_dimensions"]
            ).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            x = torch.from_numpy(source_feature).cuda()
            y = torch.from_numpy(source_target).cuda()
            train_index = torch.from_numpy(np.flatnonzero(train_rows)).cuda()
            training_horizon_tensor = torch.from_numpy(training_horizon_indices).cuda()
            horizon_tensor = torch.from_numpy(normalized_horizons).cuda()
            q = float(config["risk_quantile_level"])
            last = 0.0
            for step in range(int(model_config["steps"])):
                row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
                local_horizon_index = training_horizon_tensor[
                    torch.randint(len(training_horizon_tensor), (len(row),), device="cuda")
                ]
                prediction = model(x[row], horizon_tensor[local_horizon_index])
                error = y[row, local_horizon_index] - prediction
                loss = torch.maximum(q * error, (q - 1.0) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                if step % 500 == 0:
                    print(f"P326 risk-matched horizon q85 step={step + 1} loss={last:.7f}", flush=True)
            model.eval()
            calibration_index = np.flatnonzero(calibration_rows)
            with torch.no_grad():
                calibration_x = x[torch.from_numpy(calibration_index).cuda()]
                calibration_residuals = []
                for horizon_index in training_horizon_indices:
                    local_h = torch.full((len(calibration_x),), float(normalized_horizons[horizon_index]), device="cuda")
                    prediction = model(calibration_x, local_h).cpu().numpy()
                    calibration_residuals.append(source_target[calibration_rows, horizon_index] - prediction)
                calibration_offset = float(np.quantile(np.concatenate(calibration_residuals), q))
                source_h = torch.full((len(x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                source_score = model(x, source_h).cpu().numpy()
                p201_x = torch.from_numpy(p201_feature).cuda()
                p201_h = torch.full((len(p201_x),), float(normalized_horizons[heldout_horizon_index]), device="cuda")
                p201_score = model(p201_x, p201_h).cpu().numpy()
            source_score = np.maximum(source_score + calibration_offset, 0.0)
            p201_score = np.maximum(p201_score + calibration_offset, 0.0)
            source_local_target = source_target[:, heldout_horizon_index]
            p201_local_target = p201_target[:, heldout_horizon_index]
            frozen_selective = torch.load(
                args.runs_root / config["frozen_horizon_selective_authority"]["run"] / config["frozen_horizon_selective_authority"]["artifact"],
                map_location="cpu",
            )
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            source_metrics = _selective_metrics(
                source_score[development_rows], source_local_target[development_rows],
                source_conditions[development_rows], evaluation_ceilings,
            )
            p201_metrics = _selective_metrics(p201_score, p201_local_target, p201_conditions, evaluation_ceilings)
            decision = config["decision"]
            checks = {
                "P201_risk_matched_horizon_authority_risk": p201_metrics["maximum_unsafe_admission_rate"] <= float(decision["maximum_P201_unsafe_admission_rate"]),
                "P201_risk_matched_horizon_authority_coverage": p201_metrics["mean_admission_coverage"] >= float(decision["minimum_P201_mean_admission_coverage"]),
                "P201_ceiling_admission_monotonicity": p201_metrics["ceiling_admission_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_admission_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_width": source_feature.shape[1],
                "hidden_dimensions": model_config["hidden_dimensions"],
                "input_mean": feature_mean,
                "input_scale": feature_scale,
                "horizons_seconds": horizon_values,
                "training_horizon_indices": training_horizon_indices,
                "heldout_horizon_index": heldout_horizon_index,
                "risk_quantile_level": q,
                "calibration_offset": calibration_offset,
                "heldout_log_cost_ceilings": evaluation_ceilings,
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {
                    "horizon_example_count": int(train_rows.sum() * len(training_horizon_indices)),
                    "risk_quantile_level": q, "steps": int(model_config["steps"]), "final_pinball_loss": last,
                },
                "calibration": {
                    "horizon_example_count": int(calibration_rows.sum() * len(training_horizon_indices)),
                    "risk_quantile_level": q, "signed_log_cost_offset": calibration_offset,
                },
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
                "source_heldout_horizon_development": source_metrics,
                "P201_heldout_task_and_horizon_development": p201_metrics,
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
        if horizon_temporal_calibration_training:
            frozen_horizon = torch.load(
                args.runs_root / config["frozen_horizon_certificate"]["run"] / config["frozen_horizon_certificate"]["artifact"],
                map_location="cuda",
            )
            frozen_selective = torch.load(
                args.runs_root / config["frozen_horizon_selective_authority"]["run"] / config["frozen_horizon_selective_authority"]["artifact"],
                map_location="cuda",
            )
            feature_mean = np.asarray(frozen_horizon["input_mean"], np.float32)
            feature_scale = np.asarray(frozen_horizon["input_scale"], np.float32)
            quantile_offsets = np.asarray(frozen_horizon["calibration_offsets"], np.float32)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(frozen_horizon["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            quantile_model = HorizonMonotoneQuantileHead(
                int(frozen_horizon["input_width"]), frozen_horizon["hidden_dimensions"]
            ).cuda()
            quantile_model.load_state_dict(frozen_horizon["model_state_dict"])
            quantile_model.eval()
            selective_model = CeilingConditionedSelectiveAuthority(
                int(frozen_selective["input_width"]), frozen_selective["hidden_dimensions"],
                float(frozen_selective["maximum_residual"]),
            ).cuda()
            selective_model.load_state_dict(frozen_selective["model_state_dict"])
            selective_model.eval()
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration_rows = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
            development_rows = source_example_scenes % modulus == int(split["development_scene_remainder"])
            train_rows = ~(calibration_rows | development_rows)
            training_horizon_indices = np.asarray(config["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(config["heldout_horizon_index"])

            def expand_temporal_horizons(features, target, horizon_indices):
                expanded_feature, expanded_target, expanded_base, expanded_row = [], [], [], []
                with torch.no_grad():
                    local_x = torch.from_numpy(features).cuda()
                    for horizon_index in horizon_indices:
                        h_value = float(normalized_horizons[horizon_index])
                        h = torch.full((len(local_x),), h_value, device="cuda")
                        prediction = quantile_model(local_x, h).cpu().numpy() + quantile_offsets[None]
                        base_score = np.maximum.accumulate(prediction, axis=1)[:, 2]
                        expanded_feature.append(np.concatenate((features, np.full((len(features), 1), h_value, np.float32)), 1))
                        expanded_target.append(target[:, horizon_index])
                        expanded_base.append(base_score.astype(np.float32))
                        expanded_row.append(np.arange(len(features)))
                return np.concatenate(expanded_feature), np.concatenate(expanded_target), np.concatenate(expanded_base), np.concatenate(expanded_row)

            source_expanded, source_expanded_target, source_expanded_base, source_expanded_row = expand_temporal_horizons(
                source_feature, source_target, training_horizon_indices
            )
            p201_expanded, p201_expanded_target, p201_expanded_base, _ = expand_temporal_horizons(
                p201_feature, p201_target, np.asarray([heldout_horizon_index], np.int64)
            )
            expanded_train = train_rows[source_expanded_row]
            expanded_calibration = calibration_rows[source_expanded_row]
            expanded_development = development_rows[source_expanded_row]
            with torch.no_grad():
                source_raw_score = selective_model(
                    torch.from_numpy(source_expanded).cuda(), torch.from_numpy(source_expanded_base).cuda()
                )[0].cpu().numpy()
                p201_raw_score = selective_model(
                    torch.from_numpy(p201_expanded).cuda(), torch.from_numpy(p201_expanded_base).cuda()
                )[0].cpu().numpy()
            scale_config = config["scale_model"]
            scale_model = HorizonNonconformityScale(
                source_expanded.shape[1], scale_config["hidden_dimensions"], float(scale_config["minimum_scale"])
            ).cuda()
            optimizer = torch.optim.AdamW(
                scale_model.parameters(), lr=float(scale_config["learning_rate"]), weight_decay=float(scale_config["weight_decay"])
            )
            x = torch.from_numpy(source_expanded).cuda()
            positive_gap = torch.from_numpy(np.maximum(source_expanded_target - source_raw_score, 0.0).astype(np.float32)).cuda()
            train_index = torch.from_numpy(np.flatnonzero(expanded_train)).cuda()
            q = float(config["calibration_score_quantile"])
            last = 0.0
            for step in range(int(scale_config["steps"])):
                row = train_index[torch.randint(len(train_index), (int(scale_config["batch_size"]),), device="cuda")]
                prediction = scale_model(x[row])
                error = positive_gap[row] - prediction
                loss = torch.maximum(q * error, (q - 1.0) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                if step % 500 == 0:
                    print(f"P325 temporal nonconformity scale step={step + 1} loss={last:.7f}", flush=True)
            scale_model.eval()
            with torch.no_grad():
                source_scale = scale_model(x).cpu().numpy()
                p201_scale = scale_model(torch.from_numpy(p201_expanded).cuda()).cpu().numpy()
            normalized_residual = (
                source_expanded_target[expanded_calibration] - source_raw_score[expanded_calibration]
            ) / source_scale[expanded_calibration]
            normalized_threshold = max(0.0, float(np.quantile(normalized_residual, q)))
            source_score = source_raw_score + normalized_threshold * source_scale
            p201_score = p201_raw_score + normalized_threshold * p201_scale
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            source_conditions_expanded = np.concatenate([source_conditions for _ in training_horizon_indices])
            source_metrics = _selective_metrics(
                source_score[expanded_development], source_expanded_target[expanded_development],
                source_conditions_expanded[expanded_development], evaluation_ceilings,
            )
            p201_metrics = _selective_metrics(p201_score, p201_expanded_target, p201_conditions, evaluation_ceilings)
            p324_global_margin = float(frozen_selective["calibration_margin"])
            p201_p324_metrics = _selective_metrics(
                p201_raw_score + p324_global_margin, p201_expanded_target, p201_conditions, evaluation_ceilings
            )
            p201_base_metrics = _selective_metrics(
                p201_expanded_base, p201_expanded_target, p201_conditions, evaluation_ceilings
            )
            decision = config["decision"]
            checks = {
                "P201_time_varying_horizon_authority_risk": p201_metrics["maximum_unsafe_admission_rate"] <= float(decision["maximum_P201_unsafe_admission_rate"]),
                "P201_time_varying_horizon_authority_coverage": p201_metrics["mean_admission_coverage"] >= float(decision["minimum_P201_mean_admission_coverage"]),
                "P201_ceiling_admission_monotonicity": p201_metrics["ceiling_admission_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_admission_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "model_state_dict": scale_model.state_dict(),
                "input_width": source_expanded.shape[1],
                "hidden_dimensions": scale_config["hidden_dimensions"],
                "minimum_scale": float(scale_config["minimum_scale"]),
                "normalized_calibration_threshold": normalized_threshold,
                "calibration_score_quantile": q,
                "heldout_log_cost_ceilings": evaluation_ceilings,
                "frozen_horizon_certificate": config["frozen_horizon_certificate"],
                "frozen_horizon_selective_authority": config["frozen_horizon_selective_authority"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {
                    "horizon_example_count": int(expanded_train.sum()), "steps": int(scale_config["steps"]),
                    "final_pinball_loss": last, "positive_gap_mean": float(positive_gap[train_index].mean().detach()),
                },
                "calibration": {
                    "horizon_example_count": int(expanded_calibration.sum()), "score_quantile": q,
                    "normalized_nonconformity_threshold": normalized_threshold,
                    "source_scale_mean": float(np.mean(source_scale[expanded_calibration])),
                    "source_scale_minimum": float(np.min(source_scale[expanded_calibration])),
                    "source_scale_maximum": float(np.max(source_scale[expanded_calibration])),
                },
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
                "source_development": source_metrics,
                "P201_post_hoc_development": p201_metrics,
                "P201_frozen_P324_global_margin_baseline": p201_p324_metrics,
                "P201_frozen_horizon_q95_baseline": p201_base_metrics,
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
        if horizon_selective_authority_training:
            frozen_horizon = torch.load(
                args.runs_root / config["frozen_horizon_certificate"]["run"] / config["frozen_horizon_certificate"]["artifact"],
                map_location="cuda",
            )
            feature_mean = np.asarray(frozen_horizon["input_mean"], np.float32)
            feature_scale = np.asarray(frozen_horizon["input_scale"], np.float32)
            quantile_offsets = np.asarray(frozen_horizon["calibration_offsets"], np.float32)
            source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
            horizon_values = np.asarray(frozen_horizon["horizons_seconds"], np.float32)
            normalized_horizons = horizon_values / float(horizon_values.max())
            quantile_model = HorizonMonotoneQuantileHead(
                int(frozen_horizon["input_width"]), frozen_horizon["hidden_dimensions"]
            ).cuda()
            quantile_model.load_state_dict(frozen_horizon["model_state_dict"])
            quantile_model.eval()
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration_rows = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
            development_rows = source_example_scenes % modulus == int(split["development_scene_remainder"])
            train_rows = ~(calibration_rows | development_rows)
            training_horizon_indices = np.asarray(config["training_horizon_indices"], np.int64)
            heldout_horizon_index = int(config["heldout_horizon_index"])

            def expand_horizons(features, target, horizon_indices):
                expanded_feature, expanded_target, expanded_base, expanded_row = [], [], [], []
                with torch.no_grad():
                    x = torch.from_numpy(features).cuda()
                    for horizon_index in horizon_indices:
                        h_value = float(normalized_horizons[horizon_index])
                        h = torch.full((len(x),), h_value, device="cuda")
                        prediction = quantile_model(x, h).cpu().numpy() + quantile_offsets[None]
                        base_score = np.maximum.accumulate(prediction, axis=1)[:, 2]
                        expanded_feature.append(np.concatenate((features, np.full((len(features), 1), h_value, np.float32)), 1))
                        expanded_target.append(target[:, horizon_index])
                        expanded_base.append(base_score.astype(np.float32))
                        expanded_row.append(np.arange(len(features)))
                return np.concatenate(expanded_feature), np.concatenate(expanded_target), np.concatenate(expanded_base), np.concatenate(expanded_row)

            source_expanded, source_expanded_target, source_expanded_base, source_expanded_row = expand_horizons(
                source_feature, source_target, training_horizon_indices
            )
            p201_expanded, p201_expanded_target, p201_expanded_base, _ = expand_horizons(
                p201_feature, p201_target, np.asarray([heldout_horizon_index], np.int64)
            )
            expanded_train = train_rows[source_expanded_row]
            expanded_calibration = calibration_rows[source_expanded_row]
            expanded_development = development_rows[source_expanded_row]
            anchor_levels = np.asarray(config["training_ceiling_quantile_levels"], np.float32)
            evaluation_levels = np.asarray(config["heldout_ceiling_quantile_levels"], np.float32)
            anchor_ceilings = np.quantile(source_expanded_target[expanded_train], anchor_levels).astype(np.float32)
            evaluation_ceilings = np.quantile(source_expanded_target[expanded_train], evaluation_levels).astype(np.float32)
            model_config = config["selective_model"]
            model = CeilingConditionedSelectiveAuthority(
                source_expanded.shape[1], model_config["hidden_dimensions"], float(model_config["maximum_residual"])
            ).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            x = torch.from_numpy(source_expanded).cuda()
            y = torch.from_numpy(source_expanded_target).cuda()
            frozen_score = torch.from_numpy(source_expanded_base).cuda()
            ceiling_tensor = torch.from_numpy(anchor_ceilings).cuda()
            train_index = torch.from_numpy(np.flatnonzero(expanded_train)).cuda()
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
                    print(f"P324 horizon-selective authority step={step + 1} loss={last:.7f}", flush=True)
            model.eval()
            with torch.no_grad():
                source_score = model(x, frozen_score)[0].cpu().numpy()
                p201_score = model(
                    torch.from_numpy(p201_expanded).cuda(), torch.from_numpy(p201_expanded_base).cuda()
                )[0].cpu().numpy()
            calibration_residual = source_expanded_target[expanded_calibration] - source_score[expanded_calibration]
            calibration_margin = max(0.0, float(np.quantile(calibration_residual, float(config["calibration_score_quantile"]))))
            source_score += calibration_margin
            p201_score += calibration_margin
            source_conditions_expanded = np.concatenate([source_conditions for _ in training_horizon_indices])
            source_metrics = _selective_metrics(
                source_score[expanded_development], source_expanded_target[expanded_development],
                source_conditions_expanded[expanded_development], evaluation_ceilings,
            )
            p201_metrics = _selective_metrics(
                p201_score, p201_expanded_target, p201_conditions, evaluation_ceilings
            )
            source_base_metrics = _selective_metrics(
                source_expanded_base[expanded_development], source_expanded_target[expanded_development],
                source_conditions_expanded[expanded_development], evaluation_ceilings,
            )
            p201_base_metrics = _selective_metrics(
                p201_expanded_base, p201_expanded_target, p201_conditions, evaluation_ceilings
            )
            decision = config["decision"]
            checks = {
                "P201_horizon_selective_authority_risk": p201_metrics["maximum_unsafe_admission_rate"] <= float(decision["maximum_P201_unsafe_admission_rate"]),
                "P201_horizon_selective_authority_coverage": p201_metrics["mean_admission_coverage"] >= float(decision["minimum_P201_mean_admission_coverage"]),
                "P201_ceiling_admission_monotonicity": p201_metrics["ceiling_admission_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_admission_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_width": source_expanded.shape[1],
                "hidden_dimensions": model_config["hidden_dimensions"],
                "maximum_residual": float(model_config["maximum_residual"]),
                "calibration_margin": calibration_margin,
                "training_ceiling_quantile_levels": anchor_levels,
                "training_log_cost_ceilings": anchor_ceilings,
                "heldout_ceiling_quantile_levels": evaluation_levels,
                "heldout_log_cost_ceilings": evaluation_ceilings,
                "frozen_horizon_certificate": config["frozen_horizon_certificate"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {"horizon_example_count": int(expanded_train.sum()), "steps": int(model_config["steps"]), "final_loss": last, "training_log_cost_ceilings": [float(value) for value in anchor_ceilings]},
                "calibration": {"horizon_example_count": int(expanded_calibration.sum()), "score_quantile": float(config["calibration_score_quantile"]), "nonnegative_log_cost_margin": calibration_margin},
                "heldout_horizon_seconds": float(horizon_values[heldout_horizon_index]),
                "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
                "source_development": source_metrics,
                "source_frozen_horizon_q95_baseline": source_base_metrics,
                "P201_post_hoc_development": p201_metrics,
                "P201_frozen_horizon_q95_baseline": p201_base_metrics,
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
        split = config["split"]
        modulus = int(split["scene_modulus"])
        calibration = source_example_scenes % modulus == int(split["calibration_scene_remainder"])
        development = source_example_scenes % modulus == int(split["development_scene_remainder"])
        train = ~(calibration | development)
        feature_mean = source_feature[train].mean(0)
        feature_scale = source_feature[train].std(0).clip(min=1e-5)
        source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
        p201_feature = ((p201_feature - feature_mean) / feature_scale).astype(np.float32)
        model_config = config["horizon_model"]
        model = HorizonMonotoneQuantileHead(source_feature.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
        )
        x = torch.from_numpy(source_feature).cuda()
        y = torch.from_numpy(source_target).cuda()
        horizon_values = np.asarray(config["horizons_seconds"], np.float32)
        normalized_horizons = horizon_values / float(horizon_values.max())
        training_horizon_indices = np.asarray(config["training_horizon_indices"], np.int64)
        heldout_horizon_index = int(config["heldout_horizon_index"])
        horizon_tensor = torch.from_numpy(normalized_horizons).cuda()
        training_horizon_tensor = torch.from_numpy(training_horizon_indices).cuda()
        quantiles = np.asarray(config["quantile_levels"], np.float32)
        q = torch.from_numpy(quantiles).cuda()
        train_index = torch.from_numpy(np.flatnonzero(train)).cuda()
        last = 0.0
        for step in range(int(model_config["steps"])):
            row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
            local_horizon_index = training_horizon_tensor[
                torch.randint(len(training_horizon_tensor), (len(row),), device="cuda")
            ]
            prediction = model(x[row], horizon_tensor[local_horizon_index])
            error = y[row, local_horizon_index, None] - prediction
            loss = torch.maximum(q[None] * error, (q[None] - 1) * error).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
            if step % 500 == 0:
                print(f"P322 horizon quantile step={step + 1} pinball={last:.7f}", flush=True)
        model.eval()
        calibration_predictions, calibration_targets = [], []
        calibration_x = x[torch.from_numpy(np.flatnonzero(calibration)).cuda()]
        with torch.no_grad():
            for horizon_index in training_horizon_indices:
                h = torch.full((len(calibration_x),), float(normalized_horizons[horizon_index]), device="cuda")
                calibration_predictions.append(model(calibration_x, h).cpu().numpy())
                calibration_targets.append(source_target[calibration, horizon_index])
        calibration_prediction = np.concatenate(calibration_predictions)
        calibration_target = np.concatenate(calibration_targets)
        residual = calibration_target[:, None] - calibration_prediction
        offsets = np.asarray([
            np.quantile(residual[:, index], float(level)) for index, level in enumerate(quantiles)
        ], np.float32)

        def horizon_metrics(features, target, conditions, horizon_index):
            with torch.no_grad():
                h = torch.full((len(features),), float(normalized_horizons[horizon_index]), device="cuda")
                prediction = model(torch.from_numpy(features).cuda(), h).cpu().numpy()
            prediction = np.maximum.accumulate(prediction + offsets[None], axis=1)
            local_target = target[:, horizon_index]
            coverage = np.mean(local_target[:, None] <= prediction, axis=0)
            error = local_target[:, None] - prediction
            pinball = np.maximum(quantiles[None] * error, (quantiles[None] - 1) * error)
            by_condition = {}
            for progress, command in np.unique(conditions, axis=0):
                local = np.isclose(conditions[:, 0], progress) & np.isclose(conditions[:, 1], command)
                by_condition[f"progress={float(progress)},command={float(command)}"] = {
                    "example_count": int(local.sum()),
                    "empirical_coverages": [float(value) for value in np.mean(local_target[local, None] <= prediction[local], axis=0)],
                    "median_absolute_log_cost_error": float(np.mean(np.abs(prediction[local, 0] - local_target[local]))),
                }
            return {
                "example_count": int(len(local_target)),
                "horizon_seconds": float(horizon_values[horizon_index]),
                "quantile_levels": [float(value) for value in quantiles],
                "empirical_coverages": [float(value) for value in coverage],
                "maximum_quantile_undercoverage": float(np.max(quantiles - coverage)),
                "median_absolute_log_cost_error": float(np.mean(np.abs(prediction[:, 0] - local_target))),
                "mean_pinball_loss": float(np.mean(pinball)),
                "mean_q95_minus_q50_width": float(np.mean(prediction[:, 2] - prediction[:, 0])),
                "by_task_condition": by_condition,
            }

        source_metrics = horizon_metrics(
            source_feature[development], source_target[development], source_conditions[development], heldout_horizon_index
        )
        p201_metrics = horizon_metrics(p201_feature, p201_target, p201_conditions, heldout_horizon_index)
        with torch.no_grad():
            all_horizon_predictions = []
            p201_x = torch.from_numpy(p201_feature).cuda()
            for horizon_value in normalized_horizons:
                h = torch.full((len(p201_x),), float(horizon_value), device="cuda")
                prediction = model(p201_x, h).cpu().numpy() + offsets[None]
                all_horizon_predictions.append(np.maximum.accumulate(prediction, axis=1))
        all_horizon_predictions = np.stack(all_horizon_predictions, axis=1)
        horizon_violations = int(np.sum(np.diff(all_horizon_predictions, axis=1) < -1e-8))
        quantile_violations = int(np.sum(np.diff(all_horizon_predictions, axis=2) < -1e-8))
        p201_metrics["horizon_monotonicity_violations"] = horizon_violations
        p201_metrics["quantile_order_violations"] = quantile_violations
        decision = config["decision"]
        checks = {
            "P201_heldout_horizon_quantile_coverage": p201_metrics["maximum_quantile_undercoverage"] <= float(decision["maximum_P201_quantile_undercoverage"]),
            "P201_heldout_horizon_median_fidelity": p201_metrics["median_absolute_log_cost_error"] <= float(decision["maximum_P201_median_absolute_log_cost_error"]),
            "P201_horizon_quantile_monotonicity": horizon_violations <= int(decision["maximum_P201_horizon_monotonicity_violations"]) and quantile_violations == 0,
        }
        verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_width": source_feature.shape[1],
            "hidden_dimensions": model_config["hidden_dimensions"],
            "input_mean": feature_mean,
            "input_scale": feature_scale,
            "quantile_levels": quantiles,
            "horizons_seconds": horizon_values,
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
            "training": {
                "row_count": int(train.sum()),
                "horizon_example_count": int(train.sum() * len(training_horizon_indices)),
                "training_horizons_seconds": [float(horizon_values[index]) for index in training_horizon_indices],
                "steps": int(model_config["steps"]),
                "final_pinball_loss": last,
            },
            "calibration": {
                "horizon_example_count": int(calibration.sum() * len(training_horizon_indices)),
                "additive_offsets": [float(value) for value in offsets],
            },
            "source_heldout_horizon_development": source_metrics,
            "P201_heldout_task_and_horizon_development": p201_metrics,
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
    if (selective_authority_training or selective_authority_confirmation or task_projection_evaluation
            or action_pair_editor_training or groupwise_pair_editor_training):
        certificate = torch.load(
            args.runs_root / config["frozen_certificate"]["run"] / config["frozen_certificate"]["artifact"],
            map_location="cuda",
        )
        feature_mean = np.asarray(certificate["input_mean"], np.float32)
        feature_scale = np.asarray(certificate["input_scale"], np.float32)
        offsets = np.asarray(certificate["calibration_offsets"], np.float32)
        source_feature = ((source_feature - feature_mean) / feature_scale).astype(np.float32)
        p201_feature_raw = p201_feature
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

        if groupwise_pair_editor_training:
            frozen_selective = torch.load(
                args.runs_root / config["frozen_selective_authority"]["run"] / config["frozen_selective_authority"]["artifact"],
                map_location="cuda",
            )
            selective_model = CeilingConditionedSelectiveAuthority(
                int(frozen_selective["input_width"]), frozen_selective["hidden_dimensions"],
                float(frozen_selective["maximum_residual"]),
            ).cuda()
            selective_model.load_state_dict(frozen_selective["model_state_dict"])
            selective_model.eval()
            selective_margin = float(frozen_selective["calibration_margin"])
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            source_pair = _pair_examples(
                source_descriptor, source_selector_score, source_costs, source_queries, source_scenes,
                progress_model, maneuver_model, config["training_progress_preferences"],
                config["training_lateral_commands"], float(config["lateral_preference_weight"]),
            )
            p201_pair = _pair_examples(
                p201_descriptor, p201_selector_score, p201_costs, p201_queries, p201_scenes,
                progress_model, maneuver_model, config["heldout_progress_preferences"],
                config["heldout_lateral_commands"], float(config["lateral_preference_weight"]),
            )
            (source_pair_feature, source_pair_target, source_pair_scenes, _, source_nominal,
             source_pair_utility, pair_indices) = source_pair
            (p201_pair_feature, p201_pair_target, _, _, p201_nominal,
             p201_pair_utility, _) = p201_pair
            feature_width = source_pair_feature.shape[-1]
            pair_count = source_pair_feature.shape[1]
            source_pair_feature = ((source_pair_feature - feature_mean) / feature_scale).astype(np.float32)
            p201_pair_feature = ((p201_pair_feature - feature_mean) / feature_scale).astype(np.float32)

            def frozen_set_scores(features):
                flat = features.reshape(-1, features.shape[-1])
                chunks = []
                with torch.no_grad():
                    for begin in range(0, len(flat), 4096):
                        local = torch.from_numpy(flat[begin:begin + 4096]).cuda()
                        prediction = quantile_model(local).cpu().numpy()
                        q95 = np.maximum.accumulate(prediction + offsets[None], axis=1)[:, 2].astype(np.float32)
                        chunks.append((selective_model(
                            local, torch.from_numpy(q95).cuda()
                        )[0].cpu().numpy() + selective_margin).astype(np.float32))
                return np.concatenate(chunks).reshape(features.shape[:2])

            source_pair_base = frozen_set_scores(source_pair_feature)
            p201_pair_base = frozen_set_scores(p201_pair_feature)

            def augment(features, base_score, utility):
                utility_unit = (utility - utility.min(1, keepdims=True)) / np.maximum(
                    utility.max(1, keepdims=True) - utility.min(1, keepdims=True), 1e-5
                )
                return np.concatenate((features, base_score[:, :, None], utility_unit[:, :, None]), 2).astype(np.float32), utility_unit.astype(np.float32)

            source_augmented, source_utility_unit = augment(source_pair_feature, source_pair_base, source_pair_utility)
            p201_augmented, p201_utility_unit = augment(p201_pair_feature, p201_pair_base, p201_pair_utility)
            row_scenes = source_pair_scenes[:, 0]
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration_rows = row_scenes % modulus == int(split["calibration_scene_remainder"])
            development_rows = row_scenes % modulus == int(split["development_scene_remainder"])
            train_rows = ~(calibration_rows | development_rows)
            model_config = config["groupwise_editor_model"]
            selector = GroupwisePairSelector(source_augmented.shape[2], int(model_config["hidden_width"])).cuda()
            optimizer = torch.optim.AdamW(
                selector.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            x = torch.from_numpy(source_augmented).cuda()
            target = torch.from_numpy(source_pair_target).cuda()
            utility = torch.from_numpy(source_utility_unit).cuda()
            train_index = torch.from_numpy(np.flatnonzero(train_rows)).cuda()
            last_selector = 0.0
            for step in range(int(model_config["selector_steps"])):
                row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
                composite = target[row] + float(model_config["utility_preservation_weight"]) * utility[row]
                oracle = torch.argmin(composite, 1)
                logits = selector(x[row])
                probability = torch.softmax(logits / float(model_config["selection_temperature"]), 1)
                regret = (probability * (composite - composite.min(1, keepdim=True).values)).sum(1).mean()
                loss = F.cross_entropy(logits, oracle) + float(model_config["decision_regret_weight"]) * regret
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_selector = float(loss.detach())
                if step % 500 == 0:
                    print(f"P321 groupwise selector step={step + 1} loss={last_selector:.7f}", flush=True)
            selector.eval()
            with torch.no_grad():
                source_choice = torch.argmax(selector(x), 1).cpu().numpy()
                p201_choice = torch.argmax(selector(torch.from_numpy(p201_augmented).cuda()), 1).cpu().numpy()

            def selected_features(augmented, choice):
                row = np.arange(len(augmented))
                return np.concatenate((augmented[row, choice], augmented.mean(1)), 1).astype(np.float32)

            source_selected_feature = selected_features(source_augmented, source_choice)
            p201_selected_feature = selected_features(p201_augmented, p201_choice)
            source_selected_target = source_pair_target[np.arange(len(source_pair_target)), source_choice]
            p201_selected_target = p201_pair_target[np.arange(len(p201_pair_target)), p201_choice]
            risk_model = SelectedPairQuantileHead(
                source_selected_feature.shape[1], model_config["risk_hidden_dimensions"]
            ).cuda()
            risk_optimizer = torch.optim.AdamW(
                risk_model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            selected_x = torch.from_numpy(source_selected_feature).cuda()
            selected_y = torch.from_numpy(source_selected_target).cuda()
            quantile = float(config["selected_pair_quantile"])
            last_risk = 0.0
            for step in range(int(model_config["risk_steps"])):
                row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
                prediction = risk_model(selected_x[row])
                error = selected_y[row] - prediction
                loss = torch.maximum(quantile * error, (quantile - 1) * error).mean()
                risk_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                risk_optimizer.step()
                last_risk = float(loss.detach())
                if step % 500 == 0:
                    print(f"P321 selected-pair q90 step={step + 1} pinball={last_risk:.7f}", flush=True)
            risk_model.eval()
            with torch.no_grad():
                source_selected_score = risk_model(selected_x).cpu().numpy()
                p201_selected_score = risk_model(torch.from_numpy(p201_selected_feature).cuda()).cpu().numpy()
            selected_calibration_residual = source_selected_target[calibration_rows] - source_selected_score[calibration_rows]
            selected_margin = max(0.0, float(np.quantile(selected_calibration_residual, quantile)))
            source_selected_score += selected_margin
            p201_selected_score += selected_margin

            def groupwise_metrics(selected_score, selected_target, selected_choice, nominal_choice, nominal_base, ceilings):
                row = np.arange(len(selected_score))
                by_ceiling = {}
                coverages, nominal_coverages, unsafe_rates = [], [], []
                previous = None
                monotonicity_violations = 0
                for ceiling in ceilings:
                    authorized = selected_score <= float(ceiling)
                    nominal_authorized = nominal_base[row, nominal_choice] <= float(ceiling)
                    unsafe = authorized & (selected_target > ceiling)
                    if previous is not None:
                        monotonicity_violations += int(np.sum(previous & ~authorized))
                    previous = authorized
                    count = int(authorized.sum())
                    coverage = float(np.mean(authorized))
                    nominal_coverage = float(np.mean(nominal_authorized))
                    unsafe_rate = float(unsafe.sum() / max(count, 1))
                    by_ceiling[str(float(ceiling))] = {
                        "log_cost_ceiling": float(ceiling),
                        "actual_cost_ceiling": float(np.expm1(ceiling)),
                        "groupwise_pair_authority_coverage": coverage,
                        "nominal_pair_authority_coverage": nominal_coverage,
                        "authority_coverage_gain": coverage - nominal_coverage,
                        "unsafe_groupwise_pair_rate": unsafe_rate,
                        "pair_edit_rate_among_authorized": float(np.sum(authorized & (selected_choice != nominal_choice)) / max(count, 1)),
                        "mean_selected_actual_cost": float(np.mean(np.expm1(selected_target[authorized]))) if count else None,
                    }
                    coverages.append(coverage)
                    nominal_coverages.append(nominal_coverage)
                    unsafe_rates.append(unsafe_rate)
                return {
                    "request_example_count": int(len(selected_score)),
                    "candidate_pair_count": int(pair_count),
                    "mean_groupwise_pair_authority_coverage": float(np.mean(coverages)),
                    "mean_nominal_pair_authority_coverage": float(np.mean(nominal_coverages)),
                    "mean_authority_coverage_gain": float(np.mean(np.asarray(coverages) - np.asarray(nominal_coverages))),
                    "maximum_unsafe_groupwise_pair_rate": float(np.max(unsafe_rates)),
                    "ceiling_authority_monotonicity_violations": monotonicity_violations,
                    "by_ceiling": by_ceiling,
                }

            source_metrics = groupwise_metrics(
                source_selected_score[development_rows], source_selected_target[development_rows],
                source_choice[development_rows], source_nominal[development_rows],
                source_pair_base[development_rows], evaluation_ceilings,
            )
            p201_metrics = groupwise_metrics(
                p201_selected_score, p201_selected_target, p201_choice, p201_nominal,
                p201_pair_base, evaluation_ceilings,
            )
            decision = config["decision"]
            checks = {
                "P201_groupwise_editor_risk": p201_metrics["maximum_unsafe_groupwise_pair_rate"] <= float(decision["maximum_P201_unsafe_groupwise_pair_rate"]),
                "P201_groupwise_editor_coverage_gain": p201_metrics["mean_authority_coverage_gain"] >= float(decision["minimum_P201_mean_authority_coverage_gain"]),
                "P201_groupwise_editor_monotonicity": p201_metrics["ceiling_authority_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_authority_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "selector_state_dict": selector.state_dict(),
                "risk_model_state_dict": risk_model.state_dict(),
                "pair_input_width": int(source_augmented.shape[2]),
                "hidden_width": int(model_config["hidden_width"]),
                "risk_input_width": int(source_selected_feature.shape[1]),
                "risk_hidden_dimensions": model_config["risk_hidden_dimensions"],
                "selected_pair_quantile": quantile,
                "selected_pair_calibration_margin": selected_margin,
                "pair_indices": pair_indices,
                "heldout_log_cost_ceilings": evaluation_ceilings,
                "frozen_selective_authority": config["frozen_selective_authority"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {
                    "row_count": int(train_rows.sum()),
                    "pair_example_count": int(train_rows.sum() * pair_count),
                    "selector_steps": int(model_config["selector_steps"]),
                    "risk_steps": int(model_config["risk_steps"]),
                    "final_selector_loss": last_selector,
                    "final_risk_pinball_loss": last_risk,
                },
                "calibration": {"row_count": int(calibration_rows.sum()), "selected_pair_quantile": quantile, "nonnegative_log_cost_margin": selected_margin},
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
            return

        if action_pair_editor_training:
            frozen_selective = torch.load(
                args.runs_root / config["frozen_selective_authority"]["run"] / config["frozen_selective_authority"]["artifact"],
                map_location="cuda",
            )
            selective_model = CeilingConditionedSelectiveAuthority(
                int(frozen_selective["input_width"]), frozen_selective["hidden_dimensions"],
                float(frozen_selective["maximum_residual"]),
            ).cuda()
            selective_model.load_state_dict(frozen_selective["model_state_dict"])
            selective_model.eval()
            selective_margin = float(frozen_selective["calibration_margin"])
            anchor_ceilings = np.asarray(frozen_selective["training_log_cost_ceilings"], np.float32)
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            source_pair = _pair_examples(
                source_descriptor, source_selector_score, source_costs, source_queries, source_scenes,
                progress_model, maneuver_model, config["training_progress_preferences"],
                config["training_lateral_commands"], float(config["lateral_preference_weight"]),
            )
            p201_pair = _pair_examples(
                p201_descriptor, p201_selector_score, p201_costs, p201_queries, p201_scenes,
                progress_model, maneuver_model, config["heldout_progress_preferences"],
                config["heldout_lateral_commands"], float(config["lateral_preference_weight"]),
            )
            (source_pair_feature, source_pair_target, source_pair_scenes, _, source_nominal,
             source_pair_utility, pair_indices) = source_pair
            (p201_pair_feature, p201_pair_target, _, _, p201_nominal,
             p201_pair_utility, _) = p201_pair
            feature_width = source_pair_feature.shape[-1]
            pair_count = source_pair_feature.shape[1]
            source_pair_feature = ((source_pair_feature.reshape(-1, feature_width) - feature_mean) / feature_scale).astype(np.float32)
            p201_pair_feature = ((p201_pair_feature.reshape(-1, feature_width) - feature_mean) / feature_scale).astype(np.float32)

            def frozen_pair_scores(features):
                chunks = []
                with torch.no_grad():
                    for begin in range(0, len(features), 4096):
                        local = torch.from_numpy(features[begin:begin + 4096]).cuda()
                        prediction = quantile_model(local).cpu().numpy()
                        q95 = np.maximum.accumulate(prediction + offsets[None], axis=1)[:, 2].astype(np.float32)
                        score = selective_model(local, torch.from_numpy(q95).cuda())[0].cpu().numpy() + selective_margin
                        chunks.append(score.astype(np.float32))
                return np.concatenate(chunks)

            source_pair_base = frozen_pair_scores(source_pair_feature)
            p201_pair_base = frozen_pair_scores(p201_pair_feature)
            source_pair_target_flat = source_pair_target.reshape(-1)
            source_pair_scenes_flat = source_pair_scenes.reshape(-1)
            split = config["split"]
            modulus = int(split["scene_modulus"])
            calibration = source_pair_scenes_flat % modulus == int(split["calibration_scene_remainder"])
            development_rows = source_pair_scenes[:, 0] % modulus == int(split["development_scene_remainder"])
            train = ~(calibration | (source_pair_scenes_flat % modulus == int(split["development_scene_remainder"])))
            model_config = config["pair_editor_model"]
            pair_model = CeilingConditionedSelectiveAuthority(
                feature_width, model_config["hidden_dimensions"], float(model_config["maximum_residual"])
            ).cuda()
            optimizer = torch.optim.AdamW(
                pair_model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"])
            )
            x = torch.from_numpy(source_pair_feature).cuda()
            y = torch.from_numpy(source_pair_target_flat).cuda()
            frozen_score = torch.from_numpy(source_pair_base).cuda()
            ceiling_tensor = torch.from_numpy(anchor_ceilings).cuda()
            train_index = torch.from_numpy(np.flatnonzero(train)).cuda()
            temperature = float(model_config["admission_temperature"])
            last = 0.0
            for step in range(int(model_config["steps"])):
                row = train_index[torch.randint(len(train_index), (int(model_config["batch_size"]),), device="cuda")]
                ceiling = ceiling_tensor[torch.randint(len(ceiling_tensor), (len(row),), device="cuda")]
                risk_score, residual = pair_model(x[row], frozen_score[row])
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
                    print(f"P320 action-pair editor step={step + 1} loss={last:.7f}", flush=True)
            pair_model.eval()

            def edited_pair_scores(features, base_score):
                chunks = []
                with torch.no_grad():
                    for begin in range(0, len(features), 4096):
                        chunks.append(pair_model(
                            torch.from_numpy(features[begin:begin + 4096]).cuda(),
                            torch.from_numpy(base_score[begin:begin + 4096]).cuda(),
                        )[0].cpu().numpy())
                return np.concatenate(chunks)

            source_pair_score = edited_pair_scores(source_pair_feature, source_pair_base)
            p201_pair_score = edited_pair_scores(p201_pair_feature, p201_pair_base)
            pair_calibration_residual = source_pair_target_flat[calibration] - source_pair_score[calibration]
            pair_margin = max(0.0, float(np.quantile(pair_calibration_residual, float(config["calibration_score_quantile"]))))
            source_pair_score = source_pair_score.reshape(-1, pair_count) + pair_margin
            source_pair_base = source_pair_base.reshape(-1, pair_count)
            p201_pair_score = p201_pair_score.reshape(-1, pair_count) + pair_margin
            p201_pair_base = p201_pair_base.reshape(-1, pair_count)

            def editor_metrics(score, base_score, target, nominal, utility, ceilings):
                by_ceiling = {}
                coverages, nominal_coverages, unsafe_rates = [], [], []
                previous_authorized = None
                monotonicity_violations = 0
                row = np.arange(len(score))
                for ceiling in ceilings:
                    feasible = score <= float(ceiling)
                    authorized = feasible.any(1)
                    chosen = np.argmin(np.where(feasible, utility, np.inf), axis=1)
                    nominal_authorized = base_score[row, nominal] <= float(ceiling)
                    unsafe = authorized & (target[row, chosen] > ceiling)
                    if previous_authorized is not None:
                        monotonicity_violations += int(np.sum(previous_authorized & ~authorized))
                    previous_authorized = authorized
                    count = int(authorized.sum())
                    coverage = float(np.mean(authorized))
                    nominal_coverage = float(np.mean(nominal_authorized))
                    unsafe_rate = float(unsafe.sum() / max(count, 1))
                    by_ceiling[str(float(ceiling))] = {
                        "log_cost_ceiling": float(ceiling),
                        "actual_cost_ceiling": float(np.expm1(ceiling)),
                        "edited_pair_authority_coverage": coverage,
                        "nominal_pair_authority_coverage": nominal_coverage,
                        "authority_coverage_gain": coverage - nominal_coverage,
                        "unsafe_edited_pair_rate": unsafe_rate,
                        "pair_edit_rate_among_authorized": float(np.sum(authorized & (chosen != nominal)) / max(count, 1)),
                        "mean_selected_actual_cost": float(np.mean(np.expm1(target[row[authorized], chosen[authorized]]))) if count else None,
                    }
                    coverages.append(coverage)
                    nominal_coverages.append(nominal_coverage)
                    unsafe_rates.append(unsafe_rate)
                return {
                    "request_example_count": int(len(score)),
                    "candidate_pair_count": int(score.shape[1]),
                    "mean_edited_pair_authority_coverage": float(np.mean(coverages)),
                    "mean_nominal_pair_authority_coverage": float(np.mean(nominal_coverages)),
                    "mean_authority_coverage_gain": float(np.mean(np.asarray(coverages) - np.asarray(nominal_coverages))),
                    "maximum_unsafe_edited_pair_rate": float(np.max(unsafe_rates)),
                    "ceiling_authority_monotonicity_violations": monotonicity_violations,
                    "by_ceiling": by_ceiling,
                }

            source_metrics = editor_metrics(
                source_pair_score[development_rows], source_pair_base[development_rows],
                source_pair_target[development_rows], source_nominal[development_rows],
                source_pair_utility[development_rows], evaluation_ceilings,
            )
            p201_metrics = editor_metrics(
                p201_pair_score, p201_pair_base, p201_pair_target, p201_nominal,
                p201_pair_utility, evaluation_ceilings,
            )
            decision = config["decision"]
            checks = {
                "P201_pair_editor_risk": p201_metrics["maximum_unsafe_edited_pair_rate"] <= float(decision["maximum_P201_unsafe_edited_pair_rate"]),
                "P201_pair_editor_coverage_gain": p201_metrics["mean_authority_coverage_gain"] >= float(decision["minimum_P201_mean_authority_coverage_gain"]),
                "P201_pair_editor_monotonicity": p201_metrics["ceiling_authority_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_authority_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            torch.save({
                "model_state_dict": pair_model.state_dict(),
                "input_width": feature_width,
                "hidden_dimensions": model_config["hidden_dimensions"],
                "maximum_residual": float(model_config["maximum_residual"]),
                "calibration_margin": pair_margin,
                "pair_indices": pair_indices,
                "training_log_cost_ceilings": anchor_ceilings,
                "heldout_log_cost_ceilings": evaluation_ceilings,
                "frozen_selective_authority": config["frozen_selective_authority"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {"example_count": int(train.sum()), "steps": int(model_config["steps"]), "final_loss": last},
                "calibration": {"example_count": int(calibration.sum()), "score_quantile": float(config["calibration_score_quantile"]), "nonnegative_log_cost_margin": pair_margin},
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
            return

        if task_projection_evaluation:
            frozen_selective = torch.load(
                args.runs_root / config["frozen_selective_authority"]["run"] / config["frozen_selective_authority"]["artifact"],
                map_location="cuda",
            )
            selective_model = CeilingConditionedSelectiveAuthority(
                int(frozen_selective["input_width"]),
                frozen_selective["hidden_dimensions"],
                float(frozen_selective["maximum_residual"]),
            ).cuda()
            selective_model.load_state_dict(frozen_selective["model_state_dict"])
            selective_model.eval()
            calibration_margin = float(frozen_selective["calibration_margin"])
            ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            endpoint_feature, endpoint_target, _, endpoint_conditions = _task_examples(
                p201_descriptor, p201_selector_score, p201_costs, p201_queries, p201_scenes,
                progress_model, maneuver_model, config["training_progress_preferences"],
                config["training_lateral_commands"], float(config["lateral_preference_weight"]),
                int(config["selected_action_count"]),
            )
            group_count = len(p201_groups)
            endpoint_count = len(config["training_progress_preferences"]) * len(config["training_lateral_commands"])
            request_count = len(config["heldout_progress_preferences"]) * len(config["heldout_lateral_commands"])
            candidate_feature = np.concatenate((
                endpoint_feature.reshape(endpoint_count, group_count, -1),
                p201_feature_raw.reshape(request_count, group_count, -1),
            ), axis=0).transpose(1, 0, 2)
            candidate_target = np.concatenate((
                endpoint_target.reshape(endpoint_count, group_count),
                p201_target.reshape(request_count, group_count),
            ), axis=0).T
            candidate_conditions = np.concatenate((
                endpoint_conditions.reshape(endpoint_count, group_count, 2),
                p201_conditions.reshape(request_count, group_count, 2),
            ), axis=0).transpose(1, 0, 2)
            candidate_feature_flat = ((candidate_feature.reshape(-1, candidate_feature.shape[-1]) - feature_mean) / feature_scale).astype(np.float32)
            with torch.no_grad():
                candidate_quantiles = quantile_model(torch.from_numpy(candidate_feature_flat).cuda()).cpu().numpy()
            candidate_frozen = np.maximum.accumulate(candidate_quantiles + offsets[None], axis=1)[:, 2].astype(np.float32)
            with torch.no_grad():
                candidate_score = selective_model(
                    torch.from_numpy(candidate_feature_flat).cuda(), torch.from_numpy(candidate_frozen).cuda()
                )[0].cpu().numpy() + calibration_margin
            candidate_score = candidate_score.reshape(group_count, endpoint_count + request_count)
            candidate_target_cost = np.expm1(candidate_target)
            lateral_weight = float(config["task_projection_lateral_weight"])
            by_ceiling = {}
            projection_coverages, exact_coverages, unsafe_rates = [], [], []
            previous_authorized = None
            monotonicity_violations = 0
            for ceiling in ceilings:
                feasible = candidate_score <= float(ceiling)
                authorized_rows, exact_rows, unsafe_rows, deviations, selected_costs = [], [], [], [], []
                for request_index in range(request_count):
                    request_condition = candidate_conditions[:, endpoint_count + request_index]
                    distance = (
                        np.square(candidate_conditions[:, :, 0] - request_condition[:, None, 0])
                        + lateral_weight * np.square(candidate_conditions[:, :, 1] - request_condition[:, None, 1])
                    )
                    projected_distance = np.where(feasible, distance, np.inf)
                    authorized = feasible.any(1)
                    chosen = np.argmin(projected_distance, axis=1)
                    row = np.arange(group_count)
                    exact = feasible[:, endpoint_count + request_index]
                    chosen_target = candidate_target[row, chosen]
                    authorized_rows.append(authorized)
                    exact_rows.append(exact)
                    unsafe_rows.append(authorized & (chosen_target > ceiling))
                    deviations.append(np.where(authorized, np.sqrt(distance[row, chosen]), 0.0))
                    selected_costs.append(np.where(authorized, candidate_target_cost[row, chosen], 0.0))
                authorized = np.concatenate(authorized_rows)
                exact = np.concatenate(exact_rows)
                unsafe = np.concatenate(unsafe_rows)
                deviation = np.concatenate(deviations)
                selected_cost = np.concatenate(selected_costs)
                if previous_authorized is not None:
                    monotonicity_violations += int(np.sum(previous_authorized & ~authorized))
                previous_authorized = authorized
                authorized_count = int(authorized.sum())
                coverage = float(np.mean(authorized))
                exact_coverage = float(np.mean(exact))
                unsafe_rate = float(unsafe.sum() / max(authorized_count, 1))
                by_ceiling[str(float(ceiling))] = {
                    "log_cost_ceiling": float(ceiling),
                    "actual_cost_ceiling": float(np.expm1(ceiling)),
                    "projected_authority_coverage": coverage,
                    "exact_request_authority_coverage": exact_coverage,
                    "authority_coverage_gain": coverage - exact_coverage,
                    "unsafe_projected_authority_rate": unsafe_rate,
                    "exact_request_retention_rate_among_authorized": float(np.sum(authorized & exact) / max(authorized_count, 1)),
                    "mean_task_deviation_among_authorized": float(np.mean(deviation[authorized])) if authorized_count else 0.0,
                    "mean_selected_actual_cost": float(np.mean(selected_cost[authorized])) if authorized_count else None,
                }
                projection_coverages.append(coverage)
                exact_coverages.append(exact_coverage)
                unsafe_rates.append(unsafe_rate)
            metrics = {
                "request_example_count": int(group_count * request_count),
                "candidate_task_count": int(endpoint_count + request_count),
                "mean_projected_authority_coverage": float(np.mean(projection_coverages)),
                "mean_exact_request_authority_coverage": float(np.mean(exact_coverages)),
                "mean_authority_coverage_gain": float(np.mean(np.asarray(projection_coverages) - np.asarray(exact_coverages))),
                "maximum_unsafe_projected_authority_rate": float(np.max(unsafe_rates)),
                "ceiling_authority_monotonicity_violations": monotonicity_violations,
                "by_ceiling": by_ceiling,
            }
            decision = config["decision"]
            checks = {
                "P201_projected_authority_risk": metrics["maximum_unsafe_projected_authority_rate"] <= float(decision["maximum_P201_unsafe_projected_authority_rate"]),
                "P201_projected_authority_coverage_gain": metrics["mean_authority_coverage_gain"] >= float(decision["minimum_P201_mean_authority_coverage_gain"]),
                "P201_projected_authority_monotonicity": metrics["ceiling_authority_monotonicity_violations"] <= int(decision["maximum_P201_ceiling_authority_monotonicity_violations"]),
            }
            verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {"example_count": 0, "steps": 0},
                "P201_post_hoc_development": metrics,
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

        if selective_authority_confirmation:
            frozen_selective = torch.load(
                args.runs_root / config["frozen_selective_authority"]["run"] / config["frozen_selective_authority"]["artifact"],
                map_location="cuda",
            )
            model = CeilingConditionedSelectiveAuthority(
                int(frozen_selective["input_width"]),
                frozen_selective["hidden_dimensions"],
                float(frozen_selective["maximum_residual"]),
            ).cuda()
            model.load_state_dict(frozen_selective["model_state_dict"])
            model.eval()
            evaluation_ceilings = np.asarray(frozen_selective["heldout_log_cost_ceilings"], np.float32)
            calibration_margin = float(frozen_selective["calibration_margin"])
            with torch.no_grad():
                p201_score = model(
                    torch.from_numpy(p201_feature).cuda(), torch.from_numpy(p201_frozen_score).cuda()
                )[0].cpu().numpy() + calibration_margin
            p201_metrics = _selective_metrics(p201_score, p201_target, p201_conditions, evaluation_ceilings)
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
                "input_width": int(frozen_selective["input_width"]),
                "hidden_dimensions": frozen_selective["hidden_dimensions"],
                "maximum_residual": float(frozen_selective["maximum_residual"]),
                "calibration_margin": calibration_margin,
                "training_ceiling_quantile_levels": frozen_selective["training_ceiling_quantile_levels"],
                "training_log_cost_ceilings": frozen_selective["training_log_cost_ceilings"],
                "heldout_ceiling_quantile_levels": frozen_selective["heldout_ceiling_quantile_levels"],
                "heldout_log_cost_ceilings": evaluation_ceilings,
                "frozen_certificate": config["frozen_certificate"],
                "frozen_selective_authority": config["frozen_selective_authority"],
            }, run_dir / config["model_artifact"])
            summary = {
                "schema_version": config["output_schema_version"],
                "task_id": config["task_id"],
                "hypothesis_id": config["hypothesis_id"],
                "status": "done",
                "verdict": verdict,
                "role": config["role"],
                "training": {"example_count": 0, "steps": 0, "final_loss": None},
                "calibration": {
                    "example_count": 0,
                    "nonnegative_log_cost_margin": calibration_margin,
                    "reused_from": config["frozen_selective_authority"],
                },
                "heldout_log_cost_ceilings": [float(value) for value in evaluation_ceilings],
                "source_development": None,
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
