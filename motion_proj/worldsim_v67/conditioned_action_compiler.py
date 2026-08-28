"""Budget/horizon-conditioned decision-focused trajectory action compiler."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, FEATURE_NAMES, action_features


CONDITIONED_FEATURE_NAMES = FEATURE_NAMES + ("selected_fraction", "horizon_seconds")


def conditioned_padded_cases(
    arrays: Mapping[str, np.ndarray], selected_fractions: list[float], horizon_seconds_by_domain: list[float],
    mean: np.ndarray | None = None, scale: np.ndarray | None = None, base_score_key: str = "qmean",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    base = action_features(arrays)
    cases = np.asarray(arrays["case_index"], dtype=np.int64)
    domains = np.asarray(arrays.get("domain_index", np.zeros(len(cases))), dtype=np.int64)
    unique_cases = np.unique(cases)
    maximum_actions = max(int(np.count_nonzero(cases == case)) for case in unique_cases)
    row_count = len(unique_cases) * len(selected_fractions)
    raw = np.zeros((row_count, maximum_actions, base.shape[1] + 2), dtype=np.float32)
    payload = {
        "qmean": np.zeros((row_count, maximum_actions), dtype=np.float32),
        "target": np.zeros((row_count, maximum_actions), dtype=np.float32),
        "mask": np.zeros((row_count, maximum_actions), dtype=np.float32),
        "domain": np.zeros(row_count, dtype=np.int64),
        "selected_fraction": np.zeros(row_count, dtype=np.float32),
        "action_indices": np.empty(row_count, dtype=object),
    }
    row = 0
    horizons = np.asarray(horizon_seconds_by_domain, dtype=np.float32)
    for fraction in selected_fractions:
        for case in unique_cases:
            members = np.flatnonzero(cases == case)
            count = len(members)
            domain = int(domains[members[0]])
            raw[row, :count, :-2] = base[members]
            raw[row, :count, -2] = float(fraction)
            raw[row, :count, -1] = float(horizons[domain])
            payload["qmean"][row, :count] = np.asarray(arrays[base_score_key])[members]
            payload["target"][row, :count] = np.asarray(arrays["target_cost"])[members]
            payload["mask"][row, :count] = 1.0
            payload["domain"][row] = domain
            payload["selected_fraction"][row] = float(fraction)
            payload["action_indices"][row] = members
            row += 1
    valid = payload["mask"].astype(bool)
    if mean is None:
        mean = raw[valid].mean(axis=0)
        scale = raw[valid].std(axis=0).clip(min=1e-4)
    payload["features"] = ((raw - mean) / scale).astype(np.float32) * payload["mask"][:, :, None]
    return payload, np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)


def train_conditioned_action_compiler(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, Any], seed: int,
) -> tuple[BoundedListwiseCompiler, np.ndarray, np.ndarray, dict[str, Any]]:
    padded, mean, scale = conditioned_padded_cases(
        arrays, [float(x) for x in config["training_selected_fractions"]],
        [float(x) for x in config["training_horizon_seconds_by_domain"]],
        base_score_key=str(config.get("base_score_key", "qmean")),
    )
    features = torch.from_numpy(padded["features"]).cuda()
    qmean = torch.from_numpy(padded["qmean"]).cuda()
    target = torch.from_numpy(padded["target"]).cuda()
    mask = torch.from_numpy(padded["mask"]).cuda()
    domains = torch.from_numpy(padded["domain"]).cuda()
    fractions = torch.from_numpy(padded["selected_fraction"]).cuda()
    action_count = mask.sum(dim=1)
    selected_count = torch.floor(action_count * fractions).clamp(min=1)
    pair_valid = (
        (mask[:, :, None] * mask[:, None, :]).bool()
        & (target[:, :, None] - target[:, None, :]).abs().ge(float(config["pairwise_minimum_target_gap"]))
    )
    pair_sign = torch.sign(target[:, :, None] - target[:, None, :])
    torch.manual_seed(int(seed))
    model = BoundedListwiseCompiler(
        features.shape[-1], list(config["hidden_dimensions"]), float(config["maximum_residual_cost"])
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    final = {}
    for _ in range(int(config["epochs"])):
        score, residual = model(features, qmean, mask)
        if "residual_budget_anchor_fraction" in config:
            anchor = float(config["residual_budget_anchor_fraction"])
            if "residual_budget_peak_fraction" in config:
                peak = float(config["residual_budget_peak_fraction"])
                upper = float(config["residual_budget_upper_anchor_fraction"])
                rising = (fractions - anchor) / (peak - anchor)
                falling = (upper - fractions) / (upper - peak)
                amplitude = torch.minimum(rising, falling).clamp(0.0, 1.0)[:, None]
            else:
                full = float(config["residual_budget_full_fraction"])
                amplitude = ((fractions - anchor) / (full - anchor)).clamp(0.0, 1.0)[:, None]
            residual = residual * amplitude
            score = (qmean + residual).clamp(0.0, 1.0)
        regression_elements = torch.nn.functional.smooth_l1_loss(
            score, target, beta=float(config["huber_beta"]), reduction="none"
        )
        regression_case = (regression_elements * mask).sum(dim=1) / action_count
        score_delta = score[:, :, None] - score[:, None, :]
        pair_loss = torch.nn.functional.softplus(-(score_delta * pair_sign) / float(config["ranking_temperature"]))
        pairwise_case = (pair_loss * pair_valid).sum(dim=(1, 2)) / pair_valid.sum(dim=(1, 2)).clamp(min=1)
        soft_rank = 1.0 + (
            torch.sigmoid(score_delta / float(config["soft_rank_temperature"])) * mask[:, None, :]
        ).sum(dim=2) - 0.5
        selected_weight = torch.sigmoid(
            (selected_count[:, None] + 0.5 - soft_rank) / float(config["soft_selection_temperature"])
        ) * mask
        selected_cost = (selected_weight * target).sum(dim=1) / selected_weight.sum(dim=1).clamp(min=1e-5)
        domain_losses = []
        for domain in torch.unique(domains):
            inside = domains == domain
            domain_losses.append(
                float(config["listwise_weight"]) * selected_cost[inside].mean()
                + float(config["ranking_weight"]) * pairwise_case[inside].mean()
                + float(config["regression_weight"]) * regression_case[inside].mean()
            )
        domain_losses_t = torch.stack(domain_losses)
        residual_penalty = (residual.square() * mask).sum() / mask.sum()
        if config.get("domain_aggregation", "mean_variance") == "smooth_max":
            domain_temperature = float(config["domain_smooth_max_temperature"])
            domain_objective = domain_temperature * torch.logsumexp(domain_losses_t / domain_temperature, dim=0)
        else:
            domain_objective = domain_losses_t.mean() + float(config["domain_loss_variance_weight"]) * domain_losses_t.var(unbiased=False)
        loss = domain_objective + float(config["residual_regularization_weight"]) * residual_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "total_loss": float(loss.detach().cpu()),
            "soft_selected_cost": float(selected_cost.mean().detach().cpu()),
            "pairwise_loss": float(pairwise_case.mean().detach().cpu()),
            "regression_loss": float(regression_case.mean().detach().cpu()),
            "residual_rms": float(torch.sqrt(residual_penalty).detach().cpu()),
            "minimum_domain_loss": float(domain_losses_t.min().detach().cpu()),
            "maximum_domain_loss": float(domain_losses_t.max().detach().cpu()),
        }
    final.update(
        train_conditioned_case_count=int(len(padded["domain"])),
        train_action_rows=int(mask.sum().cpu()), development_domain_count=int(len(np.unique(padded["domain"]))),
    )
    return model.eval(), mean, scale, final


def score_conditioned_action_compiler(
    model: BoundedListwiseCompiler, arrays: Mapping[str, np.ndarray], selected_fraction: float,
    horizon_seconds_by_domain: list[float], mean: np.ndarray, scale: np.ndarray, base_score_key: str = "qmean",
    residual_budget_anchor_fraction: float | None = None, residual_budget_full_fraction: float | None = None,
    residual_budget_peak_fraction: float | None = None, residual_budget_upper_anchor_fraction: float | None = None,
) -> np.ndarray:
    padded, _, _ = conditioned_padded_cases(
        arrays, [float(selected_fraction)], horizon_seconds_by_domain, mean, scale, base_score_key=base_score_key
    )
    with torch.inference_mode():
        scores, residual = model(
            torch.from_numpy(padded["features"]).cuda(), torch.from_numpy(padded["qmean"]).cuda(),
            torch.from_numpy(padded["mask"]).cuda(),
        )
        if residual_budget_anchor_fraction is not None:
            if residual_budget_peak_fraction is not None:
                rising = (float(selected_fraction) - float(residual_budget_anchor_fraction)) / (
                    float(residual_budget_peak_fraction) - float(residual_budget_anchor_fraction)
                )
                falling = (float(residual_budget_upper_anchor_fraction) - float(selected_fraction)) / (
                    float(residual_budget_upper_anchor_fraction) - float(residual_budget_peak_fraction)
                )
                amplitude = np.clip(min(rising, falling), 0.0, 1.0)
            else:
                amplitude = np.clip(
                    (float(selected_fraction) - float(residual_budget_anchor_fraction))
                    / (float(residual_budget_full_fraction) - float(residual_budget_anchor_fraction)), 0.0, 1.0,
                )
            scores = (torch.from_numpy(padded["qmean"]).cuda() + float(amplitude) * residual).clamp(0.0, 1.0)
    flat = np.zeros(len(arrays["qmean"]), dtype=np.float32)
    score_np = scores.cpu().numpy()
    for row, members in enumerate(padded["action_indices"]):
        flat[members] = score_np[row, :len(members)]
    return flat
