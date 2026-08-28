"""Bounded listwise trajectory-action compiler."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


FEATURE_NAMES = (
    "qmean",
    "qstd",
    "q_min",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q_max",
    "progress_ratio",
    "absolute_lateral_offset_m",
    "log_visited_count",
)


class BoundedListwiseCompiler(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: list[int], maximum_residual: float) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current = int(feature_count)
        for width in hidden_dimensions:
            layers.extend((torch.nn.Linear(current, int(width)), torch.nn.SiLU()))
            current = int(width)
        layers.append(torch.nn.Linear(current, 1))
        self.network = torch.nn.Sequential(*layers)
        self.maximum_residual = float(maximum_residual)

    def forward(
        self, features: torch.Tensor, qmean: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.tanh(self.network(features).squeeze(-1)) * self.maximum_residual
        center = (raw * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        residual = (raw - center) * mask
        return (qmean + residual).clamp(0.0, 1.0), residual


def action_features(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(arrays["qmean"], dtype=np.float32)[:, None],
            np.asarray(arrays["qstd"], dtype=np.float32)[:, None],
            np.asarray(arrays["quantiles"], dtype=np.float32),
            np.asarray(arrays["progress_ratio"], dtype=np.float32)[:, None],
            np.abs(np.asarray(arrays["lateral_offset_m"], dtype=np.float32))[:, None],
            np.log1p(np.asarray(arrays["visited_count"], dtype=np.float32))[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def padded_cases(
    arrays: Mapping[str, np.ndarray], mean: np.ndarray | None = None, scale: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    features = action_features(arrays)
    if mean is None:
        mean = features.mean(axis=0)
        scale = features.std(axis=0).clip(min=1e-4)
    normalized = (features - mean) / scale
    cases = np.asarray(arrays["case_index"], dtype=np.int64)
    unique_cases = np.unique(cases)
    maximum_actions = max(int(np.count_nonzero(cases == case)) for case in unique_cases)
    payload = {
        "features": np.zeros((len(unique_cases), maximum_actions, features.shape[1]), dtype=np.float32),
        "qmean": np.zeros((len(unique_cases), maximum_actions), dtype=np.float32),
        "target": np.zeros((len(unique_cases), maximum_actions), dtype=np.float32),
        "unsafe": np.zeros((len(unique_cases), maximum_actions), dtype=np.float32),
        "mask": np.zeros((len(unique_cases), maximum_actions), dtype=np.float32),
        "domain": np.zeros(len(unique_cases), dtype=np.int64),
        "action_indices": np.empty(len(unique_cases), dtype=object),
    }
    domains = np.asarray(arrays.get("domain_index", np.zeros(len(cases))), dtype=np.int64)
    for row, case in enumerate(unique_cases):
        members = np.flatnonzero(cases == case)
        count = len(members)
        payload["features"][row, :count] = normalized[members]
        payload["qmean"][row, :count] = np.asarray(arrays["qmean"])[members]
        payload["target"][row, :count] = np.asarray(arrays["target_cost"])[members]
        payload["unsafe"][row, :count] = np.asarray(arrays["unsafe"], dtype=np.float32)[members]
        payload["mask"][row, :count] = 1.0
        payload["domain"][row] = domains[members[0]]
        payload["action_indices"][row] = members
    return payload, np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)


def train_listwise_compiler(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, Any], seed: int
) -> tuple[BoundedListwiseCompiler, np.ndarray, np.ndarray, dict[str, Any]]:
    padded, mean, scale = padded_cases(arrays)
    features = torch.from_numpy(padded["features"]).cuda()
    qmean = torch.from_numpy(padded["qmean"]).cuda()
    target = torch.from_numpy(padded["target"]).cuda()
    unsafe = torch.from_numpy(padded["unsafe"]).cuda()
    mask = torch.from_numpy(padded["mask"]).cuda()
    domains = torch.from_numpy(padded["domain"]).cuda()
    action_count = mask.sum(dim=1)
    selected_count = torch.floor(action_count * float(config["selected_fraction"])).clamp(min=1)
    pair_valid = (
        (mask[:, :, None] * mask[:, None, :]).bool()
        & (target[:, :, None] - target[:, None, :]).abs().ge(float(config["pairwise_minimum_target_gap"]))
    )
    pair_sign = torch.sign(target[:, :, None] - target[:, None, :])
    torch.manual_seed(int(seed))
    model = BoundedListwiseCompiler(
        features.shape[-1], list(config["hidden_dimensions"]), float(config["maximum_residual_cost"])
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    final = {}
    for _ in range(int(config["epochs"])):
        score, residual = model(features, qmean, mask)
        regression_elements = torch.nn.functional.smooth_l1_loss(
            score, target, beta=float(config["huber_beta"]), reduction="none"
        )
        regression_case = (regression_elements * mask).sum(dim=1) / action_count
        score_delta = score[:, :, None] - score[:, None, :]
        pair_loss = torch.nn.functional.softplus(
            -(score_delta * pair_sign) / float(config["ranking_temperature"])
        )
        pairwise_case = (pair_loss * pair_valid).sum(dim=(1, 2)) / pair_valid.sum(dim=(1, 2)).clamp(min=1)
        soft_rank = 1.0 + (
            torch.sigmoid(score_delta / float(config["soft_rank_temperature"]))
            * mask[:, None, :]
        ).sum(dim=2) - 0.5
        selected_weight = torch.sigmoid(
            (selected_count[:, None] + 0.5 - soft_rank)
            / float(config["soft_selection_temperature"])
        ) * mask
        listwise_case = (selected_weight * target).sum(dim=1) / selected_weight.sum(dim=1).clamp(min=1e-5)
        unsafe_listwise_case = (selected_weight * unsafe).sum(dim=1) / selected_weight.sum(dim=1).clamp(min=1e-5)
        risk_aversion = float(config.get("entropic_risk_aversion", 1.0))
        entropic_case = torch.log(
            (selected_weight * torch.exp(risk_aversion * target)).sum(dim=1)
            / selected_weight.sum(dim=1).clamp(min=1e-5)
        ) / risk_aversion
        domain_losses = []
        for domain in torch.unique(domains):
            inside = domains == domain
            domain_losses.append(
                float(config["regression_weight"]) * regression_case[inside].mean()
                + float(config["ranking_weight"]) * pairwise_case[inside].mean()
                + float(config["listwise_weight"]) * listwise_case[inside].mean()
                + float(config.get("unsafe_listwise_weight", 0.0)) * unsafe_listwise_case[inside].mean()
                + float(config.get("entropic_listwise_weight", 0.0)) * entropic_case[inside].mean()
            )
        domain_losses_t = torch.stack(domain_losses)
        domain_mean = domain_losses_t.mean()
        domain_variance = domain_losses_t.var(unbiased=False)
        residual_penalty = (residual.square() * mask).sum() / mask.sum()
        loss = (
            domain_mean
            + float(config["domain_loss_variance_weight"]) * domain_variance
            + float(config["residual_regularization_weight"]) * residual_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "total_loss": float(loss.detach().cpu()),
            "domain_mean_loss": float(domain_mean.detach().cpu()),
            "domain_loss_variance": float(domain_variance.detach().cpu()),
            "regression_loss": float(regression_case.mean().detach().cpu()),
            "pairwise_loss": float(pairwise_case.mean().detach().cpu()),
            "soft_selected_cost": float(listwise_case.mean().detach().cpu()),
            "soft_selected_unsafe_rate": float(unsafe_listwise_case.mean().detach().cpu()),
            "soft_selected_entropic_risk": float(entropic_case.mean().detach().cpu()),
            "residual_rms": float(torch.sqrt(residual_penalty).detach().cpu()),
        }
    final.update(
        train_case_count=int(len(padded["domain"])),
        train_action_count=int(mask.sum().cpu()),
        development_domain_count=int(len(np.unique(padded["domain"]))),
    )
    return model.eval(), mean, scale, final


def score_listwise_compiler(
    model: BoundedListwiseCompiler,
    arrays: Mapping[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    padded, _, _ = padded_cases(arrays, mean, scale)
    with torch.inference_mode():
        scores, _ = model(
            torch.from_numpy(padded["features"]).cuda(),
            torch.from_numpy(padded["qmean"]).cuda(),
            torch.from_numpy(padded["mask"]).cuda(),
        )
    flat = np.zeros(len(arrays["qmean"]), dtype=np.float32)
    score_np = scores.cpu().numpy()
    for row, members in enumerate(padded["action_indices"]):
        flat[members] = score_np[row, : len(members)]
    return flat
