"""Trajectory-conditioned visited-state reliability model."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


FEATURE_NAMES = (
    "qmean",
    "log_visited_count",
    "progress_ratio",
    "lateral_offset_scaled",
    "absolute_lateral_offset_scaled",
    "qmean_case_delta",
    "qmean_case_zscore",
    "progress_lateral_interaction",
)


class TrajectoryReliabilityHead(torch.nn.Module):
    """Low-capacity residual expected-cost model conditioned on an Ego path."""

    def __init__(self, input_dimension: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        dimensions = [int(input_dimension), *(int(v) for v in hidden_dimensions), 1]
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(torch.nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(torch.nn.GELU())
        self.residual = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor, qmean: torch.Tensor) -> torch.Tensor:
        base = torch.logit(qmean.clamp(1e-4, 1.0 - 1e-4))
        return base + self.residual(values).reshape(-1)


class LatticeResidualAdapter(torch.nn.Module):
    """Bounded action-lattice bias that keeps qmean as the dominant ranking signal."""

    def __init__(self, action_count: int, maximum_residual_cost: float) -> None:
        super().__init__()
        self.action_bias = torch.nn.Parameter(torch.zeros(int(action_count)))
        self.maximum_residual_cost = float(maximum_residual_cost)

    def forward(
        self, qmean: torch.Tensor, actions: torch.Tensor, cases: torch.Tensor
    ) -> torch.Tensor:
        raw = self.maximum_residual_cost * torch.tanh(self.action_bias[actions])
        centered = torch.empty_like(raw)
        for case in torch.unique(cases):
            members = cases == case
            centered[members] = raw[members] - raw[members].mean()
        residual = centered.clamp(
            -self.maximum_residual_cost, self.maximum_residual_cost
        )
        return (qmean + residual).clamp(0.0, 1.0)


def feature_matrix(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    qmean = np.asarray(arrays["qmean"], dtype=np.float32)
    cases = np.asarray(arrays["case_index"])
    case_mean = np.zeros_like(qmean)
    case_std = np.zeros_like(qmean)
    for case in np.unique(cases):
        members = cases == case
        case_mean[members] = qmean[members].mean()
        case_std[members] = max(float(qmean[members].std()), 1e-4)
    progress = np.asarray(arrays["progress_ratio"], dtype=np.float32)
    lateral = np.asarray(arrays["lateral_offset_m"], dtype=np.float32) / 1.5
    return np.stack(
        (
            qmean,
            np.log1p(np.asarray(arrays["visited_count"], dtype=np.float32)),
            progress,
            lateral,
            np.abs(lateral),
            qmean - case_mean,
            (qmean - case_mean) / case_std,
            progress * lateral,
        ),
        axis=1,
    ).astype(np.float32)


def _ranking_pairs(
    target: np.ndarray, cases: np.ndarray, minimum_gap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right, signs = [], [], []
    for case in np.unique(cases):
        members = np.flatnonzero(cases == case)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                delta = float(target[members[i]] - target[members[j]])
                if abs(delta) < float(minimum_gap):
                    continue
                left.append(int(members[i]))
                right.append(int(members[j]))
                signs.append(float(np.sign(delta)))
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(signs, dtype=np.float32),
    )


def train_head(
    arrays: Mapping[str, np.ndarray], model_config: Mapping[str, Any], seed: int
) -> tuple[TrajectoryReliabilityHead, np.ndarray, np.ndarray, dict[str, Any]]:
    values = feature_matrix(arrays)
    target_np = np.asarray(arrays["target_cost"], dtype=np.float32)
    unsafe_np = np.asarray(arrays["unsafe"], dtype=np.float32)
    qmean_np = np.asarray(arrays["qmean"], dtype=np.float32)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-5)
    x = torch.from_numpy((values - mean) / scale).cuda()
    target = torch.from_numpy(target_np).cuda()
    unsafe = torch.from_numpy(unsafe_np).cuda()
    qmean = torch.from_numpy(qmean_np).cuda()
    left_np, right_np, signs_np = _ranking_pairs(
        target_np,
        np.asarray(arrays["case_index"]),
        float(model_config["pairwise_minimum_target_gap"]),
    )
    left = torch.from_numpy(left_np).cuda()
    right = torch.from_numpy(right_np).cuda()
    signs = torch.from_numpy(signs_np).cuda()
    torch.manual_seed(int(seed))
    model = TrajectoryReliabilityHead(x.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final = {}
    model.train()
    for _ in range(int(model_config["epochs"])):
        logits = model(x, qmean)
        prediction = torch.sigmoid(logits)
        regression = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=float(model_config["huber_beta"])
        )
        unsafe_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, unsafe)
        pair_delta = (prediction[left] - prediction[right]) * signs
        ranking = torch.nn.functional.softplus(
            -pair_delta / float(model_config["ranking_temperature"])
        ).mean()
        residual = (logits - torch.logit(qmean.clamp(1e-4, 1.0 - 1e-4))).square().mean()
        loss = (
            float(model_config["regression_weight"]) * regression
            + float(model_config["unsafe_weight"]) * unsafe_loss
            + float(model_config["ranking_weight"]) * ranking
            + float(model_config["residual_regularization_weight"]) * residual
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "total_loss": float(loss.detach().cpu()),
            "regression_loss": float(regression.detach().cpu()),
            "unsafe_loss": float(unsafe_loss.detach().cpu()),
            "ranking_loss": float(ranking.detach().cpu()),
            "residual_regularization": float(residual.detach().cpu()),
        }
    model.eval()
    final.update(
        train_row_count=int(len(values)),
        pair_count=int(len(left_np)),
    )
    return model, mean, scale, final


def score_head(
    model: TrajectoryReliabilityHead,
    arrays: Mapping[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = feature_matrix(arrays)
    qmean = np.asarray(arrays["qmean"], dtype=np.float32)
    with torch.inference_mode():
        logits = model(
            torch.from_numpy((values - mean) / scale).cuda(),
            torch.from_numpy(qmean).cuda(),
        )
        return torch.sigmoid(logits).float().cpu().numpy()


def train_lattice_adapter(
    arrays: Mapping[str, np.ndarray], model_config: Mapping[str, Any], seed: int
) -> tuple[LatticeResidualAdapter, dict[str, Any]]:
    target_np = np.asarray(arrays["target_cost"], dtype=np.float32)
    target = torch.from_numpy(target_np).cuda()
    qmean = torch.from_numpy(np.asarray(arrays["qmean"], dtype=np.float32)).cuda()
    actions = torch.from_numpy(np.asarray(arrays["action_index"], dtype=np.int64)).cuda()
    cases_np = np.asarray(arrays["case_index"], dtype=np.int64)
    cases = torch.from_numpy(cases_np).cuda()
    left_np, right_np, signs_np = _ranking_pairs(
        target_np, cases_np, float(model_config["pairwise_minimum_target_gap"])
    )
    left = torch.from_numpy(left_np).cuda()
    right = torch.from_numpy(right_np).cuda()
    signs = torch.from_numpy(signs_np).cuda()
    torch.manual_seed(int(seed))
    model = LatticeResidualAdapter(
        int(model_config["action_count"]),
        float(model_config["maximum_residual_cost"]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final = {}
    model.train()
    for _ in range(int(model_config["epochs"])):
        prediction = model(qmean, actions, cases)
        regression = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=float(model_config["huber_beta"])
        )
        pair_delta = (prediction[left] - prediction[right]) * signs
        ranking = torch.nn.functional.softplus(
            -pair_delta / float(model_config["ranking_temperature"])
        ).mean()
        residual_regularization = model.action_bias.square().mean()
        loss = (
            float(model_config["regression_weight"]) * regression
            + float(model_config["ranking_weight"]) * ranking
            + float(model_config["residual_regularization_weight"])
            * residual_regularization
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "total_loss": float(loss.detach().cpu()),
            "regression_loss": float(regression.detach().cpu()),
            "ranking_loss": float(ranking.detach().cpu()),
            "residual_regularization": float(residual_regularization.detach().cpu()),
        }
    model.eval()
    with torch.inference_mode():
        learned_bias = (
            model.maximum_residual_cost * torch.tanh(model.action_bias)
        ).cpu().numpy()
    final.update(
        train_row_count=int(len(target_np)),
        pair_count=int(len(left_np)),
        learned_action_bias=[float(v) for v in learned_bias],
        maximum_residual_cost=float(model.maximum_residual_cost),
    )
    return model, final


def score_lattice_adapter(
    model: LatticeResidualAdapter, arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    with torch.inference_mode():
        return model(
            torch.from_numpy(np.asarray(arrays["qmean"], dtype=np.float32)).cuda(),
            torch.from_numpy(np.asarray(arrays["action_index"], dtype=np.int64)).cuda(),
            torch.from_numpy(np.asarray(arrays["case_index"], dtype=np.int64)).cuda(),
        ).float().cpu().numpy()
