"""Low-capacity Actor-token outcome models for the V6.5 train-only probe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


class ActorOutcomeMLP(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimensions: tuple[int, int]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.GELU(),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]),
            nn.GELU(),
            nn.Linear(hidden_dimensions[1], 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


@dataclass
class ActorOutcomeFit:
    model: ActorOutcomeMLP
    mean: np.ndarray
    scale: np.ndarray
    epoch_losses: list[float]


def fit_actor_outcome(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_dimensions: tuple[int, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> ActorOutcomeFit:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((features - mean) / scale).astype(np.float32)
    device = torch.device("cuda")
    values = torch.from_numpy(standardized).to(device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    model = ActorOutcomeMLP(features.shape[1], hidden_dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = float(np.count_nonzero(labels))
    negatives = float(labels.shape[0] - positives)
    positive_weight = torch.tensor(negatives / max(1.0, positives), device=device)
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(target.shape[0], device=device)
        total, steps = 0.0, 0
        model.train()
        for offset in range(0, target.shape[0], batch_size):
            chosen = permutation[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(values[chosen])
            loss = F.binary_cross_entropy_with_logits(
                logits, target[chosen], pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        losses.append(total / max(1, steps))
    return ActorOutcomeFit(model=model, mean=mean, scale=scale, epoch_losses=losses)


def score_actor_outcome(fit: ActorOutcomeFit, features: np.ndarray) -> np.ndarray:
    standardized = ((features - fit.mean) / fit.scale).astype(np.float32)
    outputs = []
    fit.model.eval()
    with torch.inference_mode():
        for offset in range(0, standardized.shape[0], 131072):
            batch = torch.from_numpy(standardized[offset : offset + 131072]).cuda()
            outputs.append(torch.sigmoid(fit.model(batch)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def fit_actor_cost(
    features: np.ndarray,
    costs: np.ndarray,
    *,
    hidden_dimensions: tuple[int, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> ActorOutcomeFit:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((features - mean) / scale).astype(np.float32)
    device = torch.device("cuda")
    values = torch.from_numpy(standardized).to(device)
    target = torch.as_tensor(costs, dtype=torch.float32, device=device)
    model = ActorOutcomeMLP(features.shape[1], hidden_dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(target.shape[0], device=device)
        total, steps = 0.0, 0
        model.train()
        for offset in range(0, target.shape[0], batch_size):
            chosen = permutation[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = torch.sigmoid(model(values[chosen]))
            loss = F.smooth_l1_loss(prediction, target[chosen])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        losses.append(total / max(1, steps))
    return ActorOutcomeFit(model=model, mean=mean, scale=scale, epoch_losses=losses)


def continuous_cost_metrics(costs: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    cost_order = np.argsort(np.argsort(costs, kind="stable"), kind="stable").astype(np.float64)
    prediction_order = np.argsort(np.argsort(predictions, kind="stable"), kind="stable").astype(np.float64)
    spearman = float(np.corrcoef(cost_order, prediction_order)[0, 1])
    residual = predictions - costs
    return {
        "spearman": spearman,
        "mse": float(np.mean(np.square(residual))),
        "mae": float(np.mean(np.abs(residual))),
    }


def selected_cost_metrics(
    costs: np.ndarray,
    predictions: np.ndarray,
    scene_index: np.ndarray,
    *,
    coverage: float,
) -> dict[str, object]:
    selected_count = max(1, int(np.floor(float(coverage) * predictions.shape[0])))
    selected = np.argsort(predictions, kind="stable")[:selected_count]
    scene_rows = []
    for scene in np.unique(scene_index):
        members = np.flatnonzero(scene_index == scene)
        local_count = max(1, int(np.floor(float(coverage) * members.shape[0])))
        local = members[np.argsort(predictions[members], kind="stable")[:local_count]]
        scene_rows.append({
            "scene_index": int(scene),
            "eligible_count": int(members.shape[0]),
            "selected_count": int(local_count),
            "selected_mean_cost": float(np.mean(costs[local])),
        })
    return {
        "eligible_count": int(predictions.shape[0]),
        "selected_count": int(selected_count),
        "realized_coverage": float(selected_count / predictions.shape[0]),
        "selected_mean_cost": float(np.mean(costs[selected])),
        "scene_rows": scene_rows,
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    auroc = float(roc_auc_score(labels, scores)) if np.unique(labels).shape[0] == 2 else float("nan")
    return {
        "auroc": auroc,
        "auprc": float(average_precision_score(labels, scores)),
    }


def selected_outcome_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    scene_index: np.ndarray,
    *,
    coverage: float,
) -> dict[str, object]:
    selected_count = max(1, int(np.floor(float(coverage) * scores.shape[0])))
    selected = np.argsort(scores, kind="stable")[:selected_count]
    scene_rows = []
    for scene in np.unique(scene_index):
        members = np.flatnonzero(scene_index == scene)
        local_count = max(1, int(np.floor(float(coverage) * members.shape[0])))
        local = members[np.argsort(scores[members], kind="stable")[:local_count]]
        conflicts = int(np.count_nonzero(labels[local]))
        scene_rows.append(
            {
                "scene_index": int(scene),
                "eligible_count": int(members.shape[0]),
                "selected_count": int(local_count),
                "outcome_count": conflicts,
                "selected_outcome_rate": float(conflicts / local_count),
            }
        )
    conflicts = int(np.count_nonzero(labels[selected]))
    return {
        "eligible_count": int(scores.shape[0]),
        "selected_count": int(selected_count),
        "realized_coverage": float(selected_count / scores.shape[0]),
        "outcome_count": conflicts,
        "selected_outcome_rate": float(conflicts / selected_count),
        "scene_rows": scene_rows,
    }
