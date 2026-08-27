"""Low-capacity trajectory-conditioned residual over the frozen V6.4 trunk."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


class TrajectoryResidualValidity(nn.Module):
    """Apply a trajectory-only FiLM residual to frozen 64D native features."""

    def __init__(self, native_dimension: int, trajectory_dimension: int) -> None:
        super().__init__()
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dimension, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.film = nn.Linear(16, native_dimension * 2)
        self.delta_head = nn.Sequential(
            nn.Linear(native_dimension + 16, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self, native_hidden: torch.Tensor, base_logit: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.trajectory_encoder(trajectory)
        gamma, beta = self.film(encoded).chunk(2, dim=1)
        interaction = native_hidden * torch.tanh(gamma) + beta
        delta = self.delta_head(torch.cat((interaction, encoded), dim=1)).squeeze(1)
        return base_logit + delta


class MonotoneTaskRiskResidual(nn.Module):
    """Add task risk near a trajectory without reducing frozen physical risk."""

    def __init__(self, native_dimension: int, trajectory_dimension: int) -> None:
        super().__init__()
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dimension, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.film = nn.Linear(16, native_dimension * 2)
        self.task_head = nn.Sequential(
            nn.Linear(native_dimension + 16, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.task_head[-1].weight)
        nn.init.constant_(self.task_head[-1].bias, -4.0)

    def forward(
        self,
        native_hidden: torch.Tensor,
        base_logit: torch.Tensor,
        trajectory: torch.Tensor,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.trajectory_encoder(trajectory)
        gamma, beta = self.film(encoded).chunk(2, dim=1)
        interaction = native_hidden * torch.tanh(gamma) + beta
        task_risk = F.softplus(
            self.task_head(torch.cat((interaction, encoded), dim=1)).squeeze(1)
        )
        return base_logit + relevance.clamp(0.0, 1.0) * task_risk


@dataclass
class FitResult:
    model: TrajectoryResidualValidity
    trajectory_mean: np.ndarray
    trajectory_scale: np.ndarray
    epoch_losses: list[float]


@dataclass
class MonotoneFitResult:
    model: MonotoneTaskRiskResidual
    trajectory_mean: np.ndarray
    trajectory_scale: np.ndarray
    epoch_losses: list[float]


def fit_trajectory_residual(
    native_hidden: np.ndarray,
    base_logit: np.ndarray,
    trajectory: np.ndarray,
    hidden_free: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    focal_gamma: float,
    focal_alpha: float,
    seed: int,
) -> FitResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    mean = trajectory.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = trajectory.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((trajectory - mean) / scale).astype(np.float32)
    hidden = torch.as_tensor(native_hidden, dtype=torch.float16, device=device)
    base = torch.as_tensor(base_logit, dtype=torch.float32, device=device)
    task = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    target = torch.as_tensor(hidden_free, dtype=torch.float32, device=device)
    model = TrajectoryResidualValidity(hidden.shape[1], task.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scaler = torch.cuda.amp.GradScaler()
    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        permutation = torch.randperm(target.shape[0], device=device)
        total = 0.0
        steps = 0
        for offset in range(0, target.shape[0], int(batch_size)):
            chosen = permutation[offset : offset + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(hidden[chosen], base[chosen], task[chosen])
                labels = target[chosen]
                probability = torch.sigmoid(logits)
                p_t = probability * labels + (1.0 - probability) * (1.0 - labels)
                alpha_t = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
                loss = (
                    alpha_t
                    * (1.0 - p_t).pow(focal_gamma)
                    * F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                ).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            steps += 1
        losses.append(total / max(1, steps))
    return FitResult(model=model, trajectory_mean=mean, trajectory_scale=scale, epoch_losses=losses)


def score_trajectory_residual(
    fit: FitResult,
    native_hidden: np.ndarray,
    base_logit: np.ndarray,
    trajectory: np.ndarray,
    *,
    batch_size: int = 131072,
) -> np.ndarray:
    device = torch.device("cuda")
    standardized = ((trajectory - fit.trajectory_mean) / fit.trajectory_scale).astype(np.float32)
    fit.model.eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, base_logit.shape[0], batch_size):
            stop = offset + batch_size
            hidden = torch.as_tensor(native_hidden[offset:stop], dtype=torch.float16, device=device)
            base = torch.as_tensor(base_logit[offset:stop], dtype=torch.float32, device=device)
            task = torch.as_tensor(standardized[offset:stop], dtype=torch.float32, device=device)
            with torch.cuda.amp.autocast():
                outputs.append(torch.sigmoid(fit.model(hidden, base, task)).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def fit_monotone_task_risk(
    native_hidden: np.ndarray,
    base_logit: np.ndarray,
    trajectory: np.ndarray,
    hidden_free: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    focal_gamma: float,
    focal_alpha: float,
    minimum_task_weight: float,
    seed: int,
) -> MonotoneFitResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    mean = trajectory.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = trajectory.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((trajectory - mean) / scale).astype(np.float32)
    hidden = torch.as_tensor(native_hidden, dtype=torch.float16, device=device)
    base = torch.as_tensor(base_logit, dtype=torch.float32, device=device)
    task = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    relevance = torch.as_tensor(trajectory[:, -1], dtype=torch.float32, device=device)
    target = torch.as_tensor(hidden_free, dtype=torch.float32, device=device)
    weights = float(minimum_task_weight) + (1.0 - float(minimum_task_weight)) * relevance
    model = MonotoneTaskRiskResidual(hidden.shape[1], task.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scaler = torch.cuda.amp.GradScaler()
    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        permutation = torch.randperm(target.shape[0], device=device)
        total = 0.0
        steps = 0
        for offset in range(0, target.shape[0], int(batch_size)):
            chosen = permutation[offset : offset + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(
                    hidden[chosen], base[chosen], task[chosen], relevance[chosen]
                )
                labels = target[chosen]
                probability = torch.sigmoid(logits)
                p_t = probability * labels + (1.0 - probability) * (1.0 - labels)
                alpha_t = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
                point_loss = (
                    alpha_t
                    * (1.0 - p_t).pow(focal_gamma)
                    * F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                )
                loss = (point_loss * weights[chosen]).sum() / weights[chosen].sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            steps += 1
        losses.append(total / max(1, steps))
    return MonotoneFitResult(model=model, trajectory_mean=mean, trajectory_scale=scale, epoch_losses=losses)


def score_monotone_task_risk(
    fit: MonotoneFitResult,
    native_hidden: np.ndarray,
    base_logit: np.ndarray,
    trajectory: np.ndarray,
    *,
    batch_size: int = 131072,
) -> np.ndarray:
    device = torch.device("cuda")
    standardized = ((trajectory - fit.trajectory_mean) / fit.trajectory_scale).astype(np.float32)
    fit.model.eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, base_logit.shape[0], batch_size):
            stop = offset + batch_size
            hidden = torch.as_tensor(native_hidden[offset:stop], dtype=torch.float16, device=device)
            base = torch.as_tensor(base_logit[offset:stop], dtype=torch.float32, device=device)
            task = torch.as_tensor(standardized[offset:stop], dtype=torch.float32, device=device)
            relevance = torch.as_tensor(
                trajectory[offset:stop, -1], dtype=torch.float32, device=device
            )
            with torch.cuda.amp.autocast():
                outputs.append(
                    torch.sigmoid(fit.model(hidden, base, task, relevance)).float().cpu().numpy()
                )
    return np.concatenate(outputs).astype(np.float32)


def ranking_metrics(hidden_free: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(hidden_free, dtype=np.uint8)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def fixed_opportunity_metrics(
    hidden_free: np.ndarray,
    route: np.ndarray,
    scores: np.ndarray,
    scene_index: np.ndarray,
    unit_index: np.ndarray,
    *,
    coverage: float,
    tail_fraction: float,
) -> dict[str, object]:
    unit_rows = []
    for scene in np.unique(scene_index):
        for unit in np.unique(unit_index[scene_index == scene]):
            mask = (scene_index == scene) & (unit_index == unit)
            local_scores = scores[mask]
            local_route = route[mask]
            local_hidden = hidden_free[mask]
            selected_count = max(1, int(math.floor(float(coverage) * local_scores.size)))
            selected = np.argsort(local_scores, kind="stable")[:selected_count]
            route_eligible = int(local_route.sum())
            route_conflicts = int(np.count_nonzero(local_route[selected] & local_hidden[selected]))
            unit_rows.append(
                {
                    "scene_index": int(scene),
                    "unit_index": int(unit),
                    "eligible_count": int(local_scores.size),
                    "selected_count": int(selected_count),
                    "route_eligible_count": route_eligible,
                    "route_conflict_count": route_conflicts,
                    "fixed_route_conflict_density": float(
                        route_conflicts / route_eligible if route_eligible else 0.0
                    ),
                }
            )
    route_eligible = sum(row["route_eligible_count"] for row in unit_rows)
    route_conflicts = sum(row["route_conflict_count"] for row in unit_rows)
    densities = np.asarray([row["fixed_route_conflict_density"] for row in unit_rows])
    tail_count = max(1, int(math.ceil(densities.size * float(tail_fraction))))
    scene_rows = []
    for scene in sorted({row["scene_index"] for row in unit_rows}):
        rows = [row for row in unit_rows if row["scene_index"] == scene]
        denominator = sum(row["route_eligible_count"] for row in rows)
        conflicts = sum(row["route_conflict_count"] for row in rows)
        scene_rows.append(
            {
                "scene_index": scene,
                "route_eligible_count": denominator,
                "route_conflict_count": conflicts,
                "fixed_route_conflict_density": float(conflicts / denominator if denominator else 0.0),
            }
        )
    return {
        "mean_realized_coverage": float(
            sum(row["selected_count"] for row in unit_rows)
            / sum(row["eligible_count"] for row in unit_rows)
        ),
        "route_eligible_count": route_eligible,
        "route_conflict_count": route_conflicts,
        "pooled_fixed_route_conflict_density": float(
            route_conflicts / route_eligible if route_eligible else 0.0
        ),
        "worst_tail_fixed_route_cvar": float(np.sort(densities)[::-1][:tail_count].mean()),
        "tail_count": tail_count,
        "scene_rows": scene_rows,
        "unit_rows": unit_rows,
    }
