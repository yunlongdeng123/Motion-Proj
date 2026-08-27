"""Low-capacity continuous-context admission model for WorldSim V6.5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CoverageMLP(nn.Module):
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
class CoverageFit:
    model: CoverageMLP
    mean: np.ndarray
    scale: np.ndarray
    minimum_coverage: float
    maximum_coverage: float
    epoch_losses: list[float]


def fit_coverage_model(
    features: np.ndarray,
    target_coverage: np.ndarray,
    *,
    hidden_dimensions: tuple[int, int],
    minimum_coverage: float,
    maximum_coverage: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> CoverageFit:
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(target_coverage, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    values = ((values - mean) / scale).astype(np.float32)
    normalized_target = np.clip(
        (targets - minimum_coverage) / (maximum_coverage - minimum_coverage), 0.0, 1.0
    ).astype(np.float32)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    x = torch.from_numpy(values).to(device)
    y = torch.from_numpy(normalized_target).to(device)
    model = CoverageMLP(values.shape[1], hidden_dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.sigmoid(model(x))
        loss = F.smooth_l1_loss(prediction, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return CoverageFit(
        model=model,
        mean=mean,
        scale=scale,
        minimum_coverage=float(minimum_coverage),
        maximum_coverage=float(maximum_coverage),
        epoch_losses=losses,
    )


def predict_coverage(fit: CoverageFit, features: np.ndarray) -> np.ndarray:
    values = ((np.asarray(features, dtype=np.float32) - fit.mean) / fit.scale).astype(np.float32)
    fit.model.eval()
    with torch.inference_mode():
        normalized = torch.sigmoid(fit.model(torch.from_numpy(values).cuda())).cpu().numpy()
    return (
        fit.minimum_coverage
        + (fit.maximum_coverage - fit.minimum_coverage) * normalized
    ).astype(np.float32)
