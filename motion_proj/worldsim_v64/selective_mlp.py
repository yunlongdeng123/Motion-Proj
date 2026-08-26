"""Fixed full-native-feature MLP for hidden-FREE risk ranking."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _RiskNetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimensions: tuple[int, int], dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]),
            nn.GELU(),
            nn.Linear(hidden_dimensions[1], 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(1)


class NativeBoundarySelectiveMLP:
    """Small fixed MLP trained directly on 17 logits plus 256 BEV features."""

    def __init__(
        self,
        *,
        hidden_dimensions: tuple[int, int],
        dropout: float,
        focal_gamma: float,
        focal_alpha: float,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.hidden_dimensions = tuple(int(value) for value in hidden_dimensions)
        self.dropout = float(dropout)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.model: _RiskNetwork | None = None
        self.fit_summary: dict[str, object] = {}

    def fit(self, features: np.ndarray, hidden_free: np.ndarray) -> "NativeBoundarySelectiveMLP":
        started = time.monotonic()
        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(hidden_free, dtype=np.float32)
        self.mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        values = ((values - self.mean) / self.scale).astype(np.float32)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        device = torch.device("cuda")
        self.model = _RiskNetwork(values.shape[1], self.hidden_dimensions, self.dropout).to(device)
        x = torch.from_numpy(values).to(device)
        y = torch.from_numpy(labels).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler()
        losses = []
        for _ in range(self.epochs):
            self.model.train()
            permutation = torch.randperm(x.shape[0], device=device)
            epoch_loss = 0.0
            step_count = 0
            for offset in range(0, x.shape[0], self.batch_size):
                indices = permutation[offset : offset + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    logits = self.model(x[indices])
                    targets = y[indices]
                    probability = torch.sigmoid(logits)
                    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
                    alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
                    loss = (
                        alpha_t
                        * (1.0 - p_t).pow(self.focal_gamma)
                        * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                    ).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.detach())
                step_count += 1
            losses.append(epoch_loss / step_count)
        self.model.eval()
        self.fit_summary = {
            "point_count": int(values.shape[0]),
            "feature_dimension": int(values.shape[1]),
            "positive_count": int(labels.sum()),
            "positive_prevalence": float(labels.mean()),
            "epoch_losses": losses,
            "wall_seconds": time.monotonic() - started,
            "device": str(device),
        }
        del x, y
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return self

    def score(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        del logits
        if self.model is None or self.mean is None or self.scale is None:
            raise RuntimeError("selective MLP is not fitted")
        values = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        outputs = []
        with torch.inference_mode():
            for offset in range(0, values.shape[0], 131072):
                batch = torch.from_numpy(values[offset : offset + 131072]).to(device)
                outputs.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)
