"""NuScenes-only selective repairability and validity--hazard factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn


VALIDITY_FEATURE_NAMES = (
    "log_query_points",
    "log_canonical_surfels",
    "query_surface_mean_m",
    "query_surface_p95_m",
    "clean_keep_fraction",
    "unknown_query_fraction",
    "temporal_support_mean",
    "temporal_support_p10",
    "view_support_mean",
    "completion_candidate_fraction",
    "compiled_to_query_ratio",
    "sensor_range_m",
    "observation_frame_count",
)

HAZARD_FEATURE_NAMES = (
    "minimum_ttc_s",
    "minimum_clearance_m",
    "closing_speed_mps",
    "hard_brake_score",
    "crossing_probability",
)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def payload(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class SharedTwoHead(nn.Module):
    def __init__(self, validity_dim: int, hazard_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(validity_dim + hazard_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.repair_head = nn.Linear(hidden_dim // 2, 1)
        self.hazard_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self, validity: torch.Tensor, hazard: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(torch.cat([validity, hazard], dim=-1))
        return self.repair_head(hidden).squeeze(-1), self.hazard_head(hidden).squeeze(-1)


class FactorizedTwoHead(nn.Module):
    def __init__(self, validity_dim: int, hazard_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.repair_head = SmallMLP(validity_dim, hidden_dim)
        self.hazard_head = SmallMLP(hazard_dim, max(hidden_dim // 2, 8))

    def forward(
        self, validity: torch.Tensor, hazard: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.repair_head(validity), self.hazard_head(hazard)


def rows_to_arrays(rows: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    validity = np.asarray(
        [[float(row["runtime_features"][key]) for key in VALIDITY_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    hazard = np.asarray(
        [[float(row["hazard_features"][key]) for key in HAZARD_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    return {
        "validity": validity,
        "hazard": hazard,
        "repairable": np.asarray(
            [bool(row["target_supported_repairable"]) for row in rows], dtype=np.float32
        ),
        "hazardous": np.asarray([bool(row["hazardous"]) for row in rows], dtype=np.float32),
        "query_chamfer": np.asarray(
            [float(row["query_only"]["symmetric_chamfer_m"]) for row in rows],
            dtype=np.float32,
        ),
        "compiled_chamfer": np.asarray(
            [float(row["after"]["symmetric_chamfer_m"]) for row in rows],
            dtype=np.float32,
        ),
    }


def _positive_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum().clamp_min(1.0)
    negatives = (len(labels) - labels.sum()).clamp_min(1.0)
    return (negatives / positives).detach()


def train_model(
    model: nn.Module,
    train: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    validity = torch.as_tensor(train["validity"], dtype=torch.float32, device=device)
    hazard = torch.as_tensor(train["hazard"], dtype=torch.float32, device=device)
    repairable = torch.as_tensor(train["repairable"], dtype=torch.float32, device=device)
    hazardous = torch.as_tensor(train["hazardous"], dtype=torch.float32, device=device)
    repair_loss = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(repairable))
    hazard_loss = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(hazardous))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    batch_size = int(config["batch_size"])
    history = []
    model.train()
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(len(validity), generator=generator).to(device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            repair_logit, hazard_logit = model(validity[index], hazard[index])
            loss_repair = repair_loss(repair_logit, repairable[index])
            loss_hazard = hazard_loss(hazard_logit, hazardous[index])
            loss = loss_repair + float(config["hazard_loss_weight"]) * loss_hazard
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        if epoch in {0, int(config["epochs"]) - 1} or (epoch + 1) % 20 == 0:
            history.append({"epoch": epoch + 1, "loss": epoch_loss / max(batches, 1)})
    return history


def predict(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        repair, hazard = model(
            torch.as_tensor(arrays["validity"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["hazard"], dtype=torch.float32, device=device),
        )
    return torch.sigmoid(repair).cpu().numpy(), torch.sigmoid(hazard).cpu().numpy()


def calibrate_crc_threshold(
    repair_scores: np.ndarray,
    repairable: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """CRC for monotone false-repair loss; guarantee is nuScenes-only."""
    failures = 1.0 - repairable
    candidates = np.concatenate(
        [np.unique(repair_scores), np.asarray([1.0000001], dtype=np.float32)]
    )
    for threshold in np.sort(candidates):
        selected = repair_scores >= threshold
        adjusted_risk = (float(np.sum(failures[selected])) + 1.0) / (len(failures) + 1.0)
        if adjusted_risk <= alpha:
            return {
                "threshold": float(threshold),
                "crc_adjusted_false_repair_risk": adjusted_risk,
                "coverage": float(np.mean(selected)),
                "selective_failure_rate": float(
                    np.mean(failures[selected]) if np.any(selected) else 0.0
                ),
            }
    raise RuntimeError("CRC threshold search failed to find the all-abstain candidate")


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def evaluate_scores(
    arrays: Mapping[str, np.ndarray],
    repair_scores: np.ndarray,
    hazard_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    repairable = arrays["repairable"]
    failures = 1.0 - repairable
    selected = repair_scores >= threshold
    hazardous = arrays["hazardous"].astype(bool)
    query = arrays["query_chamfer"]
    compiled = arrays["compiled_chamfer"]
    selective_surface = np.where(selected, compiled, query)
    return {
        "actor_count": len(repairable),
        "repairable_count": int(np.sum(repairable)),
        "hazard_actor_count": int(np.sum(hazardous)),
        "repairability": _binary_metrics(repairable, repair_scores),
        "hazard": _binary_metrics(arrays["hazardous"], hazard_scores),
        "coverage": float(np.mean(selected)),
        "false_repair_rate": float(np.mean(selected * failures)),
        "selective_failure_rate": float(
            np.mean(failures[selected]) if np.any(selected) else 0.0
        ),
        "hazard_actor_coverage": float(
            np.mean(selected[hazardous]) if np.any(hazardous) else 0.0
        ),
        "hazard_selected_failure_rate": float(
            np.mean(failures[selected & hazardous]) if np.any(selected & hazardous) else 0.0
        ),
        "mean_query_chamfer_m": float(np.mean(query)),
        "mean_always_repair_chamfer_m": float(np.mean(compiled)),
        "mean_selective_surface_chamfer_m": float(np.mean(selective_surface)),
        "mean_selected_gain_m": float(
            np.mean(query[selected] - compiled[selected]) if np.any(selected) else 0.0
        ),
        "actor_retention": 1.0,
        "hazard_label_retention": 1.0,
    }


def paired_input_leakage(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    original_repair, original_hazard = predict(model, arrays, device)
    paired = dict(arrays)
    paired["hazard"] = arrays["hazard"][::-1].copy()
    hazard_swap_repair, _ = predict(model, paired, device)
    paired = dict(arrays)
    paired["validity"] = arrays["validity"][::-1].copy()
    _, validity_swap_hazard = predict(model, paired, device)
    return {
        "hazard_swap_repair_score_shift": float(
            np.mean(np.abs(original_repair - hazard_swap_repair))
        ),
        "validity_swap_hazard_score_shift": float(
            np.mean(np.abs(original_hazard - validity_swap_hazard))
        ),
    }
