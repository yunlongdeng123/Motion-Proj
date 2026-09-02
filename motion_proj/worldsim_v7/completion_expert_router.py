"""Two-stage routing between frozen completion experts for WorldSim V7."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v7.completion_responsibility import COMPLETION_STATES
from motion_proj.worldsim_v7.selective_validity_hazard import (
    HAZARD_FEATURE_NAMES,
    VALIDITY_FEATURE_NAMES,
)


P17R_SCORE_FEATURE_NAMES = (
    "p17r_completion_fraction",
    "p17r_score_mean",
    "p17r_score_standard_deviation",
    "p17r_score_minimum",
    "p17r_score_maximum",
)
ROUTER_FEATURE_NAMES = (
    tuple(f"validity::{name}" for name in VALIDITY_FEATURE_NAMES)
    + tuple(f"hazard::{name}" for name in HAZARD_FEATURE_NAMES)
    + P17R_SCORE_FEATURE_NAMES
)
EXPERT_ACTIONS = ("always_complete", "p17r")
ALWAYS_COMPLETE, P17R = range(2)


class TwoExpertRouter(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(ROUTER_FEATURE_NAMES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(EXPERT_ACTIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def router_features(
    actor_row: Mapping[str, Any],
    p17r_scores: np.ndarray,
    p17r_threshold: float,
) -> np.ndarray:
    scores = np.asarray(p17r_scores, dtype=np.float32)
    if len(scores):
        score_features = (
            float(np.mean(scores >= float(p17r_threshold))),
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.min(scores)),
            float(np.max(scores)),
        )
    else:
        score_features = (0.0, 0.0, 0.0, 0.0, 0.0)
    values = (
        [float(actor_row["runtime_features"][name]) for name in VALIDITY_FEATURE_NAMES]
        + [float(actor_row["hazard_features"][name]) for name in HAZARD_FEATURE_NAMES]
        + list(score_features)
    )
    features = np.asarray(values, dtype=np.float32)
    if features.shape != (len(ROUTER_FEATURE_NAMES),):
        raise ValueError(f"unexpected router feature shape: {features.shape}")
    return features


def p17r_dominance_label(p17r_row: Mapping[str, Any]) -> int:
    baseline_chamfer = float(p17r_row["baseline_chamfer_m"])
    p17r_chamfer = float(p17r_row["p16_chamfer_m"])
    baseline_early = int(p17r_row["baseline_attribution"]["new_early_count"])
    p17r_early = int(p17r_row["p16_attribution"]["new_early_count"])
    both_no_worse = p17r_chamfer <= baseline_chamfer and p17r_early <= baseline_early
    one_strict = p17r_chamfer < baseline_chamfer or p17r_early < baseline_early
    return P17R if both_no_worse and one_strict else ALWAYS_COMPLETE


def route_actor_row(
    actor_row: Mapping[str, Any],
    p17r_row: Mapping[str, Any],
    action: int,
    action_probabilities: np.ndarray,
) -> dict[str, Any]:
    if action not in (ALWAYS_COMPLETE, P17R):
        raise ValueError(f"unknown expert action: {action}")
    routed = dict(p17r_row)
    routed["expert_action"] = EXPERT_ACTIONS[action]
    routed["expert_action_probabilities"] = {
        name: float(action_probabilities[index])
        for index, name in enumerate(EXPERT_ACTIONS)
    }
    if action == P17R:
        return routed
    routed["predicted_state_counts"] = {
        name: int(routed["candidate_count"]) if name == "OCCUPIED" else 0
        for name in COMPLETION_STATES
    }
    routed["mean_unknown_score"] = 0.0
    routed["p16_chamfer_m"] = float(routed["baseline_chamfer_m"])
    routed["p16_composite_gain_m"] = float(routed["baseline_composite_gain_m"])
    routed["p16_quality"] = dict(actor_row["after"])
    routed["p16_attribution"] = dict(routed["baseline_attribution"])
    return routed


def predict_actions(
    model: TwoExpertRouter,
    standardizer: Any,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    values = torch.as_tensor(
        standardizer.transform(np.asarray(features, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(values), dim=1)
    return (
        probabilities.argmax(dim=1).cpu().numpy().astype(np.int64),
        probabilities.cpu().numpy().astype(np.float32),
    )


def routing_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(EXPERT_ACTIONS), len(EXPERT_ACTIONS)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return {
        "actor_count": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)) if len(labels) else 0.0,
        "label_counts": {
            name: int(np.count_nonzero(labels == index))
            for index, name in enumerate(EXPERT_ACTIONS)
        },
        "prediction_counts": {
            name: int(np.count_nonzero(predictions == index))
            for index, name in enumerate(EXPERT_ACTIONS)
        },
        "confusion_true_rows_predicted_columns": confusion.tolist(),
    }
