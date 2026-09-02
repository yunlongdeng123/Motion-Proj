"""Source-trained three-state responsibility for V7 completion candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v7.av2_four_action_compiler import (
    COMPLETION_FEATURE_NAMES,
    _nearest,
    _quality,
    _voxel_unique,
)
from motion_proj.worldsim_v7.visible_failure_attribution import (
    _surface_contradiction,
    attribute_actor,
)


COMPLETION_STATES = ("FREE", "OCCUPIED", "UNKNOWN")
FREE, OCCUPIED, UNKNOWN = range(3)


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        mean = np.mean(features, axis=0, dtype=np.float64)
        scale = np.std(features, axis=0, dtype=np.float64)
        return cls(mean.astype(np.float32), np.maximum(scale, 1.0e-6).astype(np.float32))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FeatureStandardizer":
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["scale"], dtype=np.float32),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray((features - self.mean) / self.scale, dtype=np.float32)

    def payload(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class CompletionResponsibilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(COMPLETION_STATES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def completion_labels(
    diagnostics: Mapping[str, Any],
    attribution_config: Mapping[str, Any],
    target_support_distance_m: float,
    device: torch.device,
) -> np.ndarray:
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32)
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    contradicted = _surface_contradiction(
        candidates,
        np.asarray(diagnostics["target"], dtype=np.float32),
        np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32),
        attribution_config,
        device,
    )
    target_distance, _ = _nearest(
        candidates,
        np.asarray(diagnostics["target"], dtype=np.float32),
        device,
        int(attribution_config["distance_chunk_size"]),
    )
    labels = np.full(len(candidates), UNKNOWN, dtype=np.int64)
    labels[(target_distance <= float(target_support_distance_m)) & ~contradicted] = OCCUPIED
    labels[contradicted] = FREE
    return labels


def predict_completion(
    model: CompletionResponsibilityMLP,
    standardizer: FeatureStandardizer,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(features):
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, len(COMPLETION_STATES)), dtype=np.float32),
        )
    tensor = torch.as_tensor(
        standardizer.transform(features), dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)
    return (
        probabilities.argmax(dim=1).cpu().numpy().astype(np.int64),
        probabilities.cpu().numpy().astype(np.float32),
    )


def apply_completion_policy(
    actor_row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    predicted_states: np.ndarray,
    probabilities: np.ndarray,
    compiler_config: Mapping[str, Any],
    attribution_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32)
    selected = predicted_states == OCCUPIED
    completed = candidates[selected]
    layers = [
        np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
        np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
        completed.reshape(-1, 3),
    ]
    compiled = _voxel_unique(
        np.concatenate(layers, axis=0),
        float(compiler_config["compiler_geometry"]["output_voxel_size_m"]),
    )
    target = np.asarray(diagnostics["target"], dtype=np.float32)
    quality = _quality(compiled, target, compiler_config, device)
    p16_diagnostics = dict(diagnostics)
    p16_diagnostics["compiled"] = compiled
    p16_diagnostics["completed"] = completed
    baseline_attribution = attribute_actor(
        actor_row,
        diagnostics,
        attribution_config,
        float(compiler_config["compiler_geometry"]["output_voxel_size_m"]),
        device,
    )
    p16_attribution = attribute_actor(
        actor_row,
        p16_diagnostics,
        attribution_config,
        float(compiler_config["compiler_geometry"]["output_voxel_size_m"]),
        device,
    )
    query_chamfer = float(actor_row["query_only"]["symmetric_chamfer_m"])
    baseline_chamfer = float(actor_row["after"]["symmetric_chamfer_m"])
    state_counts = {
        name: int(np.count_nonzero(predicted_states == index))
        for index, name in enumerate(COMPLETION_STATES)
    }
    return {
        "track_id": str(actor_row["track_id"]),
        "category": str(actor_row["category"]),
        "hazardous": bool(actor_row["hazardous"]),
        "candidate_count": int(len(candidates)),
        "predicted_state_counts": state_counts,
        "mean_unknown_score": float(np.mean(probabilities[:, UNKNOWN])) if len(probabilities) else 0.0,
        "query_chamfer_m": query_chamfer,
        "baseline_chamfer_m": baseline_chamfer,
        "p16_chamfer_m": float(quality["symmetric_chamfer_m"]),
        "baseline_composite_gain_m": query_chamfer - baseline_chamfer,
        "p16_composite_gain_m": query_chamfer - float(quality["symmetric_chamfer_m"]),
        "p16_quality": quality,
        "baseline_attribution": baseline_attribution,
        "p16_attribution": p16_attribution,
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(COMPLETION_STATES), len(COMPLETION_STATES)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class = {}
    f1_values = []
    for index, name in enumerate(COMPLETION_STATES):
        true_positive = int(confusion[index, index])
        support = int(np.sum(confusion[index]))
        predicted = int(np.sum(confusion[:, index]))
        precision = true_positive / max(predicted, 1)
        recall = true_positive / max(support, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {
        "candidate_count": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)) if len(labels) else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "per_class": per_class,
    }


def summarize_actor_policy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("completion policy has no Actor rows")

    def summarize_variant(name: str) -> dict[str, Any]:
        attribution_key = f"{name}_attribution"
        chamfer_key = f"{name}_chamfer_m"
        gain_key = f"{name}_composite_gain_m"
        target_rays = sum(int(row[attribution_key]["target_ray_count"]) for row in rows)
        new_early = sum(int(row[attribution_key]["new_early_count"]) for row in rows)
        new_hits = sum(int(row[attribution_key]["new_hit_count"]) for row in rows)
        return {
            "mean_chamfer_m": float(np.mean([row[chamfer_key] for row in rows])),
            "mean_composite_gain_m": float(np.mean([row[gain_key] for row in rows])),
            "target_ray_count": int(target_rays),
            "new_early_count": int(new_early),
            "new_early_rate": float(new_early / max(target_rays, 1)),
            "new_hit_count": int(new_hits),
            "new_hit_to_new_early": float(new_hits / max(new_early, 1)),
        }

    def stratum(hazardous: bool, name: str) -> dict[str, Any]:
        selected = [row for row in rows if bool(row["hazardous"]) is hazardous]
        attribution_key = f"{name}_attribution"
        rays = sum(int(row[attribution_key]["target_ray_count"]) for row in selected)
        early = sum(int(row[attribution_key]["new_early_count"]) for row in selected)
        hits = sum(int(row[attribution_key]["new_hit_count"]) for row in selected)
        return {
            "actor_count": len(selected),
            "target_ray_count": int(rays),
            "new_early_count": int(early),
            "new_early_rate": float(early / max(rays, 1)),
            "new_hit_count": int(hits),
            "mean_chamfer_m": float(np.mean([row[f"{name}_chamfer_m"] for row in selected])) if selected else 0.0,
        }

    state_counts = {
        state: sum(int(row["predicted_state_counts"][state]) for row in rows)
        for state in COMPLETION_STATES
    }
    return {
        "actor_count": len(rows),
        "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in rows)),
        "candidate_count": int(sum(int(row["candidate_count"]) for row in rows)),
        "predicted_state_counts": state_counts,
        "completion_coverage": float(
            state_counts["OCCUPIED"] / max(sum(state_counts.values()), 1)
        ),
        "mean_unknown_score": float(np.mean([row["mean_unknown_score"] for row in rows])),
        "mean_query_chamfer_m": float(np.mean([row["query_chamfer_m"] for row in rows])),
        "baseline": summarize_variant("baseline"),
        "p16": summarize_variant("p16"),
        "hazard": {
            "baseline": stratum(True, "baseline"),
            "p16": stratum(True, "p16"),
        },
        "clear": {
            "baseline": stratum(False, "baseline"),
            "p16": stratum(False, "p16"),
        },
    }
