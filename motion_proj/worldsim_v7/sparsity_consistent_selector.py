"""Source-only sparsity augmentation and score consistency for V7 P6-C."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v7.selective_validity_hazard import (
    SmallMLP,
    VALIDITY_FEATURE_NAMES,
    rows_to_arrays,
)


_FEATURE_INDEX = {name: index for index, name in enumerate(VALIDITY_FEATURE_NAMES)}


def opportunity_view_arrays(
    rows: list[Mapping[str, Any]], scale: float
) -> dict[str, np.ndarray]:
    arrays = rows_to_arrays(rows)
    validity = arrays["validity"].copy()
    factor = float(scale)
    surfel_index = _FEATURE_INDEX["log_canonical_surfels"]
    surfels = np.maximum(np.expm1(validity[:, surfel_index]), 0.0)
    validity[:, surfel_index] = np.log1p(surfels * factor)
    for name in (
        "temporal_support_mean",
        "temporal_support_p10",
        "view_support_mean",
        "observation_frame_count",
    ):
        validity[:, _FEATURE_INDEX[name]] *= factor
    arrays["validity"] = validity.astype(np.float32)
    return arrays


def fit_source_view_standardizer_values(
    rows: list[Mapping[str, Any]], scales: Sequence[float]
) -> np.ndarray:
    views = [rows_to_arrays(rows)["validity"]]
    views.extend(opportunity_view_arrays(rows, scale)["validity"] for scale in scales)
    return np.concatenate(views, axis=0).astype(np.float32)


def train_sparsity_consistent_model(
    model: SmallMLP,
    original: Mapping[str, np.ndarray],
    augmented: list[Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
    consistency_weight: float,
    device: torch.device,
) -> list[dict[str, float]]:
    original_validity = torch.as_tensor(
        original["validity"], dtype=torch.float32, device=device
    )
    augmented_validity = [
        torch.as_tensor(view["validity"], dtype=torch.float32, device=device)
        for view in augmented
    ]
    repairable = torch.as_tensor(
        original["repairable"], dtype=torch.float32, device=device
    )
    positives = repairable.sum().clamp_min(1.0)
    negatives = (len(repairable) - repairable.sum()).clamp_min(1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).detach())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    batch_size = int(config["batch_size"])
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(len(original_validity), generator=generator).to(device)
        total_loss = 0.0
        total_supervised = 0.0
        total_consistency = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            original_logits = model(original_validity[index])
            augmented_logits = [model(view[index]) for view in augmented_validity]
            supervised = torch.stack(
                [loss_fn(original_logits, repairable[index])]
                + [loss_fn(logits, repairable[index]) for logits in augmented_logits]
            ).mean()
            original_probability = torch.sigmoid(original_logits)
            consistency = torch.stack(
                [
                    torch.mean(
                        (torch.sigmoid(logits) - original_probability) ** 2
                    )
                    for logits in augmented_logits
                ]
            ).mean()
            loss = supervised + float(consistency_weight) * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_supervised += float(supervised.detach().cpu())
            total_consistency += float(consistency.detach().cpu())
            batches += 1
        if epoch in {0, int(config["epochs"]) - 1} or (epoch + 1) % 20 == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": total_loss / max(batches, 1),
                    "supervised_loss": total_supervised / max(batches, 1),
                    "consistency_loss": total_consistency / max(batches, 1),
                }
            )
    return history


def predict_validity(
    model: SmallMLP, arrays: Mapping[str, np.ndarray], device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(
            torch.as_tensor(arrays["validity"], dtype=torch.float32, device=device)
        )
    return torch.sigmoid(logits).cpu().numpy()


def mean_intervention_score_shift(
    model: SmallMLP,
    original: Mapping[str, np.ndarray],
    augmented: list[Mapping[str, np.ndarray]],
    device: torch.device,
) -> float:
    reference = predict_validity(model, original, device)
    return float(
        np.mean(
            [
                np.mean(np.abs(predict_validity(model, view, device) - reference))
                for view in augmented
            ]
        )
    )
