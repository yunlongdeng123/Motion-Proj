"""Opportunity-normalized validity representation for the V7-F11 recovery."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from motion_proj.worldsim_v7.selective_validity_hazard import (
    SmallMLP,
    rows_to_arrays,
)


OPPORTUNITY_VALIDITY_FEATURE_NAMES = (
    "log_query_points",
    "log_canonical_surfels_per_observation",
    "query_surface_mean_m",
    "query_surface_p95_m",
    "clean_keep_fraction",
    "unknown_query_fraction",
    "temporal_support_fraction_mean",
    "temporal_support_fraction_p10",
    "view_support_fraction_mean",
    "completion_candidate_fraction",
    "compiled_to_query_ratio",
    "sensor_range_m",
)


def _opportunity_features(
    runtime: Mapping[str, Any], opportunity_scale: float = 1.0
) -> list[float]:
    frames = max(float(runtime["observation_frame_count"]), 1.0)
    canonical_surfels = max(float(np.expm1(runtime["log_canonical_surfels"])), 0.0)
    scale = float(opportunity_scale)
    scaled_frames = frames * scale
    scaled_surfels = canonical_surfels * scale
    scaled_temporal_mean = float(runtime["temporal_support_mean"]) * scale
    scaled_temporal_p10 = float(runtime["temporal_support_p10"]) * scale
    scaled_view_mean = float(runtime["view_support_mean"]) * scale
    return [
        float(runtime["log_query_points"]),
        float(np.log1p(scaled_surfels / scaled_frames)),
        float(runtime["query_surface_mean_m"]),
        float(runtime["query_surface_p95_m"]),
        float(runtime["clean_keep_fraction"]),
        float(runtime["unknown_query_fraction"]),
        scaled_temporal_mean / scaled_frames,
        scaled_temporal_p10 / scaled_frames,
        scaled_view_mean / scaled_frames,
        float(runtime["completion_candidate_fraction"]),
        float(runtime["compiled_to_query_ratio"]),
        float(runtime["sensor_range_m"]),
    ]


def rows_to_opportunity_arrays(
    rows: list[Mapping[str, Any]], opportunity_scale: float = 1.0
) -> dict[str, np.ndarray]:
    arrays = rows_to_arrays(rows)
    arrays["validity"] = np.asarray(
        [
            _opportunity_features(row["runtime_features"], opportunity_scale)
            for row in rows
        ],
        dtype=np.float32,
    )
    return arrays


def maximum_opportunity_feature_shift(
    rows: list[Mapping[str, Any]], scale_factors: Sequence[float]
) -> float:
    reference = rows_to_opportunity_arrays(rows)["validity"]
    return max(
        float(
            np.max(
                np.abs(
                    reference
                    - rows_to_opportunity_arrays(rows, opportunity_scale=scale)["validity"]
                )
            )
        )
        for scale in scale_factors
    )


def train_validity_model(
    model: SmallMLP,
    train: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    validity = torch.as_tensor(train["validity"], dtype=torch.float32, device=device)
    repairable = torch.as_tensor(train["repairable"], dtype=torch.float32, device=device)
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
        permutation = torch.randperm(len(validity), generator=generator).to(device)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            loss = loss_fn(model(validity[index]), repairable[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        if epoch in {0, int(config["epochs"]) - 1} or (epoch + 1) % 20 == 0:
            history.append({"epoch": epoch + 1, "loss": loss_sum / max(batches, 1)})
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
