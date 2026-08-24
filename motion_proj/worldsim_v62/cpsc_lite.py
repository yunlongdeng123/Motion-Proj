"""Streaming P2/P4 loader, CPSC-Lite network, and development losses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
    project_feasible_tristate,
)


@dataclass
class UnitArrays:
    scene: str
    target_frame: int
    prior_features: np.ndarray
    query_features: np.ndarray
    prior_tristate: np.ndarray
    target_class: np.ndarray
    method_class: np.ndarray
    dropout_class: np.ndarray
    query_type: np.ndarray
    contradiction: np.ndarray
    actor_bound: np.ndarray
    prior_valid: np.ndarray

    @property
    def query_count(self) -> int:
        return int(self.target_class.shape[0])


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def load_unit_arrays(
    p2_run: Path, p4_run: Path, scene: str, target_frame: int
) -> UnitArrays:
    query_path = p2_run / "units" / scene / f"f{target_frame:03d}" / "QUERIES.npz"
    sidecar_path = (
        p4_run
        / "units"
        / scene
        / f"f{target_frame:03d}"
        / "IRWM_PRIOR_SIDECAR.npz"
    )
    with np.load(query_path, allow_pickle=False) as query:
        query_indices = np.asarray(query["query_indices"], dtype=np.int32)
        grid_shape = np.asarray(query["grid_shape"], dtype=np.float32)
        method_class = np.asarray(query["method_class_index"], dtype=np.int64)
        dropout_class = np.asarray(query["dropout_class_index"], dtype=np.int64)
        target_class = np.asarray(query["target_class_index"], dtype=np.int64)
        query_type = np.asarray(query["query_type"], dtype=np.int64)
        contradiction = np.asarray(query["method_contradiction"], dtype=bool)
        actor_id = np.asarray(query["actor_id"], dtype=np.int32)
        actor_current = np.asarray(query["actor_current_support"], dtype=bool)
        actor_swept = np.asarray(query["actor_swept_support"], dtype=bool)
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        query_to_prior = np.asarray(sidecar["query_to_prior_cell"], dtype=np.int64)
        unique_logits = np.asarray(sidecar["prior_logits"], dtype=np.float32)
        query_to_bev = np.asarray(sidecar["query_to_bev_cell"], dtype=np.int64)
        unique_bev = np.asarray(sidecar["bev_features"], dtype=np.float32)
        prior_valid = np.asarray(sidecar["query_source_valid"], dtype=bool)

    query_count = int(query_indices.shape[0])
    raw_logits = np.zeros((query_count, unique_logits.shape[1]), dtype=np.float32)
    bev_features = np.zeros((query_count, unique_bev.shape[1]), dtype=np.float32)
    valid_prior_mapping = query_to_prior >= 0
    valid_bev_mapping = query_to_bev >= 0
    raw_logits[valid_prior_mapping] = unique_logits[query_to_prior[valid_prior_mapping]]
    bev_features[valid_bev_mapping] = unique_bev[query_to_bev[valid_bev_mapping]]
    prior_valid = prior_valid & valid_prior_mapping & valid_bev_mapping

    semantic_probabilities = _stable_softmax(raw_logits)
    entropy = -np.sum(
        semantic_probabilities * np.log(np.clip(semantic_probabilities, 1e-8, None)),
        axis=1,
    ) / np.log(float(semantic_probabilities.shape[1]))
    prior_tristate = np.stack(
        (
            semantic_probabilities[:, 0],
            semantic_probabilities[:, 1:].sum(axis=1),
            np.zeros(query_count, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    prior_tristate[~prior_valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    entropy[~prior_valid] = 1.0

    normalized_coordinates = 2.0 * (
        (query_indices.astype(np.float32) + 0.5) / grid_shape[None]
    ) - 1.0
    method_one_hot = np.eye(3, dtype=np.float32)[method_class]
    actor_bound = actor_id >= 0
    actor_features = np.stack((actor_bound, actor_current, actor_swept), axis=1).astype(
        np.float32
    )
    prior_method_residual = prior_tristate - method_one_hot
    prior_features = np.concatenate(
        (
            raw_logits,
            entropy[:, None].astype(np.float32),
            prior_tristate,
            prior_valid[:, None].astype(np.float32),
            bev_features,
        ),
        axis=1,
    ).astype(np.float32)
    query_features = np.concatenate(
        (
            normalized_coordinates,
            method_one_hot,
            contradiction[:, None].astype(np.float32),
            actor_features,
            prior_method_residual,
        ),
        axis=1,
    ).astype(np.float32)
    return UnitArrays(
        scene=scene,
        target_frame=int(target_frame),
        prior_features=prior_features,
        query_features=query_features,
        prior_tristate=prior_tristate,
        target_class=target_class,
        method_class=method_class,
        dropout_class=dropout_class,
        query_type=query_type,
        contradiction=contradiction,
        actor_bound=actor_bound,
        prior_valid=prior_valid,
    )


def iter_unit_batches(
    unit: UnitArrays,
    batch_size: int,
    *,
    rng: np.random.Generator | None,
) -> Iterator[dict[str, np.ndarray]]:
    order = np.arange(unit.query_count, dtype=np.int64)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, unit.query_count, int(batch_size)):
        selected = order[start : start + int(batch_size)]
        yield {
            "prior_features": unit.prior_features[selected],
            "query_features": unit.query_features[selected],
            "prior_tristate": unit.prior_tristate[selected],
            "target_class": unit.target_class[selected],
            "method_class": unit.method_class[selected],
            "dropout_class": unit.dropout_class[selected],
            "query_type": unit.query_type[selected],
            "contradiction": unit.contradiction[selected],
            "actor_bound": unit.actor_bound[selected],
            "prior_valid": unit.prior_valid[selected],
        }


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, values: Tensor) -> Tensor:
        return self.norm(values + self.layers(values))


class CPSCLite(nn.Module):
    def __init__(
        self,
        prior_feature_dimension: int,
        query_feature_dimension: int,
        *,
        hidden_width: int = 256,
        decoder_layers: int = 4,
        residual_blocks: int = 2,
        projection_iterations: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.prior_adapter = nn.Sequential(
            nn.LayerNorm(prior_feature_dimension),
            nn.Linear(prior_feature_dimension, hidden_width),
            nn.GELU(),
        )
        self.query_adapter = nn.Sequential(
            nn.LayerNorm(query_feature_dimension),
            nn.Linear(query_feature_dimension, hidden_width),
            nn.GELU(),
        )
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_width, hidden_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_width),
                )
                for _ in range(int(decoder_layers))
            ]
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_width, dropout) for _ in range(int(residual_blocks))]
        )
        self.evidence_head = nn.Linear(hidden_width, 3)
        self.constraint_trust_head = nn.Linear(hidden_width, 1)
        self.projection_updates = nn.ModuleList(
            [nn.Linear(hidden_width + 3, 3) for _ in range(int(projection_iterations))]
        )

    def forward(
        self,
        prior_features: Tensor,
        query_features: Tensor,
        *,
        observed_free: Tensor,
        observed_occupied: Tensor,
        contradiction: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.prior_adapter(prior_features) + self.query_adapter(query_features)
        for layer in self.decoder:
            hidden = hidden + layer(hidden)
        for block in self.residual_blocks:
            hidden = block(hidden)

        evidence = F.softplus(self.evidence_head(hidden))
        alpha = evidence + 1.0
        base_probabilities = alpha / alpha.sum(dim=-1, keepdim=True)
        probabilities = base_probabilities
        trust = torch.sigmoid(self.constraint_trust_head(hidden))
        constrained = torch.zeros_like(observed_free, dtype=torch.bool)
        for update in self.projection_updates:
            delta = update(torch.cat((hidden, probabilities), dim=-1))
            logits = torch.log(probabilities.clamp_min(1e-8)) + trust * delta
            projected = project_feasible_tristate(
                logits,
                observed_free=observed_free,
                observed_occupied=observed_occupied,
                contradiction=contradiction,
            )
            probabilities = projected.probabilities
            constrained = projected.constrained
        return {
            "probabilities": probabilities,
            "base_probabilities": base_probabilities,
            "alpha": alpha,
            "evidence": evidence,
            "uncertainty": 3.0 / alpha.sum(dim=-1),
            "constraint_trust": trust.squeeze(-1),
            "constrained": constrained,
        }


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def _dirichlet_kl_uniform(alpha: Tensor) -> Tensor:
    classes = int(alpha.shape[-1])
    sum_alpha = alpha.sum(dim=-1)
    return (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=-1)
        - torch.lgamma(torch.tensor(float(classes), device=alpha.device))
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha)[..., None])).sum(
            dim=-1
        )
    )


def compute_cpsc_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    loss_config: dict[str, Any],
    epoch: int,
) -> dict[str, Tensor]:
    probabilities = outputs["probabilities"].float()
    base_probabilities = outputs["base_probabilities"].float()
    alpha = outputs["alpha"].float()
    target = batch["target_class"].long()
    method = batch["method_class"].long()
    dropout = batch["dropout_class"].long()
    contradiction = batch["contradiction"].bool()
    observed_free = (method == FREE_INDEX) & ~contradiction
    observed_occupied = (method == OCCUPIED_INDEX) & ~contradiction
    method_unknown = method == UNKNOWN_INDEX

    hard_expected = torch.full_like(target, UNKNOWN_INDEX)
    hard_expected[observed_free] = FREE_INDEX
    hard_expected[observed_occupied] = OCCUPIED_INDEX
    constrained = observed_free | observed_occupied | contradiction
    supervised = ~(constrained & (target != hard_expected))
    nll = -torch.log(
        probabilities.gather(1, target[:, None]).squeeze(1).clamp_min(1e-8)
    )
    class_weights = torch.as_tensor(
        loss_config["class_weights"], device=probabilities.device, dtype=probabilities.dtype
    )
    weighted = nll * class_weights[target]
    query_loss = weighted[supervised].sum() / class_weights[target][supervised].sum().clamp_min(
        1.0
    )

    one_hot = F.one_hot(target, num_classes=3).to(alpha.dtype)
    sum_alpha = alpha.sum(dim=-1)
    expected_nll = (
        one_hot * (torch.digamma(sum_alpha)[..., None] - torch.digamma(alpha))
    ).sum(dim=-1)
    adjusted_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0
    anneal = min(1.0, float(epoch + 1) / float(loss_config["evidential_anneal_epochs"]))
    evidential_loss = _masked_mean(
        expected_nll + anneal * _dirichlet_kl_uniform(adjusted_alpha), supervised
    )

    hidden_free_mask = method_unknown & (dropout == FREE_INDEX) & ~contradiction
    hidden_free_loss = _masked_mean(probabilities[:, OCCUPIED_INDEX], hidden_free_mask)
    safe_occ_mask = (
        method_unknown
        & (dropout == OCCUPIED_INDEX)
        & (target == OCCUPIED_INDEX)
        & ~contradiction
    )
    safe_occ_loss = _masked_mean(
        -torch.log(probabilities[:, OCCUPIED_INDEX].clamp_min(1e-8)), safe_occ_mask
    )
    actor_temporal_mask = (
        method_unknown
        & batch["actor_bound"].bool()
        & (dropout != UNKNOWN_INDEX)
        & ~contradiction
    )
    dropout_nll = -torch.log(
        probabilities.gather(1, dropout[:, None]).squeeze(1).clamp_min(1e-8)
    )
    actor_temporal_loss = _masked_mean(dropout_nll, actor_temporal_mask)

    prior_preserve_mask = safe_occ_mask & batch["prior_valid"].bool()
    prior_cross_entropy = -(
        batch["prior_tristate"].to(probabilities.dtype)
        * torch.log(base_probabilities.clamp_min(1e-8))
    ).sum(dim=-1)
    prior_preserve_loss = _masked_mean(prior_cross_entropy, prior_preserve_mask)

    total = (
        float(loss_config["query_weight"]) * query_loss
        + float(loss_config["evidential_weight"]) * evidential_loss
        + float(loss_config["hidden_free_weight"]) * hidden_free_loss
        + float(loss_config["safe_occ_weight"]) * safe_occ_loss
        + float(loss_config["actor_temporal_weight"]) * actor_temporal_loss
        + float(loss_config["prior_preserve_weight"]) * prior_preserve_loss
    )
    return {
        "total": total,
        "query": query_loss,
        "evidential": evidential_loss,
        "hidden_free": hidden_free_loss,
        "safe_occ": safe_occ_loss,
        "actor_temporal": actor_temporal_loss,
        "prior_preserve": prior_preserve_loss,
        "supervised_count": supervised.sum(),
        "hidden_free_count": hidden_free_mask.sum(),
        "safe_occ_count": safe_occ_mask.sum(),
        "actor_temporal_count": actor_temporal_mask.sum(),
        "hard_conflict_count": (~supervised).sum(),
    }


def projection_only_probabilities(
    prior_tristate: Tensor,
    method_class: Tensor,
    contradiction: Tensor,
) -> Tensor:
    return project_feasible_tristate(
        torch.log(prior_tristate.clamp_min(1e-8)),
        observed_free=(method_class == FREE_INDEX) & ~contradiction,
        observed_occupied=(method_class == OCCUPIED_INDEX) & ~contradiction,
        contradiction=contradiction,
    ).probabilities
