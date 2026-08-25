"""Native surface compiler network and deterministic surface data interface."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


EVIDENCE_TO_CLASS = np.asarray([UNKNOWN_INDEX, FREE_INDEX, OCCUPIED_INDEX], dtype=np.int64)
FEATURE_SLICES = {
    "method_one_hot": slice(276, 279),
    "method_contradiction": slice(279, 280),
    "method_behind_hit": slice(280, 281),
    "signed_distance": slice(281, 283),
    "ray": slice(289, 294),
    "temporal": slice(294, 298),
    "actor": slice(298, 302),
    "authority": slice(306, 311),
}
POINT_FEATURE_DIMENSION = 311


def cvar_tail(values: Tensor, alpha: float) -> Tensor:
    """Differentiable empirical upper-tail CVaR with a frozen alpha."""
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    count = max(1, int(math.ceil((1.0 - float(alpha)) * flat.numel())))
    return torch.topk(flat, count, largest=True, sorted=False).values.mean()


def aggregate_tail(values: Tensor, alpha: float, aggregator: str) -> Tensor:
    """Reduce one risk set with a frozen matched-ablation operator."""
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    if aggregator == "mean":
        return flat.mean()
    if aggregator == "max":
        return flat.max()
    if aggregator == "cvar":
        return cvar_tail(flat, alpha)
    raise ValueError(f"unsupported risk aggregator: {aggregator}")


def _segment_mean(values: Tensor, index: Tensor, count: int) -> Tensor:
    output = values.new_zeros((count, values.shape[1]))
    output.index_add_(0, index, values)
    denominator = torch.bincount(index, minlength=count).to(values.dtype).clamp_min(1.0)
    return output / denominator[:, None]


class SurfaceNeighborBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden: Tensor, edge_index: Tensor) -> Tensor:
        aggregate = torch.zeros_like(hidden)
        degree = hidden.new_zeros((hidden.shape[0],))
        if edge_index.numel():
            source, target = edge_index
            aggregate.index_add_(0, target, hidden[source])
            degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)[:, None]
        return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=1)))


class SurfNCC(nn.Module):
    """Two-block surface encoder with patch attention and one proposal token."""

    def __init__(
        self,
        input_dimension: int,
        *,
        hidden_dimension: int = 256,
        neighbor_blocks: int = 2,
        patch_transformer_layers: int = 2,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.neighbor_blocks = nn.ModuleList(
            [SurfaceNeighborBlock(hidden_dimension) for _ in range(neighbor_blocks)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=2 * hidden_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(layer, patch_transformer_layers)
        self.proposal_token = nn.Parameter(torch.zeros(1, 1, hidden_dimension))
        nn.init.normal_(self.proposal_token, std=0.02)
        self.state_head = nn.Linear(hidden_dimension, 3)
        self.hidden_free_head = nn.Linear(hidden_dimension, 1)
        self.authority_head = nn.Linear(hidden_dimension, 1)
        self.patch_risk_head = nn.Linear(hidden_dimension, 1)
        self.proposal_risk_head = nn.Linear(hidden_dimension, 1)

    def encode_surface_points(
        self,
        point_features: Tensor,
        edge_index: Tensor,
        patch_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode a bounded point chunk and pool complete deterministic patches."""
        hidden = self.point_mlp(point_features)
        for block in self.neighbor_blocks:
            hidden = block(hidden, edge_index)
        patch_count = int(patch_index.max().item()) + 1
        return hidden, _segment_mean(hidden, patch_index, patch_count)

    def encode_proposal_patches(
        self,
        patch_tokens: Tensor,
        patch_proposal_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one proposal token over the complete patch set of each proposal."""
        proposal_count = int(patch_proposal_index.max().item()) + 1
        encoded_patches = torch.zeros_like(patch_tokens)
        proposal_tokens = []
        for proposal in range(proposal_count):
            selected_patches = torch.nonzero(
                patch_proposal_index == proposal, as_tuple=False
            ).squeeze(1)
            tokens = torch.cat(
                (
                    self.proposal_token.expand(1, -1, -1),
                    patch_tokens[selected_patches][None],
                ),
                dim=1,
            )
            encoded = self.patch_encoder(tokens)[0]
            proposal_tokens.append(encoded[0])
            encoded_patches[selected_patches] = encoded[1:]
        return encoded_patches, torch.stack(proposal_tokens, dim=0)

    def decode_surface_points(
        self,
        hidden: Tensor,
        patch_index: Tensor,
        patch_proposal_index: Tensor,
        encoded_patches: Tensor,
        proposal_tokens: Tensor,
        method_class: Tensor,
        contradiction: Tensor,
        *,
        cvar_alpha: float,
    ) -> dict[str, Tensor]:
        """Decode a point chunk using patch/proposal context assembled upstream."""
        point_proposal_index = patch_proposal_index[patch_index]
        fused = (
            hidden
            + encoded_patches[patch_index]
            + proposal_tokens[point_proposal_index]
        )

        state_logits = self.state_head(fused)
        base_probabilities = torch.softmax(state_logits, dim=-1)
        projected = project_feasible_tristate(
            state_logits,
            observed_free=(method_class == FREE_INDEX) & ~contradiction,
            observed_occupied=(method_class == OCCUPIED_INDEX) & ~contradiction,
            contradiction=contradiction,
        )
        probabilities = projected.probabilities
        hidden_free_logits = self.hidden_free_head(fused).squeeze(1)
        authority_logits = self.authority_head(fused).squeeze(1)
        hidden_free = torch.sigmoid(hidden_free_logits)
        authority = torch.sigmoid(authority_logits)
        point_risk = torch.clamp(
            probabilities[:, OCCUPIED_INDEX] * hidden_free + 0.25 * (1.0 - authority),
            min=0.0,
            max=1.0,
        )
        patch_count = int(patch_index.max().item()) + 1
        proposal_count = int(patch_proposal_index.max().item()) + 1
        patch_cvar = torch.stack(
            [
                cvar_tail(point_risk[patch_index == patch], cvar_alpha)
                for patch in range(patch_count)
            ]
        )
        proposal_cvar = torch.stack(
            [
                patch_cvar[patch_proposal_index == proposal].max()
                for proposal in range(proposal_count)
            ]
        )
        patch_risk_head = torch.sigmoid(
            self.patch_risk_head(encoded_patches)
        ).squeeze(1)
        proposal_head_target = torch.stack(
            [
                patch_risk_head[patch_proposal_index == proposal].max()
                for proposal in range(proposal_count)
            ]
        )
        return {
            "base_probabilities": base_probabilities,
            "probabilities": probabilities,
            "constrained": projected.constrained,
            "hidden_free": hidden_free,
            "hidden_free_logits": hidden_free_logits,
            "authority": authority,
            "authority_logits": authority_logits,
            "point_risk": point_risk,
            "patch_cvar": patch_cvar,
            "proposal_cvar": proposal_cvar,
            "patch_risk_head": patch_risk_head,
            "proposal_head_target": proposal_head_target,
            "proposal_risk_head": torch.sigmoid(
                self.proposal_risk_head(proposal_tokens)
            ).squeeze(1),
        }

    def forward(
        self,
        point_features: Tensor,
        edge_index: Tensor,
        patch_index: Tensor,
        method_class: Tensor,
        contradiction: Tensor,
        *,
        cvar_alpha: float,
        patch_proposal_index: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden, patch_tokens = self.encode_surface_points(
            point_features, edge_index, patch_index
        )
        patch_count = int(patch_tokens.shape[0])
        if patch_proposal_index is None:
            patch_proposal_index = torch.zeros(
                patch_count, device=patch_index.device, dtype=torch.long
            )
        encoded_patches, proposal_tokens = self.encode_proposal_patches(
            patch_tokens, patch_proposal_index
        )
        return self.decode_surface_points(
            hidden,
            patch_index,
            patch_proposal_index,
            encoded_patches,
            proposal_tokens,
            method_class,
            contradiction,
            cvar_alpha=cvar_alpha,
        )


@dataclass
class SurfaceUnit:
    scene: str
    target_frame: int
    arrays: dict[str, np.ndarray]
    native_logits: np.ndarray
    native_bev: np.ndarray
    native_entropy: np.ndarray
    native_margin: np.ndarray
    native_source_valid: np.ndarray
    target_class: np.ndarray
    method_class: np.ndarray
    proposal_index: np.ndarray
    proposal_ids: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_surface_unit(
    p3_run: Path,
    native_run: Path,
    scene: str,
    target_frame: int,
    *,
    native_split: str = "development",
) -> SurfaceUnit:
    unit_dir = p3_run / "units" / scene / f"f{target_frame:03d}"
    with np.load(unit_dir / "SURFACE_POINTS.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    native_dir = native_run / "units" / native_split / scene / f"f{target_frame:03d}"
    native_indices = np.asarray(arrays["native_indices"], dtype=np.int64)
    native_valid = np.asarray(arrays["native_valid"], dtype=bool)
    logits_grid = np.load(native_dir / "NATIVE_LOGITS.npy", mmap_mode="r")
    bev_grid = np.load(native_dir / "BEV_LATENT.npy", mmap_mode="r")
    entropy_grid = np.load(native_dir / "ENTROPY.npy", mmap_mode="r")
    margin_grid = np.load(native_dir / "MARGIN.npy", mmap_mode="r")
    valid_grid = np.load(native_dir / "SOURCE_VALID.npy", mmap_mode="r")
    clipped = np.stack(
        [
            np.clip(native_indices[:, axis], 0, logits_grid.shape[axis] - 1)
            for axis in range(3)
        ],
        axis=1,
    )
    idx = tuple(clipped.T)
    native_logits = np.asarray(logits_grid[idx], dtype=np.float32)
    native_bev = np.asarray(bev_grid[clipped[:, 0], clipped[:, 1]], dtype=np.float32)
    native_entropy = np.asarray(entropy_grid[idx], dtype=np.float32)
    native_margin = np.asarray(margin_grid[idx], dtype=np.float32)
    native_source_valid = native_valid & np.asarray(valid_grid[idx], dtype=bool)
    native_logits[~native_source_valid] = 0.0
    native_bev[~native_source_valid] = 0.0
    native_entropy[~native_source_valid] = 1.0
    native_margin[~native_source_valid] = 0.0

    surfaces = _read_jsonl(unit_dir / "SURFACE_REGISTRY.jsonl")
    proposal_ids = sorted({row["proposal_id"] for row in surfaces})
    proposal_lookup = {value: index for index, value in enumerate(proposal_ids)}
    surface_to_proposal = np.empty(len(surfaces), dtype=np.int32)
    for row in surfaces:
        surface_to_proposal[int(row["surface_index"])] = proposal_lookup[row["proposal_id"]]
    proposal_index = surface_to_proposal[np.asarray(arrays["surface_index"], dtype=np.int64)]
    return SurfaceUnit(
        scene=scene,
        target_frame=int(target_frame),
        arrays=arrays,
        native_logits=native_logits,
        native_bev=native_bev,
        native_entropy=native_entropy,
        native_margin=native_margin,
        native_source_valid=native_source_valid,
        target_class=EVIDENCE_TO_CLASS[np.asarray(arrays["target_state"], dtype=np.int64)],
        method_class=EVIDENCE_TO_CLASS[np.asarray(arrays["method_state"], dtype=np.int64)],
        proposal_index=proposal_index,
        proposal_ids=proposal_ids,
    )


def assemble_point_features(unit: SurfaceUnit, selected: np.ndarray) -> np.ndarray:
    arrays = unit.arrays
    chosen = np.asarray(selected, dtype=np.int64)
    method_one_hot = np.eye(3, dtype=np.float32)[unit.method_class[chosen]]
    hard = np.concatenate(
        (
            method_one_hot,
            np.asarray(arrays["method_contradiction"])[chosen, None].astype(np.float32),
            np.asarray(arrays["method_behind_hit"])[chosen, None].astype(np.float32),
        ),
        axis=1,
    )
    temporal = np.stack(
        [
            np.asarray(arrays[name], dtype=np.float32)[chosen]
            for name in (
                "temporal_free_count",
                "temporal_occ_count",
                "temporal_unknown_count",
                "temporal_contradiction_count",
            )
        ],
        axis=1,
    ) / float(arrays["temporal_state_by_sweep"].shape[1])
    actor = np.stack(
        (
            np.asarray(arrays["actor_id"])[chosen] >= 0,
            np.asarray(arrays["actor_current_support"])[chosen],
            np.asarray(arrays["actor_swept_support"])[chosen],
            np.asarray(arrays["actor_observed_hit_support"])[chosen],
        ),
        axis=1,
    ).astype(np.float32)
    surface_type = np.eye(4, dtype=np.float32)[
        np.asarray(arrays["surface_type"], dtype=np.int64)[chosen]
    ]
    authority = np.stack(
        [
            ((np.asarray(arrays["authority_bits"], dtype=np.uint8)[chosen] >> bit) & 1)
            for bit in range(5)
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate(
        (
            unit.native_logits[chosen],
            unit.native_bev[chosen],
            unit.native_entropy[chosen, None],
            unit.native_margin[chosen, None],
            unit.native_source_valid[chosen, None].astype(np.float32),
            hard,
            np.asarray(arrays["signed_distance_free_m"], dtype=np.float32)[chosen, None],
            np.asarray(arrays["signed_distance_occupied_m"], dtype=np.float32)[chosen, None],
            np.asarray(arrays["patch_local_coordinates_m"], dtype=np.float32)[chosen],
            np.asarray(arrays["normals"], dtype=np.float32)[chosen],
            np.asarray(arrays["ray_direction"], dtype=np.float32)[chosen],
            np.asarray(arrays["ray_distance_m"], dtype=np.float32)[chosen, None],
            np.asarray(arrays["ray_hit_order"], dtype=np.float32)[chosen, None],
            temporal,
            actor,
            surface_type,
            authority,
        ),
        axis=1,
    ).astype(np.float32)


def build_surface_edges(
    grid_indices: np.ndarray,
    neighborhood_index: np.ndarray,
    shape: tuple[int, int, int] = (300, 300, 40),
) -> np.ndarray:
    """Build exact 6-neighbor edges inside deterministic local patches."""
    points = np.asarray(grid_indices, dtype=np.int64)
    neighborhoods = np.asarray(neighborhood_index, dtype=np.int64)
    linear = np.ravel_multi_index(points.T, shape)
    volume = int(np.prod(shape))
    lookup = {
        int(neighborhood * volume + cell): index
        for index, (neighborhood, cell) in enumerate(zip(neighborhoods, linear))
    }
    offsets = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.int64)
    sources: list[int] = []
    targets: list[int] = []
    bounds = np.asarray(shape, dtype=np.int64)
    for point_index, point in enumerate(points):
        for offset in offsets:
            neighbor = point + offset
            if np.any(neighbor >= bounds):
                continue
            cell = int(np.ravel_multi_index(neighbor, shape))
            neighbor_index = lookup.get(
                int(neighborhoods[point_index] * volume + cell)
            )
            if neighbor_index is not None:
                sources.extend((point_index, neighbor_index))
                targets.extend((neighbor_index, point_index))
    return np.asarray((sources, targets), dtype=np.int64)


def _selected_batch(unit: SurfaceUnit, selected: np.ndarray) -> dict[str, np.ndarray]:
    selected = np.asarray(selected, dtype=np.int64)
    global_patch = np.asarray(unit.arrays["patch_index"], dtype=np.int64)[selected]
    unique_patch, local_patch = np.unique(global_patch, return_inverse=True)
    patch_proposal_global = np.asarray(
        [unit.proposal_index[selected[np.flatnonzero(global_patch == patch)[0]]] for patch in unique_patch],
        dtype=np.int64,
    )
    proposal_global_index, patch_proposal_index = np.unique(
        patch_proposal_global, return_inverse=True
    )
    point_proposal_index = patch_proposal_index[local_patch].astype(np.int64)
    actor_points = np.asarray(unit.arrays["actor_id"], dtype=np.int32) >= 0
    proposal_actor = np.asarray(
        [
            np.any(actor_points[unit.proposal_index == proposal])
            for proposal in proposal_global_index
        ],
        dtype=bool,
    )
    proposal_chunk_point_count = np.bincount(
        point_proposal_index, minlength=proposal_global_index.shape[0]
    ).astype(np.int64)
    proposal_point_count = np.asarray(
        [np.count_nonzero(unit.proposal_index == proposal) for proposal in proposal_global_index],
        dtype=np.int64,
    )
    full_contradiction = np.asarray(
        unit.arrays["method_contradiction"], dtype=bool
    )
    proposal_unsafe = []
    proposal_safe = []
    for proposal in proposal_global_index:
        proposal_points = unit.proposal_index == proposal
        hidden_free = np.any(
            proposal_points
            & (unit.target_class == FREE_INDEX)
            & (unit.method_class == UNKNOWN_INDEX)
            & ~full_contradiction
        )
        target_occupied = np.any(
            proposal_points & (unit.target_class == OCCUPIED_INDEX)
        )
        proposal_unsafe.append(bool(hidden_free))
        proposal_safe.append(bool(target_occupied and not hidden_free))
    edge_index = build_surface_edges(
        np.asarray(unit.arrays["grid_indices"])[selected],
        np.asarray(unit.arrays["patch_index"])[selected],
    )
    return {
        "selected": selected,
        "point_features": assemble_point_features(unit, selected),
        "edge_index": edge_index,
        "patch_index": local_patch.astype(np.int64),
        "patch_global_index": unique_patch.astype(np.int64),
        "patch_proposal_index": patch_proposal_index.astype(np.int64),
        "point_proposal_index": point_proposal_index,
        "proposal_global_index": proposal_global_index.astype(np.int64),
        "proposal_actor": proposal_actor,
        "proposal_point_count": proposal_point_count,
        "proposal_chunk_point_count": proposal_chunk_point_count,
        "proposal_safe": np.asarray(proposal_safe, dtype=bool),
        "proposal_unsafe": np.asarray(proposal_unsafe, dtype=bool),
        "target_class": unit.target_class[selected],
        "method_class": unit.method_class[selected],
        "contradiction": np.asarray(unit.arrays["method_contradiction"])[selected].astype(bool),
        "authority_target": (np.asarray(unit.arrays["authority_bits"])[selected] != 0),
    }


def proposal_batches(
    unit: SurfaceUnit, proposal_index: int, point_limit: int
) -> list[dict[str, np.ndarray]]:
    selected = np.flatnonzero(unit.proposal_index == int(proposal_index))
    patch_values = np.asarray(unit.arrays["patch_index"], dtype=np.int64)[selected]
    groups: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    count = 0
    for patch in np.unique(patch_values):
        members = selected[patch_values == patch]
        if kept and count + members.shape[0] > int(point_limit):
            groups.append(np.concatenate(kept))
            kept = []
            count = 0
        kept.append(members)
        count += int(members.shape[0])
    if kept:
        groups.append(np.concatenate(kept))
    return [_selected_batch(unit, group) for group in groups]


def proposal_batch(unit: SurfaceUnit, proposal_index: int, point_limit: int) -> dict[str, np.ndarray]:
    return proposal_batches(unit, proposal_index, point_limit)[0]


def packed_unit_batches(unit: SurfaceUnit, point_limit: int) -> list[dict[str, np.ndarray]]:
    global_patch = np.asarray(unit.arrays["patch_index"], dtype=np.int64)
    groups: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    count = 0
    patch_order = sorted(
        np.unique(global_patch).tolist(),
        key=lambda patch: (
            int(unit.proposal_index[np.flatnonzero(global_patch == patch)[0]]),
            int(patch),
        ),
    )
    for patch in patch_order:
        members = np.flatnonzero(global_patch == patch)
        if kept and count + members.shape[0] > int(point_limit):
            groups.append(np.concatenate(kept))
            kept = []
            count = 0
        kept.append(members)
        count += int(members.shape[0])
    if kept:
        groups.append(np.concatenate(kept))
    return [_selected_batch(unit, group) for group in groups]


def capacity_proposal_indices(unit: SurfaceUnit) -> list[int]:
    counts = np.bincount(unit.proposal_index, minlength=len(unit.proposal_ids))
    actor_point = np.asarray(unit.arrays["actor_id"], dtype=np.int32) >= 0
    actor_by_proposal = np.zeros(len(unit.proposal_ids), dtype=bool)
    np.logical_or.at(actor_by_proposal, unit.proposal_index, actor_point)
    result = []
    for actor in (False, True):
        candidates = np.flatnonzero(actor_by_proposal == actor)
        if candidates.shape[0]:
            result.append(int(candidates[np.argmax(counts[candidates])]))
    return result


def apply_structural_dropout(
    unit: SurfaceUnit,
    batch: dict[str, np.ndarray],
    rng: np.random.Generator,
    support_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask one applicable evidence family without changing target supervision."""
    features = np.asarray(batch["point_features"], dtype=np.float32).copy()
    selected = np.asarray(batch["selected"], dtype=np.int64)
    arrays = unit.arrays
    actor_applicable = bool(
        np.any(np.asarray(arrays["actor_observed_hit_support"])[selected])
    )
    families = ["ray_bundle", "spatial_block", "temporal_window", "surface_patch"]
    if actor_applicable:
        families.append("actor_support")
    family = families[int(rng.integers(0, len(families)))]
    point_count = selected.shape[0]
    mask = np.zeros(point_count, dtype=bool)
    effective_method = np.asarray(batch["method_class"], dtype=np.int64).copy()
    effective_contradiction = np.asarray(batch["contradiction"], dtype=bool).copy()
    detail: dict[str, Any] = {"family": family, "support_fraction": float(support_fraction)}

    if family == "ray_bundle":
        values = np.asarray(arrays["ray_bundle_id"])[selected]
        unique = np.unique(values)
        count = max(1, int(math.ceil(float(support_fraction) * unique.shape[0])))
        dropped = rng.choice(unique, size=count, replace=False)
        mask = np.isin(values, dropped)
        features[mask, FEATURE_SLICES["ray"]] = 0.0
        detail["dropped_bundle_count"] = int(count)
        detail["dropped_bundle_ids"] = sorted(int(value) for value in dropped)
    elif family == "spatial_block":
        coordinates = np.asarray(arrays["coordinates_m"], dtype=np.float32)[selected]
        seed_position = int(rng.integers(0, point_count))
        seed = coordinates[seed_position]
        distance = np.max(np.abs(coordinates - seed[None]), axis=1)
        count = max(1, int(math.ceil(float(support_fraction) * point_count)))
        order = np.lexsort((np.arange(point_count), distance))
        mask[order[:count]] = True
        detail["dropped_point_count"] = int(count)
        detail["seed_unit_point_index"] = int(selected[seed_position])
        detail["seed_coordinate_m"] = seed.tolist()
        detail["maximum_selected_linf_distance_m"] = float(distance[order[count - 1]])
    elif family == "surface_patch":
        values = np.asarray(batch["patch_index"], dtype=np.int64)
        unique = np.unique(values)
        count = max(1, int(math.ceil(float(support_fraction) * unique.shape[0])))
        dropped = rng.choice(unique, size=count, replace=False)
        mask = np.isin(values, dropped)
        detail["dropped_patch_count"] = int(count)
        patch_global = np.asarray(batch["patch_global_index"], dtype=np.int64)
        detail["dropped_patch_ids"] = sorted(
            int(patch_global[int(value)]) for value in dropped
        )
    elif family == "actor_support":
        support = np.flatnonzero(np.asarray(arrays["actor_observed_hit_support"])[selected])
        count = max(1, int(math.ceil(float(support_fraction) * support.shape[0])))
        dropped = rng.choice(support, size=count, replace=False)
        mask[dropped] = True
        features[mask, 301] = 0.0
        detail["dropped_actor_observation_count"] = int(count)
        detail["dropped_unit_point_indices"] = sorted(
            int(value) for value in selected[dropped]
        )
    else:
        states = np.asarray(arrays["temporal_state_by_sweep"], dtype=np.uint8)[selected].copy()
        contradictions = np.asarray(
            arrays["temporal_contradiction_by_sweep"], dtype=bool
        )[selected].copy()
        sweep_count = int(states.shape[1])
        count = max(1, int(math.ceil(float(support_fraction) * sweep_count)))
        dropped = np.sort(rng.choice(sweep_count, size=count, replace=False))
        keep = np.ones(sweep_count, dtype=bool)
        keep[dropped] = False
        remaining = states[:, keep]
        remaining_contradiction = contradictions[:, keep]
        free = np.any(remaining == 1, axis=1)
        occupied = np.any(remaining == 2, axis=1)
        combined_contradiction = np.any(remaining_contradiction, axis=1) | (free & occupied)
        method_class = np.full(point_count, UNKNOWN_INDEX, dtype=np.int64)
        method_class[free & ~occupied] = FREE_INDEX
        method_class[occupied & ~free] = OCCUPIED_INDEX
        features[:, FEATURE_SLICES["method_one_hot"]] = np.eye(3, dtype=np.float32)[
            method_class
        ]
        features[:, FEATURE_SLICES["method_contradiction"]] = combined_contradiction[:, None]
        features[:, FEATURE_SLICES["method_behind_hit"]] = 0.0
        features[:, FEATURE_SLICES["signed_distance"]] = 0.0
        temporal = np.stack(
            (
                np.count_nonzero(remaining == 1, axis=1),
                np.count_nonzero(remaining == 2, axis=1),
                np.count_nonzero(remaining == 0, axis=1),
                np.count_nonzero(remaining_contradiction, axis=1),
            ),
            axis=1,
        ).astype(np.float32) / float(sweep_count)
        features[:, FEATURE_SLICES["temporal"]] = temporal
        features[:, 301] = 0.0
        detail["dropped_sweeps"] = dropped.tolist()
        effective_method = method_class
        effective_contradiction = combined_contradiction
        mask[:] = True

    if family != "temporal_window":
        features[mask, FEATURE_SLICES["method_one_hot"]] = np.asarray(
            [0.0, 0.0, 1.0], dtype=np.float32
        )
        features[mask, FEATURE_SLICES["method_contradiction"]] = 0.0
        features[mask, FEATURE_SLICES["method_behind_hit"]] = 0.0
        features[mask, FEATURE_SLICES["signed_distance"]] = 0.0
        features[mask, FEATURE_SLICES["temporal"]] = 0.0
        features[mask, 301] = 0.0
        effective_method[mask] = UNKNOWN_INDEX
        effective_contradiction[mask] = False
    detail["masked_point_count"] = int(np.count_nonzero(mask))
    detail["point_count"] = int(point_count)
    detail["_method_class"] = effective_method
    detail["_contradiction"] = effective_contradiction
    return features, detail


def plan_packed_structural_dropout(
    unit: SurfaceUnit,
    rng: np.random.Generator,
    support_fraction: float,
) -> dict[int, dict[str, Any]]:
    """Freeze one semantic mask over each complete proposal before patch chunking."""
    arrays = unit.arrays
    plans: dict[int, dict[str, Any]] = {}
    for proposal in range(len(unit.proposal_ids)):
        selected = np.flatnonzero(unit.proposal_index == proposal)
        actor_applicable = bool(
            np.any(np.asarray(arrays["actor_observed_hit_support"])[selected])
        )
        families = ["ray_bundle", "spatial_block", "temporal_window", "surface_patch"]
        if actor_applicable:
            families.append("actor_support")
        family = families[int(rng.integers(0, len(families)))]
        plan: dict[str, Any] = {
            "family": family,
            "support_fraction": float(support_fraction),
            "proposal_global_index": int(proposal),
            "proposal_id": unit.proposal_ids[proposal],
            "point_count": int(selected.shape[0]),
        }
        masked = np.empty((0,), dtype=np.int64)
        if family == "ray_bundle":
            values = np.asarray(arrays["ray_bundle_id"])[selected]
            unique = np.unique(values)
            count = max(1, int(math.ceil(float(support_fraction) * unique.shape[0])))
            dropped = rng.choice(unique, size=count, replace=False)
            masked = selected[np.isin(values, dropped)]
            plan["dropped_bundle_count"] = int(count)
            plan["dropped_bundle_ids"] = sorted(int(value) for value in dropped)
        elif family == "spatial_block":
            coordinates = np.asarray(arrays["coordinates_m"], dtype=np.float32)[selected]
            seed_position = int(rng.integers(0, selected.shape[0]))
            seed = coordinates[seed_position]
            distance = np.max(np.abs(coordinates - seed[None]), axis=1)
            count = max(1, int(math.ceil(float(support_fraction) * selected.shape[0])))
            order = np.lexsort((selected, distance))
            masked = selected[order[:count]]
            plan["dropped_point_count"] = int(count)
            plan["seed_unit_point_index"] = int(selected[seed_position])
            plan["seed_coordinate_m"] = seed.tolist()
            plan["maximum_selected_linf_distance_m"] = float(
                distance[order[count - 1]]
            )
        elif family == "surface_patch":
            values = np.asarray(arrays["patch_index"], dtype=np.int64)[selected]
            unique = np.unique(values)
            count = max(1, int(math.ceil(float(support_fraction) * unique.shape[0])))
            dropped = rng.choice(unique, size=count, replace=False)
            masked = selected[np.isin(values, dropped)]
            plan["dropped_patch_count"] = int(count)
            plan["dropped_patch_ids"] = sorted(int(value) for value in dropped)
        elif family == "actor_support":
            support = np.flatnonzero(
                np.asarray(arrays["actor_observed_hit_support"])[selected]
            )
            count = max(1, int(math.ceil(float(support_fraction) * support.shape[0])))
            dropped = rng.choice(support, size=count, replace=False)
            masked = selected[dropped]
            plan["dropped_actor_observation_count"] = int(count)
            plan["dropped_unit_point_indices"] = sorted(int(value) for value in masked)
        else:
            sweep_count = int(arrays["temporal_state_by_sweep"].shape[1])
            count = max(1, int(math.ceil(float(support_fraction) * sweep_count)))
            dropped = np.sort(rng.choice(sweep_count, size=count, replace=False))
            plan["dropped_sweeps"] = dropped.tolist()
            plan["_dropped_sweeps"] = dropped
            masked = selected
        plan["masked_point_count"] = int(masked.shape[0])
        plan["_masked_unit_point_indices"] = masked
        plans[proposal] = plan
    return plans


def structural_dropout_records(
    plans: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in plans[proposal].items() if not key.startswith("_")}
        for proposal in sorted(plans)
    ]


def apply_packed_structural_dropout(
    unit: SurfaceUnit,
    batch: dict[str, np.ndarray],
    plans: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply complete-proposal masks consistently to one packed patch chunk."""
    features = np.asarray(batch["point_features"], dtype=np.float32).copy()
    method = np.asarray(batch["method_class"], dtype=np.int64).copy()
    contradiction = np.asarray(batch["contradiction"], dtype=bool).copy()
    point_proposal = np.asarray(batch["point_proposal_index"], dtype=np.int64)
    selected_unit = np.asarray(batch["selected"], dtype=np.int64)
    arrays = unit.arrays
    raw_authority = np.asarray(arrays["authority_bits"], dtype=np.uint8)
    authority_target = raw_authority[selected_unit] != 0
    for local_proposal in np.unique(point_proposal):
        positions = np.flatnonzero(point_proposal == local_proposal)
        proposal = int(batch["proposal_global_index"][local_proposal])
        plan = plans[proposal]
        family = str(plan["family"])
        if family == "temporal_window":
            chosen = selected_unit[positions]
            states = np.asarray(
                arrays["temporal_state_by_sweep"], dtype=np.uint8
            )[chosen]
            temporal_contradiction = np.asarray(
                arrays["temporal_contradiction_by_sweep"], dtype=bool
            )[chosen]
            dropped = np.asarray(plan["_dropped_sweeps"], dtype=np.int64)
            keep = np.ones(states.shape[1], dtype=bool)
            keep[dropped] = False
            remaining = states[:, keep]
            remaining_contradiction = temporal_contradiction[:, keep]
            free = np.any(remaining == 1, axis=1)
            occupied = np.any(remaining == 2, axis=1)
            combined_contradiction = np.any(
                remaining_contradiction, axis=1
            ) | (free & occupied)
            method_class = np.full(positions.shape[0], UNKNOWN_INDEX, dtype=np.int64)
            method_class[free & ~occupied] = FREE_INDEX
            method_class[occupied & ~free] = OCCUPIED_INDEX
            features[positions, FEATURE_SLICES["method_one_hot"]] = np.eye(
                3, dtype=np.float32
            )[method_class]
            features[positions, FEATURE_SLICES["method_contradiction"]] = (
                combined_contradiction[:, None]
            )
            features[positions, FEATURE_SLICES["method_behind_hit"]] = 0.0
            features[positions, FEATURE_SLICES["signed_distance"]] = 0.0
            temporal = np.stack(
                (
                    np.count_nonzero(remaining == 1, axis=1),
                    np.count_nonzero(remaining == 2, axis=1),
                    np.count_nonzero(remaining == 0, axis=1),
                    np.count_nonzero(remaining_contradiction, axis=1),
                ),
                axis=1,
            ).astype(np.float32) / float(states.shape[1])
            features[positions, FEATURE_SLICES["temporal"]] = temporal
            features[positions, 301] = 0.0
            authority_bits = (
                (method_class == OCCUPIED_INDEX).astype(np.uint8)
                | ((np.count_nonzero(remaining == 2, axis=1) >= 2).astype(np.uint8) << 1)
                | (raw_authority[chosen] & np.uint8(0b11100))
            )
            features[positions, FEATURE_SLICES["authority"]] = np.stack(
                [((authority_bits >> bit) & 1) for bit in range(5)], axis=1
            ).astype(np.float32)
            authority_target[positions] = authority_bits != 0
            method[positions] = method_class
            contradiction[positions] = combined_contradiction
            continue

        masked_unit = np.asarray(plan["_masked_unit_point_indices"], dtype=np.int64)
        local_mask = np.isin(selected_unit[positions], masked_unit)
        masked_positions = positions[local_mask]
        if family == "ray_bundle":
            features[masked_positions, FEATURE_SLICES["ray"]] = 0.0
        elif family == "actor_support":
            features[masked_positions, 301] = 0.0
        features[masked_positions, FEATURE_SLICES["method_one_hot"]] = np.asarray(
            [0.0, 0.0, 1.0], dtype=np.float32
        )
        features[masked_positions, FEATURE_SLICES["method_contradiction"]] = 0.0
        features[masked_positions, FEATURE_SLICES["method_behind_hit"]] = 0.0
        features[masked_positions, FEATURE_SLICES["signed_distance"]] = 0.0
        features[masked_positions, FEATURE_SLICES["temporal"]] = 0.0
        features[masked_positions, 301] = 0.0
        authority_bits = raw_authority[selected_unit[masked_positions]] & np.uint8(
            0b11100
        )
        features[masked_positions, FEATURE_SLICES["authority"]] = np.stack(
            [((authority_bits >> bit) & 1) for bit in range(5)], axis=1
        ).astype(np.float32)
        authority_target[masked_positions] = authority_bits != 0
        method[masked_positions] = UNKNOWN_INDEX
        contradiction[masked_positions] = False
    return features, method, contradiction, authority_target


def compute_surfncc_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    cvar_alpha: float,
    weights: dict[str, float],
    hidden_free_aggregator: str = "cvar",
) -> dict[str, Tensor]:
    probabilities = outputs["probabilities"].float()
    target = batch["target_class"].long()
    method = batch["method_class"].long()
    contradiction = batch["contradiction"].bool()
    hard_mismatch = (
        ((method == FREE_INDEX) & (target != FREE_INDEX))
        | ((method == OCCUPIED_INDEX) & (target != OCCUPIED_INDEX))
        | contradiction
    )
    supervised = ~hard_mismatch
    state = probabilities.sum() * 0.0
    if bool(supervised.any()):
        state = F.nll_loss(
            torch.log(probabilities[supervised].clamp_min(1e-8)), target[supervised]
        )
    hidden_free_mask = (target == FREE_INDEX) & (method == UNKNOWN_INDEX) & ~contradiction
    retention_mask = (target == OCCUPIED_INDEX) & (method == UNKNOWN_INDEX) & ~contradiction
    hidden_free_tails = []
    retention_tails = []
    point_proposal = batch.get("point_proposal_index")
    if point_proposal is None:
        hidden_free_tails.append(
            aggregate_tail(
                probabilities[hidden_free_mask, OCCUPIED_INDEX],
                cvar_alpha,
                hidden_free_aggregator,
            )
        )
        retention_tails.append(
            cvar_tail(1.0 - probabilities[retention_mask, OCCUPIED_INDEX], cvar_alpha)
        )
    else:
        proposal_count = int(point_proposal.max().item()) + 1
        for proposal in range(proposal_count):
            selected = point_proposal == proposal
            if bool((selected & hidden_free_mask).any()):
                hidden_free_tails.append(
                    aggregate_tail(
                        probabilities[selected & hidden_free_mask, OCCUPIED_INDEX],
                        cvar_alpha,
                        hidden_free_aggregator,
                    )
                )
            if bool((selected & retention_mask).any()):
                retention_tails.append(
                    cvar_tail(
                        1.0 - probabilities[selected & retention_mask, OCCUPIED_INDEX],
                        cvar_alpha,
                    )
                )
    hidden_free_tail = (
        torch.stack(hidden_free_tails).mean()
        if hidden_free_tails
        else probabilities.sum() * 0.0
    ) + F.binary_cross_entropy_with_logits(
        outputs["hidden_free_logits"].float(), hidden_free_mask.float()
    )
    retention = (
        torch.stack(retention_tails).mean()
        if retention_tails
        else probabilities.sum() * 0.0
    )
    authority = F.binary_cross_entropy_with_logits(
        outputs["authority_logits"].float(), batch["authority_target"].float()
    )
    edge_index = batch["edge_index"]
    consistency = probabilities.sum() * 0.0
    if edge_index.numel():
        consistency = torch.abs(
            probabilities[edge_index[0]] - probabilities[edge_index[1]]
        ).mean()
    head_consistency = F.mse_loss(
        outputs["patch_risk_head"].float(), outputs["patch_cvar"].detach().float()
    ) + F.mse_loss(
        outputs["proposal_risk_head"].float(),
        outputs.get("proposal_head_target", outputs["proposal_cvar"]).detach().float(),
    )
    ranking = probabilities.sum() * 0.0
    total = (
        float(weights["state"]) * state
        + float(weights["hidden_free_tail"]) * hidden_free_tail
        + float(weights["safe_occ_retention"]) * retention
        + float(weights["proposal_rank"]) * ranking
        + float(weights["surface_consistency"]) * (consistency + head_consistency)
        + float(weights["authority"]) * authority
    )
    return {
        "total": total,
        "state": state,
        "hidden_free_tail": hidden_free_tail,
        "safe_occ_retention": retention,
        "proposal_rank": ranking,
        "surface_consistency": consistency + head_consistency,
        "authority": authority,
        "supervised_count": supervised.sum(),
        "hidden_free_count": hidden_free_mask.sum(),
        "safe_occ_count": retention_mask.sum(),
    }


def proposal_rank_loss(
    safe_outputs: dict[str, Tensor],
    unsafe_outputs: dict[str, Tensor],
    margin: float,
) -> Tensor:
    return torch.relu(
        float(margin)
        + safe_outputs["proposal_cvar"].float()
        - unsafe_outputs["proposal_cvar"].float()
    )
