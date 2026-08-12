"""Schema and deterministic persistence for a per-actor Gaussian evidence field."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from .beta_fusion import BetaEvidence, prior_from_probability, update_from_counts


CORE_POSITIVE = np.int8(1)

IDENTITY_FIELDS = {
    "gaussian_id",
    "base_model",
    "base_index",
    "hard_instance_id",
}
EVIDENCE_FIELDS = {
    "alpha",
    "beta",
    "posterior",
    "uncertainty",
    "authenticity",
    "source_instance_opacity",
    "mask_evidence",
    "positive_count",
    "negative_count",
    "mask_weight",
    "visibility_weight",
    "depth_weight",
    "lidar_weight",
    "joint_weight",
}
SCALAR_FIELDS = {"actor_instance_id", "actor_token"}
REQUIRED_FIELDS = IDENTITY_FIELDS | EVIDENCE_FIELDS | SCALAR_FIELDS


def saturating_confidence(value: np.ndarray, *, scale: float) -> np.ndarray:
    """Map non-negative contribution mass to [0, 1) without dataset peeking."""

    mass = np.asarray(value, dtype=np.float64)
    divisor = float(scale)
    if not np.isfinite(mass).all() or np.any(mass < 0.0):
        raise ValueError("contribution mass must be finite and non-negative")
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("saturation scale must be finite and positive")
    return 1.0 - np.exp(-mass / divisor)


def build_evidence_state(
    *,
    instance_field: Mapping[str, np.ndarray],
    semantic_sidecar: Mapping[str, np.ndarray],
    actor_instance_id: int,
    actor_token: str,
    prior_strength: float,
    unassigned_probability: float,
    visibility_saturation_mass: float,
    mask_confidence_floor: float,
    depth_confidence_floor: float,
    lidar_confidence_floor: float,
    observed_authenticity: float,
) -> dict[str, np.ndarray]:
    """Initialize from O1 and fuse frozen mask/visibility/depth/LiDAR evidence."""

    gaussian_id = np.asarray(instance_field["gaussian_id"], dtype=np.int64)
    total = gaussian_id.size
    if gaussian_id.ndim != 1 or not np.array_equal(gaussian_id, np.arange(total)):
        raise ValueError("V3.3 Gaussian identity/order is invalid")
    required_sidecar = {
        "semantic_score",
        "visible_mass",
        "depth_consistency_rate",
        "num_positive_views",
        "num_negative_views",
        "labels",
    }
    missing = required_sidecar - set(semantic_sidecar)
    if missing:
        raise ValueError(f"semantic sidecar is missing fields: {sorted(missing)}")
    for name in required_sidecar:
        if np.asarray(semantic_sidecar[name]).shape != (total,):
            raise ValueError(f"semantic sidecar {name} shape differs from O1")

    floors = {
        "mask_confidence_floor": mask_confidence_floor,
        "depth_confidence_floor": depth_confidence_floor,
        "lidar_confidence_floor": lidar_confidence_floor,
        "observed_authenticity": observed_authenticity,
    }
    for name, value in floors.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")

    o1_probability = np.asarray(instance_field["instance_opacity"], dtype=np.float64)
    if o1_probability.shape != (total,):
        raise ValueError("V3.3 O1 opacity shape is invalid")
    unassigned = float(unassigned_probability)
    if not np.isfinite(unassigned) or not 0.0 < unassigned < 1.0:
        raise ValueError("unassigned_probability must lie strictly in (0, 1)")
    hard_instance_id = np.asarray(instance_field["hard_instance_id"], dtype=np.int32)
    if hard_instance_id.shape != (total,):
        raise ValueError("V3.3 hard-instance identity shape is invalid")
    # O1 stores one opacity per Gaussian for whichever actor owns that row.  A
    # per-actor evidence state must not inherit another actor's O1 probability.
    source_probability = np.where(
        hard_instance_id == int(actor_instance_id), o1_probability, unassigned
    )
    mask_evidence = np.asarray(semantic_sidecar["semantic_score"], dtype=np.float64)
    if not np.isfinite(mask_evidence).all() or np.any(
        (mask_evidence < 0.0) | (mask_evidence > 1.0)
    ):
        raise ValueError("semantic_score must lie in [0, 1]")
    mask_weight = float(mask_confidence_floor) + (
        1.0 - float(mask_confidence_floor)
    ) * np.abs(2.0 * mask_evidence - 1.0)
    visibility_weight = saturating_confidence(
        np.asarray(semantic_sidecar["visible_mass"], dtype=np.float64),
        scale=visibility_saturation_mass,
    )
    depth_rate = np.asarray(
        semantic_sidecar["depth_consistency_rate"], dtype=np.float64
    )
    if not np.isfinite(depth_rate).all() or np.any((depth_rate < 0.0) | (depth_rate > 1.0)):
        raise ValueError("depth_consistency_rate must lie in [0, 1]")
    depth_weight = float(depth_confidence_floor) + (
        1.0 - float(depth_confidence_floor)
    ) * depth_rate
    lidar_supported = np.asarray(semantic_sidecar["labels"], dtype=np.int8) == CORE_POSITIVE
    lidar_weight = np.where(lidar_supported, 1.0, float(lidar_confidence_floor))
    positive = np.asarray(semantic_sidecar["num_positive_views"], dtype=np.float64)
    negative = np.asarray(semantic_sidecar["num_negative_views"], dtype=np.float64)

    prior = prior_from_probability(source_probability, strength=prior_strength)
    factors = {
        "mask": mask_weight,
        "visibility": visibility_weight,
        "depth": depth_weight,
        "lidar": lidar_weight,
    }
    posterior = update_from_counts(
        prior,
        positive_count=positive,
        negative_count=negative,
        factors=factors,
    )
    joint_weight = mask_weight * visibility_weight * depth_weight * lidar_weight
    authenticity = np.maximum(
        lidar_supported.astype(np.float64),
        float(observed_authenticity) * visibility_weight * depth_weight,
    )
    state = {
        "gaussian_id": gaussian_id.copy(),
        "base_model": np.asarray(instance_field["base_model"], dtype=np.int8).copy(),
        "base_index": np.asarray(instance_field["base_index"], dtype=np.int64).copy(),
        "hard_instance_id": hard_instance_id.copy(),
        "actor_instance_id": np.asarray(int(actor_instance_id), dtype=np.int32),
        "actor_token": np.asarray(str(actor_token), dtype="<U64"),
        "alpha": posterior.alpha.astype(np.float32),
        "beta": posterior.beta.astype(np.float32),
        "posterior": posterior.posterior.astype(np.float32),
        "uncertainty": posterior.uncertainty.astype(np.float32),
        "authenticity": authenticity.astype(np.float32),
        "source_instance_opacity": source_probability.astype(np.float32),
        "mask_evidence": mask_evidence.astype(np.float32),
        "positive_count": positive.astype(np.float32),
        "negative_count": negative.astype(np.float32),
        "mask_weight": mask_weight.astype(np.float32),
        "visibility_weight": visibility_weight.astype(np.float32),
        "depth_weight": depth_weight.astype(np.float32),
        "lidar_weight": lidar_weight.astype(np.float32),
        "joint_weight": joint_weight.astype(np.float32),
    }
    validate_evidence_state(state)
    return state


def validate_evidence_state(state: Mapping[str, np.ndarray]) -> None:
    missing = REQUIRED_FIELDS - set(state)
    if missing:
        raise ValueError(f"evidence state is missing fields: {sorted(missing)}")
    gaussian_id = np.asarray(state["gaussian_id"], dtype=np.int64)
    total = gaussian_id.size
    if gaussian_id.ndim != 1 or not np.array_equal(gaussian_id, np.arange(total)):
        raise ValueError("gaussian_id must be a contiguous global index")
    for name in (IDENTITY_FIELDS | EVIDENCE_FIELDS) - {"gaussian_id"}:
        if np.asarray(state[name]).shape != (total,):
            raise ValueError(f"evidence state {name} shape is invalid")
    if np.asarray(state["actor_instance_id"]).shape != () or np.asarray(
        state["actor_token"]
    ).shape != ():
        raise ValueError("actor identity fields must be scalar")
    evidence = BetaEvidence(state["alpha"], state["beta"])
    np.testing.assert_allclose(state["posterior"], evidence.posterior, atol=2e-6)
    np.testing.assert_allclose(state["uncertainty"], evidence.uncertainty, atol=2e-6)
    for name in EVIDENCE_FIELDS - {"alpha", "beta", "positive_count", "negative_count"}:
        value = np.asarray(state[name], dtype=np.float64)
        if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"evidence state {name} must lie in [0, 1]")
    for name in ("positive_count", "negative_count"):
        value = np.asarray(state[name], dtype=np.float64)
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"evidence state {name} must be non-negative")


def atomic_save_evidence_state(
    path: str | Path, state: Mapping[str, np.ndarray]
) -> None:
    """Write byte-stable NPZ evidence without touching the RGB checkpoint."""

    validate_evidence_state(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(state):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(state[name]), allow_pickle=False
                )
                entry = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(
                    entry,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    os.replace(temporary, target)


def load_evidence_state(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        state = {name: arrays[name] for name in arrays.files}
    validate_evidence_state(state)
    return state


def evidence_state_summary(state: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_evidence_state(state)
    posterior = np.asarray(state["posterior"], dtype=np.float64)
    uncertainty = np.asarray(state["uncertainty"], dtype=np.float64)
    return {
        "gaussian_count": int(posterior.size),
        "actor_instance_id": int(np.asarray(state["actor_instance_id"]).item()),
        "actor_token": str(np.asarray(state["actor_token"]).item()),
        "posterior_mean": float(posterior.mean()),
        "posterior_above_half_count": int((posterior >= 0.5).sum()),
        "uncertainty_mean": float(uncertainty.mean()),
        "evidence_strength_mean": float(
            np.mean(np.asarray(state["alpha"]) + np.asarray(state["beta"]))
        ),
        "lidar_supported_count": int((np.asarray(state["lidar_weight"]) == 1.0).sum()),
    }
