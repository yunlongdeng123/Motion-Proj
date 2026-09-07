"""Target-free Actor candidate visual feature cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ACTOR_VISUAL_CACHE_SCHEMA_VERSION = "worldsim_v72.actor_visual_cache.v1"


@dataclass(frozen=True)
class ActorVisualCache:
    track_id: str
    scene_id: str
    window_id: str
    backbone_id: str
    candidates_actor_m: np.ndarray
    base_features: np.ndarray
    input_evidence_fou: np.ndarray
    opportunity_count: np.ndarray
    visual_features: np.ndarray
    confidence_sum: np.ndarray
    observation_count: np.ndarray
    world_from_actor: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        candidates = np.asarray(self.candidates_actor_m, dtype=np.float32).reshape(-1, 3)
        count = len(candidates)
        base = np.asarray(self.base_features, dtype=np.float32)
        evidence = np.asarray(self.input_evidence_fou, dtype=np.float32)
        opportunity = np.asarray(self.opportunity_count, dtype=np.int32).reshape(-1)
        visual = np.asarray(self.visual_features, dtype=np.float32)
        confidence = np.asarray(self.confidence_sum, dtype=np.float32).reshape(-1)
        observations = np.asarray(self.observation_count, dtype=np.int32).reshape(-1)
        transform = np.asarray(self.world_from_actor, dtype=np.float64)
        if base.ndim != 2 or visual.ndim != 2 or base.shape[0] != count or visual.shape[0] != count:
            raise ValueError("Actor feature row count does not match candidates")
        if evidence.shape != (count, 3):
            raise ValueError("input_evidence_fou must have shape (N,3)")
        if any(len(value) != count for value in (opportunity, confidence, observations)):
            raise ValueError("Actor candidate scalar row count mismatch")
        if transform.shape != (4, 4):
            raise ValueError("world_from_actor must have shape (4,4)")
        if np.any(confidence < 0.0) or np.any(observations < 0) or np.any(opportunity < 0):
            raise ValueError("confidence/count/opportunity must be non-negative")
        object.__setattr__(self, "candidates_actor_m", candidates)
        object.__setattr__(self, "base_features", base)
        object.__setattr__(self, "input_evidence_fou", evidence)
        object.__setattr__(self, "opportunity_count", opportunity)
        object.__setattr__(self, "visual_features", visual)
        object.__setattr__(self, "confidence_sum", confidence)
        object.__setattr__(self, "observation_count", observations)
        object.__setattr__(self, "world_from_actor", transform)
        object.__setattr__(self, "provenance", dict(self.provenance))


_ARRAY_FIELDS = (
    "candidates_actor_m",
    "base_features",
    "input_evidence_fou",
    "opportunity_count",
    "visual_features",
    "confidence_sum",
    "observation_count",
    "world_from_actor",
)


def save_actor_visual_cache(value: ActorVisualCache, output_path: Path) -> tuple[Path, Path]:
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("Actor visual cache must use .npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output_path.with_suffix(".json")
    temporary_npz = output_path.with_suffix(".tmp.npz")
    temporary_json = sidecar.with_suffix(".tmp.json")
    np.savez_compressed(
        temporary_npz,
        schema_version=np.asarray(ACTOR_VISUAL_CACHE_SCHEMA_VERSION),
        **{name: getattr(value, name) for name in _ARRAY_FIELDS},
    )
    metadata = {
        "schema_version": ACTOR_VISUAL_CACHE_SCHEMA_VERSION,
        "track_id": value.track_id,
        "scene_id": value.scene_id,
        "window_id": value.window_id,
        "backbone_id": value.backbone_id,
        "provenance": dict(value.provenance),
    }
    temporary_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_npz.replace(output_path)
    temporary_json.replace(sidecar)
    return output_path, sidecar


def load_actor_visual_cache(input_path: Path) -> ActorVisualCache:
    input_path = Path(input_path)
    metadata = json.loads(input_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ACTOR_VISUAL_CACHE_SCHEMA_VERSION:
        raise ValueError("Actor visual cache JSON version mismatch")
    with np.load(input_path, allow_pickle=False) as payload:
        if str(payload["schema_version"]) != ACTOR_VISUAL_CACHE_SCHEMA_VERSION:
            raise ValueError("Actor visual cache NPZ version mismatch")
        arrays = {name: payload[name] for name in _ARRAY_FIELDS}
    return ActorVisualCache(
        track_id=str(metadata["track_id"]),
        scene_id=str(metadata["scene_id"]),
        window_id=str(metadata["window_id"]),
        backbone_id=str(metadata["backbone_id"]),
        provenance=dict(metadata["provenance"]),
        **arrays,
    )
