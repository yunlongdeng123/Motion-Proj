"""Appearance-owned RGB observations, separate from physical/evidence caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


APPEARANCE_CACHE_SCHEMA_VERSION = "worldsim_v72.appearance_cache.v1"


@dataclass(frozen=True)
class ActorAppearanceCache:
    track_id: str
    scene_id: str
    window_id: str
    backbone_id: str
    rgb: np.ndarray
    confidence_sum: np.ndarray
    observation_count: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.float32).reshape(-1, 3)
        confidence = np.asarray(self.confidence_sum, dtype=np.float32).reshape(-1)
        count = np.asarray(self.observation_count, dtype=np.int32).reshape(-1)
        if len(rgb) != len(confidence) or len(rgb) != len(count):
            raise ValueError("appearance cache row mismatch")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "confidence_sum", confidence)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(self, "provenance", dict(self.provenance))


def save_actor_appearance_cache(value: ActorAppearanceCache, output_path: Path) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output_path.with_suffix(".json")
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(APPEARANCE_CACHE_SCHEMA_VERSION),
        rgb=value.rgb,
        confidence_sum=value.confidence_sum,
        observation_count=value.observation_count,
    )
    metadata = {
        "schema_version": APPEARANCE_CACHE_SCHEMA_VERSION,
        "track_id": value.track_id,
        "scene_id": value.scene_id,
        "window_id": value.window_id,
        "backbone_id": value.backbone_id,
        "provenance": dict(value.provenance),
    }
    temporary_json = sidecar.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    temporary_json.replace(sidecar)
    return output_path, sidecar


def load_actor_appearance_cache(input_path: Path) -> ActorAppearanceCache:
    input_path = Path(input_path)
    metadata = json.loads(input_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != APPEARANCE_CACHE_SCHEMA_VERSION:
        raise ValueError("appearance cache version mismatch")
    with np.load(input_path, allow_pickle=False) as payload:
        if str(payload["schema_version"]) != APPEARANCE_CACHE_SCHEMA_VERSION:
            raise ValueError("appearance NPZ version mismatch")
        arrays = {name: payload[name] for name in ("rgb", "confidence_sum", "observation_count")}
    return ActorAppearanceCache(
        track_id=str(metadata["track_id"]),
        scene_id=str(metadata["scene_id"]),
        window_id=str(metadata["window_id"]),
        backbone_id=str(metadata["backbone_id"]),
        provenance=dict(metadata["provenance"]),
        **arrays,
    )
