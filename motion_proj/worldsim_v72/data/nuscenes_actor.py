"""Streaming nuScenes actor pose lookup for known SE(3) trajectory inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import ijson
import numpy as np

from motion_proj.worldsim_v72.data.nuscenes_camera import _transform


@dataclass(frozen=True)
class ActorPose:
    track_id: str
    annotation_id: str
    world_from_actor: np.ndarray
    size_wlh_m: np.ndarray


def load_actor_poses_at_sample(
    metadata_root: Path,
    sample_id: str,
    track_ids: Iterable[str],
) -> dict[str, ActorPose]:
    """Read only requested actor poses without materializing the 0.58 GB annotation JSON."""

    result = load_actor_poses_for_samples(metadata_root, {str(sample_id): track_ids})
    return {track_id: pose for (found_sample, track_id), pose in result.items() if found_sample == str(sample_id)}


def load_actor_poses_for_samples(
    metadata_root: Path,
    tracks_by_sample: Mapping[str, Iterable[str]],
) -> dict[tuple[str, str], ActorPose]:
    """Resolve many sample/track requests in one streaming pass over annotations."""

    requested = {
        str(sample_id): {str(value) for value in track_ids}
        for sample_id, track_ids in tracks_by_sample.items()
    }
    remaining = sum(len(track_ids) for track_ids in requested.values())
    found: dict[tuple[str, str], ActorPose] = {}
    annotation_path = Path(metadata_root) / "sample_annotation.json"
    with annotation_path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            sample = str(row["sample_token"])
            track_id = str(row["instance_token"])
            if sample not in requested or track_id not in requested[sample]:
                continue
            found[(sample, track_id)] = ActorPose(
                track_id=track_id,
                annotation_id=str(row["token"]),
                world_from_actor=_transform(row["translation"], row["rotation"]),
                size_wlh_m=np.asarray(row["size"], dtype=np.float32),
            )
            remaining -= 1
            if remaining == 0:
                break
    return found
