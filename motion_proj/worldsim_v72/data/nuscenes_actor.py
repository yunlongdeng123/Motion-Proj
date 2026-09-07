"""Streaming nuScenes actor pose lookup for known SE(3) trajectory inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

    requested = {str(value) for value in track_ids}
    found: dict[str, ActorPose] = {}
    annotation_path = Path(metadata_root) / "sample_annotation.json"
    with annotation_path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            track_id = str(row["instance_token"])
            if str(row["sample_token"]) != str(sample_id) or track_id not in requested:
                continue
            found[track_id] = ActorPose(
                track_id=track_id,
                annotation_id=str(row["token"]),
                world_from_actor=_transform(row["translation"], row["rotation"]),
                size_wlh_m=np.asarray(row["size"], dtype=np.float32),
            )
            if len(found) == len(requested):
                break
    return found
