from __future__ import annotations

from pathlib import Path

import pytest

from scripts.materialize_worldsim_v4_m2_scene_config import M2MaterializationError
from scripts.materialize_worldsim_v4_m2_validation_scene_config import _request_row


def _row(tmp_path: Path, *, frame: int) -> dict:
    mask = tmp_path / "mask.npz"
    image = tmp_path / "image.jpg"
    mask.write_bytes(b"mask")
    image.write_bytes(b"image")
    from scripts.materialize_worldsim_v4_m2_scene_config import sha256_file

    return {
        "role": "high_support",
        "frame": frame,
        "camera_id": 2,
        "positive_pixels": 10,
        "mask": str(mask),
        "mask_sha256": sha256_file(mask),
        "source_image": str(image),
        "source_image_sha256": sha256_file(image),
    }


def test_validation_request_uses_only_train_supports(tmp_path: Path) -> None:
    result = _request_row(
        scene="scene-val",
        role="high_support",
        actor={"dataset_instance_id": 9},
        row=_row(tmp_path, frame=12),
        protocol={
            "target_remainder": 2,
            "heldout_remainder": 4,
            "support_offsets": [-1, 1],
        },
    )
    assert result["support_views"] == [[11, 2], [13, 2]]
    assert all(frame % 5 in {1, 3} for frame, _ in result["support_views"])


def test_validation_request_rejects_nonfrozen_target_remainder(tmp_path: Path) -> None:
    with pytest.raises(M2MaterializationError, match="target remainder drift"):
        _request_row(
            scene="scene-val",
            role="high_support",
            actor={"dataset_instance_id": 9},
            row=_row(tmp_path, frame=14),
            protocol={
                "target_remainder": 2,
                "heldout_remainder": 4,
                "support_offsets": [-1, 1],
            },
        )
