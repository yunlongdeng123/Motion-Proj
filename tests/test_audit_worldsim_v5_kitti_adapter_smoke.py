from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.audit_worldsim_v5_kitti_adapter_smoke import (
    KittiAdapterSmokeError,
    frame_indices,
    image_probe,
    lidar_probe,
    stereo_baseline_m,
)


def _calibration() -> dict[str, np.ndarray]:
    return {
        "P2": np.asarray(
            [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0, 0, 1, 0]],
            dtype=np.float64,
        ),
        "P3": np.asarray(
            [
                [100.0, 0.0, 50.0, -54.0],
                [0.0, 100.0, 40.0, 0.0],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        ),
        "R_rect": np.eye(4),
        "T_velo_cam": np.eye(4),
    }


def test_frame_identity_image_and_lidar_decode_probes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for frame in (0, 1, 2):
        Image.new("RGB", (8, 4), color=(frame, 0, 0)).save(
            image_dir / f"{frame:06d}.png"
        )
    assert frame_indices(image_dir, ".png") == [0, 1, 2]
    rows = image_probe([image_dir / "000000.png", image_dir / "000002.png"])
    assert {(row["width"], row["height"]) for row in rows} == {(8, 4)}

    lidar = tmp_path / "000000.bin"
    np.asarray([[1.0, 2.0, 5.0, 0.5], [-1.0, 0.0, -2.0, 0.1]], np.float32).tofile(
        lidar
    )
    probes = lidar_probe([lidar], _calibration())
    assert probes[0]["point_count"] == 2
    assert probes[0]["front_projectable_p2"] == 1
    assert probes[0]["finite_projected_pixels"] is True


def test_stereo_baseline_gate_is_metric_and_fail_closed() -> None:
    assert stereo_baseline_m(_calibration()) == pytest.approx(0.54)
    calibration = _calibration()
    calibration["P3"][0, 3] = -1.0
    with pytest.raises(KittiAdapterSmokeError, match="baseline"):
        stereo_baseline_m(calibration)
