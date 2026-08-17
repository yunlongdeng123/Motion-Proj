from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.feature_sidecar import (
    array_sha256,
    feature_mean_std_correction1,
    pca_patch_grid,
    record_chain_sha256,
    select_h_uplift_records,
    sidecar_relative_path,
    standardize_in_place,
    validate_sidecar_identity,
)
from scripts.run_worldsim_v51_h_feature_pca import validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_b_h_feature_pca_v1.yaml"


def _records() -> list[dict[str, object]]:
    return [
        {
            "role": "historical_diagnostic",
            "scene": scene,
            "scene_index": scene_index,
            "frame": frame,
            "camera": camera,
            "width": 1600,
            "height": 900,
            "path": f"/{scene}/{frame:03d}_{camera}.jpg",
            "bytes": 1,
            "sha256": f"{scene}-{frame}-{camera}",
        }
        for scene, scene_index in (("scene-b", 2), ("scene-a", 1))
        for frame in (40, 0)
        for camera in (1, 0)
    ]


def test_select_h_records_uses_frozen_scene_frame_camera_order() -> None:
    selected = select_h_uplift_records(
        {"records": list(reversed(_records()))},
        scenes=["scene-a", "scene-b"],
        frames=[0, 40],
        cameras=[0, 1],
    )
    assert [
        (row["scene"], row["frame"], row["camera"]) for row in selected
    ] == [
        (scene, frame, camera)
        for scene in ("scene-a", "scene-b")
        for frame in (0, 40)
        for camera in (0, 1)
    ]
    assert record_chain_sha256(selected) == record_chain_sha256(selected)


def test_select_h_records_rejects_missing_grid_cell() -> None:
    with pytest.raises(ValueError, match="不完整"):
        select_h_uplift_records(
            {"records": _records()[:-1]},
            scenes=["scene-a", "scene-b"],
            frames=[0, 40],
            cameras=[0, 1],
        )


def test_correction1_statistics_and_in_place_standardization() -> None:
    source = np.asarray(
        [[1.0, 7.0, -2.0], [2.0, 5.0, 0.0], [4.0, 3.0, 2.0], [8.0, 1.0, 4.0]],
        dtype=np.float32,
    )
    mean, std = feature_mean_std_correction1(source, chunk_rows=2)
    np.testing.assert_allclose(mean, source.mean(axis=0, dtype=np.float64), rtol=0, atol=0)
    np.testing.assert_allclose(std, source.std(axis=0, ddof=1, dtype=np.float64))
    target = source.copy()
    standardize_in_place(target, mean=mean, std=std, chunk_rows=2)
    np.testing.assert_allclose(target.mean(axis=0), np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(target.std(axis=0, ddof=1), np.ones(3), atol=1e-7)


def test_pca_grid_shape_dtype_and_identity() -> None:
    rows = np.arange(24, dtype=np.float32).reshape(6, 4)
    components = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    grid = pca_patch_grid(
        rows,
        pca_mean=np.ones(4, dtype=np.float32),
        components=components,
        grid_hw=[2, 3],
    )
    assert grid.shape == (2, 2, 3)
    assert grid.dtype == np.float32
    record = {
        "shape": [2, 2, 3],
        "dtype": "float32",
        "content_sha256": array_sha256(grid),
    }
    validate_sidecar_identity(record, grid)
    assert sidecar_relative_path(
        {"scene": "scene-0471", "frame": 40, "camera": 2}
    ).as_posix() == "artifacts/features/scene-0471/040_2.npz"


def test_h_feature_pca_config_binds_exact_45_view_contract_and_locks() -> None:
    config, records, manifest = validate_config(CONFIG)
    assert len(records) == 45
    assert manifest["record_count"] == 240
    assert config["views"]["expected_patch_count"] == 328320
    assert config["views"]["expected_raw_memmap_bytes"] == 2017198080
    assert config["pca"]["standardization_std_correction"] == 1
    assert config["pca"]["random_state"] == 20260814
    assert config["pca"]["subsampling_applied"] is False
    assert config["model"]["repeat_first_image_exact"] is True
    assert config["locks"]["membership_proxy_read"] is False
    assert config["locks"]["method_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
