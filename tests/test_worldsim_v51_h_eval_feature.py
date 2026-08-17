from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_h_eval_feature import _transform, validate_config
from motion_proj.worldsim_v51.feature_sidecar import array_sha256
from motion_proj.worldsim_v51.protocol import sha256_file


CONFIG = ROOT / "configs/worldsim_v51/stage_b_h_eval_feature_v1.yaml"


def test_frozen_transform_matches_explicit_standardize_and_project() -> None:
    raw = np.asarray(
        [
            [3.0, 2.0, 5.0, -1.0],
            [5.0, 4.0, 1.0, 3.0],
            [7.0, 6.0, -3.0, 7.0],
            [9.0, 8.0, -7.0, 11.0],
            [11.0, 10.0, -11.0, 15.0],
            [13.0, 12.0, -15.0, 19.0],
        ],
        dtype=np.float32,
    )
    original = raw.copy()
    state = {
        "feature_mean": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "feature_std": np.asarray([2.0, 4.0, 2.0, 8.0], dtype=np.float64),
        "pca_mean": np.asarray([0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        "components": np.asarray(
            [[1.0, 0.0, 0.5, 0.0], [0.0, -1.0, 0.0, 0.25]],
            dtype=np.float32,
        ),
    }
    config = {"views": {"patch_grid_hw": [2, 3]}}

    observed = _transform(raw, state, config)
    standardized = (
        (raw.astype(np.float64) - state["feature_mean"]) / state["feature_std"]
    ).astype(np.float32)
    expected_rows = (standardized - state["pca_mean"]) @ state["components"].T
    expected = expected_rows.reshape(2, 3, 2).transpose(2, 0, 1)

    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(raw, original)
    assert observed.shape == (2, 2, 3)
    assert observed.dtype == np.float32


def test_h_eval_feature_config_binds_exact_heldout_contract_and_locks() -> None:
    config, records, manifest, state = validate_config(CONFIG)

    assert manifest["record_count"] == 240
    assert len(records) == 45
    assert {
        (str(row["scene"]), int(row["frame"]), int(row["camera"]))
        for row in records
    } == {
        (scene, frame, camera)
        for scene in ("scene-0471", "scene-1087", "scene-0379")
        for frame in (2, 42, 82, 122, 162)
        for camera in (0, 1, 2)
    }
    assert all(int(row["frame"]) % 5 == 2 for row in records)
    assert all(int(row["frame"]) % 5 != 4 for row in records)
    assert config["pca_transform"]["fit"] is False
    assert state["components"].shape == (40, 1536)
    assert config["locks"]["h_heldout_pixels_read_for_feature_extraction"] is True
    assert config["locks"]["membership_proxy_read"] is False
    assert config["locks"]["renderer_start"] is False
    assert config["locks"]["uplift_feature_read"] is False
    assert config["locks"]["method_quality_read"] is False
    assert config["locks"]["screening_quality_read"] is False
    assert config["locks"]["confirmation_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"


def test_h_eval_feature_freeze_binds_terminal_manifest_and_sidecars() -> None:
    freeze = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/stage_b_h_eval_feature_freeze_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    run = Path(freeze["canonical_run"]["path"])
    for relative, expected in freeze["canonical_run"]["hashes"].items():
        assert (run / relative).is_file()
        assert sha256_file(run / relative) == expected

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["report"]["first_raw_repeat_bit_exact"] is True
    assert summary["report"]["first_transform_repeat_bit_exact"] is True
    assert summary["report"]["pca_fit"] is False
    manifest = json.loads(
        (run / "artifacts/h_eval_feature_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["records"]) == 45
    for record in manifest["records"]:
        path = run / record["path"]
        assert sha256_file(path) == record["file_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["feature"]
            feature = np.asarray(archive["feature"])
            assert feature.shape == (40, 64, 114)
            assert feature.dtype == np.float32
            assert np.isfinite(feature).all()
            assert array_sha256(feature) == record["content_sha256"]
    assert freeze["locks"]["membership_proxy_read"] is False
    assert freeze["locks"]["renderer_started"] is False
    assert freeze["locks"]["uplift_feature_read"] is False
    assert freeze["locks"]["method_quality_read"] is False
    assert freeze["locks"]["validation_quality_read"] is False
    assert freeze["locks"]["test_quality_read"] is False
    assert freeze["locks"]["m2_status"] == "pending"
    assert freeze["locks"]["m3_status"] == "pending"
