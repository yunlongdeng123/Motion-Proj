from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v51.protocol import sha256_file
from scripts.run_worldsim_v51_d0_progressive_operator import (
    _view_names,
    load_observation_matrices,
    validate_config,
)


CONFIG = ROOT / "configs/worldsim_v51/stage_d_progressive_operator_v1.yaml"
FREEZE = ROOT / "configs/worldsim_v51/stage_d_progressive_operator_freeze_v1.yaml"


def test_d0_operator_config_preserves_method_and_all_quality_locks() -> None:
    config, freeze = validate_config(CONFIG)
    assert freeze["status"] == "done"
    assert config["method"]["invariant_baseline"] == "U2_B3"
    assert config["method"]["progressive_thresholds"] == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert config["method"]["maximum_logical_distance"] == 2
    assert config["method"]["parameter_search"] is False
    assert config["locks"]["parse_v5_quality_diagnostics"] is False
    assert config["locks"]["render_evaluation_views"] is False
    assert config["locks"]["h_quality_read"] is False
    assert config["locks"]["screening_quality_read"] is False
    assert config["locks"]["confirmation_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
    assert len(_view_names(config)) == 15


def test_observation_loader_uses_manifest_validity_and_neutral_fill(tmp_path: Path) -> None:
    run = tmp_path / "run"
    observation = run / "artifacts/observations/f000_c0.npz"
    atomic_save_npz(
        observation,
        {
            "gaussian_id": np.asarray([0, 2, 3]),
            "mask_quality_accepted": np.asarray([1, 1, 0], dtype=np.int8),
            "reliability": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "sam_probability": np.asarray([0.8, 0.2, 0.9], dtype=np.float32),
            "sam_probability_available": np.asarray([1, 1, 1], dtype=np.int8),
            "visibility": np.asarray([0.5, 0.7, 0.9], dtype=np.float32),
        },
    )
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "inventory": [
                    {
                        "path": "artifacts/observations/f000_c0.npz",
                        "bytes": observation.stat().st_size,
                        "sha256": sha256_file(observation),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    probability, visibility, reports = load_observation_matrices(
        unary_run=run,
        unary_manifest_path=manifest,
        unary_manifest_sha256=sha256_file(manifest),
        gaussian_count=4,
        view_names=["f000_c0.npz"],
    )
    np.testing.assert_allclose(probability[:, 0], [0.8, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(visibility[:, 0], [0.5, 0.0, 0.0, 0.0])
    assert reports[0]["row_count"] == 3
    assert reports[0]["valid_row_count"] == 1


def test_d0_operator_config_rejects_future_quality_unlock(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["locks"]["validation_quality_read"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        validate_config(path)
    except Exception as error:
        assert "quality lock" in str(error) or "lock drift" in str(error)
    else:
        raise AssertionError("validation quality unlock must fail closed")


def test_d0_operator_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_worldsim_v51_d0_progressive_operator.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_d0_operator_freeze_binds_audited_no_quality_run() -> None:
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "done"
    assert freeze["canonical_run"]["status"] == "done"
    run = Path(freeze["canonical_run"]["path"])
    for relative, expected in freeze["canonical_run"]["hashes"].items():
        assert (run / relative).is_file()
        assert sha256_file(run / relative) == expected
    assert freeze["repeatability_probe"]["byte_exact"] is True
    assert [scene["scene"] for scene in freeze["scenes"]] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    for scene in freeze["scenes"]:
        assert scene["final_background_count"] + scene["final_actor_count"] + scene[
            "final_unknown_count"
        ] == scene["gaussian_count"]
    assert freeze["locks"]["quality_read"] is False
    assert freeze["locks"]["h_quality_read"] is False
    assert freeze["locks"]["screening_quality_read"] is False
    assert freeze["locks"]["validation_quality_read"] is False
    assert freeze["locks"]["test_quality_read"] is False
    assert freeze["locks"]["m2_status"] == "pending"
    assert freeze["locks"]["m3_status"] == "pending"
