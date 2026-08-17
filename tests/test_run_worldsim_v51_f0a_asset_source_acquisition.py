from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.protocol import ProtocolError
from scripts.run_worldsim_v51_f0a_asset_source_acquisition import (
    _validate_config,
    select_train_only_records,
)


def test_f0a_asset_source_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0a_asset_source_acquisition.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0a_asset_source_auditor_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_worldsim_v51_f0a_asset_source_acquisition.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


def test_f0a_asset_source_freeze_is_parseable_and_terminal() -> None:
    freeze = yaml.safe_load(
        (
            ROOT
            / "configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_freeze_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert freeze["status"] == "done"
    assert freeze["canonical_run"]["conclusion"] == (
        "f0a_assets_and_sources_frozen_environment_setup_required"
    )
    assert freeze["governance"]["materialization_authorized"] is False


def test_f0a_asset_source_formal_config_validates() -> None:
    config = _validate_config(
        ROOT / "configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_v1.yaml"
    )
    assert config["decision"]["materialization_authorized"] is False
    assert config["locks"]["validation_quality_read"] is False


def test_select_train_only_records_uses_scene_frame_camera_order() -> None:
    records = [
        {"scene": "b", "frame": 1, "camera": 0, "role": "h"},
        {"scene": "a", "frame": 1, "camera": 0, "role": "h"},
        {"scene": "a", "frame": 0, "camera": 0, "role": "h"},
        {"scene": "b", "frame": 0, "camera": 0, "role": "h"},
        {"scene": "a", "frame": 0, "camera": 0, "role": "evaluation"},
    ]
    selected = select_train_only_records(records, ["a", "b"], [0, 1], [0], "h")
    assert [(row["scene"], row["frame"], row["camera"]) for row in selected] == [
        ("a", 0, 0),
        ("a", 1, 0),
        ("b", 0, 0),
        ("b", 1, 0),
    ]


def test_select_train_only_records_fails_closed_on_missing_view() -> None:
    with pytest.raises(ProtocolError, match="denominator drift"):
        select_train_only_records(
            [{"scene": "a", "frame": 0, "camera": 0, "role": "h"}],
            ["a"],
            [0, 1],
            [0],
            "h",
        )
