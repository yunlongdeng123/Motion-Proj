from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0b_three_view_association_parity import (
    _arm_command,
    _parity_report,
    _validate_config,
)


def test_f0b_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0b_three_view_association_parity.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0b_formal_config_validates_and_locks_train_only_inputs() -> None:
    config = _validate_config(
        ROOT
        / "configs/worldsim_v51/stage_f_f0b_three_view_association_parity_v1.yaml"
    )
    assert config["input_provenance"]["split"] == "train_only"
    assert config["sources"]["deva"]["identity"] == (
        "contained_in_frozen_gaussian_grouping_tree"
    )
    assert [row["frame"] for row in config["inputs"]] == [0, 40, 80]
    assert config["locks"]["input_image_pixels_decoded_count"] == 9
    assert config["locks"]["quality_read"] is False
    assert config["decision"]["materialization_authorized"] is False


def test_f0b_arm_contract_changes_batch_only() -> None:
    config = _validate_config(
        ROOT
        / "configs/worldsim_v51/stage_f_f0b_three_view_association_parity_v1.yaml"
    )
    commands = [
        _arm_command(config, arm, Path("/input"), Path(f"/output/{arm['name']}"))
        for arm in config["execution"]["arms"]
    ]
    batch_index = commands[0].index("--SAM_NUM_POINTS_PER_BATCH") + 1
    side_index = commands[0].index("--SAM_NUM_POINTS_PER_SIDE") + 1
    assert [command[batch_index] for command in commands] == ["32", "16", "32"]
    assert [command[side_index] for command in commands] == ["32", "32", "32"]
    normalized = []
    for command in commands:
        row = list(command)
        row[row.index("--output") + 1] = "/output"
        normalized.append(row)
    assert normalized[0] == normalized[2]
    normalized[1][batch_index] = "32"
    assert normalized[0] == normalized[1]


def _arm(name: str, suffix: str = "") -> dict:
    return {
        "name": name,
        "masks": [{"sha256": f"mask-{index}{suffix}"} for index in range(3)],
        "metadata_sha256": f"metadata{suffix}",
        "nonzero_mask_count": 2,
        "stable_short_ids": [1],
        "positive_short_id_presence": {"1": 2},
    }


def test_f0b_parity_gate_requires_exact_batch_and_repeat_outputs() -> None:
    arms = [
        _arm("primary_batch32"),
        _arm("parity_batch16"),
        _arm("repeat_batch32"),
    ]
    decision = {"minimum_nonzero_masks": 1, "minimum_stable_short_id_frames": 2}
    assert _parity_report(arms, decision)["all_required"] is True
    arms[1] = _arm("parity_batch16", "-drift")
    report = _parity_report(arms, decision)
    assert report["batch_mask_exact"] is False
    assert report["batch_metadata_exact"] is False
    assert report["all_required"] is False
