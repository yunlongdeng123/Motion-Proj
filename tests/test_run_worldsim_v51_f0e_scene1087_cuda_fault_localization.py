from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0b_three_view_association_parity import _arm_command
from scripts.run_worldsim_v51_f0e_scene1087_cuda_fault_localization import (
    _outcome,
    _validate_config,
)
from scripts.audit_worldsim_v51_f0e_scene1087_cuda_fault_localization import audit


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_v1.yaml"


def test_f0e_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0e_scene1087_cuda_fault_localization.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0e_config_locks_exact_cross_camera_replays() -> None:
    config = _validate_config(CONFIG)
    assert [(row["frame"], row["camera"]) for row in config["inputs"]] == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]
    assert config["environment"]["CUDA_LAUNCH_BLOCKING"] == "1"
    assert [row["sam_num_points_per_batch"] for row in config["execution"]["attempts"]] == [64, 64]
    assert config["locks"]["smaller_batch_retry"] is False
    assert config["locks"]["quality_read"] is False


def test_f0e_official_cli_keeps_frozen_method_parameters() -> None:
    config = _validate_config(CONFIG)
    for attempt in config["execution"]["attempts"]:
        command = _arm_command(config, attempt, Path("/input"), Path("/output"))
        assert command[command.index("--SAM_NUM_POINTS_PER_SIDE") + 1] == "32"
        assert command[command.index("--SAM_NUM_POINTS_PER_BATCH") + 1] == "64"
        assert command[command.index("--size") + 1] == "480"


def test_f0e_outcome_matrix_distinguishes_repeatability_boundaries() -> None:
    failed = {"classification": "expected_cublas_internal_failure"}
    assert _outcome([failed, failed])[0] == "both_expected_failure"
    succeeded = {
        "classification": "success",
        "masks": [{"sha256": "a"}, {"sha256": "b"}, {"sha256": "c"}],
        "metadata": {"sha256": "m"},
    }
    assert _outcome([succeeded, succeeded])[0] == "both_success_exact"
    assert _outcome([succeeded, failed])[0] == "mixed"
    drifted = {
        "classification": "success",
        "masks": [{"sha256": "x"}, {"sha256": "b"}, {"sha256": "c"}],
        "metadata": {"sha256": "m"},
    }
    assert _outcome([succeeded, drifted])[0] == "success_nonexact"


def test_f0e_r036_independent_audit_replays_mixed_outcome() -> None:
    result = audit(
        CONFIG,
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T130000Z__m1-stage-f-f0e-cuda-localization-s20260814-r036"
        ),
    )
    assert result["status"] == "pass"
    assert result["audited_run_status"] == "done"
    assert result["outcome"] == "mixed"
    assert [row["classification"] for row in result["attempts"]] == [
        "expected_cublas_internal_failure",
        "success",
    ]
    assert result["quality_read"] is False


def test_f0e_freeze_keeps_mixed_outcome_out_of_materialization() -> None:
    freeze = yaml.safe_load(
        (
            ROOT
            / "configs/worldsim_v51/"
            "stage_f_f0e_scene1087_cuda_fault_localization_freeze_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert freeze["status"] == "done"
    assert freeze["canonical_run"]["outcome"] == "mixed"
    assert freeze["independent_audit"]["status"] == "pass"
    assert freeze["interpretation"]["failure"] == "V51-F62"
    assert freeze["governance"]["full_materialization_authorized"] is False
    assert freeze["governance"]["smaller_batch_retry"] is False
    assert freeze["governance"]["next_phase"] == (
        "cuda_runtime_health_and_reproducibility_gate_preregistration"
    )
