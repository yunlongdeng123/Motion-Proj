from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0f_cuda_runtime_health_reproducibility import (
    _outcome,
    _validate_config,
)
from scripts.audit_worldsim_v51_f0f_cuda_runtime_health_reproducibility import audit


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_v1.yaml"


def test_f0f_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_worldsim_v51_f0f_cuda_runtime_health_reproducibility.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0f_config_locks_abab_control_target_contract() -> None:
    config = _validate_config(CONFIG)
    attempts = config["execution"]["attempts"]
    assert [row["name"] for row in attempts] == [
        "control_replay_1",
        "target_replay_1",
        "control_replay_2",
        "target_replay_2",
    ]
    assert [row["sam_num_points_per_batch"] for row in attempts] == [64] * 4
    assert config["environment"]["CUDA_LAUNCH_BLOCKING"] == "1"
    assert config["locks"]["gpu_reset"] is False
    assert config["locks"]["quality_read"] is False


def _success(suffix: str = "") -> dict:
    return {
        "classification": "success",
        "masks": [{"sha256": f"a{suffix}"}, {"sha256": f"b{suffix}"}, {"sha256": f"c{suffix}"}],
        "metadata": {"sha256": f"m{suffix}"},
    }


def test_f0f_outcome_matrix_separates_control_and_target_instability() -> None:
    success = _success()
    failed = {"classification": "expected_cublas_internal_failure"}
    assert _outcome([success, success, success, success])[0] == "all_success_exact"
    assert _outcome([success, failed, success, success])[0] == "control_stable_target_failure"
    assert _outcome([failed, success, success, success])[0] == "control_failure"
    assert _outcome([success, success, success, _success("-drift")])[0] == "success_nonexact"


def test_f0f_r037_independent_audit_replays_control_target_boundary() -> None:
    result = audit(
        CONFIG,
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T140000Z__m1-stage-f-f0f-runtime-repro-s20260814-r037"
        ),
    )
    assert result["status"] == "pass"
    assert result["outcome"] == "control_stable_target_failure"
    assert result["pair_checks"] == {
        "control_both_success": True,
        "control_exact": True,
        "target_both_success": False,
        "target_exact": False,
    }
    assert [row["classification"] for row in result["attempts"]] == [
        "success",
        "expected_cublas_internal_failure",
        "success",
        "expected_cublas_internal_failure",
    ]
    assert result["quality_read"] is False


def test_f0f_freeze_keeps_target_instability_out_of_materialization() -> None:
    freeze = yaml.safe_load(
        (
            ROOT
            / "configs/worldsim_v51/"
            "stage_f_f0f_cuda_runtime_health_reproducibility_freeze_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert freeze["status"] == "done"
    assert freeze["canonical_run"]["outcome"] == "control_stable_target_failure"
    assert freeze["control"]["exact_pair"] is True
    assert freeze["target"]["classifications"] == [
        "expected_cublas_internal_failure",
        "expected_cublas_internal_failure",
    ]
    assert freeze["interpretation"]["failure"] == "V51-F62"
    assert freeze["governance"]["full_materialization_authorized"] is False
    assert freeze["governance"]["next_phase"] == (
        "source_neutral_target_tensor_allocator_instrumentation_preregistration"
    )
