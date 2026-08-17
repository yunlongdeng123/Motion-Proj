from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0f_cuda_runtime_health_reproducibility import (
    _outcome,
    _validate_config,
)


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
