from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0b_three_view_association_parity import _arm_command
from scripts.run_worldsim_v51_f0c_upstream_batch_association_repeatability import (
    _repeatability_report,
    _validate_config,
)
from scripts.audit_worldsim_v51_f0c_upstream_batch_association_repeatability import audit


CONFIG = (
    ROOT
    / "configs/worldsim_v51/stage_f_f0c_upstream_batch_association_repeatability_v1.yaml"
)


def test_f0c_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0c_upstream_batch_association_repeatability.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0c_auditor_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_worldsim_v51_f0c_upstream_batch_association_repeatability.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


def test_f0c_formal_config_restores_upstream_batch_and_locks_headroom() -> None:
    config = _validate_config(CONFIG)
    assert [row["sam_num_points_per_batch"] for row in config["execution"]["arms"]] == [64, 64]
    assert config["execution"]["interpretation"]["upstream_default_batch_restored"] is True
    assert config["resources"]["required_nvidia_total_mib"] == 24576
    assert config["resources"]["required_nvidia_headroom_mib"] == 256
    assert config["resources"]["maximum_nvidia_peak_mib"] == 24320
    assert config["locks"]["smaller_batch_retry"] is False


def test_f0c_official_cli_uses_grid32_batch64_in_both_arms() -> None:
    config = _validate_config(CONFIG)
    for arm in config["execution"]["arms"]:
        command = _arm_command(config, arm, Path("/input"), Path("/output"))
        assert command[command.index("--SAM_NUM_POINTS_PER_SIDE") + 1] == "32"
        assert command[command.index("--SAM_NUM_POINTS_PER_BATCH") + 1] == "64"


def _arm(name: str, suffix: str = "") -> dict:
    return {
        "name": name,
        "masks": [{"sha256": f"mask-{index}{suffix}"} for index in range(3)],
        "metadata_sha256": f"metadata{suffix}",
        "nonzero_mask_count": 3,
        "positive_short_id_presence": {"7": 3},
    }


def test_f0c_repeatability_gate_requires_exact_outputs_and_association() -> None:
    decision = {"minimum_nonzero_masks": 1, "minimum_stable_short_id_frames": 2}
    passing = _repeatability_report(
        [_arm("primary_batch64"), _arm("repeat_batch64")], decision
    )
    assert passing["all_required_before_resource"] is True
    failing = _repeatability_report(
        [_arm("primary_batch64"), _arm("repeat_batch64", "-drift")], decision
    )
    assert failing["all_masks_exact"] is False
    assert failing["metadata_exact"] is False
    assert failing["all_required_before_resource"] is False


def test_f0c_r034_independent_audit_passes() -> None:
    result = audit(
        CONFIG,
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T110000Z__m1-stage-f-f0c-upstream-batch-s20260814-r034"
        ),
    )
    assert result["status"] == "pass"
    assert result["audited_run_status"] == "done"
    assert result["repeatability"]["all_required_before_resource"] is True
    assert result["resources"]["nvidia_headroom_mib"] >= 256
    assert result["quality_read"] is False
