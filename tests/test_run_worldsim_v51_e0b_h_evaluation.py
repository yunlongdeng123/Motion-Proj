from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/worldsim_v51/stage_e_e0b_h_evaluation_v1.yaml"


def test_e0b_h_runner_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_e0b_h_evaluation.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout
    assert "--device" in result.stdout


def test_e0b_h_config_binds_three_arms_two_gates_and_locks() -> None:
    from scripts.run_worldsim_v51_e0b_h_evaluation import validate_config

    config, operator, d0_manifest, _ = validate_config(CONFIG)
    assert operator["status"] == "done"
    assert d0_manifest["status"] == "rejected"
    assert config["arms"]["primary_comparator"] == "U2_B3_G0"
    assert config["arms"]["mechanism_comparator"] == "D0"
    assert config["arms"]["candidate"] == "E0B"
    assert config["evaluation"]["expected_total_view_count"] == 12
    assert config["decision"]["pass_requires"] == "primary_and_mechanism_gates"
    assert config["locks"]["parameter_search"] is False
    assert config["locks"]["e1_panogs_execution"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
