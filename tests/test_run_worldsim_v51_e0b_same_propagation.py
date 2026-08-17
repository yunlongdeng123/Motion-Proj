from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/worldsim_v51/stage_e_e0b_same_propagation_v1.yaml"


def test_e0b_runner_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_e0b_same_propagation.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_e0b_config_binds_level_same_propagation_and_locks() -> None:
    from scripts.run_worldsim_v51_e0b_same_propagation import validate_config

    config, e0a, d0 = validate_config(CONFIG)
    assert config["level_selection"]["selected_level"] == "fine_q50"
    assert config["level_selection"]["quality_metric_consumed"] is False
    assert config["method"]["progressive_thresholds"] == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert config["future_h_gate_preregistration_boundary"]["candidate"] == "E0B"
    assert e0a["gate"]["pass"] is True
    assert d0["status"] == "done"
    assert config["locks"]["h_quality_read"] is False
    assert config["locks"]["e1_panogs_execution"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
