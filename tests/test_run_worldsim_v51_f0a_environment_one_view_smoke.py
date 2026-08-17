from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import (
    _validate_config,
)


def test_f0a_environment_smoke_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0a_environment_one_view_smoke.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0a_environment_smoke_formal_config_validates() -> None:
    config = _validate_config(
        ROOT / "configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_v2.yaml"
    )
    assert config["one_view"]["interpretation"]["association_capability_claim"] is False
    assert config["decision"]["materialization_authorized"] is False
