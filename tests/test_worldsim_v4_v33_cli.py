from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "build_worldsim_v4_v33_actor_registry.py",
    "build_worldsim_v4_v33_registration.py",
    "finalize_worldsim_v4_v33_scene_chain.py",
    "materialize_worldsim_v4_v33_instance_config.py",
    "materialize_worldsim_v4_v33_replay.py",
    "materialize_worldsim_v4_v33_semantic_config.py",
    "materialize_worldsim_v4_v33_spatial_config.py",
    "run_worldsim_v4_v33_instance_field.py",
    "run_worldsim_v4_v33_abstain_scene.py",
    "run_worldsim_v4_v33_semantic_lift.py",
    "run_worldsim_v4_v33_spatial_delta.py",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_cli_help_resolves_project_package_without_pythonpath(entrypoint: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / entrypoint), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
