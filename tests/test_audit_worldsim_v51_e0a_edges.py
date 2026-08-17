from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_e0a_edge_auditor_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_worldsim_v51_e0a_edges.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout
    assert "--output" in result.stdout
