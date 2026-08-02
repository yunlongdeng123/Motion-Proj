import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_dr_v2_blocked_run.py"


def test_refuses_non_v2_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "TASK" / "instance"
    run_dir.mkdir(parents=True)
    (run_dir / "terminal.json").write_text('{"status":"blocked"}')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run_dir), "--task-id", "TASK"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "V2 根目录" in result.stderr


def test_source_has_terminal_guard_and_atomic_writes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'ALLOWED_TERMINAL = {"blocked", "rejected"}' in source
    assert "os.replace(temporary, path)" in source
    assert 'path.name == "artifacts.json"' in source
