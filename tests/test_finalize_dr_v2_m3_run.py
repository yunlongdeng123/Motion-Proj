import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_dr_v2_m3_run.py"
SPEC = importlib.util.spec_from_file_location("finalize_dr_v2_m3_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dynamic_editing_v2"
    run = root / "TASK" / "instance-r1"
    (run / "stages").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "terminal.json").write_text('{"status":"running"}\n', encoding="utf-8")
    (run / "manifest.json").write_text(
        '{"task_id":"TASK","status":"running"}\n', encoding="utf-8"
    )
    (run / "logs" / "stage.log").write_text("evidence\n", encoding="utf-8")
    return root, run


def test_finalize_blocked_run_is_immutable_and_auditable(tmp_path: Path) -> None:
    root, run = make_run(tmp_path)
    result = MODULE.finalize(
        run_dir=run,
        task_id="TASK",
        status="blocked",
        summary="Environment failure only; no method-quality conclusion.",
        failure_code="CUDA_EXTENSION_ARCH_MISMATCH",
        failure_detail="The binary has no SM 8.6 kernel image.",
        evidence=["logs/stage.log"],
        allowed_root=root,
    )
    assert result["terminal"]["failure"]["code"] == "CUDA_EXTENSION_ARCH_MISMATCH"
    assert json.loads((run / "manifest.json").read_text())["status"] == "blocked"
    artifacts = json.loads((run / "artifacts.json").read_text())
    assert any(row["path"] == "logs/stage.log" for row in artifacts["files"])
    with pytest.raises(RuntimeError, match="already immutable"):
        MODULE.finalize(
            run_dir=run,
            task_id="TASK",
            status="blocked",
            summary="duplicate",
            failure_code="DUPLICATE",
            failure_detail="must fail",
            allowed_root=root,
        )


def test_done_requires_successful_named_stages(tmp_path: Path) -> None:
    root, run = make_run(tmp_path)
    (run / "stages" / "smoke.json").write_text(
        '{"stage":"smoke","status":"blocked"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="did not succeed"):
        MODULE.finalize(
            run_dir=run,
            task_id="TASK",
            status="done",
            summary="done",
            required_stages=["smoke"],
            allowed_root=root,
        )


def test_refuses_run_outside_allowed_root(tmp_path: Path) -> None:
    root, run = make_run(tmp_path)
    with pytest.raises(RuntimeError, match="outside the V2 root"):
        MODULE.finalize(
            run_dir=run,
            task_id="TASK",
            status="blocked",
            summary="blocked",
            failure_code="X",
            failure_detail="Y",
            allowed_root=tmp_path / "different-root",
        )
