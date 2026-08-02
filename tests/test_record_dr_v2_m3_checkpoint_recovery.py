import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "record_dr_v2_m3_checkpoint_recovery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "record_dr_v2_m3_checkpoint_recovery", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def contract(checkpoint: Path) -> tuple[dict, dict]:
    checkpoint.write_bytes(b"checkpoint")
    formal = {
        "status": "blocked",
        "stop_reason": "memory.current/memory.max >= 0.90 twice",
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    terminal = {
        "status": "blocked",
        "failure": {
            "code": "M3_FORMAL_RENDER_CGROUP_MEMORY_GUARD",
            "detail": "checkpoint-intact_oom0_oom-kill0",
        },
    }
    return formal, terminal


def test_recovery_contract_is_narrow_and_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    formal, terminal = contract(checkpoint)
    MODULE.validate_recovery_contract(
        formal=formal,
        terminal=terminal,
        checkpoint=checkpoint,
        checkpoint_step=30_000,
    )
    with pytest.raises(RuntimeError, match="step mismatch"):
        MODULE.validate_recovery_contract(
            formal=formal,
            terminal=terminal,
            checkpoint=checkpoint,
            checkpoint_step=29_999,
        )
    with pytest.raises(RuntimeError, match="failure code"):
        MODULE.validate_recovery_contract(
            formal=formal,
            terminal={"failure": {"code": "OTHER", "detail": "oom0_oom-kill0"}},
            checkpoint=checkpoint,
            checkpoint_step=30_000,
        )


def test_reusable_stage_and_atomic_overwrite_guard(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"status":"done","value":1}\n', encoding="utf-8")
    payload = MODULE.reusable_stage("profile", source, json.loads(source.read_text()))
    assert payload["status"] == "done"
    assert payload["source_status"] == "done"
    assert payload["value"] == 1
    target = tmp_path / "target.json"
    MODULE.atomic_json(target, payload)
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        MODULE.atomic_json(target, payload)
