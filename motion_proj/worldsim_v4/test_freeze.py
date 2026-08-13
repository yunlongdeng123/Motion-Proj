"""Committed V4 test-freeze and exact-once execution contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


FREEZE_SCHEMA = "worldsim_v4_test_freeze_v1"
ATTEMPT_SCHEMA = "worldsim_v4_test_attempt_v1"
TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"


class TestFreezeError(RuntimeError):
    """A committed test-freeze or exact-once contract is invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_mapping(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TestFreezeError(f"JSON root is not a mapping: {path}")
    return value


def git(project_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(project_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise TestFreezeError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def committed_freeze(
    freeze_path: str | Path,
    project_root: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify that the exact freeze bytes are the sole commit after source."""

    freeze_path = Path(freeze_path).resolve()
    project_root = Path(project_root).resolve()
    freeze = load_mapping(freeze_path)
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise TestFreezeError("V4 test-freeze schema drift")
    if freeze.get("task_id") != TASK_ID or freeze.get("status") != "frozen":
        raise TestFreezeError("V4 test-freeze task/status drift")
    if freeze.get("test_authorized") is not True:
        raise TestFreezeError("V4 test-freeze does not authorize test")
    if int(freeze.get("test_read_count", -1)) != 1:
        raise TestFreezeError("V4 test read count must be exactly one")
    if git(project_root, "status", "--porcelain"):
        raise TestFreezeError("test execution requires a clean worktree")
    try:
        relative = freeze_path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise TestFreezeError("test-freeze must be inside the project") from error
    head = git(project_root, "rev-parse", "HEAD")
    parent = git(project_root, "rev-parse", "HEAD^")
    committed_blob = subprocess.run(
        ["git", "-C", str(project_root), "show", f"HEAD:{relative}"],
        capture_output=True,
        check=False,
    )
    if committed_blob.returncode or committed_blob.stdout != freeze_path.read_bytes():
        raise TestFreezeError("working V4_TEST_FREEZE.json differs from HEAD")
    if parent != freeze.get("source_commit"):
        raise TestFreezeError("freeze commit parent differs from frozen source commit")
    if git(project_root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD") != relative:
        raise TestFreezeError("freeze commit must contain only V4_TEST_FREEZE.json")
    return freeze, {
        "freeze_commit": head,
        "source_commit": parent,
        "freeze_sha256": sha256_file(freeze_path),
        "relative_path": relative,
    }


def validate_execution_plan(freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = freeze.get("execution_plan")
    scene_order = freeze.get("scene_order")
    if not isinstance(plan, list) or not isinstance(scene_order, list):
        raise TestFreezeError("test execution plan/scene order is missing")
    if len(plan) != 18 or len(scene_order) != 18 or len(set(scene_order)) != 18:
        raise TestFreezeError("test execution plan must contain 18 unique scenes")
    if [row.get("scene") for row in plan] != scene_order:
        raise TestFreezeError("test execution order drift")
    attempt_ids = [row.get("attempt_id") for row in plan]
    run_dirs = [row.get("run_dir") for row in plan]
    if len(set(attempt_ids)) != 18 or len(set(run_dirs)) != 18:
        raise TestFreezeError("attempt IDs and run directories must be unique")
    for ordinal, row in enumerate(plan, start=1):
        if row.get("ordinal") != ordinal:
            raise TestFreezeError("test execution ordinals must be contiguous")
        if not Path(str(row.get("run_dir", ""))).is_absolute():
            raise TestFreezeError("test run directories must be absolute")
    ledger_dir = Path(str(freeze.get("ledger_dir", "")))
    if not ledger_dir.is_absolute():
        raise TestFreezeError("test ledger directory must be absolute")
    return [dict(row) for row in plan]


def validate_test_attempt(
    *,
    freeze_path: str | Path,
    attempt_path: str | Path,
    project_root: str | Path,
    scene: str,
    run_dir: str | Path,
    config_path: str | Path,
    inventory_path: str | Path,
) -> dict[str, Any]:
    freeze, provenance = committed_freeze(freeze_path, project_root)
    plan = validate_execution_plan(freeze)
    matches = [row for row in plan if row["scene"] == scene]
    if len(matches) != 1:
        raise TestFreezeError("scene must occur exactly once in the test plan")
    planned = matches[0]
    if Path(planned["run_dir"]).resolve() != Path(run_dir).resolve():
        raise TestFreezeError("test run directory differs from frozen plan")
    if sha256_file(config_path) != freeze.get("config_sha256"):
        raise TestFreezeError("test config SHA differs from freeze")
    if sha256_file(inventory_path) != freeze.get("test_asset_inventory_sha256"):
        raise TestFreezeError("test asset inventory SHA differs from freeze")
    attempt_path = Path(attempt_path).resolve()
    expected_attempt = (
        Path(str(freeze["ledger_dir"])).resolve()
        / "attempts"
        / f"{planned['attempt_id']}.json"
    )
    if attempt_path != expected_attempt or not attempt_path.is_file():
        raise TestFreezeError("test attempt path differs from frozen ledger")
    attempt = load_mapping(attempt_path)
    expected = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": TASK_ID,
        "scene": scene,
        "ordinal": planned["ordinal"],
        "attempt_id": planned["attempt_id"],
        "run_dir": str(Path(run_dir).resolve()),
        "freeze_sha256": provenance["freeze_sha256"],
        "freeze_commit": provenance["freeze_commit"],
        "state": "started",
    }
    if attempt != expected:
        raise TestFreezeError("test attempt contents differ from exact-once contract")
    completion = attempt_path.parent.parent / "completions" / attempt_path.name
    if completion.exists():
        raise TestFreezeError("test attempt was already completed")
    return {
        "freeze": freeze,
        "provenance": provenance,
        "plan": planned,
        "attempt": attempt,
        "attempt_sha256": sha256_file(attempt_path),
    }


def exclusive_json(path: str | Path, payload: object) -> None:
    """Create canonical JSON exactly once without replacing existing state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The existence of a partial exact-once marker is deliberately retained.
        raise
