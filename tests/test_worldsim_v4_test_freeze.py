from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from motion_proj.worldsim_v4.test_freeze import (
    ATTEMPT_SCHEMA,
    FREEZE_SCHEMA,
    TASK_ID,
    TestFreezeError as FreezeError,
    canonical_json_bytes,
    committed_freeze,
    exclusive_json,
    sha256_file,
    validate_execution_plan,
    validate_test_attempt,
)
from scripts.run_worldsim_v4_m3_test_exact_once import attempt_payload


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def frozen_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    config = root / "config.yaml"
    inventory = root / "inventory.yaml"
    config.write_text("config: 1\n", encoding="utf-8")
    inventory.write_text("inventory: 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "source"], check=True)
    source = git(root, "rev-parse", "HEAD")
    ledger = tmp_path / "ledger"
    scenes = [f"scene-{index:04d}" for index in range(18)]
    plan = [
        {
            "ordinal": index + 1,
            "scene": scene,
            "attempt_id": f"test-{index + 1:02d}-{scene}",
            "run_dir": str(tmp_path / "runs" / scene),
        }
        for index, scene in enumerate(scenes)
    ]
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "frozen",
        "source_commit": source,
        "config_sha256": sha256_file(config),
        "test_asset_inventory_sha256": sha256_file(inventory),
        "scene_order": scenes,
        "execution_plan": plan,
        "ledger_dir": str(ledger),
        "test_read_count": 1,
        "test_authorized": True,
    }
    freeze_path = root / "V4_TEST_FREEZE.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    subprocess.run(["git", "-C", str(root), "add", "V4_TEST_FREEZE.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "freeze"], check=True)
    return root, freeze_path, config, inventory, freeze


def test_committed_freeze_and_exact_attempt(tmp_path: Path) -> None:
    root, freeze_path, config, inventory, freeze = frozen_repo(tmp_path)
    provenance = committed_freeze(freeze_path, root)[1]
    planned = freeze["execution_plan"][0]
    attempt_path = (
        Path(freeze["ledger_dir"])
        / "attempts"
        / f"{planned['attempt_id']}.json"
    )
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": TASK_ID,
        "scene": planned["scene"],
        "ordinal": planned["ordinal"],
        "attempt_id": planned["attempt_id"],
        "run_dir": planned["run_dir"],
        "freeze_sha256": provenance["freeze_sha256"],
        "freeze_commit": provenance["freeze_commit"],
        "state": "started",
    }
    assert attempt_payload(planned, provenance) == attempt
    exclusive_json(attempt_path, attempt)
    result = validate_test_attempt(
        freeze_path=freeze_path,
        attempt_path=attempt_path,
        project_root=root,
        scene=planned["scene"],
        run_dir=planned["run_dir"],
        config_path=config,
        inventory_path=inventory,
    )
    assert result["attempt"] == attempt
    with pytest.raises(FileExistsError):
        exclusive_json(attempt_path, attempt)


def test_freeze_fails_on_dirty_or_nonsole_commit(tmp_path: Path) -> None:
    root, freeze_path, *_ = frozen_repo(tmp_path)
    dirty = root / "dirty"
    dirty.write_text("x", encoding="utf-8")
    with pytest.raises(FreezeError, match="clean"):
        committed_freeze(freeze_path, root)
    dirty.unlink()
    extra = root / "extra.txt"
    extra.write_text("not allowed in freeze commit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "extra.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--amend", "--no-edit", "-q"],
        check=True,
    )
    with pytest.raises(FreezeError, match="only"):
        committed_freeze(freeze_path, root)


def test_execution_plan_rejects_duplicate_run(tmp_path: Path) -> None:
    _, _, _, _, freeze = frozen_repo(tmp_path)
    freeze["execution_plan"][1]["run_dir"] = freeze["execution_plan"][0]["run_dir"]
    with pytest.raises(FreezeError, match="unique"):
        validate_execution_plan(freeze)
