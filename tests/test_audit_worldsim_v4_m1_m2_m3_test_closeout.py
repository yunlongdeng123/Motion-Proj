from __future__ import annotations

import json
from pathlib import Path
import subprocess

from motion_proj.worldsim_v4.test_freeze import canonical_json_bytes, sha256_file
from scripts.audit_worldsim_v4_m1_m2_m3_test_closeout import (
    verify_freeze_history,
    verify_run,
)


def test_closeout_accepts_docs_commit_descending_from_freeze_only_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    (root / "source.txt").write_text("source\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "source"], check=True)
    source = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    freeze = {"source_commit": source}
    freeze_path = root / "V4_TEST_FREEZE.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    subprocess.run(["git", "-C", str(root), "add", "V4_TEST_FREEZE.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "freeze"], check=True)
    freeze_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    (root / "docs.md").write_text("closeout\n")
    subprocess.run(["git", "-C", str(root), "add", "docs.md"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "docs"], check=True)

    result = verify_freeze_history(root, freeze_path, freeze)
    assert result["source_commit"] == source
    assert result["freeze_commit"] == freeze_commit
    assert result["head"] != freeze_commit


def test_verify_run_recomputes_summary_and_manifest_hashes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"status": "done", "value": 1}) + "\n"
    )
    (run / "manifest.json").write_text(json.dumps({"files": {}}) + "\n")
    (run / "status.json").write_text(
        json.dumps(
            {
                "status": "done",
                "summary_sha256": sha256_file(run / "summary.json"),
                "manifest_sha256": sha256_file(run / "manifest.json"),
            }
        )
        + "\n"
    )
    assert verify_run(run, "fixture")["value"] == 1
