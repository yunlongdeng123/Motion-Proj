from __future__ import annotations

import json
from pathlib import Path

import pytest

from motion_proj.worldsim_v33.integration_release import (
    atomic_json,
    build_content_manifest,
    extract_and_verify_archive,
    nested_get,
    sha256_file,
    validate_expectations,
    verify_release_directory,
    write_content_manifest,
    write_deterministic_archive,
)


def _release(root: Path) -> Path:
    root.mkdir()
    (root / "README.md").write_text("release\n", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets/delta.npz").write_bytes(b"npz-placeholder")
    atomic_json(root / "ledgers/decision.json", {"selected": "G0_raw_3d"})
    write_content_manifest(root)
    return root


def test_nested_expectations_pass_and_fail_closed() -> None:
    payload = {"decision": {"selected": "A4", "accepted": True}}
    assert nested_get(payload, "decision.selected") == "A4"
    validate_expectations(
        payload,
        {"decision.selected": "A4", "decision.accepted": True},
        role="test",
    )
    with pytest.raises(RuntimeError, match="expectation"):
        validate_expectations(payload, {"decision.selected": "A2"}, role="test")


def test_content_manifest_is_sorted_and_excludes_itself(tmp_path: Path) -> None:
    release = _release(tmp_path / "release")
    manifest = json.loads((release / "content_manifest.json").read_text())
    paths = [row["path"] for row in manifest["files"]]
    assert paths == sorted(paths)
    assert "content_manifest.json" not in paths
    assert manifest["full_checkpoint_copy_count"] == 0
    assert build_content_manifest(release) == manifest


def test_release_directory_detects_extra_or_tampered_file(tmp_path: Path) -> None:
    release = _release(tmp_path / "release")
    assert verify_release_directory(release)["valid"]
    (release / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set"):
        verify_release_directory(release)


def test_release_rejects_full_checkpoint_copy(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "checkpoint.pth").write_bytes(b"forbidden")
    manifest = write_content_manifest
    with pytest.raises(RuntimeError, match="完整模型"):
        manifest(release)


def test_deterministic_archive_and_exact_replay(tmp_path: Path) -> None:
    release = _release(tmp_path / "release")
    first = write_deterministic_archive(release, tmp_path / "first.zip")
    second = write_deterministic_archive(release, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert first["bytes"] == second["bytes"]
    replay = extract_and_verify_archive(first["path"], tmp_path / "replay")
    assert replay["manifest_sha256"] == sha256_file(
        release / "content_manifest.json"
    )


def test_archive_replay_rejects_existing_target(tmp_path: Path) -> None:
    release = _release(tmp_path / "release")
    archive = write_deterministic_archive(release, tmp_path / "release.zip")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        extract_and_verify_archive(archive["path"], existing)
