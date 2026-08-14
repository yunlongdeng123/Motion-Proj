from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/extract_worldsim_v5_kitti_smoke.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_kitti_extract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _archive(path: Path, component: str, directory: bool) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for sequence in ("0000", "0001"):
            name = (
                f"training/{component}/{sequence}/000000.bin"
                if directory
                else f"training/{component}/{sequence}.txt"
            )
            archive.writestr(name, f"{component}-{sequence}".encode())


def test_selective_smoke_extraction_is_atomic_and_content_addressed(tmp_path) -> None:
    archives = {}
    for component in MODULE.DEFAULT_ARCHIVES:
        path = tmp_path / f"{component}.zip"
        _archive(path, component, component not in MODULE.FILE_COMPONENTS)
        archives[component] = path
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    result = MODULE.extract(
        archives=archives,
        sequences=("0000", "0001"),
        output=output,
        manifest_path=manifest,
    )
    assert result["complete"] is True
    assert result["file_count"] == 12
    assert output.is_dir()
    assert manifest.is_file()
    with pytest.raises(MODULE.KittiSmokeExtractionError, match="禁止覆盖"):
        MODULE.extract(
            archives=archives,
            sequences=("0000", "0001"),
            output=output,
            manifest_path=manifest,
        )


def test_zip_slip_member_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../bad", b"bad")
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(MODULE.KittiSmokeExtractionError, match="不安全"):
            MODULE.safe_member(archive.infolist()[0].filename)


def test_legacy_archive_audit_only_unlocks_safe_selective_extraction(
    tmp_path, monkeypatch
) -> None:
    archives = {}
    rows = []
    for component in MODULE.DEFAULT_ARCHIVES:
        path = tmp_path / f"{component}.zip"
        _archive(path, component, component not in MODULE.FILE_COMPONENTS)
        archives[component] = path
        rows.append(
            {
                "component": component,
                "path": str(path),
                "archive_bytes": path.stat().st_size,
                "sha256": MODULE.sha256_file(path),
                "central_directory_readable": True,
                "unsafe_member_count": 0,
            }
        )
    monkeypatch.setattr(MODULE, "DEFAULT_ARCHIVES", archives)
    audit = tmp_path / "audit.json"
    payload = {
        "task_id": "WS-V5-D1-KITTI-ARCHIVE-AUDIT-01",
        "status": "blocked_dataset_adapter",
        "archives": rows,
        "gates": {
            "all_archives_present": True,
            "archive_sha256_recorded": True,
            "central_directories_readable": True,
            "expected_component_paths": True,
            "expected_sequence_sets": True,
            "no_duplicate_members": True,
            "safe_member_paths": True,
            "unencrypted_members": True,
            "sensor_frame_alignment": False,
        },
    }
    audit.write_text(json.dumps(payload), encoding="utf-8")
    evidence, digest = MODULE.audited_archives(audit)
    assert set(evidence) == set(archives)
    assert digest == MODULE.sha256_file(audit)

    payload["gates"]["safe_member_paths"] = False
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.KittiSmokeExtractionError, match="safe-extraction"):
        MODULE.audited_archives(audit)


def test_formal_extraction_closeout_has_n_a_checkpoint_and_no_quality(
    tmp_path, monkeypatch
) -> None:
    project = tmp_path / "project"
    for relative in (
        "scripts/extract_worldsim_v5_kitti_smoke.py",
        "tests/test_worldsim_v5_kitti_smoke_extraction.py",
        "docs/KITTI_TRACKING_ARCHIVE_METADATA_V5.json",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        MODULE,
        "git_output",
        lambda *args: (
            "commit" if args == ("rev-parse", "HEAD") else "branch"
        ),
    )
    run = tmp_path / "run"
    (run / "artifacts").mkdir(parents=True)
    (run / "source_snapshot").mkdir()
    (run / "resolved_config.json").write_text("{}", encoding="utf-8")
    raw_manifest = tmp_path / "raw.json"
    raw_manifest.write_text("{}", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    payload = {
        "manifest_sha256": "1" * 64,
        "sequences": ["0000", "0001"],
        "file_count": 1805,
        "uncompressed_bytes": 2104258586,
    }
    summary = MODULE.finalize_formal_run(
        run, payload, manifest_path=raw_manifest, archive_audit=audit
    )
    assert summary["status"] == "done"
    assert summary["checkpoint"] == "N/A_data_preparation"
    assert summary["quality_read"] is False
    assert (run / "manifest.json").is_file()
