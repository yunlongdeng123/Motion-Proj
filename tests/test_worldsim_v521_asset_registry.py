from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v521.asset_audit import AssetAuditError, audit_file, ensure_matched_asset


def test_hash_mismatch_never_silently_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"wrong")
    row = audit_file({"path": str(path), "bytes": 5, "sha256": "0" * 64})
    assert row["state"] == "PRESENT_HASH_MISMATCH"
    with pytest.raises(AssetAuditError):
        ensure_matched_asset(row)


def test_missing_manifested_asset_is_not_unrecoverable(tmp_path: Path) -> None:
    row = audit_file({"path": str(tmp_path / "missing.bin"), "bytes": 10, "sha256": "1" * 64})
    assert row["state"] == "MISSING_BUT_MANIFESTED"


def test_stride10_is_rejected_from_matched_cohort() -> None:
    with pytest.raises(AssetAuditError):
        ensure_matched_asset({"state": "PRESENT_EXACT", "protocol": "stride10"})
