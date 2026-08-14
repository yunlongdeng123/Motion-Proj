from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_worldsim_v5_m1_raw_preparation.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_m1_raw_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_content_address_omits_manifest_hash() -> None:
    payload = {"schema_version": 1, "complete": True}
    payload["manifest_sha256"] = MODULE.sha256_payload(payload)
    MODULE.validate_content_address(payload, Path("manifest.json"))
    payload["complete"] = False
    with pytest.raises(MODULE.RawAuditError, match="content address"):
        MODULE.validate_content_address(payload, Path("manifest.json"))


def test_audit_accepts_internal_scene_content_address(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    scene_rows = []
    mapping = {}
    for index in range(8):
        filename = f"samples/CAM_FRONT/{index}.jpg"
        sensor = raw_root / filename
        sensor.parent.mkdir(parents=True, exist_ok=True)
        sensor.write_bytes(b"sensor")
        scene = {
            "scene_name": f"scene-{index:04d}",
            "scene_index": index,
            "complete": True,
            "required_count": 1,
            "present_count": 1,
            "files": [{"filename": filename, "bytes": 6}],
        }
        scene["manifest_sha256"] = MODULE.sha256_payload(scene)
        scene_path = tmp_path / f"scene-{index:04d}.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        scene_rows.append(
            {
                "scene_name": scene["scene_name"],
                "scene_index": index,
                "required_count": 1,
                "bytes": 6,
                "manifest": str(scene_path),
                "manifest_sha256": scene["manifest_sha256"],
            }
        )
        mapping[filename] = "v1.0-trainval01_blobs.tgz"
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(mapping), encoding="utf-8")
    batch = {
        "complete": True,
        "quality_read": False,
        "scene_count": 8,
        "required_count": 8,
        "present_count": 8,
        "total_bytes": 48,
        "raw_root": str(raw_root),
        "member_shard_index": str(index_path),
        "member_shard_index_sha256": MODULE.sha256_file(index_path),
        "scenes": scene_rows,
    }
    batch["manifest_sha256"] = MODULE.sha256_payload(batch)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    result = MODULE.audit(batch_path)
    assert result["required_count"] == 8
    assert result["scenes"][0]["manifest_file_sha256"] == MODULE.sha256_file(
        Path(scene_rows[0]["manifest"])
    )
