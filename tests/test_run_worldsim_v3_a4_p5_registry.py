from __future__ import annotations

import json
import sys

import pytest

from scripts.aggregate_worldsim_v3_a4_p5 import build_aggregate
from scripts.audit_worldsim_v3_a4_p5_resume import build_resume_audit
from scripts.run_worldsim_v3_a4_p5_registry import (
    build_compact_registry,
    canonical_sha256,
    sha256_file,
)
from scripts.run_worldsim_v3_a4_p5_reload_smoke import (
    audit_actor_indices,
    index_sha256,
)


def fixture_protocol() -> dict:
    return {
        "task_id": "task",
        "profile_id": "profile",
        "scene": "scene",
        "seed": 0,
        "selected_asset": {
            "checkpoint": {"path": "/checkpoint", "sha256": "c", "bytes": 10},
            "source_config": {"path": "/config", "sha256": "f", "bytes": 2},
            "actor_registry": {
                "path": "/registry",
                "sha256": "r",
                "bytes": 3,
            },
        },
        "registry_contract": {
            "schema_version": "schema",
            "static_asset": {
                "asset_id": "static-background-000",
                "model_key": "models.Background",
                "gaussian_count": 7,
                "asset_count": 1,
                "storage": "monolithic_checkpoint_reference",
                "independently_extractable": False,
                "independent_extraction_missing_reason": "p3_chunk_not_authorized",
            },
            "actor_assets": {
                "required_compact_fields": [
                    "asset_id",
                    "rigid_model_index",
                    "instance_token",
                    "class_name",
                    "availability",
                    "selector",
                    "gaussian_count",
                    "flat_indices_sha256",
                    "source_registry_sha256",
                ]
            },
        },
        "recovery_contract": {
            "stage_order": [
                "input_audit",
                "registry_materialize",
                "reload_smoke",
                "aggregate",
                "resume_audit",
            ],
            "minimum_rerun_units": {},
            "completed_stage_policy": "never_overwrite",
        },
    }


def fixture_source_registry() -> dict:
    actors = [
        {
            "rigid_model_index": 0,
            "instance_token": "a",
            "class_name": "vehicle.car",
            "availability": "available",
            "checkpoint_tensor_slice": {
                "selector": "models.RigidNodes.points_ids[:,0] == 0",
                "gaussian_count": 2,
                "flat_indices_sha256": index_sha256([0, 2]),
                "flat_index_ranges_half_open": [[0, 1], [2, 3]],
            },
        },
        {
            "rigid_model_index": 1,
            "instance_token": "b",
            "class_name": "vehicle.car",
            "availability": "unavailable_empty_checkpoint_slice",
            "checkpoint_tensor_slice": {
                "selector": "models.RigidNodes.points_ids[:,0] == 1",
                "gaussian_count": 0,
                "flat_indices_sha256": index_sha256([]),
                "flat_index_ranges_half_open": [],
            },
        },
    ]
    payload = {"actors": actors}
    payload["actor_registry_sha256"] = canonical_sha256(payload)
    return payload


def test_compact_registry_is_reference_only_and_preserves_unavailable() -> None:
    protocol = fixture_protocol()
    source = fixture_source_registry()
    protocol["selected_asset"]["actor_registry"]["embedded_registry_sha256"] = source[
        "actor_registry_sha256"
    ]
    registry = build_compact_registry(protocol, source)
    assert registry["totals"] == {
        "static_asset_count": 1,
        "static_gaussian_count": 7,
        "actor_asset_count": 2,
        "available_actor_count": 1,
        "unavailable_actor_count": 1,
        "actor_gaussian_count": 2,
        "total_gaussian_count": 9,
    }
    assert "flat_index_ranges_half_open" not in registry["actor_assets"][0]
    assert registry["actor_assets"][1]["availability"].startswith("unavailable")
    expected = registry.pop("registry_sha256")
    assert canonical_sha256(registry) == expected


def test_actor_index_audit_recomputes_exact_hashes() -> None:
    source = fixture_source_registry()
    rows = audit_actor_indices([0, 2, 0], source["actors"])
    assert all(row["exact"] for row in rows)
    assert rows[1]["actual_availability"] == "unavailable_empty_checkpoint_slice"


def test_actor_index_audit_rejects_drift() -> None:
    source = fixture_source_registry()
    rows = audit_actor_indices([0, 1, 0], source["actors"])
    assert not all(row["exact"] for row in rows)


def test_aggregate_keeps_resume_pending(tmp_path) -> None:
    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True)
    payloads = {
        "input_audit": {"input_bytes": 10, "minimum_rerun_unit": "input"},
        "registry_materialize": {
            "registry": {"bytes": 5},
            "minimum_rerun_unit": "materialize",
        },
        "reload_smoke": {
            "reload_total_seconds": 1.0,
            "filesystem_cache": "uncontrolled",
            "minimum_rerun_unit": "reload",
            "model_gaussian_counts": {"Background": 7, "RigidNodes": 2},
            "actor_rows": [{}, {}],
            "all_actor_indices_exact": True,
        },
    }
    hashes = {}
    for name, payload in payloads.items():
        path = stages / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        hashes[name] = sha256_file(path)
    manifest = {
        "stage_hashes": hashes,
        "input_audits": {
            "selected_asset.actor_registry": {"bytes": 3},
            "selected_asset.checkpoint": {"bytes": 10},
            "selected_asset.source_config": {"bytes": 2},
        },
    }
    aggregate = build_aggregate(run_dir, manifest)
    assert aggregate["stage_ledger"]["resume_audit"]["status"] is None
    assert aggregate["stage_ledger"]["resume_audit"]["missing_reason"]


def test_resume_audit_reuses_completed_stages(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    artifacts = run_dir / "artifacts"
    stages.mkdir(parents=True)
    artifacts.mkdir()
    registry = {"value": 1}
    registry["registry_sha256"] = canonical_sha256(registry)
    registry_path = artifacts / "deployment_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    hashes = {}
    for name in ("input_audit", "reload_smoke", "aggregate"):
        path = stages / f"{name}.json"
        path.write_text(json.dumps({"stage": name}), encoding="utf-8")
        hashes[name] = sha256_file(path)
    materialize = {
        "registry": {
            "path": "artifacts/deployment_registry.json",
            "sha256": sha256_file(registry_path),
        }
    }
    materialize_path = stages / "registry_materialize.json"
    materialize_path.write_text(json.dumps(materialize), encoding="utf-8")
    hashes["registry_materialize"] = sha256_file(materialize_path)
    (run_dir / "manifest.json").write_text(
        json.dumps({"stage_hashes": hashes}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "scripts.audit_worldsim_v3_a4_p5_resume.nvidia_compute_rows", lambda: []
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    result = build_resume_audit(run_dir)
    assert result["torch_imported"] is False
    assert result["gpu_launch_observed"] is False
    assert len(result["actions"]) == 4


def test_resume_audit_rejects_stage_drift(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True)
    hashes = {}
    for name in ("input_audit", "registry_materialize", "reload_smoke", "aggregate"):
        path = stages / f"{name}.json"
        path.write_text(json.dumps({"stage": name}), encoding="utf-8")
        hashes[name] = sha256_file(path)
    (run_dir / "manifest.json").write_text(
        json.dumps({"stage_hashes": hashes}), encoding="utf-8"
    )
    (stages / "reload_smoke.json").write_text("drift", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.audit_worldsim_v3_a4_p5_resume.nvidia_compute_rows", lambda: []
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(RuntimeError, match="hash drift"):
        build_resume_audit(run_dir)
