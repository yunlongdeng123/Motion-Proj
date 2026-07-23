import hashlib
import json
from datetime import datetime, timezone

import pytest

from motion_proj.runtime.atomic import (
    atomic_merge_json_object,
    atomic_write_json,
    atomic_write_text,
)
from motion_proj.runtime.fingerprint import file_fingerprint
from motion_proj.runtime.v71_contract import (
    RunContractError,
    V71RunContract,
    build_retrospective_evidence_index,
    compute_artifact_set_hash,
    generate_run_id,
    validate_optional_branch_not_triggered,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _base_manifest(run_id: str, world_hash: str, render_hash: str, artifact_set_hash: str):
    return {
        "schema_version": 1,
        "task_id": "V7-EV-10",
        "run_id": run_id,
        "parent_run_id": None,
        "command": ["pytest"],
        "plan_version": "V7.1",
        "code_commit": "a" * 40,
        "code_dirty": False,
        "dirty_diff_hash": "b" * 64,
        "config_fingerprint": "c" * 64,
        "data_fingerprint": "d" * 64,
        "split_fingerprint": "e" * 64,
        "proposal_fingerprint": "f" * 64,
        "third_party_commit": "not_applicable",
        "checkpoint_hashes": {},
        "scene_ids": [],
        "actor_ids": [],
        "camera_ids": [],
        "time_range": None,
        "seed": 0,
        "environment": {"python": "test"},
        "cuda": "not_used",
        "gpu": "not_used",
        "world_state_schema_version": "smoke",
        "canonicalization_version": "smoke",
        "world_state_hash": world_hash,
        "render_request_hash": render_hash,
        "artifact_set_hash": artifact_set_hash,
        "safety_geometry_version": "not_applicable",
        "observation_evidence_version": "retrospective",
        "render_support_version": "not_applicable",
        "certificate_version": "not_applicable",
        "scenario_effect_version": "not_applicable",
        "provenance_version": "retrospective",
        "recovery_policy": "not_applicable",
        "started_at": "2026-07-23T00:00:00+00:00",
        "ended_at": None,
        "exit_reason": None,
        "terminal_status": "running",
        "status": "running",
    }


def _materialize_contract(tmp_path, run_id="v71_ev10_test"):
    world_hash = _digest(b"world")
    render_hash = _digest(b"render")
    artifact_bytes = b"artifact\n"
    artifact_hash = _digest(artifact_bytes)
    rows = [
        {
            "relative_path": "artifacts/value.txt",
            "world_state_hash": world_hash,
            "render_request_hash": render_hash,
            "artifact_hash": artifact_hash,
            "size_bytes": len(artifact_bytes),
            "media_type": "text/plain",
        }
    ]
    artifact_set_hash = compute_artifact_set_hash(rows)
    manifest = _base_manifest(run_id, world_hash, render_hash, artifact_set_hash)
    contract = V71RunContract.initialize(tmp_path, run_id, manifest)
    run_dir = contract.run_dir
    atomic_write_text(str(run_dir / "resolved.yaml"), "schema_version: 1\n")
    for name in ("code", "environment", "data", "third_party", "checkpoints"):
        atomic_write_json(str(run_dir / "fingerprints" / f"{name}.json"), {"name": name})
    atomic_write_json(str(run_dir / "proposal_bank.json"), {"proposals": []})
    atomic_write_json(str(run_dir / "actor_registry.json"), {"actors": []})
    atomic_write_json(
        str(run_dir / "world_state_manifest.json"),
        {"world_state_hash": world_hash},
    )
    atomic_write_json(
        str(run_dir / "render_request_manifest.json"),
        {
            "world_state_hash": world_hash,
            "render_request_hash": render_hash,
        },
    )
    (run_dir / "artifacts" / "value.txt").write_bytes(artifact_bytes)
    atomic_write_json(
        str(run_dir / "artifact_manifest.json"),
        {
            "world_state_hash": world_hash,
            "render_request_hash": render_hash,
            "artifact_set_hash": artifact_set_hash,
            "artifacts": rows,
        },
    )
    atomic_write_text(str(run_dir / "metrics.jsonl"), '{"step": 0, "ok": true}\n')
    atomic_write_json(str(run_dir / "summary.json"), {"ok": True})
    return contract


def test_complete_contract_validates_bytes_and_single_marker(tmp_path):
    contract = _materialize_contract(tmp_path)

    contract.finalize("COMPLETE", exit_reason="smoke_passed")

    assert contract.validate() == "COMPLETE"
    assert [path.name for path in contract.run_dir.iterdir() if path.name == "COMPLETE"] == [
        "COMPLETE"
    ]


def test_run_id_reuse_fails_closed(tmp_path):
    contract = _materialize_contract(tmp_path)
    manifest = json.loads((contract.run_dir / "manifest.json").read_text())

    with pytest.raises(RunContractError, match="禁止复用"):
        V71RunContract.initialize(tmp_path, contract.run_dir.name, manifest)


def test_conflicting_terminal_markers_fail_closed(tmp_path):
    contract = _materialize_contract(tmp_path)
    (contract.run_dir / "COMPLETE").write_text("first\n")
    (contract.run_dir / "FAILED").write_text("second\n")

    with pytest.raises(RunContractError, match="恰好一个"):
        contract.validate()


@pytest.mark.parametrize(
    "field",
    ["world_state_hash", "render_request_hash", "artifact_set_hash"],
)
def test_missing_three_layer_hash_prevents_complete(tmp_path, field):
    contract = _materialize_contract(tmp_path)
    path = contract.run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest[field]
    atomic_write_json(str(path), manifest)

    with pytest.raises(RunContractError, match="manifest 缺少字段"):
        contract.finalize("COMPLETE", exit_reason="must_not_complete")


def test_missing_artifact_hash_prevents_complete(tmp_path):
    contract = _materialize_contract(tmp_path)
    path = contract.run_dir / "artifact_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["artifacts"][0]["artifact_hash"]
    atomic_write_json(str(path), manifest)

    with pytest.raises(RunContractError, match="artifact row 缺少字段"):
        contract.finalize("COMPLETE", exit_reason="must_not_complete")


def test_missing_summary_prevents_complete(tmp_path):
    contract = _materialize_contract(tmp_path)
    (contract.run_dir / "summary.json").unlink()

    with pytest.raises(RunContractError, match="缺少必需 artifact"):
        contract.finalize("COMPLETE", exit_reason="must_not_complete")


def test_artifact_byte_corruption_prevents_complete(tmp_path):
    contract = _materialize_contract(tmp_path)
    (contract.run_dir / "artifacts" / "value.txt").write_text("corrupt!\n")

    with pytest.raises(RunContractError, match="artifact bytes hash 不一致"):
        contract.finalize("COMPLETE", exit_reason="must_not_complete")


def test_optional_branch_not_triggered_has_no_run_or_marker(tmp_path):
    parent_summary = tmp_path / "summary.json"
    atomic_write_json(
        str(parent_summary),
        {
            "h2_generation_branch": "not_triggered",
            "trigger_reason": "materiality_not_met",
        },
    )
    branch_root = tmp_path / "V7-H2-12D"

    validate_optional_branch_not_triggered(parent_summary, branch_root)
    branch_root.mkdir()
    with pytest.raises(RunContractError, match="禁止创建 run 目录"):
        validate_optional_branch_not_triggered(parent_summary, branch_root)


def test_evidence_index_records_hashes_and_explicit_missing_fields(tmp_path):
    roots = {}
    for task in ("B0", "O0", "S0", "C0", "L0", "U0"):
        root = tmp_path / task
        root.mkdir()
        (root / "evidence.json").write_text(f'{{"task": "{task}"}}\n')
        roots[task] = root
    output = tmp_path / "V7_EVIDENCE_INDEX.json"

    index = build_retrospective_evidence_index(
        output,
        roots,
        index_code_state={"commit": "test", "dirty": False},
    )

    assert index["totals"]["stage_count"] == 6
    assert index["totals"]["file_count"] == 6
    assert file_fingerprint(str(output))
    assert index["stages"]["B0"]["evidence_mode"] == "retrospective"
    missing = {
        row["field"]: row["status"]
        for row in index["stages"]["B0"]["formal_contract_missing"]
    }
    assert missing["run_start_commit"] == "unknown_not_inferred"
    assert missing["terminal_marker"] == "missing"
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        build_retrospective_evidence_index(
            output,
            roots,
            index_code_state={"commit": "test"},
        )


def test_s0_summary_merge_preserves_other_scenes(tmp_path):
    path = tmp_path / "s0_edit_summary.json"
    atomic_write_json(str(path), {"3": {"actor": 35}, "5": {"actor": 34}})

    merged = atomic_merge_json_object(str(path), {"4": {"actor": 8}})

    assert merged == {
        "3": {"actor": 35},
        "4": {"actor": 8},
        "5": {"actor": 34},
    }


def test_run_id_contains_timestamp_seed_and_config_hash():
    run_id = generate_run_id(
        "V7-EV-10",
        "PILOT-3",
        7,
        "a" * 64,
        now=datetime(2026, 7, 23, 1, 2, 3, 456789, tzinfo=timezone.utc),
    )

    assert run_id == "v71_v7-ev-10__pilot-3__s7__20260723T010203456789Z__aaaaaaaa"
