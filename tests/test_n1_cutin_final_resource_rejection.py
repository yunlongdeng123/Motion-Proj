"""Final cut-in resource-contract rejection 的不可变裁决测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from motion_proj.runtime.fingerprint import file_fingerprint
from scripts.adjudicate_n1_cutin_final_resource_contract import adjudicate


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_parent(tmp_path: Path, *, cgroup_current: int = 1_523_929_088) -> Path:
    parent = tmp_path / "parent"
    (parent / "stages").mkdir(parents=True)
    (parent / "FAILED").write_text("orchestrator_or_worker_failure\n", encoding="utf-8")
    (parent / "resolved.yaml").write_text("task_id: N1-EVENT-CUTIN-FINAL-01\n", encoding="utf-8")
    _write_json(
        parent / "preflight.json",
        {
            "formal": True,
            "code_dirty": False,
            "code_commit": "parent-clean-commit",
            "memory": {
                "process_rss_bytes": 20_705_280,
                "cgroup_memory_current_bytes": cgroup_current,
            },
        },
    )
    _write_json(
        parent / "failure.json",
        {
            "terminal_status": "FAILED",
            "exit_reason": "orchestrator_or_worker_failure",
        },
    )
    _write_json(
        parent / "stages" / "K4_REGRESSION.json",
        {"passed": True, "n2_authorized": False},
    )
    return parent


def _config(parent: Path, output_root: Path) -> dict:
    names = ("preflight.json", "failure.json", "resolved.yaml", "stages/K4_REGRESSION.json")
    return {
        "schema_version": "n1-cutin-final-resource-adjudication-v1",
        "task_id": "N1-EVENT-CUTIN-FINAL-RESOURCE-AUDIT-01",
        "seed": 0,
        "require_clean_git": False,
        "parent_run": str(parent),
        "parent_expected_terminal": "FAILED",
        "expected_parent_code_commit": "parent-clean-commit",
        "expected_parent_failure_exit_reason": "orchestrator_or_worker_failure",
        "parent_artifact_sha256": {
            name: file_fingerprint(str(parent / name)) for name in names
        },
        "resource_contract": {
            "max_start_cgroup_current_bytes": 1_350_000_000,
            "expected_parent_preflight_cgroup_current_bytes": 1_523_929_088,
        },
        "run_root": str(output_root),
        "decision": {
            "terminal_status": "REJECTED",
            "research_verdict": "stop_nuscenes_cutin_mining",
            "n2_authorized": False,
        },
    }


def test_resource_contract_failure_creates_rejected_adjudication(tmp_path: Path) -> None:
    parent = _make_parent(tmp_path)
    config_path = tmp_path / "resource-rejection.yaml"
    config_path.write_text(yaml.safe_dump(_config(parent, tmp_path / "out")), encoding="utf-8")

    run_dir = adjudicate(config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert (run_dir / "REJECTED").is_file()
    assert summary["terminal_status"] == "REJECTED"
    assert summary["evaluation_scene_count_started"] == 0
    assert summary["human_audit_created"] is False
    assert summary["n2_authorized"] is False
    assert summary["resource_contract"]["start_contract_excess_bytes"] == 173_929_088


def test_resource_adjudication_rejects_nonviolating_preflight(tmp_path: Path) -> None:
    parent = _make_parent(tmp_path, cgroup_current=1_350_000_000)
    config = _config(parent, tmp_path / "out")
    config["resource_contract"]["expected_parent_preflight_cgroup_current_bytes"] = 1_350_000_000
    config["parent_artifact_sha256"]["preflight.json"] = file_fingerprint(
        str(parent / "preflight.json")
    )
    config_path = tmp_path / "resource-rejection.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="并未违反启动资源合同"):
        adjudicate(config_path)
