#!/usr/bin/env python3
"""Independently audit the r045 Trace3D exact-asset recovery preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.audit_worldsim_v51_f0b_three_view_association_parity import _load_json, _load_jsonl, _load_yaml
from scripts.audit_worldsim_v51_f0c_upstream_batch_association_repeatability import _manifest_inventory
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"
EXPECTED_RUN_NAME = "20260818T220000Z__m1-stage-g-g0-trace3d-source-recovery-s20260814-r045"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _page_marker_count(path: Path) -> int:
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF"):
        raise ProtocolError("r045 paper magic drift")
    return len(re.findall(rb"/Type\s*/Page\b", payload))


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    if run_dir.name != EXPECTED_RUN_NAME:
        raise ProtocolError("r045 run identity drift")
    config = _load_yaml(config_path)
    conclusion = config["decision"]["expected_conclusion"]
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if status.get("task_id") != TASK_ID or status.get("status") != "done" or status.get("conclusion") != conclusion:
        raise ProtocolError("r045 terminal drift")
    source_commit = status["source_commit"]
    source_tree = _git(PROJECT, "show", "-s", "--format=%T", source_commit)
    resolved_path = run_dir / "resolved_config.yaml"
    committed_config = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"])
    if resolved_path.read_bytes() != committed_config:
        raise ProtocolError("r045 resolved config drift")

    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r045 event drift")
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "artifacts/source_report.json"
    resources_path = run_dir / "artifacts/resources.json"
    summary = _load_json(summary_path)
    report = _load_json(report_path)
    resources = _load_json(resources_path)
    if (
        summary.get("status") != "done" or summary.get("conclusion") != conclusion
        or summary.get("source_commit") != source_commit or summary.get("source_tree") != source_tree
        or summary.get("source_report") != report or summary.get("resources") != resources
        or not all(summary.get("resource_checks", {}).values())
    ):
        raise ProtocolError("r045 summary/report/resource drift")

    false_fields = (
        "network_access", "assets_redownloaded", "source_code_execution", "submodules_initialized",
        "model_download", "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation",
    )
    for field in false_fields:
        if summary.get(field) is not False:
            raise ProtocolError(f"r045 summary false-lock drift: {field}")
    if report.get("assets_reused_exact") is not True or report.get("assets_redownloaded") is not False:
        raise ProtocolError("r045 asset reuse declaration drift")
    for field in ("network_access", "source_code_execution", "submodules_initialized", "model_download", "image_pixels_read", "mask_pixels_read", "quality_metrics_read"):
        if report.get(field) is not False:
            raise ProtocolError(f"r045 report false-lock drift: {field}")

    paper_spec = config["published_assets"]["paper"]
    paper_path = Path(paper_spec["path"])
    paper = report["paper"]
    if (
        paper_path.stat().st_size != int(paper_spec["bytes"])
        or sha256_file(paper_path) != paper_spec["sha256"]
        or paper.get("bytes") != int(paper_spec["bytes"])
        or paper.get("sha256") != paper_spec["sha256"]
        or paper.get("page_marker_count") != _page_marker_count(paper_path)
        or paper.get("page_marker_count") != 11
        or paper.get("page_count_method") != "standard_library_regex_type_page"
    ):
        raise ProtocolError("r045 paper audit drift")

    repo_spec = config["published_assets"]["repository"]
    repo_path = Path(repo_spec["path"])
    repository = report["repository"]
    commit = _git(repo_path, "rev-parse", "HEAD")
    tree = _git(repo_path, "rev-parse", "HEAD^{tree}")
    status_porcelain = _git(repo_path, "status", "--porcelain")
    remote_url = _git(repo_path, "remote", "get-url", "origin")
    submodule_lines = [line for line in _git(repo_path, "submodule", "status").splitlines() if line]
    if (
        commit != repo_spec["commit"] or tree != repo_spec["tree"] or status_porcelain
        or remote_url != repo_spec["url"] or len(submodule_lines) != int(repo_spec["submodule_pointer_count"])
        or any(not line.startswith("-") for line in submodule_lines)
        or repository.get("commit") != commit or repository.get("tree") != tree
        or repository.get("status_porcelain") != status_porcelain or repository.get("submodule_status") != submodule_lines
        or repository.get("submodules_initialized") is not False
    ):
        raise ProtocolError("r045 repository audit drift")

    method_evidence = repository["method_evidence"]
    if set(method_evidence) != set(config["method_source_map"]):
        raise ProtocolError("r045 method evidence denominator drift")
    verified_source_bytes = 0
    for contract, spec in config["method_source_map"].items():
        evidence = method_evidence[contract]
        path = repo_path / spec["path"]
        source = path.read_text(encoding="utf-8", errors="replace")
        if (
            evidence.get("path") != spec["path"] or evidence.get("bytes") != path.stat().st_size
            or evidence.get("sha256") != sha256_file(path) or evidence.get("required_markers") != spec["required_markers"]
            or evidence.get("all_markers_present") is not True
            or any(marker not in source for marker in spec["required_markers"])
        ):
            raise ProtocolError(f"r045 method evidence drift: {contract}")
        verified_source_bytes += path.stat().st_size

    boundary = report["normative_adapter_boundary"]
    if boundary != config["normative_adapter_boundary"] or boundary.get("immutable_base") is not True:
        raise ProtocolError("r045 immutable-base boundary drift")
    if summary.get("next_action") != config["decision"]["next_action"] or summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise ProtocolError("r045 next-action/governance drift")

    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("task_id") != TASK_ID or manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r045 manifest inventory drift")
    return {
        "schema_version": "worldsim_v51_stage_g_g0_r045_audit_v1", "task_id": TASK_ID, "status": "pass",
        "conclusion": conclusion, "run_dir": str(run_dir), "source_commit": source_commit, "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "source_report": {"bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "entry_count": len(inventory), "logical_bytes": sum(int(row["bytes"]) for row in inventory)},
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "paper": {"bytes": paper_path.stat().st_size, "sha256": sha256_file(paper_path), "page_marker_count": 11},
        "repository": {"commit": commit, "tree": tree, "clean": True, "submodule_pointer_count": len(submodule_lines), "submodules_initialized": False, "verified_method_source_bytes": verified_source_bytes},
        "resources": resources, "resource_checks": summary["resource_checks"],
        "assets_reused_exact": True, "assets_redownloaded": False, "source_code_execution": False,
        "image_pixels_read": False, "mask_pixels_read": False, "quality_metrics_read": False,
        "training": False, "gaussian_mutation": False, "failure_ledger_delta": "V51-F64_resolved",
        "next_action": config["decision"]["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_v2.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.config.resolve(), args.run_dir.resolve())
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
