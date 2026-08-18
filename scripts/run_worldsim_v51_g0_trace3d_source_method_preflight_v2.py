#!/usr/bin/env python3
"""Re-audit already-published Trace3D assets without executing upstream code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_h_uplift import _inventory, _utc_now, _write_json, _write_jsonl, _write_text


SCHEMA = "worldsim_v51_stage_g_g0_trace3d_source_method_preflight_v2"
TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"
EXPECTED_ONLY_CHANGES = [
    "reuse_and_reverify_exact_published_assets",
    "replace_external_pdfinfo_with_standard_library_pdf_page_marker_count",
    "preserve_all_source_method_adapter_and_read_locks",
]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _pdf_page_marker_count(path: Path) -> int:
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF"):
        raise ProtocolError("Trace3D paper is not a PDF")
    count = len(re.findall(rb"/Type\s*/Page\b", payload))
    if count <= 0:
        raise ProtocolError("Trace3D paper has no standard page markers")
    return count


def _validate_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("G0 Trace3D recovery config drift")
    auth = config["authorization"]["r044_closeout"]
    closeout = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "r044 closeout", int(auth["bytes"])))
    if (
        closeout.get("status") != auth["required_status"]
        or closeout["run"].get("failure") != auth["required_failure"]
        or closeout["recovery"].get("next_phase") != auth["required_next_phase"]
    ):
        raise ProtocolError("G0 Trace3D recovery authorization drift")
    paper = config["published_assets"]["paper"]
    repo = config["published_assets"]["repository"]
    if paper["sha256"] != closeout["published_assets"]["paper"]["sha256"]:
        raise ProtocolError("Trace3D paper recovery identity drift")
    if repo["commit"] != closeout["published_assets"]["repository"]["commit"] or repo["tree"] != closeout["published_assets"]["repository"]["tree"]:
        raise ProtocolError("Trace3D repository recovery identity drift")
    recovery = config["recovery"]
    if recovery.get("reuse_exact_published_assets") is not True or recovery.get("redownload_assets") is not False or recovery.get("only_changes") != EXPECTED_ONLY_CHANGES:
        raise ProtocolError("Trace3D recovery scope drift")
    boundary = config["normative_adapter_boundary"]
    if boundary.get("immutable_base") is not True or "gaussian_split" not in boundary["initially_forbidden"] or "no_quality_disagreement_diagnostic" not in boundary["initially_allowed"]:
        raise ProtocolError("Trace3D immutable-base boundary drift")
    false_locks = (
        "network_access", "source_code_execution", "submodule_initialization", "model_download",
        "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation",
        "h_quality_read", "screening_quality_read", "confirmation_quality_read", "validation_quality_read",
        "test_quality_read", "kitti_method_tuning",
    )
    if any(config["locks"].get(name) is not False for name in false_locks):
        raise ProtocolError("Trace3D source-only lock drift")
    return config


def _audit_method_sources(repo_path: Path, source_map: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for contract, spec in source_map.items():
        relative = spec["path"]
        path = repo_path / relative
        if not path.is_file():
            raise ProtocolError(f"Trace3D method source missing: {relative}")
        source = path.read_text(encoding="utf-8", errors="replace")
        markers = spec["required_markers"]
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise ProtocolError(f"Trace3D method markers missing in {relative}: {missing}")
        evidence[contract] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "required_markers": markers,
            "all_markers_present": True,
        }
    return evidence


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})
    started = time.perf_counter()

    paper_spec = config["published_assets"]["paper"]
    repo_spec = config["published_assets"]["repository"]
    paper_path = Path(paper_spec["path"])
    repo_path = Path(repo_spec["path"])
    _verify(paper_path, paper_spec["sha256"], "published Trace3D paper", int(paper_spec["bytes"]))
    if not repo_path.is_dir():
        raise ProtocolError("published Trace3D repository missing")
    page_count = _pdf_page_marker_count(paper_path)
    page_gate = paper_spec["page_marker_count"]
    if not int(page_gate["minimum"]) <= page_count <= int(page_gate["maximum"]):
        raise ProtocolError(f"Trace3D paper page-marker count out of gate: {page_count}")

    commit = _git(repo_path, "rev-parse", "HEAD")
    tree = _git(repo_path, "rev-parse", "HEAD^{tree}")
    status_porcelain = _git(repo_path, "status", "--porcelain")
    remote_url = _git(repo_path, "remote", "get-url", "origin")
    if commit != repo_spec["commit"] or tree != repo_spec["tree"] or status_porcelain or remote_url != repo_spec["url"]:
        raise ProtocolError("published Trace3D repository identity drift")
    submodule_lines = [line for line in _git(repo_path, "submodule", "status").splitlines() if line]
    if len(submodule_lines) != int(repo_spec["submodule_pointer_count"]) or any(not line.startswith("-") for line in submodule_lines):
        raise ProtocolError(f"Trace3D submodule initialization drift: {submodule_lines}")
    license_path = repo_path / "LICENSE"
    if "Apache License" not in license_path.read_text(encoding="utf-8"):
        raise ProtocolError("Trace3D license drift")
    method_evidence = _audit_method_sources(repo_path, config["method_source_map"])

    source_report = {
        "schema_version": "worldsim_v51_g0_trace3d_source_report_v2",
        "task_id": TASK_ID,
        "recovery_from": config["authorization"]["r044_closeout"],
        "paper": {
            "path": str(paper_path), "bytes": paper_path.stat().st_size, "sha256": sha256_file(paper_path),
            "page_marker_count": page_count, "page_count_method": page_gate["method"], "official_url": paper_spec["official_url"],
        },
        "repository": {
            "path": str(repo_path), "url": remote_url, "commit": commit, "tree": tree,
            "status_porcelain": status_porcelain, "license": repo_spec["license_expected"],
            "license_sha256": sha256_file(license_path), "submodule_status": submodule_lines,
            "submodules_initialized": False, "method_evidence": method_evidence,
        },
        "paper_method_contract": config["paper_method_contract"],
        "normative_adapter_boundary": config["normative_adapter_boundary"],
        "network_access": False, "assets_reused_exact": True, "assets_redownloaded": False,
        "source_code_execution": False, "submodules_initialized": False, "model_download": False,
        "image_pixels_read": False, "mask_pixels_read": False, "quality_metrics_read": False,
    }
    _write_json(run_dir / "artifacts/source_report.json", source_report)
    resources = {
        "wall_seconds": time.perf_counter() - started,
        "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text().strip()),
        "disk_free_after_bytes": shutil.disk_usage(run_dir).free,
    }
    limits = config["resources"]
    checks = {
        "wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]),
        "cgroup": resources["cgroup_memory_current_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]),
        "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"]),
    }
    if not all(checks.values()):
        raise ProtocolError(f"Trace3D source resource gate: {checks}")
    _write_json(run_dir / "artifacts/resources.json", resources)
    summary = {
        "schema_version": "worldsim_v51_g0_trace3d_source_summary_v2", "task_id": TASK_ID, "status": "done",
        "conclusion": config["decision"]["expected_conclusion"], "source_commit": identity["commit"], "source_tree": identity["tree"],
        "source_report": source_report, "resources": resources, "resource_checks": checks,
        "network_access": False, "assets_reused_exact": True, "assets_redownloaded": False,
        "source_code_execution": False, "submodules_initialized": False, "model_download": False,
        "image_pixels_read": False, "mask_pixels_read": False, "quality_metrics_read": False,
        "training": False, "gaussian_mutation": False, "next_action": config["decision"]["next_action"],
        "m2_status": "pending", "m3_status": "pending",
    }
    _write_json(run_dir / "summary.json", summary)
    events.append({"event": "run_completed", "at_utc": _utc_now()})
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "manifest.json", {"task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "conclusion": summary["conclusion"], "source_commit": identity["commit"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_v2.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
