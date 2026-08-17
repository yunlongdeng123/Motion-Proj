#!/usr/bin/env python3
"""Acquire and audit official Trace3D source without executing model code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_h_uplift import _inventory, _utc_now, _write_json, _write_jsonl, _write_text


SCHEMA = "worldsim_v51_stage_g_g0_trace3d_source_method_preflight_v1"
TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _validate_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("G0 Trace3D config drift")
    auth = config["authorization"]["gaussian_grouping_closeout"]
    freeze = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "Gaussian Grouping closeout", int(auth["bytes"])))
    if (
        freeze.get("status") != auth["required_status"]
        or freeze["governance"].get("next_task") != auth["required_next_task"]
        or freeze["governance"].get("next_phase") != auth["required_next_phase"]
    ):
        raise ProtocolError("G0 Trace3D authorization drift")
    repo = config["official_source"]["repository"]
    if repo["commit"] != "7465ad94d8e7e988513c1326bbc015e8b59cc442" or repo.get("initialize_submodules") is not False:
        raise ProtocolError("Trace3D repository identity drift")
    boundary = config["normative_adapter_boundary"]
    if boundary.get("immutable_base") is not True or "gaussian_split" not in boundary["initially_forbidden"] or "no_quality_disagreement_diagnostic" not in boundary["initially_allowed"]:
        raise ProtocolError("Trace3D immutable-base boundary drift")
    if any(config["locks"][name] is not False for name in ("source_code_execution", "submodule_initialization", "model_download", "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation")):
        raise ProtocolError("Trace3D source-only lock drift")
    return config


def _download_atomic(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise ProtocolError(f"refusing existing paper target/partial: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size < 1_000_000 or not partial.read_bytes()[:4] == b"%PDF":
        raise ProtocolError("Trace3D paper download invalid")
    os.replace(partial, target)


def _clone_atomic(url: str, commit: str, target: Path) -> None:
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise ProtocolError(f"refusing existing repository target/partial: {target}")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(partial)], check=True)
    subprocess.run(["git", "-C", str(partial), "checkout", "--detach", commit], check=True)
    if _git(partial, "rev-parse", "HEAD") != commit or _git(partial, "status", "--porcelain"):
        raise ProtocolError("Trace3D checkout identity drift")
    os.replace(partial, target)


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
    paper_spec = config["official_source"]["paper"]
    repo_spec = config["official_source"]["repository"]
    paper_path = Path(paper_spec["target_path"])
    repo_path = Path(repo_spec["target_path"])
    _download_atomic(paper_spec["official_url"], paper_path)
    _clone_atomic(repo_spec["url"], repo_spec["commit"], repo_path)
    page_count_text = subprocess.check_output(["pdfinfo", str(paper_path)], text=True)
    page_line = next(line for line in page_count_text.splitlines() if line.startswith("Pages:"))
    paper = {"path": str(paper_path), "bytes": paper_path.stat().st_size, "sha256": sha256_file(paper_path), "page_count": int(page_line.split(":", 1)[1].strip()), "official_url": paper_spec["official_url"]}
    license_path = repo_path / "LICENSE"
    if "Apache License" not in license_path.read_text(encoding="utf-8"):
        raise ProtocolError("Trace3D license drift")
    required_files = ["LICENSE", "README.md", "requirements.txt", ".gitmodules", "train.py", "train_gaus.py", "merge_patches.py", "remove_ab_gaus.py", "get_sam_masks.py"]
    files = {}
    for name in required_files:
        path = repo_path / name
        if not path.is_file():
            raise ProtocolError(f"Trace3D required source missing: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    python_files = sorted(repo_path.rglob("*.py"))
    source_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in python_files)
    keyword_counts = {keyword: source_text.lower().count(keyword) for keyword in ("trace", "ambiguous", "weight_matrix", "merge", "prune", "densif")}
    if keyword_counts["trace"] == 0 or keyword_counts["merge"] == 0 or keyword_counts["prune"] == 0:
        raise ProtocolError(f"Trace3D method marker drift: {keyword_counts}")
    repository = {
        "path": str(repo_path), "url": repo_spec["url"], "commit": _git(repo_path, "rev-parse", "HEAD"),
        "tree": _git(repo_path, "rev-parse", "HEAD^{tree}"), "status_porcelain": _git(repo_path, "status", "--porcelain"),
        "license": repo_spec["license_expected"], "license_sha256": sha256_file(license_path), "files": files,
        "python_file_count": len(python_files), "keyword_counts": keyword_counts,
        "submodule_status": _git(repo_path, "submodule", "status"), "submodules_initialized": False,
    }
    if repository["commit"] != repo_spec["commit"] or repository["status_porcelain"]:
        raise ProtocolError("Trace3D repository audit drift")
    source_report = {
        "schema_version": "worldsim_v51_g0_trace3d_source_report_v1", "task_id": TASK_ID,
        "paper": paper, "repository": repository, "paper_method_contract": config["paper_method_contract"],
        "normative_adapter_boundary": config["normative_adapter_boundary"],
        "source_code_execution": False, "submodules_initialized": False, "model_download": False,
        "image_pixels_read": False, "mask_pixels_read": False, "quality_metrics_read": False,
    }
    _write_json(run_dir / "artifacts/source_report.json", source_report)
    resources = {"wall_seconds": time.perf_counter() - started, "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text().strip()), "disk_free_after_bytes": shutil.disk_usage(run_dir).free}
    limits = config["resources"]
    checks = {"wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]), "cgroup": resources["cgroup_memory_current_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]), "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"])}
    if not all(checks.values()):
        raise ProtocolError(f"Trace3D source resource gate: {checks}")
    _write_json(run_dir / "artifacts/resources.json", resources)
    summary = {
        "schema_version": "worldsim_v51_g0_trace3d_source_summary_v1", "task_id": TASK_ID, "status": "done",
        "conclusion": config["decision"]["expected_conclusion"], "source_commit": identity["commit"], "source_tree": identity["tree"],
        "source_report": source_report, "resources": resources, "resource_checks": checks,
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
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
