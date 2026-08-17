#!/usr/bin/env python3
"""独立审计 Stage F F0a r032 的环境、资源与单视图输出合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import sha256_file


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
CONCLUSION = (
    "f0a_environment_and_one_view_resource_smoke_done_"
    "grid_quality_batch_parity_and_association_smoke_required"
)


class AuditError(RuntimeError):
    """r032 证据不再满足预注册合同时抛出。"""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"YAML root must be a mapping: {path}")
    return payload


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def _assert_subset(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise AuditError(f"payload mapping drift: {label}")
        for key, value in expected.items():
            if key not in actual:
                raise AuditError(f"payload field missing: {label}/{key}")
            _assert_subset(actual[key], value, f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditError(f"payload list drift: {label}")
        for index, value in enumerate(expected):
            _assert_subset(actual[index], value, f"{label}/{index}")
        return
    if actual != expected:
        raise AuditError(f"payload value drift: {label}: {actual!r} != {expected!r}")


def _verify_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    expected = {str(row["path"]): row for row in manifest["inventory"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    }
    if set(observed) != set(expected):
        raise AuditError("r032 manifest inventory coverage drift")
    total_bytes = 0
    for relative, path in observed.items():
        row = expected[relative]
        size = path.stat().st_size
        if size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AuditError(f"r032 manifest identity drift: {relative}")
        total_bytes += size
    return {"entry_count": len(observed), "logical_bytes": total_bytes}


def _resource_replay(run_dir: Path, summary: Mapping[str, Any]) -> dict[str, int]:
    samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in samples if "monitor_error" not in row]
    if not valid or len(samples) != int(summary["resources"]["sample_count"]):
        raise AuditError("r032 resource sample denominator drift")
    gpu_peak = max(int(row["gpu_used_mib"]) for row in valid)
    cgroup_peak = max(int(row["cgroup_memory_current_bytes"]) for row in valid)
    resources = summary["resources"]
    if gpu_peak != int(resources["nvidia_peak_mib"]):
        raise AuditError("r032 sampled NVIDIA peak drift")
    if cgroup_peak != int(resources["cgroup_memory_peak_bytes"]):
        raise AuditError("r032 sampled cgroup peak drift")
    if len(samples) - len(valid) != int(resources["monitor_error_count"]):
        raise AuditError("r032 resource monitor error count drift")
    return {
        "sample_count": len(samples),
        "monitor_error_count": len(samples) - len(valid),
        "nvidia_peak_mib": gpu_peak,
        "cgroup_memory_peak_bytes": cgroup_peak,
    }


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_f_f0a_environment_one_view_smoke_v6":
        raise AuditError("r032 config schema drift")
    status = _json(run_dir / "status.json")
    manifest = _json(run_dir / "manifest.json")
    summary = _json(run_dir / "summary.json")
    report = _json(run_dir / "artifacts/environment_smoke_report.json")
    one_view = _json(run_dir / "artifacts/one_view_report.json")
    environment = _json(run_dir / "artifacts/environment_lock.json")
    for label, payload in (
        ("status", status),
        ("manifest", manifest),
        ("summary", summary),
        ("report", report),
    ):
        if payload.get("status") != "done" or payload.get("task_id") != TASK_ID:
            raise AuditError(f"r032 {label} terminal drift")
    if any(payload.get("conclusion") != CONCLUSION for payload in (status, summary, report)):
        raise AuditError("r032 conclusion drift")
    if (run_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"
    ) != config_path.read_text(encoding="utf-8"):
        raise AuditError("r032 resolved config is not byte-exact")

    inventory = _verify_inventory(run_dir, manifest)
    if inventory != {"entry_count": 13, "logical_bytes": 157563}:
        raise AuditError(f"r032 inventory denominator drift: {inventory}")
    tree = _git(PROJECT, "rev-parse", f"{summary['source_commit']}^{{tree}}")
    if status["source_commit"] != summary["source_commit"] or tree != summary["source_tree"]:
        raise AuditError("r032 project source identity drift")
    for key, value in report.items():
        if key != "schema_version":
            if key not in summary:
                raise AuditError(f"summary omits report field: {key}")
            _assert_subset(summary[key], value, f"summary/report/{key}")
    _assert_subset(summary["one_view"], one_view, "summary/one_view")
    _assert_subset(summary["environment"], environment, "summary/environment")

    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        source_root = Path(spec["path"])
        if _git(source_root, "rev-parse", "HEAD") != spec["commit"]:
            raise AuditError(f"external source commit drift: {name}")
        if _git(source_root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise AuditError(f"external source tree drift: {name}")
        if _git(source_root, "status", "--porcelain"):
            raise AuditError(f"external source checkout not clean: {name}")

    hidden_specs = config["hidden_torchvision_assets"]["assets"]
    observed_hidden = {row["name"]: row for row in environment["hidden_torchvision_assets"]["assets"]}
    if set(observed_hidden) != set(hidden_specs):
        raise AuditError("hidden torchvision asset denominator drift")
    assets = []
    for name, spec in hidden_specs.items():
        row = observed_hidden[name]
        path = Path(row["path"])
        if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
            raise AuditError(f"hidden torchvision asset identity drift: {name}")
        if Path(f"{path}{config['hidden_torchvision_assets']['partial_suffix']}").exists():
            raise AuditError(f"hidden torchvision partial remains: {name}")
        assets.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    for row in environment["wheel_records"]:
        path = Path(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise AuditError(f"wheel identity drift: {row['distribution']}")
    if environment["pinned_import_versions"] != {
        "supervision": "0.14.0",
        "pulp": "2.7.0",
        "gurobipy": "12.0.3",
    }:
        raise AuditError("isolated environment import version drift")
    if environment["solver_smokes"]["gurobi"]["status"] != 2 or environment[
        "solver_smokes"
    ]["gurobi"]["solution"] != 1.0:
        raise AuditError("Gurobi solver smoke drift")
    if environment["solver_smokes"]["pulp"]["status_name"] != "Optimal" or environment[
        "solver_smokes"
    ]["pulp"]["solution"] != 1.0:
        raise AuditError("PuLP solver smoke drift")

    mask_path = Path(one_view["mask_path"])
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    labels, counts = np.unique(mask, return_counts=True)
    histogram = {str(int(label)): int(count) for label, count in zip(labels, counts)}
    if list(mask.shape) != [900, 1600] or str(mask.dtype) != "uint8":
        raise AuditError("r032 output mask schema drift")
    if histogram != one_view["unique_label_histogram"] or histogram != {"0": 1440000}:
        raise AuditError("r032 one-view boundary histogram drift")
    metadata = _json(Path(one_view["metadata_path"]))
    if not isinstance(metadata.get("annotations"), list) or len(metadata["annotations"]) != 1:
        raise AuditError("r032 pred.json denominator drift")
    command = one_view["command"]
    side_index = command.index("--SAM_NUM_POINTS_PER_SIDE")
    batch_index = command.index("--SAM_NUM_POINTS_PER_BATCH")
    if command[side_index + 1] != "32" or command[batch_index + 1] != "32":
        raise AuditError("r032 SAM grid/batch command drift")
    stdout = (run_dir / "artifacts/one_view_stdout.log").read_text(encoding="utf-8")
    stderr = (run_dir / "artifacts/one_view_stderr.log").read_text(encoding="utf-8")
    if "'SAM_NUM_POINTS_PER_SIDE': 32" not in stdout or "'SAM_NUM_POINTS_PER_BATCH': 32" not in stdout:
        raise AuditError("r032 upstream configuration stdout drift")
    if "Downloading:" in stderr or "Traceback" in stderr:
        raise AuditError("r032 upstream stderr provenance/runtime drift")

    resources = _resource_replay(run_dir, summary)
    if not all(summary["resource_checks"].values()):
        raise AuditError("r032 resource gate drift")
    if resources["nvidia_peak_mib"] > int(config["resources"]["maximum_nvidia_peak_mib"]):
        raise AuditError("r032 NVIDIA ceiling drift")
    false_locks = (
        "materialization_authorized",
        "identity_training_authorized",
        "quality_read",
        "parameter_search",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
        "f1_execution",
        "f2_execution",
        "association_capability_claim",
    )
    for name in false_locks:
        if summary.get(name) is not False:
            raise AuditError(f"r032 research lock drift: {name}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("r032 M2/M3 lock drift")

    return {
        "schema_version": "worldsim_v51_f0a_environment_one_view_audit_v1",
        "task_id": TASK_ID,
        "run": str(run_dir),
        "status": "pass",
        "conclusion": CONCLUSION,
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "inventory": inventory,
        "hidden_torchvision_assets": sorted(assets, key=lambda row: row["name"]),
        "wheel_count": len(environment["wheel_records"]),
        "solver_smokes_pass": True,
        "mask_sha256": sha256_file(mask_path),
        "mask_histogram": histogram,
        "metadata_sha256": sha256_file(Path(one_view["metadata_path"])),
        "annotation_count": len(metadata["annotations"]),
        "sam_num_points_per_side": 32,
        "sam_num_points_per_batch": 32,
        "resources": resources,
        "resource_checks_pass": True,
        "quality_read": False,
        "association_capability_claim": False,
        "materialization_authorized": False,
        "identity_training_authorized": False,
        "m2_status": "pending",
        "m3_status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
