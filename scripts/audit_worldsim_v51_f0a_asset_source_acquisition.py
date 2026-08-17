#!/usr/bin/env python3
"""独立审计 Stage F F0a r026 的源、权重与 train-only 输入冻结。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.feature_sidecar import record_chain_sha256


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
CONCLUSION = "f0a_assets_and_sources_frozen_environment_setup_required"


class AuditError(RuntimeError):
    """r026 冻结证据不再满足登记合同时抛出。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _verify(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise AuditError(f"identity drift: {label}: {path}")
    return path


def _assert_payload(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AuditError(f"payload field drift: {label}")
        for key, value in expected.items():
            _assert_payload(actual[key], value, f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditError(f"payload list drift: {label}")
        for index, value in enumerate(expected):
            _assert_payload(actual[index], value, f"{label}/{index}")
        return
    if actual != expected:
        raise AuditError(f"payload value drift: {label}: {actual!r} != {expected!r}")


def _verify_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    expected = {row["path"]: row for row in manifest["inventory"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    }
    if set(observed) != set(expected):
        raise AuditError("run manifest inventory coverage drift")
    total_bytes = 0
    for relative, path in observed.items():
        row = expected[relative]
        size = path.stat().st_size
        if size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise AuditError(f"run manifest identity drift: {relative}")
        total_bytes += size
    return {"entry_count": len(observed), "bytes": total_bytes}


def _replay_selected_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    input_config = config["train_only_input"]
    source_path = _verify(
        Path(input_config["image_manifest"]["path"]),
        input_config["image_manifest"]["sha256"],
        "source image manifest",
    )
    source = _json(source_path)
    if record_chain_sha256(source["records"]) != input_config["image_manifest"][
        "record_chain_sha256"
    ]:
        raise AuditError("source image record-chain drift")
    wanted = {
        (scene, int(frame), int(camera))
        for scene in input_config["scenes"]
        for frame in input_config["frames"]
        for camera in input_config["cameras"]
    }
    table = {}
    for raw in source["records"]:
        key = (str(raw["scene"]), int(raw["frame"]), int(raw["camera"]))
        if str(raw["role"]) != input_config["role"] or key not in wanted:
            continue
        if key in table:
            raise AuditError(f"duplicate selected input: {key}")
        table[key] = dict(raw)
    if set(table) != wanted:
        raise AuditError("selected input denominator drift")
    selected = [
        table[(scene, int(frame), int(camera))]
        for scene in input_config["scenes"]
        for frame in input_config["frames"]
        for camera in input_config["cameras"]
    ]
    expected_size = [int(value) for value in input_config["expected_image_size_wh"]]
    for row in selected:
        path = Path(row["path"])
        if [int(row["width"]), int(row["height"])] != expected_size:
            raise AuditError(f"selected image dimension drift: {path}")
        if path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise AuditError(f"selected image identity drift: {path}")
    return selected


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _yaml(config_path)
    status = _json(run_dir / "status.json")
    manifest = _json(run_dir / "manifest.json")
    summary = _json(run_dir / "summary.json")
    report = _json(run_dir / "artifacts/asset_source_report.json")
    for label, payload in (("status", status), ("manifest", manifest), ("summary", summary), ("report", report)):
        if payload.get("status") != "done" or payload.get("task_id") != TASK_ID:
            raise AuditError(f"r026 {label} terminal drift")
    if any(payload.get("conclusion") != CONCLUSION for payload in (status, summary, report)):
        raise AuditError("r026 conclusion drift")
    if (run_dir / "resolved_config.yaml").read_text(encoding="utf-8") != config_path.read_text(
        encoding="utf-8"
    ):
        raise AuditError("r026 resolved config is not byte-exact")
    inventory = _verify_inventory(run_dir, manifest)
    if inventory != {"entry_count": 7, "bytes": 49468}:
        raise AuditError(f"r026 inventory denominator drift: {inventory}")

    tree = _git(PROJECT, "rev-parse", f"{summary['source_commit']}^{{tree}}")
    if summary["source_commit"] != status["source_commit"] or tree != summary["source_tree"]:
        raise AuditError("r026 project source identity drift")
    for key, value in report.items():
        if key == "schema_version":
            continue
        if key not in summary:
            raise AuditError(f"summary omits report field: {key}")
        _assert_payload(summary[key], value, f"summary/report/{key}")

    source_spec = config["official_source"]["grounded_segment_anything_fork"]
    source_root = Path(source_spec["target"])
    source = report["grounded_segment_anything"]
    if _git(source_root, "rev-parse", "HEAD") != source_spec["commit"]:
        raise AuditError("Grounded Segment Anything commit drift")
    if _git(source_root, "rev-parse", "HEAD^{tree}") != source["tree"]:
        raise AuditError("Grounded Segment Anything tree drift")
    if _git(source_root, "status", "--porcelain") or not (
        source_root / source_spec["required_subtree"]
    ).is_dir():
        raise AuditError("Grounded Segment Anything clean/subtree drift")
    license_path = Path(source["license_path"])
    if _sha256(license_path) != source["license_sha256"]:
        raise AuditError("Grounded Segment Anything license drift")

    asset_specs = {row["name"]: row for row in config["assets"]["records"]}
    if len(report["assets"]) != len(asset_specs):
        raise AuditError("asset denominator drift")
    assets = []
    for row in report["assets"]:
        spec = asset_specs[row["name"]]
        path = Path(row["path"])
        if path.stat().st_size != int(spec["expected_bytes"]):
            raise AuditError(f"asset byte drift: {row['name']}")
        digest = _sha256(path)
        if digest != row["sha256"] or Path(f"{path}{config['assets']['atomic_partial_suffix']}").exists():
            raise AuditError(f"asset hash/atomic-publish drift: {row['name']}")
        assets.append({"name": row["name"], "bytes": path.stat().st_size, "sha256": digest})

    selected = _replay_selected_inputs(config)
    selected_manifest = _json(run_dir / "artifacts/train_only_image_manifest.json")
    expected_selected = {
        "schema_version": "worldsim_v51_f0a_train_only_image_manifest_v1",
        "task_id": TASK_ID,
        "order": config["train_only_input"]["order"],
        "record_count": len(selected),
        "total_bytes": sum(int(row["bytes"]) for row in selected),
        "record_chain_sha256": record_chain_sha256(selected),
        "records": selected,
        "image_pixels_decoded": False,
    }
    _assert_payload(selected_manifest, expected_selected, "selected input manifest")

    environment = report["environment"]
    for name, expected in environment["modules_expected_present"].items():
        if expected is not True or importlib.util.find_spec(name) is None:
            raise AuditError(f"required environment module drift: {name}")
    for name, expected in environment["modules_expected_absent"].items():
        if expected is not True or importlib.util.find_spec(name) is not None:
            raise AuditError(f"pre-isolation environment module drift: {name}")

    samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in samples if "monitor_error" not in row]
    resources = summary["resources"]
    if len(samples) != int(resources["sample_count"]) or not valid:
        raise AuditError("resource sample denominator drift")
    if max(int(row["gpu_used_mib"]) for row in valid) != int(resources["nvidia_peak_mib"]):
        raise AuditError("sampled NVIDIA peak drift")
    if max(int(row["cgroup_memory_current_bytes"]) for row in valid) != int(
        resources["cgroup_memory_peak_bytes"]
    ):
        raise AuditError("sampled cgroup peak drift")
    if not all(summary["resource_checks"].values()):
        raise AuditError("r026 resource gate drift")

    false_locks = (
        "materialization_authorized",
        "identity_training_authorized",
        "quality_read",
        "sam_execution",
        "deva_execution",
        "parameter_search",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
        "f1_execution",
        "f2_execution",
    )
    for name in false_locks:
        if summary.get(name) is not False:
            raise AuditError(f"quality/execution lock drift: {name}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("M2/M3 lock drift")

    return {
        "schema_version": "worldsim_v51_f0a_asset_source_audit_v1",
        "task_id": TASK_ID,
        "run": str(run_dir),
        "status": "pass",
        "conclusion": CONCLUSION,
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "inventory": inventory,
        "grounded_segment_anything_commit": source_spec["commit"],
        "grounded_segment_anything_tree": source["tree"],
        "assets": assets,
        "train_only_view_count": len(selected),
        "train_only_total_bytes": selected_manifest["total_bytes"],
        "train_only_record_chain_sha256": selected_manifest["record_chain_sha256"],
        "image_pixels_decoded": False,
        "environment_mutated": False,
        "resource_sample_count": len(samples),
        "quality_and_execution_locks_preserved": True,
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
