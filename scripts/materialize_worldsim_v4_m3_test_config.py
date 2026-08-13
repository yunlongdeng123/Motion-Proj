#!/usr/bin/env python3
"""Materialize the frozen M3 test config without reading test quality."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"


class M3TestConfigError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3TestConfigError(f"YAML root must be a mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3TestConfigError(f"JSON root must be a mapping: {path}")
    return value


def materialize(
    *,
    base_config_path: Path,
    inventory_path: Path,
    validation_run: Path,
) -> dict[str, Any]:
    config = load_yaml(base_config_path)
    inventory = load_yaml(inventory_path)
    validation_summary_path = validation_run / "summary.json"
    validation_manifest_path = validation_run / "manifest.json"
    validation_status_path = validation_run / "status.json"
    validation = load_json(validation_summary_path)
    status = load_json(validation_status_path)
    if (
        config.get("task_id") != TASK_ID
        or inventory.get("task_id") != TASK_ID
        or inventory.get("schema_version")
        != "worldsim_v4_m3_test_asset_inventory_v1"
    ):
        raise M3TestConfigError("M3 task/config/inventory drift")
    if (
        inventory.get("asset_summary", {}).get("scene_count") != 18
        or len(inventory.get("scene_order", [])) != 18
        or any(
            row.get("partition") != "test"
            for row in inventory.get("scenes", {}).values()
        )
        or inventory.get("test_quality_read") is not False
    ):
        raise M3TestConfigError("M3 test asset inventory contract drift")
    if (
        validation.get("status") != "done"
        or validation.get("validation_gate_passed") is not True
        or validation.get("test_freeze_authorized") is not True
        or validation.get("test_quality_read") is not False
        or status.get("summary_sha256") != sha256_file(validation_summary_path)
        or status.get("manifest_sha256") != sha256_file(validation_manifest_path)
        or validation.get("selected_parameters")
        != config.get("trajectory", {}).get("selected_parameters")
    ):
        raise M3TestConfigError("M3 validation freeze evidence drift")
    output = dict(config)
    output["schema_version"] = "worldsim_v4_m3_test_temporal_v1"
    output["status"] = "test_prepared"
    output["inputs"] = dict(config["inputs"])
    output["inputs"]["scene_inventory"] = {
        "path": str(inventory_path.resolve()),
        "sha256": sha256_file(inventory_path),
    }
    output["test_protocol"] = {
        "scene_order": list(inventory["scene_order"]),
        "scene_count": 18,
        "partition": "test",
        "quality_read_count": 1,
        "requires_committed_freeze": "V4_TEST_FREEZE.json",
        "exact_once_attempt_ledger": True,
        "parameter_search": False,
        "threshold_search": False,
        "validation_freeze": {
            "path": str(validation_run.resolve()),
            "summary_sha256": sha256_file(validation_summary_path),
            "manifest_sha256": sha256_file(validation_manifest_path),
            "status_sha256": sha256_file(validation_status_path),
        },
        "asset_inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
        },
        "test_quality_read": False,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = materialize(
        base_config_path=args.base_config.resolve(),
        inventory_path=args.inventory.resolve(),
        validation_run=args.validation_run.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
