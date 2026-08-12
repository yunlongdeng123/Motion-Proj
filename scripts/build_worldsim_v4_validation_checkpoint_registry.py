#!/usr/bin/env python3
"""登记六个 validation StreetGS formal checkpoint。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_ID = "WS-V4-M1-EVIDENCE-FIELD-01"


class ValidationCheckpointRegistryError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationCheckpointRegistryError(f"JSON root is not a mapping: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationCheckpointRegistryError(f"YAML root is not a mapping: {path}")
    return payload


def parse_run_bindings(values: Sequence[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        scene, separator, path = value.partition("=")
        if not separator or not scene or not path or scene in output:
            raise ValidationCheckpointRegistryError(f"invalid/duplicate run binding: {value}")
        output[scene] = Path(path).resolve()
    return output


def build_registry(
    *, config_path: Path, run_bindings: Mapping[str, Path]
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if (
        config.get("schema_version") != "worldsim_v4_streetgs_training_v1"
        or config.get("task_id") != TASK_ID
    ):
        raise ValidationCheckpointRegistryError("validation reconstruction config drift")
    expected = list(config["scenes"])
    if set(run_bindings) != set(expected):
        raise ValidationCheckpointRegistryError("validation run scene set drift")
    checkpoints = {}
    for scene in expected:
        run = Path(run_bindings[scene])
        status = load_json(run / "status.json")
        summary = load_json(run / "summary.json")
        manifest = load_json(run / "manifest.json")
        if (
            status.get("status") != "done"
            or status.get("task_id") != TASK_ID
            or status.get("scene") != scene
            or status.get("mode") != "formal"
            or status.get("summary_sha256") != sha256_file(run / "summary.json")
            or summary.get("status") != "done"
            or summary.get("mode") != "formal"
            or summary.get("scene") != scene
            or summary.get("iterations") != 30000
            or summary.get("project_git", {}).get("dirty") is not False
            or summary.get("test_quality_read") is not False
            or manifest.get("status") != "done"
            or manifest.get("scene") != scene
            or manifest.get("mode") != "formal"
            or manifest.get("test_quality_read") is not False
        ):
            raise ValidationCheckpointRegistryError(f"StreetGS formal contract drift: {scene}")
        checkpoint = summary["checkpoint"]
        checkpoint_path = Path(checkpoint["path"])
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != int(checkpoint["bytes"])
            or sha256_file(checkpoint_path) != checkpoint["sha256"]
            or checkpoint.get("step") != 30000
            or checkpoint.get("means_finite") is not True
        ):
            raise ValidationCheckpointRegistryError(f"StreetGS checkpoint drift: {scene}")
        source_config = checkpoint_path.parent / "config.yaml"
        if not source_config.is_file():
            raise ValidationCheckpointRegistryError(f"StreetGS source config missing: {scene}")
        manifest_checkpoint = manifest.get("artifacts", {}).get("work_dirs_checkpoint")
        if manifest_checkpoint != checkpoint:
            raise ValidationCheckpointRegistryError(f"manifest checkpoint drift: {scene}")
        checkpoints[scene] = {
            **checkpoint,
            "run": str(run),
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "summary_sha256": sha256_file(run / "summary.json"),
            "manifest_sha256": sha256_file(run / "manifest.json"),
            "status_sha256": sha256_file(run / "status.json"),
            "fingerprint_sha256": sha256_file(run / "fingerprint.json"),
        }
    return {
        "schema_version": "worldsim_v4_streetgs_checkpoint_registry_v1",
        "task_id": TASK_ID,
        "status": "done",
        "split": "validation",
        "partition_contract": "sample_index_mod_5",
        "scene_order": expected,
        "checkpoints": checkpoints,
        "source_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "development_content_read": False,
        "validation_quality_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }


def atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_registry(
        config_path=args.config.resolve(),
        run_bindings=parse_run_bindings(args.run),
    )
    atomic_yaml(args.output, payload)
    print(
        json.dumps(
            {
                "status": "done",
                "scene_count": len(payload["checkpoints"]),
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "test_quality_read": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
