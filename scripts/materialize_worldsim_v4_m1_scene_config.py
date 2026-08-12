#!/usr/bin/env python3
"""Bind a frozen V3.3 development chain into one M1 scene configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class M1MaterializationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M1MaterializationError(f"YAML root is not a mapping: {path}")
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M1MaterializationError(f"JSON root is not a mapping: {source}")
    return payload


def _verified(path: str | Path, expected: str, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise M1MaterializationError(f"{label} is missing: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise M1MaterializationError(
            f"{label} SHA drift: expected={expected} actual={actual}"
        )
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def _copy_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence": config["evidence"],
        "calibration": config["calibration"],
        "evaluation": config["evaluation"],
        "immutability": config["immutability"],
        "gates": config["gates"],
    }


def materialize_scene_config(
    *, project_root: Path, config_path: Path, scene: str
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v4_m1_evidence_v1":
        raise M1MaterializationError("M1 config schema drift")
    protocol = config["protocol"]
    if scene not in protocol["development_scenes"]:
        raise M1MaterializationError(f"scene is not a frozen development scene: {scene}")
    if protocol.get("test_quality_read") is not False:
        raise M1MaterializationError("M1 config does not seal test quality")

    matrix_path = project_root / config["inputs"]["baseline_matrix"]
    matrix = _load_yaml(matrix_path)
    chains = matrix["baselines"]["v33_frozen"]["executable_scene_chains"]
    if scene not in chains:
        raise M1MaterializationError(f"scene lacks a registered V3.3 chain: {scene}")
    registration = chains[scene]
    chain_file = registration["files"]["scene_chain.json"]
    chain_binding = _verified(
        chain_file["path"], chain_file["sha256"], label="V3.3 scene chain"
    )
    chain = _load_json(chain_binding["path"])
    if chain.get("scene") != scene or chain.get("test_quality_read") is not False:
        raise M1MaterializationError("V3.3 scene-chain identity/test contract drift")
    instance_stage = chain["stages"]["instance_field"]
    common = {
        "schema_version": "worldsim_v4_m1_scene_v1",
        "task_id": config["task_id"],
        "scene": scene,
        "partition": "development",
        "source_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "baseline_matrix": {
            "path": str(matrix_path),
            "sha256": sha256_file(matrix_path),
        },
        "v33_scene_chain": chain_binding,
        "v33_algorithm_commit": registration["algorithm_commit"],
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        **_copy_parameters(config),
    }
    if instance_stage["status"] == "abstain":
        if scene not in protocol["abstain_no_actor_scenes"]:
            raise M1MaterializationError("unregistered M1 abstention")
        return {
            **common,
            "status": "abstain",
            "reason": instance_stage.get("reason", "ABSTAIN_NO_ACTOR"),
            "actors": {},
        }
    if instance_stage["status"] != "done":
        raise M1MaterializationError(f"V3.3 instance stage is not terminal: {scene}")

    run_dir = Path(instance_stage["run"])
    stage_summary_path = run_dir / "stage_summary.json"
    _verified(
        stage_summary_path,
        instance_stage["summary_sha256"],
        label="V3.3 instance stage summary",
    )
    stage_summary = _load_json(stage_summary_path)
    if stage_summary.get("selected_arm") != config["inputs"]["required_v33_arm"]:
        raise M1MaterializationError("V3.3 selected arm is not frozen O1")
    if any(
        stage_summary.get(name) is not expected
        for name, expected in (
            ("development_optimization_read", False),
            ("development_content_read", True),
            ("heldout_content_read", False),
            ("test_quality_read", False),
        )
    ):
        raise M1MaterializationError("V3.3 partition provenance drift")

    resolved_path = run_dir / "resolved.yaml"
    resolved = _load_yaml(resolved_path)
    summary_path = run_dir / "instance_field" / "summary.json"
    summary = _load_json(summary_path)
    arm = summary["arms"][config["inputs"]["required_v33_arm"]]
    field_binding = _verified(
        arm["instance_field"], arm["instance_field_sha256"], label="V3.3 O1 field"
    )
    checkpoint_binding = _verified(
        resolved["inputs"]["checkpoint"],
        resolved["inputs"]["checkpoint_sha256"],
        label="base RGB checkpoint",
    )
    source_config_binding = _verified(
        resolved["inputs"]["source_config"],
        resolved["inputs"]["source_config_sha256"],
        label="DriveStudio source config",
    )
    eval_manifest = summary["evaluation_source"]
    evaluation_masks = _verified(
        eval_manifest["manifest"],
        eval_manifest["manifest_sha256"],
        label="development evaluation masks",
    )
    if not eval_manifest.get("optimization_forbidden") or eval_manifest.get(
        "partition"
    ) != "development":
        raise M1MaterializationError("development evaluation masks are not sealed")
    actors = {}
    for role, actor in resolved["actors"].items():
        actors[role] = {
            "instance_token": actor["instance_token"],
            "dataset_instance_id": int(actor["dataset_instance_id"]),
            "rigid_model_index": int(actor["rigid_model_index"]),
            "semantic_sidecar": _verified(
                actor["semantic_sidecar"],
                actor["semantic_sidecar_sha256"],
                label=f"semantic sidecar/{role}",
            ),
        }
    if not actors:
        raise M1MaterializationError("active M1 scene has no actor")
    return {
        **common,
        "status": "ready",
        "actors": actors,
        "inputs": {
            "checkpoint": checkpoint_binding,
            "drivestudio_source_config": source_config_binding,
            "v33_o1_instance_field": field_binding,
            "development_evaluation_masks": evaluation_masks,
            "processed_scene_dir": resolved["scene"]["processed_scene_dir"],
        },
        "runtime": resolved["runtimes"],
        "v33_reference": {
            "aggregate": arm["evaluation"]["aggregate"],
            "rows": arm["evaluation"]["rows"],
        },
    }


def atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v4/m1_evidence_v1.yaml"),
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    payload = materialize_scene_config(
        project_root=project_root, config_path=config_path, scene=args.scene
    )
    atomic_yaml(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "scene": args.scene, "output": str(args.output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
