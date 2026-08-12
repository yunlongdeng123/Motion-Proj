#!/usr/bin/env python3
"""把 validation V3.3 instance run 绑定为 M1 只读 scene config。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_worldsim_v4_m1_scene_config import (
    M1MaterializationError,
    _copy_parameters,
    _load_json,
    _load_yaml,
    _verified,
    atomic_yaml,
    sha256_file,
)


def _common(config: Mapping[str, Any], config_path: Path, scene: str) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v4_m1_evidence_v1":
        raise M1MaterializationError("M1 config schema drift")
    if config.get("status") != "development_frozen":
        raise M1MaterializationError("validation requires development_frozen M1 config")
    if scene not in config["protocol"]["validation_scenes"]:
        raise M1MaterializationError(f"scene is not frozen validation: {scene}")
    if config["protocol"].get("test_quality_read") is not False:
        raise M1MaterializationError("M1 validation does not seal test quality")
    return {
        "schema_version": "worldsim_v4_m1_scene_v1",
        "task_id": config["task_id"],
        "scene": scene,
        "partition": "validation",
        "source_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "development_content_read": False,
        "validation_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        **_copy_parameters(config),
    }


def materialize_ready(
    *,
    m1_config: Mapping[str, Any],
    m1_config_path: Path,
    instance_config_path: Path,
    instance_run: Path,
) -> dict[str, Any]:
    instance = _load_yaml(instance_config_path)
    if instance.get("schema_version") != "worldsim_v4_v33_instance_field_v1":
        raise M1MaterializationError("validation instance config schema drift")
    scene = str(instance.get("scene", {}).get("name"))
    common = _common(m1_config, m1_config_path, scene)
    if instance.get("task_id") != m1_config["task_id"]:
        raise M1MaterializationError("validation instance task ID drift")
    provenance = instance.get("provenance", {})
    if (
        provenance.get("evaluation_partition") != "development"
        or provenance.get("development_content_read") is not False
        or provenance.get("heldout_content_read") is not False
        or provenance.get("test_quality_read") is not False
    ):
        raise M1MaterializationError("validation instance provenance drift")
    status_binding = _verified(
        instance_run / "status.json",
        sha256_file(instance_run / "status.json"),
        label="validation instance status",
    )
    status = _load_json(status_binding["path"])
    stage_binding = _verified(
        instance_run / "stage_summary.json",
        status["stage_summary_sha256"],
        label="validation instance stage summary",
    )
    stage = _load_json(stage_binding["path"])
    if (
        status.get("status") != "done"
        or status.get("task_id") != m1_config["task_id"]
        or status.get("evaluation_partition") != "development"
        or stage.get("status") != "done"
        or stage.get("scene") != scene
        or stage.get("config_sha256") != sha256_file(instance_config_path)
        or stage.get("selected_arm") != m1_config["inputs"]["required_v33_arm"]
        or stage.get("test_quality_read") is not False
    ):
        raise M1MaterializationError("validation instance terminal contract drift")
    summary_path = instance_run / "instance_field" / "summary.json"
    summary = _load_json(summary_path)
    arm = summary["arms"][m1_config["inputs"]["required_v33_arm"]]
    field = _verified(
        arm["instance_field"], arm["instance_field_sha256"], label="validation O1 field"
    )
    evaluation = summary["evaluation_source"]
    masks = _verified(
        evaluation["manifest"],
        evaluation["manifest_sha256"],
        label="validation-scene evaluation masks",
    )
    if evaluation.get("partition") != "development" or not evaluation.get(
        "optimization_forbidden"
    ):
        raise M1MaterializationError("validation-scene evaluation masks are not sealed")
    inputs = instance["inputs"]
    checkpoint = _verified(
        inputs["checkpoint"], inputs["checkpoint_sha256"], label="validation checkpoint"
    )
    source_config = _verified(
        inputs["source_config"],
        inputs["source_config_sha256"],
        label="validation DriveStudio source config",
    )
    actors = {}
    for role, actor in instance["actors"].items():
        actors[role] = {
            "instance_token": actor["instance_token"],
            "dataset_instance_id": int(actor["dataset_instance_id"]),
            "rigid_model_index": int(actor["rigid_model_index"]),
            "semantic_sidecar": _verified(
                actor["semantic_sidecar"],
                actor["semantic_sidecar_sha256"],
                label=f"validation semantic sidecar/{role}",
            ),
        }
    if not actors:
        raise M1MaterializationError("validation ready scene has no actor")
    return {
        **common,
        "status": "ready",
        "actors": actors,
        "inputs": {
            "checkpoint": checkpoint,
            "drivestudio_source_config": source_config,
            "v33_o1_instance_field": field,
            "development_evaluation_masks": masks,
            "processed_scene_dir": instance["scene"]["processed_scene_dir"],
        },
        "runtime": instance["runtimes"],
        "v33_reference": {
            "aggregate": arm["evaluation"]["aggregate"],
            "rows": arm["evaluation"]["rows"],
        },
        "validation_instance_run": {
            "run": str(instance_run),
            "status": status_binding,
            "stage_summary": stage_binding,
        },
    }


def materialize_abstain(
    *,
    m1_config: Mapping[str, Any],
    m1_config_path: Path,
    bound_scene_path: Path,
) -> dict[str, Any]:
    binding = _verified(
        bound_scene_path,
        sha256_file(bound_scene_path),
        label="validation bound scene",
    )
    bound = _load_json(binding["path"])
    scene = str(bound.get("scene"))
    common = _common(m1_config, m1_config_path, scene)
    if (
        bound.get("schema_version") != "worldsim_v4_v33_bound_scene_v1"
        or bound.get("cohort_role") != "validation"
        or bound.get("test_quality_read") is not False
    ):
        raise M1MaterializationError("validation bound-scene contract drift")
    high = bound.get("actors", {}).get("high_support", {})
    if high.get("availability") == "available":
        raise M1MaterializationError("available validation high actor cannot abstain")
    return {
        **common,
        "status": "abstain",
        "reason": "ABSTAIN_NO_ACTOR",
        "actors": {},
        "validation_bound_scene": binding,
        "actor_availability": {
            role: actor.get("availability")
            for role, actor in bound.get("actors", {}).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-config", type=Path, required=True)
    parser.add_argument("--instance-config", type=Path)
    parser.add_argument("--instance-run", type=Path)
    parser.add_argument("--bound-scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m1_config_path = args.m1_config.resolve()
    m1_config = _load_yaml(m1_config_path)
    ready_mode = args.instance_config is not None or args.instance_run is not None
    abstain_mode = args.bound_scene is not None
    if ready_mode == abstain_mode:
        raise M1MaterializationError("choose exactly one of ready or abstain mode")
    if ready_mode:
        if args.instance_config is None or args.instance_run is None:
            raise M1MaterializationError("ready mode requires instance config and run")
        payload = materialize_ready(
            m1_config=m1_config,
            m1_config_path=m1_config_path,
            instance_config_path=args.instance_config.resolve(),
            instance_run=args.instance_run.resolve(),
        )
    else:
        payload = materialize_abstain(
            m1_config=m1_config,
            m1_config_path=m1_config_path,
            bound_scene_path=args.bound_scene.resolve(),
        )
    atomic_yaml(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "scene": payload["scene"], "output": str(args.output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
