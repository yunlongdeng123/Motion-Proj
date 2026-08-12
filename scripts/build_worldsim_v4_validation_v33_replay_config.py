#!/usr/bin/env python3
"""从 frozen V3.3 development 协议生成 validation replay 配置。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import REQUIRED_STAGES, load_yaml, sha256_file


TASK_ID = "WS-V4-M1-EVIDENCE-FIELD-01"


class ValidationReplayConfigError(RuntimeError):
    pass


def build_config(
    *,
    base: Mapping[str, Any],
    registry_path: Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        base.get("schema_version") != "worldsim_v4_v33_replay_v1"
        or tuple(base.get("algorithm", {}).get("required_stages", ()))
        != REQUIRED_STAGES
        or base.get("frame_partition", {}).get("test_quality_read") is not False
    ):
        raise ValidationReplayConfigError("base V3.3 replay contract drift")
    if (
        registry.get("schema_version")
        != "worldsim_v4_streetgs_checkpoint_registry_v1"
        or registry.get("task_id") != TASK_ID
        or registry.get("status") != "done"
        or registry.get("split") != "validation"
        or registry.get("partition_contract") != "sample_index_mod_5"
        or registry.get("test_quality_read") is not False
        or len(registry.get("checkpoints", {})) != 6
    ):
        raise ValidationReplayConfigError("validation checkpoint registry drift")
    output = copy.deepcopy(dict(base))
    output["task_id"] = TASK_ID
    output["status"] = "running"
    output["inputs"]["checkpoint_registry"] = str(registry_path)
    output["inputs"]["checkpoint_registry_sha256"] = sha256_file(registry_path)
    output["scene_source"]["split"] = "validation"
    output["scene_source"]["checkpoint_source"] = "m1_validation_streetgs_registry"
    output["validation_protocol"] = {
        "outer_cohort_role": "validation",
        "within_scene_evaluation_partition": "development",
        "development_frozen_method_selection_reused": True,
        "validation_optimization_forbidden": True,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    return output


def atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_path = args.checkpoint_registry.resolve()
    payload = build_config(
        base=load_yaml(args.base_config),
        registry_path=registry_path,
        registry=load_yaml(registry_path),
    )
    atomic_yaml(args.output, payload)
    print(
        json.dumps(
            {
                "status": "done",
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "checkpoint_registry_sha256": sha256_file(registry_path),
                "test_quality_read": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
