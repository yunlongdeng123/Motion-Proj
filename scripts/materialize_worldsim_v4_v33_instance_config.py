#!/usr/bin/env python3
"""从已验收 semantic lift 物化 V4 development-only instance-field 配置。"""

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

from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file


def _verified_file(path: str | Path, expected: str, label: str) -> Path:
    value = Path(path)
    if not value.is_file() or sha256_file(value) != expected:
        raise V33ReplayError(f"{label} 缺失或 SHA 漂移")
    return value


def build_instance_config(
    *,
    replay: Mapping[str, Any],
    template: Mapping[str, Any],
    semantic_config: Mapping[str, Any],
    semantic_config_path: Path,
    semantic_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if semantic_config.get("schema_version") != "worldsim_v4_v33_semantic_lift_v1":
        raise V33ReplayError("semantic config schema 漂移")
    if semantic_summary.get("status") != "done":
        raise V33ReplayError("semantic lift 尚未验收完成")
    if semantic_summary.get("config_sha256") != sha256_file(semantic_config_path):
        raise V33ReplayError("semantic summary/config SHA 漂移")
    for key in (
        "development_leaks",
        "heldout_leaks",
    ):
        if int(semantic_summary.get(key, -1)) != 0:
            raise V33ReplayError(f"semantic split gate 未通过: {key}")
    for key in (
        "development_content_read",
        "heldout_content_read",
        "test_quality_read",
    ):
        if semantic_summary.get(key) is not False:
            raise V33ReplayError(f"semantic provenance 未证明 {key}=false")
    if semantic_summary.get("checkpoint_sha256_before") != semantic_summary.get(
        "checkpoint_sha256_after"
    ):
        raise V33ReplayError("semantic lift 修改了 base checkpoint")
    inputs = semantic_config["inputs"]
    if semantic_summary["checkpoint_sha256_after"] != inputs["checkpoint_sha256"]:
        raise V33ReplayError("semantic checkpoint SHA 漂移")

    prompt = _verified_file(
        semantic_summary["prompt_manifest"],
        semantic_summary["prompt_manifest_sha256"],
        "prompt manifest",
    )
    masks = _verified_file(
        semantic_summary["mask_manifest"],
        semantic_summary["mask_manifest_sha256"],
        "train mask manifest",
    )
    semantics = _verified_file(
        semantic_summary["semantic_manifest"],
        semantic_summary["semantic_manifest_sha256"],
        "semantic manifest",
    )
    actors: dict[str, Any] = {}
    for role, actor in semantic_config["actors"].items():
        summary_actor = semantic_summary.get("actors", {}).get(role)
        if summary_actor is None:
            raise V33ReplayError(f"semantic summary 缺少 actor: {role}")
        sidecar = _verified_file(
            summary_actor["sidecar"], summary_actor["sha256"], f"{role} sidecar"
        )
        actors[role] = {
            **dict(actor),
            "semantic_sidecar": str(sidecar),
            "semantic_sidecar_sha256": summary_actor["sha256"],
        }

    frozen = replay["instance_field"]
    optimization = dict(template["optimization"])
    if frozen["formal_selected_arm"] != template["optimization"]["formal_selected_arm"]:
        raise V33ReplayError("frozen formal_selected_arm 与 V3.3 template 不一致")
    optimization["formal_selected_arm"] = frozen["formal_selected_arm"]
    scene = semantic_config["scene"]
    split = semantic_config["split"]
    return {
        "schema_version": "worldsim_v4_v33_instance_field_v1",
        "task_id": replay["task_id"],
        "seed": int(template["seed"]),
        "inputs": {
            **{key: inputs[key] for key in (
                "checkpoint",
                "checkpoint_sha256",
                "source_config",
                "source_config_sha256",
                "actor_registry",
                "actor_registry_sha256",
            )},
            "v32_config": str(semantic_config_path),
            "v32_config_sha256": sha256_file(semantic_config_path),
            "v32_prompt_manifest": str(prompt),
            "v32_prompt_manifest_sha256": semantic_summary[
                "prompt_manifest_sha256"
            ],
            "train_mask_manifest": str(masks),
            "train_mask_manifest_sha256": semantic_summary["mask_manifest_sha256"],
            "semantic_manifest": str(semantics),
            "semantic_manifest_sha256": semantic_summary[
                "semantic_manifest_sha256"
            ],
        },
        "scene": dict(scene),
        "split": dict(split),
        "actors": actors,
        "sam2_fallback": dict(template["sam2_fallback"]),
        "representation": dict(template["representation"]),
        "arms": dict(template["arms"]),
        "optimization": optimization,
        "evaluation": dict(template["evaluation"]),
        "outputs": dict(template["outputs"]),
        "runtimes": {
            "sam_python": replay["runtimes"]["sam2_python"],
            "drivestudio_python": replay["runtimes"]["drivestudio_python"],
            "drivestudio_checkout": replay["runtimes"]["drivestudio_checkout"],
        },
        "provenance": {
            "algorithm_commit": replay["algorithm"]["implementation_commit"],
            "formal_arm_source": "frozen_v33_scene0230_smoke",
            "evaluation_partition": frozen["evaluation_partition"],
            "base_rgb_immutable": True,
            "actor_abstentions": semantic_config["provenance"][
                "actor_abstentions"
            ],
            "development_content_read": False,
            "heldout_content_read": False,
            "test_quality_read": False,
        },
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
    parser.add_argument("--replay-config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--semantic-config", type=Path, required=True)
    parser.add_argument("--semantic-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay = load_yaml(args.replay_config)
    template_path = args.project_root / replay["instance_field"]["template"]
    if sha256_file(template_path) != replay["instance_field"]["template_sha256"]:
        raise V33ReplayError("instance-field template SHA 漂移")
    config = build_instance_config(
        replay=replay,
        template=load_yaml(template_path),
        semantic_config=load_yaml(args.semantic_config),
        semantic_config_path=args.semantic_config.resolve(),
        semantic_summary=json.loads(args.semantic_summary.read_text(encoding="utf-8")),
    )
    atomic_yaml(args.output, config)
    print(
        json.dumps(
            {
                "status": "done",
                "scene": config["scene"]["name"],
                "actor_count": len(config["actors"]),
                "evaluation_partition": config["provenance"][
                    "evaluation_partition"
                ],
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
