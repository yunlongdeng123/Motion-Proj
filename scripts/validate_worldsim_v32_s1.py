#!/usr/bin/env python
"""无卡验证 S1 配置、冻结输入、源码与权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from motion_proj.worldsim_v32.semantic_schema import (
    sha256_file,
    validate_actor_identity_contract,
    validate_disjoint_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checks = {
        "checkpoint": (config["inputs"]["checkpoint"], config["inputs"]["checkpoint_sha256"]),
        "source_config": (config["inputs"]["source_config"], config["inputs"]["source_config_sha256"]),
        "actor_registry": (config["inputs"]["actor_registry"], config["inputs"]["actor_registry_sha256"]),
        "instances_info": (
            str(Path(config["scene"]["processed_scene_dir"]) / "instances/instances_info.json"),
            config["scene"]["instances_info_sha256"],
        ),
        "sam2_checkpoint": (config["sam2"]["checkpoint"], config["sam2"]["checkpoint_sha256"]),
    }
    verified = {}
    for name, (raw_path, expected) in checks.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA 漂移: expected={expected} actual={actual}")
        verified[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}

    instances = json.loads(
        Path(checks["instances_info"][0]).read_text(encoding="utf-8")
    )
    registry = json.loads(
        Path(checks["actor_registry"][0]).read_text(encoding="utf-8")
    )
    registry_by_token = {
        str(row["instance_token"]): row for row in registry["actors"]
    }
    for role, actor_config in config["actors"].items():
        dataset_id = str(int(actor_config["dataset_instance_id"]))
        if dataset_id not in instances:
            raise RuntimeError(f"{role} dataset_instance_id 不存在: {dataset_id}")
        token = str(actor_config["instance_token"])
        if token not in registry_by_token:
            raise RuntimeError(f"{role} actor registry token 不存在: {token}")
        validate_actor_identity_contract(
            role=role,
            actor_config=actor_config,
            dataset_instance=instances[dataset_id],
            registry_actor=registry_by_token[token],
        )

    heldout = [int(value) for value in config["split"]["heldout_frames"]]
    train = [frame for frame in range(int(config["scene"]["frame_count"])) if frame not in set(heldout)]
    validate_disjoint_split(train, heldout)
    sam_root = Path(config["sam2"]["source_checkout"])
    head = subprocess.check_output(
        ["git", "-C", str(sam_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["sam2"]["source_commit"]:
        raise RuntimeError(f"SAM2 source commit 漂移: {head}")
    for runtime in ("sam_python", "drivestudio_python"):
        if not Path(config["runtimes"][runtime]).is_file():
            raise FileNotFoundError(config["runtimes"][runtime])
    print(json.dumps({
        "status": "done", "task_id": config["task_id"], "verified": verified,
        "train_frames": len(train), "heldout_frames": heldout,
        "heldout_excluded": True, "sam2_source_commit": head,
        "actor_identity_contract": "validated",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
