#!/usr/bin/env python3
"""物化 V4 B0 的六场景 V3.3 replay 输入合同，不启动训练或评测。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from motion_proj.worldsim_v4.v33_replay import (
    load_yaml,
    resolve_scene_contracts,
    sha256_file,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    rows = resolve_scene_contracts(config, project_root=args.project_root)
    payload = {
        "schema_version": "worldsim_v4_v33_replay_inputs_v1",
        "task_id": config["task_id"],
        "algorithm_commit": config["algorithm"]["implementation_commit"],
        "partition_contract": "sample_index_mod_5",
        "test_quality_read": False,
        "scene_count": len(rows),
        "scenes": rows,
        "source": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
        },
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "done",
                "scene_count": len(rows),
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
