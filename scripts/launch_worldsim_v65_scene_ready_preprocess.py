"""Preprocess each frozen scene as soon as its atomically extracted members are ready."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from scripts.prepare_dr_v2_drivestudio_scene import collect_required_many


def run(config_path: Path, repo_root: Path, prep_run: Path, poll_seconds: float) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prep = config["preparation"]
    raw_root = Path(prep["temporary_raw_root"])
    processed_root = Path(prep["processed_root"])
    metadata = Path(prep["metadata_root"]) / "v1.0-trainval"
    scenes = list(config["scenes"])
    required = {
        name: {row["filename"] for row in payload["sample_data"]}
        for name, payload in collect_required_many(
            metadata, [str(scene["name"]) for scene in scenes]
        ).items()
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{repo_root}:/root/autodl-tmp/third_party/drivestudio"
    prep_run.mkdir(parents=True, exist_ok=True)
    (prep_run / "logs").mkdir(parents=True, exist_ok=True)

    def preprocess(scene: dict) -> dict:
        name = str(scene["name"])
        index = int(scene["processed_index"])
        destination = processed_root / f"{index:03d}"
        started = time.monotonic()
        command = [
            "/root/autodl-tmp/envs/drivestudio/bin/python",
            str(repo_root / "scripts/preprocess_dr_v2_nuscenes_single.py"),
            "--data-root",
            str(raw_root),
            "--target-dir",
            str(prep["processor_target_dir"]),
            "--scene-index",
            str(index),
        ]
        with (prep_run / "logs" / f"{name}.log").open("wb") as handle:
            subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=int(prep["preprocess_timeout_seconds"]),
                check=True,
            )
        if not destination.is_dir():
            raise RuntimeError(f"processed scene missing after command: {name}")
        return {"scene": name, "processed_index": index, "wall_seconds": time.monotonic() - started}

    by_name = {str(scene["name"]): scene for scene in scenes}
    pending = set(by_name)
    completed = []
    futures: dict[concurrent.futures.Future, str] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(prep["preprocess_workers"])
    ) as executor:
        while pending or futures:
            for future, name in list(futures.items()):
                if future.done():
                    row = future.result()
                    completed.append(row)
                    del futures[future]
                    print(
                        f"preprocess complete {name} ({len(completed)}/{len(scenes)})",
                        flush=True,
                    )
            capacity = int(prep["preprocess_workers"]) - len(futures)
            if capacity > 0:
                ready = [
                    name
                    for name in sorted(pending)
                    if all((raw_root / path).is_file() for path in required[name])
                ]
                for name in ready[:capacity]:
                    pending.remove(name)
                    futures[executor.submit(preprocess, by_name[name])] = name
                    print(
                        f"preprocess launch {name} after {len(required[name])} atomic members",
                        flush=True,
                    )
            if pending or futures:
                time.sleep(float(poll_seconds))
    return {
        "task_id": str(config["task_id"]),
        "scene_count": len(completed),
        "rows": sorted(completed, key=lambda row: row["processed_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prep-run", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.repo_root.resolve(),
                args.prep_run.resolve(),
                args.poll_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
