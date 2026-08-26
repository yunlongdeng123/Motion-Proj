"""Feed native sidecar workers as soon as each tar-backed scene becomes ready."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from scripts.prepare_dr_v2_drivestudio_scene import collect_required


def run(config_path: Path, repo_root: Path, task_root: Path, run_id_prefix: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prep = config["preparation"]
    scenes = list(config["cohorts"]["fresh_confirmation"]["scenes"])
    raw_root = Path(prep["temporary_raw_root"])
    metadata = Path(prep["metadata_root"]) / "v1.0-trainval"
    processed_root = Path(prep["processed_root"])
    stage_target = Path("/root/autodl-tmp/tmp/worldsim_v64_p4c_prefetch_processed")
    stage_root = Path("/root/autodl-tmp/tmp/worldsim_v64_p4c_prefetch_processed_10Hz/trainval")
    preprocess_lock = threading.Lock()
    gpu_slots = threading.Semaphore(int(config["resources"]["maximum_scene_workers"]))
    task_root.mkdir(parents=True, exist_ok=True)
    logs = task_root / "stream_logs"
    logs.mkdir(exist_ok=True)

    requirements = {
        str(scene["name"]): [row["filename"] for row in collect_required(metadata, str(scene["name"]))["sample_data"]]
        for scene in scenes
    }

    def completed_native(run_dir: Path) -> dict[str, object] | None:
        summary_path = run_dir / "P2_SUMMARY.json"
        if not summary_path.is_file():
            return None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if bool(summary.get("passed")) and int(summary.get("target_count", 0)) == 12:
            return summary
        return None

    def process(scene: dict[str, object]) -> dict[str, object]:
        name = str(scene["name"])
        index = int(scene["processed_index"])
        canonical = processed_root / f"{index:03d}"
        run_id = f"{run_id_prefix}-{name}-s0-r1"
        run_dir = task_root / run_id
        existing = completed_native(run_dir)
        if existing is not None:
            return {
                "scene": name,
                "raw_wait_seconds": 0.0,
                "run": str(run_dir),
                "summary": existing,
                "reused_complete_native": True,
            }
        waited = time.monotonic()
        while not canonical.exists() and not all((raw_root / rel).is_file() for rel in requirements[name]):
            time.sleep(5)
        raw_wait_seconds = time.monotonic() - waited
        if not canonical.exists():
            with preprocess_lock:
                if not canonical.exists():
                    staged = stage_root / f"{index:03d}"
                    if staged.exists():
                        shutil.rmtree(staged)
                    command = [
                        "/root/autodl-tmp/envs/drivestudio/bin/python",
                        str(repo_root / "scripts/preprocess_dr_v2_nuscenes_single.py"),
                        "--data-root", str(raw_root),
                        "--target-dir", str(stage_target),
                        "--scene-index", str(index),
                    ]
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = f"{repo_root}:/root/autodl-tmp/third_party/drivestudio"
                    with (logs / f"preprocess-{name}.log").open("wb") as handle:
                        subprocess.run(command, cwd=repo_root, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
                    canonical.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged, canonical)

        existing = completed_native(run_dir)
        if existing is not None:
            return {
                "scene": name,
                "raw_wait_seconds": raw_wait_seconds,
                "run": str(run_dir),
                "summary": existing,
                "reused_complete_native": True,
            }
        with gpu_slots:
            command = [
                sys.executable,
                str(repo_root / "scripts/run_worldsim_v64_fresh_sidecars.py"),
                "--config", str(config_path),
                "--repo-root", str(repo_root),
                "--run-dir", str(run_dir),
                "--maximum-workers", "1",
                "--partitions", "fresh_confirmation",
                "--only-scene", name,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            environment["PYTHONPATH"] = str(repo_root)
            with (logs / f"native-{name}.log").open("wb") as handle:
                subprocess.run(command, cwd=repo_root, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((run_dir / "P2_SUMMARY.json").read_text(encoding="utf-8"))
        return {
            "scene": name,
            "raw_wait_seconds": raw_wait_seconds,
            "run": str(run_dir),
            "summary": summary,
            "reused_complete_native": False,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(scenes)) as pool:
        rows = list(pool.map(process, scenes))
    result = {"status": "done", "scene_count": len(rows), "rows": rows, "quality_read": False}
    (task_root / "STREAM_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.task_root.resolve(), args.run_id_prefix), indent=2))


if __name__ == "__main__":
    main()
