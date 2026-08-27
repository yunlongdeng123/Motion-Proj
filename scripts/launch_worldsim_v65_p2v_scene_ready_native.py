"""Launch P2V native inference as each preparation scene reaches its final log marker."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import yaml


def run(
    config_path: Path,
    prep_run: Path,
    repo_root: Path,
    runs_root: Path,
    run_prefix: str,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config["task_id"])
    (runs_root / "worldsim_v65" / task_id).mkdir(parents=True, exist_ok=True)
    scenes = [str(row["name"]) for row in config["cohorts"]["fresh_selection"]["scenes"]]
    pending = set(scenes)
    running: dict[concurrent.futures.Future, tuple[str, str]] = {}
    completed: dict[str, str] = {}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)

    def launch(scene: str, relative_run: str) -> dict[str, object]:
        run_dir = runs_root / relative_run
        summary_path = run_dir / "P2_SUMMARY.json"
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        command = [
            "/root/autodl-tmp/envs/motionproj/bin/python",
            str(repo_root / "scripts" / "run_worldsim_v63_p2_native_sidecars.py"),
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--partitions",
            "fresh_selection",
            "--maximum-workers",
            "1",
            "--only-scene",
            scene,
        ]
        subprocess.run(command, cwd=repo_root, env=environment, check=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while pending or running:
            for future in list(running):
                if not future.done():
                    continue
                scene, relative_run = running.pop(future)
                summary = future.result()
                if not bool(summary["passed"]):
                    raise RuntimeError(f"native scene failed scientific capability: {scene}")
                completed[scene] = relative_run
                print(f"native complete {scene} ({len(completed)}/{len(scenes)})", flush=True)

            capacity = 2 - len(running)
            if capacity > 0:
                ready = []
                for scene in sorted(pending):
                    log_path = prep_run / "logs" / f"{scene}.log"
                    if log_path.is_file() and "Processed dynamic masks" in log_path.read_text(errors="ignore"):
                        ready.append(scene)
                for scene in ready[:capacity]:
                    pending.remove(scene)
                    relative_run = (
                        f"worldsim_v65/{task_id}/"
                        f"{run_prefix}-scene-{scene.removeprefix('scene-')}-s0-r1"
                    )
                    print(f"native launch {scene} after final preprocess marker", flush=True)
                    future = executor.submit(launch, scene, relative_run)
                    running[future] = (scene, relative_run)
            if pending or running:
                time.sleep(5.0)

    return {"task_id": task_id, "scene_runs": completed, "all_passed": len(completed) == len(scenes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prep-run", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.prep_run.resolve(), args.repo_root.resolve(), args.runs_root.resolve(), args.run_prefix), indent=2))


if __name__ == "__main__":
    main()
