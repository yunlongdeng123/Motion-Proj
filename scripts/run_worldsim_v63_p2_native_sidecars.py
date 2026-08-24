#!/usr/bin/env python3
"""Launch frozen IR-WM workers and collect full native V6.3 sidecars."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_worker(task: dict[str, Any]) -> dict[str, Any]:
    command = [
        task["python"],
        task["worker_script"],
        "--plan",
        task["plan_path"],
        "--output-dir",
        task["output_dir"],
        "--report",
        task["report_path"],
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = task["repo_root"]
    environment["OMP_NUM_THREADS"] = str(task["cpu_threads"])
    environment["MKL_NUM_THREADS"] = str(task["cpu_threads"])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
    environment["TORCH_CUDA_ARCH_LIST"] = "8.6"
    environment["PATH"] = task["environment_bin"] + os.pathsep + environment.get(
        "PATH", ""
    )
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(task["timeout_seconds"]),
        env=environment,
    )
    Path(task["log_path"]).write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"IR-WM native worker failed for {task['scene']} rc={result.returncode}:\n"
            + "\n".join(result.stdout.splitlines()[-50:])
        )
    return json.loads(Path(task["report_path"]).read_text(encoding="utf-8"))


def run(
    config_path: Path,
    repo_root: Path,
    run_dir: Path,
    partitions: list[str],
    maximum_workers: int,
    only_scene: str | None,
    limit_targets: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip():
        raise RuntimeError("formal native sidecar run requires a clean worktree")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    unknown = set(partitions) - set(config["cohorts"])
    if unknown:
        raise ValueError(f"unknown partitions: {sorted(unknown)}")
    free_gib = shutil.disk_usage(run_dir.parent).free / 1024**3
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise RuntimeError(f"insufficient disk before run: {free_gib:.3f} GiB")
    run_dir.mkdir(parents=True)
    source_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo_root, text=True
    ).strip()

    scene_rows = []
    for partition in partitions:
        for scene in config["cohorts"][partition]["scenes"]:
            scene_rows.append({"partition": partition, **scene})
    if only_scene is not None:
        scene_rows = [row for row in scene_rows if row["name"] == only_scene]
        if not scene_rows:
            raise ValueError(f"unknown scene: {only_scene}")
    mode = "probe" if only_scene is not None or limit_targets is not None else "formal"

    plans_dir = run_dir / "plans"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    for path in (plans_dir, reports_dir, logs_dir):
        path.mkdir()
    tasks = []
    expected_target_count = 0
    for scene in scene_rows:
        scene_name = scene["name"]
        targets = [int(value) for value in scene["target_frames"]]
        if limit_targets is not None:
            targets = targets[: int(limit_targets)]
        expected_target_count += len(targets)
        scene_root = Path(config["inputs"]["processed_root"]) / f"{int(scene['processed_index']):03d}"
        target_rows = []
        for target_frame in targets:
            frames = [
                target_frame + int(offset)
                for offset in config["streaming"]["history_offsets"]
            ]
            metadata_indices = [
                (frame - int(config["streaming"]["metadata_origin_frame"]))
                // int(config["streaming"]["metadata_frame_stride"])
                for frame in frames
            ]
            target_rows.append(
                {
                    "target_frame": target_frame,
                    "frames": frames,
                    "metadata_indices": metadata_indices,
                }
            )
        plan = {
            "schema_version": "worldsim_v63.p2_native_worker_plan.v1",
            "task_id": config["task_id"],
            "seed": int(config["seed"]),
            "gpu": int(config["resources"]["gpu"]),
            "partition": scene["partition"],
            "scene": scene_name,
            "scene_root": str(scene_root),
            "targets": target_rows,
            "official_repo": config["backend"]["official_repo"],
            "official_config": config["backend"]["official_config"],
            "checkpoint_path": config["backend"]["checkpoint_path"],
            "temporal_metadata_path": config["backend"]["temporal_metadata_path"],
            "backend_identity": config["backend"]["identity"],
            "camera_ids": config["streaming"]["camera_ids"],
            "native_shape": config["streaming"]["native_shape"],
            "pad_size_divisor": config["streaming"]["pad_size_divisor"],
            "image_mean_bgr": config["streaming"]["image_mean_bgr"],
            "image_std": config["streaming"]["image_std"],
            "raw_logits_shape": config["output_contract"]["raw_logits_shape"],
            "grid_shape": config["output_contract"]["grid_shape"],
            "class_count": config["output_contract"]["class_count"],
            "source_origin_m": config["output_contract"]["source_origin_m"],
            "source_voxel_size_m": config["output_contract"]["source_voxel_size_m"],
        }
        slug = f"{scene['partition']}__{scene_name}"
        plan_path = plans_dir / f"{slug}.json"
        report_path = reports_dir / f"{slug}.json"
        log_path = logs_dir / f"{slug}.log"
        _write_json(plan_path, plan)
        tasks.append(
            {
                "scene": scene_name,
                "python": config["environment"]["python"],
                "worker_script": str(
                    repo_root / "scripts/run_worldsim_v63_p2_native_worker.py"
                ),
                "plan_path": str(plan_path),
                "output_dir": str(
                    run_dir / "units" / scene["partition"] / scene_name
                ),
                "report_path": str(report_path),
                "log_path": str(log_path),
                "repo_root": str(repo_root),
                "cpu_threads": int(config["resources"]["worker_cpu_threads"]),
                "gpu": int(config["resources"]["gpu"]),
                "environment_bin": str(Path(config["environment"]["prefix"]) / "bin"),
                "timeout_seconds": int(config["resources"]["worker_timeout_seconds"]),
            }
        )

    workers = max(
        1,
        min(
            int(maximum_workers),
            int(config["resources"]["maximum_scene_workers"]),
            len(tasks),
        ),
    )
    if workers == 1:
        reports = [_run_worker(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            reports = list(executor.map(_run_worker, tasks))
    reports.sort(key=lambda row: (row["partition"], row["scene"]))
    _write_jsonl(run_dir / "SCENE_REPORTS.jsonl", reports)

    target_rows = [target for report in reports for target in report["target_rows"]]
    output_rows = [
        {"partition": report["partition"], "scene": report["scene"], **output}
        for report in reports
        for output in report["outputs"]
    ]
    summary = {
        "schema_version": "worldsim_v63.p2_native_summary.v1",
        "task_id": config["task_id"],
        "mode": mode,
        "partitions": partitions,
        "scene_count": len(reports),
        "target_count": len(target_rows),
        "output_bytes": sum(int(row["bytes"]) for row in output_rows),
        "all_native_features_complete": all(
            row["native_logits_shape"] == [200, 200, 16, 17]
            and row["bev_latent_shape"] == [200, 200, 256]
            and row["finite"]
            for row in target_rows
        ),
        "prototype_used": any(report["prototype_used"] for report in reports),
        "maximum_worker_peak_gpu_memory_gib": max(
            (float(report["peak_gpu_memory_gib"]) for report in reports), default=0.0
        ),
        "worker_peak_sum_upper_bound_gib": sum(
            sorted(
                (float(report["peak_gpu_memory_gib"]) for report in reports),
                reverse=True,
            )[:workers]
        ),
        "target_evidence_read": any(report["target_evidence_read"] for report in reports),
        "calibration_quality_read": any(
            report["calibration_quality_read"] for report in reports
        ),
        "confirmation_content_read": any(
            report["confirmation_content_read"] for report in reports
        ),
        "exact_once_test_read": any(
            report["exact_once_test_read"] for report in reports
        ),
        "wall_seconds": time.monotonic() - started,
    }
    summary["passed"] = (
        len(target_rows) == expected_target_count
        and summary["all_native_features_complete"]
        and not summary["prototype_used"]
        and summary["output_bytes"] > 0
        and not summary["target_evidence_read"]
        and not summary["calibration_quality_read"]
        and not summary["confirmation_content_read"]
        and not summary["exact_once_test_read"]
    )
    _write_json(run_dir / "P2_SUMMARY.json", summary)
    _write_json(
        run_dir / "P2_MANIFEST.json",
        {
            "schema_version": "worldsim_v63.p2_native_manifest.v1",
            "task_id": config["task_id"],
            "mode": mode,
            "source_branch": source_branch,
            "source_worktree_clean": True,
            "backend_identity": config["backend"]["identity"],
            "partitions": partitions,
            "output_rows": output_rows,
            "identity_policy": "logical_path_semantic_version_backend_task_run_no_artifact_hash",
            "target_evidence_read": False,
            "calibration_quality_read": False,
            "confirmation_content_read": False,
            "exact_once_test_read": False,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", default=["development", "legacy"])
    parser.add_argument("--maximum-workers", type=int, default=2)
    parser.add_argument("--only-scene")
    parser.add_argument("--limit-targets", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.repo_root.resolve(),
                args.run_dir.resolve(),
                list(args.partitions),
                args.maximum_workers,
                args.only_scene,
                args.limit_targets,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
