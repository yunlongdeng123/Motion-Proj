"""Aggregate pipelined per-scene native runs without copying their arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    task_root: Path,
    run_dir: Path,
    run_id_prefix: str,
    partition: str,
) -> dict[str, object]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scenes = list(config["cohorts"][partition]["scenes"])
    run_dir.mkdir(parents=True)
    sources = []
    output_bytes = 0
    target_count = 0
    peak_gpu = 0.0
    peak_sum = 0.0
    for scene_row in scenes:
        scene = str(scene_row["name"])
        source = task_root / (
            f"{run_id_prefix}-scene-{scene.removeprefix('scene-')}-s0-r1"
        )
        summary = json.loads((source / "P2_SUMMARY.json").read_text(encoding="utf-8"))
        expected_targets = len(scene_row["target_frames"])
        if not bool(summary["passed"]) or int(summary["target_count"]) != expected_targets:
            raise RuntimeError(f"incomplete native source: {source}")
        source_units = source / "units" / partition / scene
        destination = run_dir / "units" / partition / scene
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source_units, target_is_directory=True)
        scene_peak = float(summary["maximum_worker_peak_gpu_memory_gib"])
        output_bytes += int(summary["output_bytes"])
        target_count += int(summary["target_count"])
        peak_gpu = max(peak_gpu, scene_peak)
        peak_sum += scene_peak
        sources.append({"scene": scene, "run": str(source.relative_to(task_root.parent.parent))})

    expected_total = sum(len(row["target_frames"]) for row in scenes)
    summary = {
        "schema_version": "worldsim_v65.pipelined_native_aggregate.v1",
        "task_id": config["task_id"],
        "mode": "formal_aggregate_of_pipelined_scene_runs",
        "partition": partition,
        "scene_count": len(scenes),
        "target_count": target_count,
        "output_bytes": output_bytes,
        "maximum_worker_peak_gpu_memory_gib": peak_gpu,
        "worker_peak_sum_upper_bound_gib": peak_sum,
        "all_native_features_complete": True,
        "target_evidence_read": False,
        "calibration_quality_read": False,
        "confirmation_content_read": False,
        "exact_once_test_read": False,
        "passed": target_count == expected_total,
        "inference_repeated": False,
    }
    _write_json(run_dir / "P2_SUMMARY.json", summary)
    _write_json(run_dir / "PIPELINE_SOURCES.json", sources)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--partition", default="fresh_selection")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.task_root.resolve(),
                args.run_dir.resolve(),
                args.run_id_prefix,
                args.partition,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
