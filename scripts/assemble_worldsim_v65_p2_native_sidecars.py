"""Assemble pipelined per-scene native runs without repeating inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config_path: Path, runs_root: Path, run_dir: Path, source_specs: list[str]) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    partition = "fresh_selection"
    scenes = [str(row["name"]) for row in config["cohorts"][partition]["scenes"]]
    sources = dict(spec.split("=", 1) for spec in source_specs)
    if set(sources) != set(scenes):
        raise ValueError(f"source scenes differ: {sorted(set(scenes) ^ set(sources))}")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("units", "plans", "reports", "logs"):
        (run_dir / name).mkdir()
    summaries = []
    source_rows = []
    for scene in scenes:
        source = runs_root / sources[scene]
        summary = json.loads((source / "P2_SUMMARY.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        unit_source = source / "units" / partition / scene
        unit_target = run_dir / "units" / partition / scene
        unit_target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(unit_source, unit_target, target_is_directory=True)
        slug = f"{partition}__{scene}"
        for folder, suffix in (("plans", ".json"), ("reports", ".json"), ("logs", ".log")):
            os.symlink(source / folder / f"{slug}{suffix}", run_dir / folder / f"{slug}{suffix}")
        source_rows.append({"scene": scene, "run": sources[scene]})
    peaks = sorted((float(row["maximum_worker_peak_gpu_memory_gib"]) for row in summaries), reverse=True)
    summary = {
        "schema_version": "worldsim_v65.p2_pipeline_native_summary.v1",
        "task_id": config["task_id"],
        "mode": "formal_aggregate_of_pipelined_scene_runs",
        "partition": partition,
        "scene_count": len(scenes),
        "target_count": sum(int(row["target_count"]) for row in summaries),
        "output_bytes": sum(int(row["output_bytes"]) for row in summaries),
        "maximum_worker_peak_gpu_memory_gib": max(peaks),
        "worker_peak_sum_upper_bound_gib": sum(peaks[:2]),
        "all_native_features_complete": all(bool(row["all_native_features_complete"]) for row in summaries),
        "target_evidence_read": any(bool(row["target_evidence_read"]) for row in summaries),
        "calibration_quality_read": any(bool(row["calibration_quality_read"]) for row in summaries),
        "confirmation_content_read": any(bool(row["confirmation_content_read"]) for row in summaries),
        "exact_once_test_read": any(bool(row["exact_once_test_read"]) for row in summaries),
        "passed": all(bool(row["passed"]) for row in summaries),
        "inference_repeated": False,
    }
    _write_json(run_dir / "PIPELINE_SOURCES.json", source_rows)
    _write_json(run_dir / "P2_SUMMARY.json", summary)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_dir.resolve(), args.source_run), indent=2))


if __name__ == "__main__":
    main()
