"""把逐场景 P6 native 输出汇总为不复制大数组的只读运行根。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _source_run(task_root: Path, partition: str, scene: str) -> Path:
    if partition == "fresh_calibration":
        timestamp = "20260826T111500Z" if scene == "scene-1045" else "20260826T113000Z"
        role = "calibration"
    else:
        timestamp = "20260826T121500Z"
        role = "confirmation"
    return task_root / f"{timestamp}__{role}-native-{scene}-s0-r1"


def run(config_path: Path, task_root: Path, run_dir: Path) -> dict[str, object]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True)
    reports = []
    output_rows = []
    source_rows = []
    peak_gpu = 0.0
    output_bytes = 0
    target_count = 0
    for partition, cohort in config["cohorts"].items():
        for scene_row in cohort["scenes"]:
            scene = str(scene_row["name"])
            source = _source_run(task_root, partition, scene)
            summary = json.loads((source / "P2_SUMMARY.json").read_text(encoding="utf-8"))
            manifest = json.loads((source / "P2_MANIFEST.json").read_text(encoding="utf-8"))
            if not summary["passed"] or int(summary["target_count"]) != 12:
                raise RuntimeError(f"incomplete native source: {source}")
            source_unit = source / "units" / partition / scene
            destination = run_dir / "units" / partition / scene
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source_unit, target_is_directory=True)
            reports.extend(
                json.loads(line)
                for line in (source / "SCENE_REPORTS.jsonl").read_text(encoding="utf-8").splitlines()
            )
            output_rows.extend(manifest["output_rows"])
            output_bytes += int(summary["output_bytes"])
            target_count += int(summary["target_count"])
            peak_gpu = max(peak_gpu, float(summary["maximum_worker_peak_gpu_memory_gib"]))
            source_rows.append(
                {"partition": partition, "scene": scene, "source_run": str(source)}
            )

    reports.sort(key=lambda row: (row["partition"], row["scene"]))
    (run_dir / "SCENE_REPORTS.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reports),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v64.p6_native_aggregate.v1",
        "task_id": config["task_id"],
        "mode": "formal",
        "partitions": list(config["cohorts"]),
        "scene_count": len(reports),
        "target_count": target_count,
        "output_bytes": output_bytes,
        "all_native_features_complete": True,
        "prototype_used": False,
        "maximum_worker_peak_gpu_memory_gib": peak_gpu,
        "target_evidence_read": False,
        "calibration_quality_read": False,
        "confirmation_content_read": False,
        "exact_once_test_read": False,
        "passed": len(reports) == 24 and target_count == 288,
    }
    _write_json(run_dir / "P2_SUMMARY.json", summary)
    _write_json(
        run_dir / "P2_MANIFEST.json",
        {
            "schema_version": "worldsim_v64.p6_native_aggregate_manifest.v1",
            "task_id": config["task_id"],
            "partitions": list(config["cohorts"]),
            "source_runs": source_rows,
            "output_rows": output_rows,
            "identity_policy": "logical_path_semantic_version_backend_task_run_no_artifact_hash",
            "target_evidence_read": False,
            "calibration_quality_read": False,
            "confirmation_content_read": False,
            "exact_once_test_read": False,
        },
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.task_root.resolve(), args.run_dir.resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
