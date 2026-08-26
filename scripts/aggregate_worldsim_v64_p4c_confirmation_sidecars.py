"""Aggregate per-scene P4C confirmation sidecars without copying arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    task_root: Path,
    run_dir: Path,
    run_id_prefix: str,
    replacement_scene: str | None,
    replacement_run_id_prefix: str | None,
) -> dict[str, object]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scenes = [str(row["name"]) for row in config["cohorts"]["fresh_confirmation"]["scenes"]]
    run_dir.mkdir(parents=True)
    reports, output_rows, source_rows = [], [], []
    output_bytes = 0
    peak_gpu = 0.0
    for scene in scenes:
        prefix = (
            replacement_run_id_prefix
            if scene == replacement_scene and replacement_run_id_prefix is not None
            else run_id_prefix
        )
        source = task_root / f"{prefix}-{scene}-s0-r1"
        summary = json.loads((source / "P2_SUMMARY.json").read_text(encoding="utf-8"))
        manifest = json.loads((source / "P2_MANIFEST.json").read_text(encoding="utf-8"))
        destination = run_dir / "units" / "fresh_confirmation" / scene
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source / "units" / "fresh_confirmation" / scene, target_is_directory=True)
        reports.extend(json.loads(line) for line in (source / "SCENE_REPORTS.jsonl").read_text().splitlines())
        output_rows.extend(manifest["output_rows"])
        output_bytes += int(summary["output_bytes"])
        peak_gpu = max(peak_gpu, float(summary["maximum_worker_peak_gpu_memory_gib"]))
        source_rows.append({"scene": scene, "source_run": str(source)})
    reports.sort(key=lambda row: row["scene"])
    (run_dir / "SCENE_REPORTS.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in reports), encoding="utf-8")
    summary = {
        "task_id": config["task_id"], "status": "done", "scene_count": len(scenes),
        "target_count": len(output_rows), "output_bytes": output_bytes,
        "maximum_worker_peak_gpu_memory_gib": peak_gpu,
        "confirmation_quality_read": False,
        "passed": len(scenes) == 8 and len(output_rows) == 96,
    }
    _write_json(run_dir / "P2_SUMMARY.json", summary)
    _write_json(run_dir / "P2_MANIFEST.json", {"source_runs": source_rows, "output_rows": output_rows})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--replacement-scene")
    parser.add_argument("--replacement-run-id-prefix")
    args = parser.parse_args()
    print(json.dumps(run(
        args.config.resolve(),
        args.task_root.resolve(),
        args.run_dir.resolve(),
        args.run_id_prefix,
        args.replacement_scene,
        args.replacement_run_id_prefix,
    ), indent=2))


if __name__ == "__main__":
    main()
