"""Generate the frozen WorldSim V7 P3-B AV2 camera evidence package."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.av2_p3b_camera_evidence import (
    render_camera_evidence,
    write_evidence_video,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p3_config = yaml.safe_load(
        (repo_root / str(config["p3_config"])).read_text(encoding="utf-8")
    )
    p2_config = yaml.safe_load(
        (repo_root / str(p3_config["p2_config"])).read_text(encoding="utf-8")
    )
    p2_config["compiler_geometry"].update(p3_config.get("compiler_overrides", {}))
    source_run = Path(str(config["source_p3_run_path"]))
    source_cases = _read_jsonl(source_run / "VISUAL_CASES.jsonl")
    evidence_rows = _read_jsonl(source_run / "ACTOR_HARD_EVIDENCE.jsonl")
    evidence_by_case = {
        (str(row["log_id"]), str(row["track_id"])): row for row in evidence_rows
    }
    grouped_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_cases:
        grouped_cases[str(row["log_id"])].append(row)

    run_dir = (
        Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "panels").mkdir()
    (run_dir / "videos").mkdir()
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    dataset_root = Path(str(p2_config["dataset_root"]))
    loader = AV2SensorDataLoader(dataset_root, dataset_root)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P3-B is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    output_rows: list[dict[str, Any]] = []

    try:
        for log_position, log_id in enumerate(sorted(grouped_cases)):
            compiled = compile_log(
                dataset_root / log_id,
                p2_config,
                device,
                include_diagnostics=True,
            )
            diagnostics_by_actor = compiled["compiled"]["diagnostics"]
            for source_case in sorted(
                grouped_cases[log_id], key=lambda row: int(row["actor_rank"])
            ):
                track_id = str(source_case["track_id"])
                diagnostics = diagnostics_by_actor[track_id]
                evidence = evidence_by_case[(log_id, track_id)]
                stem = (
                    f"q{int(source_case['qualitative_position']):02d}_"
                    f"a{int(source_case['actor_rank'])}_{track_id[:8]}"
                )
                panel_path = run_dir / "panels" / f"{stem}.png"
                video_path = run_dir / "videos" / f"{stem}.mp4"
                camera_metadata, panels = render_camera_evidence(
                    loader,
                    diagnostics,
                    evidence,
                    dataset_root / log_id,
                    config["camera"],
                    panel_path,
                )
                write_evidence_video(
                    panels,
                    video_path,
                    evidence,
                    config["video"],
                )
                output_rows.append(
                    {
                        **source_case,
                        **camera_metadata,
                        "panel_path": f"panels/{panel_path.name}",
                        "video_path": f"videos/{video_path.name}",
                        "source_point_panel": str(source_case["path"]),
                        "case_selection_changed": False,
                        "camera_selection_rule": (
                            "maximum projected query-point visibility over the frozen ordered "
                            "ring-camera list; ties use list order; RGB is decoded only afterward"
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "progress": f"{log_position + 1}/{len(grouped_cases)}",
                        "log_id": log_id,
                        "cases": len(output_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise

    _write_jsonl(run_dir / "CAMERA_CASES.jsonl", output_rows)
    main_rows = [row for row in output_rows if bool(row["main"])]
    _write_json(run_dir / "MAIN_CAMERA_PANELS.json", main_rows)
    _write_json(run_dir / "SUPPLEMENT_CAMERA_PANELS.json", output_rows)
    visible = np.asarray([int(row["visible_query_points"]) for row in output_rows])
    visibility_fraction = np.asarray(
        [float(row["query_visibility_fraction"]) for row in output_rows]
    )
    depth_points = np.asarray([int(row["depth_points_in_crop"]) for row in output_rows])
    summary = {
        "schema_version": "worldsim_v7.p3b_av2_camera_evidence.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": "supported_frozen_av2_camera_evidence_package",
        "claim_boundary": config["claim_boundary"],
        "source_p3_run": config["source_p3_run"],
        "case_count": len(output_rows),
        "main_case_count": len(main_rows),
        "video_count": len(output_rows),
        "unique_log_count": len(grouped_cases),
        "case_selection_changed": False,
        "camera_selection_read_rgb_appearance": False,
        "camera_selection_read_method_quality": False,
        "camera_selection_rule": (
            "maximum projected query-point visibility over frozen ring-camera order"
        ),
        "camera_coverage": {
            "minimum_visible_query_points": int(np.min(visible)),
            "median_visible_query_points": float(np.median(visible)),
            "minimum_visibility_fraction": float(np.min(visibility_fraction)),
            "median_visibility_fraction": float(np.median(visibility_fraction)),
            "minimum_sparse_depth_points_in_crop": int(np.min(depth_points)),
        },
        "artifact_overlay": "paired synthetic contract evidence",
        "photorealistic_reconstruction": False,
        "resources": {
            "gpu_used": True,
            "device": str(device),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": summary["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
