"""在已消费Source Selection上运行V7.1 B4非学习TSDF基线。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.tsdf_evidential import build_b4_surface


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    m1_summary = json.loads((Path(config["m1_run"]) / "summary.json").read_text(encoding="utf-8"))
    if m1_summary.get("verdict") != "m1_source_selection_rejected":
        raise RuntimeError("B4 closeout baseline expects terminal rejected M1")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B4 evaluator requires CUDA metrics")
    torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "source_selection_baseline"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    started = time.monotonic()
    try:
        index = build_v71_index(Path(config["source"]["dataset_root"]), split)
        rows: list[dict[str, Any]] = []
        for position, scene_name in enumerate(split["roles"]["selection"]):
            bundles = compile_source_scene(scene_name, index, config["actors"], compiler, device)
            for bundle in bundles:
                diagnostics = bundle["diagnostics"]
                anchors = np.concatenate(
                    [
                        np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
                        np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
                    ],
                    axis=0,
                )
                output = build_b4_surface(
                    diagnostics["build_frame_points"],
                    diagnostics["build_sensor_origins"],
                    anchors,
                    np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32),
                    **config["tsdf"],
                )
                row = evaluate_actor_surface(
                    np.asarray(diagnostics["compiled"], dtype=np.float32),
                    output,
                    np.asarray(diagnostics["target"], dtype=np.float32),
                    np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32),
                    hazardous=bool(bundle["row"]["hazardous"]),
                    device=device,
                    lateral_tolerance_m=float(config["evaluation"]["literal_lateral_tolerance_m"]),
                    depth_tolerance_m=float(config["evaluation"]["literal_depth_tolerance_m"]),
                    distance_chunk_size=int(config["evaluation"]["distance_chunk_size"]),
                )
                row.update(
                    {
                        "scene_name": scene_name,
                        "track_id": str(bundle["row"]["track_id"]),
                        "anchor_count": int(len(anchors)),
                        "output_count": int(len(output)),
                        "tsdf_added_count": int(max(len(output) - len(anchors), 0)),
                    }
                )
                rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "b4_source_selection",
                        "progress": f"{position + 1}/{len(split['roles']['selection'])}",
                        "scene": scene_name,
                        "actors": len(rows),
                    }
                ),
                flush=True,
            )
        metrics = summarize_surface_rows(rows)
        added = np.asarray([row["tsdf_added_count"] for row in rows], dtype=np.int64)
        summary = {
            "schema_version": "worldsim_v71.b4_evidential_tsdf.v1",
            "task_id": config["task_id"],
            "status": "done",
            "verdict": "b4_nonlearned_baseline_measured",
            "selection_actor_count": len(rows),
            "source_selection": metrics,
            "tsdf_added_point_count": int(added.sum()),
            "actors_with_tsdf_additions": int(np.count_nonzero(added)),
            "selection_read": True,
            "source_final_read": False,
            "external_read": False,
            "failure_ledger_refs": config["failure_ledger_refs"],
            "failure_ledger_delta": "none",
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "SOURCE_SELECTION_ACTORS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "source_selection_baseline",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"]}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "b4", "error": f"{type(error).__name__}: {error}"},
        )
        raise


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
