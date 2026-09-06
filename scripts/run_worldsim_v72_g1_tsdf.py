"""在 G0 同一 legacy cohort 上运行 Actor-local TSDF 密度曲线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v72_g0_raw_fusion as g0_runner
from motion_proj.worldsim_v71.dataset_nuscenes import (
    build_v71_index,
    compile_source_scene,
)
from motion_proj.worldsim_v71.tsdf_evidential import build_b4_surface
from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("device") != "cpu" or bool(config.get("training")):
        raise ValueError("G1 诊断必须是 CPU 且 training=false")
    if config.get("data_role") != "legacy_diagnostic":
        raise ValueError("当前 G1 runner 只允许 legacy_diagnostic")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    g0_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    started = time.monotonic()
    try:
        cohort_path = Path(config["cohort_rows"])
        cohort = g0_runner._load_cohort(cohort_path, int(config["expected_actor_count"]))
        wanted: dict[str, set[str]] = defaultdict(set)
        for row in cohort:
            wanted[str(row["scene_name"])].add(str(row["track_id"]))
        scene_names = sorted(wanted)
        index_split = {"roles": {"legacy_diagnostic": scene_names}}
        index = build_v71_index(Path(config["dataset_root"]), index_split)
        compiler = yaml.safe_load((REPO_ROOT / config["p2_config"]).read_text(encoding="utf-8"))
        _deep_update(compiler, config["compiler_overrides"])
        scene_to_log = g0_runner._scene_to_log(Path(config["scene_metadata"]))
        git_commit = g0_runner._git_commit()
        resolved = {
            **config,
            "run_id": run_id,
            "git_commit": git_commit,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "scene_count": len(scene_names),
        }
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema_version": "worldsim_v72.run_manifest.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "git_commit": git_commit,
            "data_role": config["data_role"],
            "pretrained_holdout_exposure": True,
            "actor_count": len(cohort),
            "scene_count": len(scene_names),
            "cohort_rows": str(cohort_path),
            "cohort_rows_sha256": g0_runner._sha256(cohort_path),
            "source_test_read": False,
            "external_test_read": False,
            "training": False,
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
        }
        g0_runner._write_json(run_dir / "manifest.json", manifest)
        fingerprint_payload = json.dumps(
            {
                "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "cohort_rows_sha256": manifest["cohort_rows_sha256"],
                "git_commit": git_commit,
                "task_id": config["task_id"],
            },
            sort_keys=True,
        ).encode()
        g0_runner._write_json(
            run_dir / "fingerprint.json",
            {
                "algorithm": "sha256",
                "value": hashlib.sha256(fingerprint_payload).hexdigest(),
                "scope": "config+cohort+git_commit+task_id",
            },
        )

        evaluation = config["evaluation"]
        budgets: list[int | None] = [int(value) for value in config["density_budgets"]]
        if bool(config["include_native_density"]):
            budgets.append(None)
        rows: list[dict[str, Any]] = []
        found: set[tuple[str, str]] = set()
        device = torch.device("cpu")
        torch.set_num_threads(1)
        for scene_index, scene_name in enumerate(scene_names):
            bundles = compile_source_scene(
                scene_name, index, config["actors"], compiler, device
            )
            for bundle in bundles:
                track_id = str(bundle["row"]["track_id"])
                if track_id not in wanted[scene_name]:
                    continue
                diagnostics = bundle["diagnostics"]
                anchors = np.concatenate(
                    [
                        np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
                        np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
                    ],
                    axis=0,
                )
                native_surface = build_b4_surface(
                    diagnostics["build_frame_points"],
                    diagnostics["build_sensor_origins"],
                    anchors,
                    np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32),
                    **config["tsdf"],
                )
                target = np.asarray(diagnostics["target"], dtype=np.float32)
                origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32)
                trajectory = np.asarray(diagnostics["track"].city_centers_m, dtype=np.float32)
                moving, displacement = g0_runner._moving(trajectory)
                for budget in budgets:
                    output = (
                        native_surface.copy()
                        if budget is None
                        else deterministic_farthest_point_sample(native_surface, budget)
                    )
                    metrics = evaluate_point_surface(
                        output,
                        target,
                        origins,
                        fscore_threshold_m=float(evaluation["fscore_threshold_m"]),
                        lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
                        depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
                        distance_chunk_size=int(evaluation["distance_chunk_size"]),
                        ray_chunk_size=int(evaluation["ray_chunk_size"]),
                        point_chunk_size=int(evaluation["point_chunk_size"]),
                    )
                    rows.append(
                        {
                            "scene_name": scene_name,
                            "log_id": scene_to_log[scene_name],
                            "track_id": track_id,
                            "category": str(bundle["row"]["category"]),
                            "hazardous": bool(bundle["row"]["hazardous"]),
                            "moving": moving,
                            "trajectory_max_displacement_m": displacement,
                            "size_lwh_m": np.asarray(
                                diagnostics["track"].size_lwh_m, dtype=np.float32
                            ).tolist(),
                            "density_budget": "native" if budget is None else int(budget),
                            "anchor_point_count": int(len(anchors)),
                            "native_tsdf_point_count": int(len(native_surface)),
                            "surface_generation": "build_only_actor_local_tsdf_fps" if budget is not None else "build_only_actor_local_tsdf_native",
                            "target_used_by_surface_generation": False,
                            **metrics,
                        }
                    )
                found.add((scene_name, track_id))
            print(
                json.dumps(
                    {
                        "stage": "g1_tsdf",
                        "progress": f"{scene_index + 1}/{len(scene_names)}",
                        "scene": scene_name,
                        "matched_actors": len(found),
                    }
                ),
                flush=True,
            )
        expected = {
            (str(row["scene_name"]), str(row["track_id"])) for row in cohort
        }
        if found != expected:
            missing = sorted(expected - found)
            raise RuntimeError(f"G1 未重建全部冻结 Actor，missing={missing[:5]} count={len(missing)}")

        g0_runner._write_jsonl(run_dir / "ACTORS.jsonl", rows)
        by_budget: dict[str, Any] = {}
        all_log_rows: list[dict[str, Any]] = []
        for offset, budget in enumerate(budgets):
            label = "native" if budget is None else str(budget)
            selected = [row for row in rows if str(row["density_budget"]) == label]
            log_rows, log_macro = g0_runner._log_macro_summary(
                selected,
                seed=int(config["bootstrap"]["seed"]) + offset,
                samples=int(config["bootstrap"]["samples"]),
            )
            for row in log_rows:
                all_log_rows.append({"density_budget": label, **row})
            by_budget[label] = {
                "all": g0_runner._stratum(selected),
                "hazard": g0_runner._stratum([row for row in selected if bool(row["hazardous"])]),
                "clear": g0_runner._stratum([row for row in selected if not bool(row["hazardous"])]),
                "moving": g0_runner._stratum([row for row in selected if bool(row["moving"])]),
                "quasi_static": g0_runner._stratum([row for row in selected if not bool(row["moving"])]),
                "log_macro_bootstrap": log_macro,
            }
        g0_runner._write_jsonl(run_dir / "LOGS.jsonl", all_log_rows)
        summary = {
            "schema_version": "worldsim_v72.g1_actor_tsdf.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "status": "done",
            "verdict": "diagnostic_pipeline_ready_no_route_decision",
            "baseline_id": "G1",
            "baseline_name": "actor_local_observed_tsdf",
            "data_role": config["data_role"],
            "pretrained_holdout_exposure": True,
            "actor_count": len(found),
            "scene_count": len(scene_names),
            "log_count": len({row["log_id"] for row in rows}),
            "density_budgets": ["native" if value is None else value for value in budgets],
            "metrics": by_budget,
            "full_return_metrics_supported": False,
            "full_return_metrics_reason": "legacy cohort has positive held-out returns but no no-return opportunities",
            "route_decision_allowed": False,
            "target_free_surface_generation": True,
            "source_test_read": False,
            "external_test_read": False,
            "training": False,
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
            "resources": {
                "device": "cpu",
                "peak_gpu_memory_gib": 0.0,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        g0_runner._write_json(run_dir / "summary.json", summary)
        g0_runner._write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "diagnostic", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        g0_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "g1", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
