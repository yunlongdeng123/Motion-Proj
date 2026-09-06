"""在已暴露 M8 cohort 上运行低内存 G0 raw-fusion 诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _load_cohort(path: Path, expected_actor_count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    identities = [(str(row["scene_name"]), str(row["track_id"])) for row in rows]
    if len(rows) != int(expected_actor_count):
        raise RuntimeError(f"cohort actor count {len(rows)} != {expected_actor_count}")
    if len(set(identities)) != len(identities):
        raise RuntimeError("cohort 含重复 Actor identity")
    return rows


def _scene_to_log(path: Path) -> dict[str, str]:
    return {
        str(row["name"]): str(row["log_token"])
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def _moving(trajectory: np.ndarray, threshold_m: float = 0.5) -> tuple[bool, float]:
    trajectory = np.asarray(trajectory, dtype=np.float32).reshape(-1, 3)
    if len(trajectory) < 2:
        return False, 0.0
    displacement = float(np.linalg.norm(trajectory - trajectory[:1], axis=1).max())
    return displacement > float(threshold_m), displacement


def _stratum(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "actor_count": 0,
            "log_count": 0,
            "ray_count": 0,
            "mean_surface_point_count": None,
            "mean_symmetric_cd_l1_m": None,
            "mean_surface_to_target_m": None,
            "mean_target_to_surface_m": None,
            "mean_fscore": None,
            "observable_rate": None,
            "early_rate": None,
            "hit_recall": None,
            "late_rate": None,
            "miss_rate": None,
        }
    rays = sum(int(row["conditional_return"]["ray_count"]) for row in rows)
    observable = sum(int(row["conditional_return"]["observable_count"]) for row in rows)
    early = sum(int(row["conditional_return"]["early_count"]) for row in rows)
    hit = sum(int(row["conditional_return"]["hit_count"]) for row in rows)
    late = sum(int(row["conditional_return"]["late_count"]) for row in rows)
    miss = sum(int(row["conditional_return"]["miss_count"]) for row in rows)
    return {
        "actor_count": len(rows),
        "log_count": len({str(row["log_id"]) for row in rows}),
        "ray_count": rays,
        "mean_surface_point_count": float(
            np.mean([row["geometry"]["surface_point_count"] for row in rows])
        ),
        "mean_symmetric_cd_l1_m": float(
            np.mean([row["geometry"]["symmetric_cd_l1_m"] for row in rows])
        ),
        "mean_surface_to_target_m": float(
            np.mean([row["geometry"]["surface_to_target_mean_m"] for row in rows])
        ),
        "mean_target_to_surface_m": float(
            np.mean([row["geometry"]["target_to_surface_mean_m"] for row in rows])
        ),
        "mean_fscore": float(
            np.mean([row["geometry"]["fscore_at_threshold"] for row in rows])
        ),
        "observable_rate": observable / max(rays, 1),
        "early_rate": early / max(rays, 1),
        "hit_recall": hit / max(rays, 1),
        "late_rate": late / max(rays, 1),
        "miss_rate": miss / max(rays, 1),
    }


def _log_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["log_id"])].append(row)
    return [{"log_id": log_id, **_stratum(items)} for log_id, items in sorted(grouped.items())]


def _bootstrap_mean_ci(
    rows: list[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float],
    *,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    values = np.asarray([value(row) for row in rows], dtype=np.float64)
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def _log_macro_summary(
    rows: list[Mapping[str, Any]], *, seed: int, samples: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logs = _log_rows(rows)
    rng = np.random.default_rng(int(seed))
    metrics = {
        key: _bootstrap_mean_ci(
            logs,
            lambda row, metric=key: float(row[metric]),
            rng=rng,
            samples=int(samples),
        )
        for key in ("mean_symmetric_cd_l1_m", "mean_fscore", "early_rate", "hit_recall", "observable_rate")
    }
    return logs, {"log_count": len(logs), "cluster_unit": "driving_log", "metrics": metrics}


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("device") != "cpu" or bool(config.get("training")):
        raise ValueError("G0 诊断必须是 CPU 且 training=false")
    if config.get("data_role") != "legacy_diagnostic":
        raise ValueError("当前 G0 runner 只允许 legacy_diagnostic")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    started = time.monotonic()
    try:
        cohort_path = Path(config["cohort_rows"])
        cohort = _load_cohort(cohort_path, int(config["expected_actor_count"]))
        scene_to_log = _scene_to_log(Path(config["scene_metadata"]))
        git_commit = _git_commit()
        resolved = {
            **config,
            "run_id": run_id,
            "git_commit": git_commit,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
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
            "pretrained_holdout_exposure": bool(config["pretrained_holdout_exposure"]),
            "actor_count": len(cohort),
            "cohort_rows": str(cohort_path),
            "cohort_rows_sha256": _sha256(cohort_path),
            "cache_root": str(config["cache_root"]),
            "source_test_read": False,
            "external_test_read": False,
            "training": False,
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
        }
        _write_json(run_dir / "manifest.json", manifest)
        fingerprint_payload = json.dumps(
            {
                "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "cohort_rows_sha256": manifest["cohort_rows_sha256"],
                "git_commit": git_commit,
                "task_id": config["task_id"],
            },
            sort_keys=True,
        ).encode()
        _write_json(
            run_dir / "fingerprint.json",
            {
                "algorithm": "sha256",
                "value": hashlib.sha256(fingerprint_payload).hexdigest(),
                "scope": "config+cohort+git_commit+task_id",
            },
        )

        evaluation = config["evaluation"]
        rows: list[dict[str, Any]] = []
        cache_root = Path(config["cache_root"])
        budgets: list[int | None] = [int(value) for value in config["density_budgets"]]
        if bool(config["include_native_density"]):
            budgets.append(None)
        for actor_index, identity in enumerate(cohort):
            scene_name = str(identity["scene_name"])
            track_id = str(identity["track_id"])
            cache_path = cache_root / scene_name / f"{track_id}.npz"
            with np.load(cache_path, allow_pickle=False) as payload:
                canonical = np.asarray(payload["canonical"], dtype=np.float32)
                target = np.asarray(payload["target"], dtype=np.float32)
                origins = np.asarray(payload["target_sensor_origins"], dtype=np.float32)
                hazardous = bool(payload["hazardous"])
                category = str(payload["category"])
                trajectory = np.asarray(payload["trajectory_xyz_m"], dtype=np.float32)
                size_lwh_m = np.asarray(payload["size_lwh_m"], dtype=np.float32)
            moving, displacement = _moving(trajectory)
            for budget in budgets:
                output = (
                    canonical.copy()
                    if budget is None
                    else deterministic_farthest_point_sample(canonical, budget)
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
                        "category": category,
                        "hazardous": hazardous,
                        "moving": moving,
                        "trajectory_max_displacement_m": displacement,
                        "size_lwh_m": size_lwh_m.tolist(),
                        "density_budget": "native" if budget is None else int(budget),
                        "native_input_point_count": int(len(canonical)),
                        "surface_generation": "build_only_raw_fusion_fps" if budget is not None else "build_only_raw_fusion_native",
                        "target_used_by_surface_generation": False,
                        **metrics,
                    }
                )
            if (actor_index + 1) % 10 == 0 or actor_index + 1 == len(cohort):
                print(
                    json.dumps(
                        {"stage": "g0_raw_fusion", "progress": f"{actor_index + 1}/{len(cohort)}"}
                    ),
                    flush=True,
                )

        _write_jsonl(run_dir / "ACTORS.jsonl", rows)
        by_budget: dict[str, Any] = {}
        all_log_rows: list[dict[str, Any]] = []
        for offset, budget in enumerate(budgets):
            label = "native" if budget is None else str(budget)
            selected = [row for row in rows if str(row["density_budget"]) == label]
            log_rows, log_macro = _log_macro_summary(
                selected,
                seed=int(config["bootstrap"]["seed"]) + offset,
                samples=int(config["bootstrap"]["samples"]),
            )
            for row in log_rows:
                all_log_rows.append({"density_budget": label, **row})
            by_budget[label] = {
                "all": _stratum(selected),
                "hazard": _stratum([row for row in selected if bool(row["hazardous"])]),
                "clear": _stratum([row for row in selected if not bool(row["hazardous"])]),
                "moving": _stratum([row for row in selected if bool(row["moving"])]),
                "quasi_static": _stratum([row for row in selected if not bool(row["moving"])]),
                "log_macro_bootstrap": log_macro,
            }
        _write_jsonl(run_dir / "LOGS.jsonl", all_log_rows)
        summary = {
            "schema_version": "worldsim_v72.g0_raw_fusion.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "status": "done",
            "verdict": "diagnostic_pipeline_ready_no_route_decision",
            "baseline_id": "G0",
            "baseline_name": "raw_actor_canonical_fusion",
            "data_role": config["data_role"],
            "pretrained_holdout_exposure": True,
            "actor_count": len(cohort),
            "log_count": len({row["log_id"] for row in rows}),
            "density_budgets": ["native" if value is None else value for value in budgets],
            "metrics": by_budget,
            "full_return_metrics_supported": False,
            "full_return_metrics_reason": "legacy v1 cohort has positive held-out returns but no no-return opportunities",
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
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "diagnostic",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "g0", "error": f"{type(error).__name__}: {error}"},
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
