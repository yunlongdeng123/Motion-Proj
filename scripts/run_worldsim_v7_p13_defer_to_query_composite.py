"""Compose frozen repair selectors with a query-surface fallback."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity(row: dict[str, Any], log_field: str = "log_id") -> tuple[str, str]:
    return str(row[log_field]), str(row["track_id"])


def _policy_metrics(
    rows: list[dict[str, Any]], selected: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    mask = np.asarray([selected(row) for row in rows], dtype=np.bool_)
    hazardous = np.asarray([bool(row["hazardous"]) for row in rows], dtype=np.bool_)
    visibility_failure = np.asarray(
        [not bool(row["visibility_safe"]) for row in rows], dtype=np.bool_
    )
    query = np.asarray([float(row["query_chamfer_m"]) for row in rows], dtype=np.float64)
    compiled = np.asarray([float(row["compiled_chamfer_m"]) for row in rows], dtype=np.float64)
    worsened = compiled > query
    composite = np.where(mask, compiled, query)
    selected_count = int(mask.sum())
    hazard_total = int(hazardous.sum())
    selected_hazard = int((mask & hazardous).sum())
    selected_visible_failure = int((mask & visibility_failure).sum())
    selected_chamfer_worsened = int((mask & worsened).sum())
    selected_gains = query[mask] - compiled[mask]
    return {
        "selected_actor_count": selected_count,
        "repair_coverage": selected_count / len(rows),
        "selected_hazard_actor_count": selected_hazard,
        "hazard_repair_coverage": selected_hazard / hazard_total if hazard_total else None,
        "actor_retention": 1.0,
        "hazard_actor_retention": 1.0,
        "selected_visible_failure_count": selected_visible_failure,
        "selected_visible_failure_rate": (
            selected_visible_failure / selected_count if selected_count else None
        ),
        "population_introduced_visible_failure_count": selected_visible_failure,
        "population_introduced_visible_failure_rate": selected_visible_failure / len(rows),
        "selected_chamfer_worsened_count": selected_chamfer_worsened,
        "selected_chamfer_worsened_rate": (
            selected_chamfer_worsened / selected_count if selected_count else None
        ),
        "population_chamfer_worsened_rate": selected_chamfer_worsened / len(rows),
        "selected_mean_chamfer_gain_m": (
            float(np.mean(selected_gains)) if selected_count else None
        ),
        "query_mean_chamfer_m": float(np.mean(query)),
        "composite_mean_chamfer_m": float(np.mean(composite)),
        "composite_mean_chamfer_gain_m": float(np.mean(query - composite)),
    }


def _pareto_frontier(policies: dict[str, dict[str, Any]]) -> list[str]:
    frontier: list[str] = []
    for name, metrics in policies.items():
        risk = float(metrics["population_introduced_visible_failure_rate"])
        gain = float(metrics["composite_mean_chamfer_gain_m"])
        dominated = False
        for other_name, other in policies.items():
            if other_name == name:
                continue
            other_risk = float(other["population_introduced_visible_failure_rate"])
            other_gain = float(other["composite_mean_chamfer_gain_m"])
            if other_risk <= risk and other_gain >= gain and (
                other_risk < risk or other_gain > gain
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    try:
        p10_rows = _read_jsonl(Path(config["p10_run"]) / "ACTOR_AUTHORITY_ROWS.jsonl")
        p11_rows = _read_jsonl(
            Path(config["p11_run"]) / "ACTOR_PROVENANCE_AUTHORITY_ROWS.jsonl"
        )
        p12_rows = _read_jsonl(Path(config["p12_run"]) / "EXTERNAL_DEVELOPMENT_ROWS.jsonl")
        p12_summary = _read_json(Path(config["p12_run"]) / "summary.json")
        p10 = {_identity(row): row for row in p10_rows}
        p11 = {_identity(row): row for row in p11_rows}
        p12 = {_identity(row, "scene_or_log"): row for row in p12_rows}
        if not (len(p10) == len(p10_rows) and len(p11) == len(p11_rows) and len(p12) == len(p12_rows)):
            raise RuntimeError("Duplicate frozen Actor identity")
        if not (set(p10) == set(p11) == set(p12)):
            raise RuntimeError("P10/P11/P12 frozen identity sets differ")

        threshold = float(p12_summary["frozen_visibility_threshold"])
        joined: list[dict[str, Any]] = []
        for key in sorted(p10):
            authority = p10[key]
            provenance = p11[key]
            visibility = p12[key]
            if not (
                bool(authority["hazardous"])
                == bool(provenance["hazardous"])
                == bool(visibility["hazardous"])
            ):
                raise RuntimeError(f"Hazard metadata mismatch for {key}")
            joined.append(
                {
                    "log_id": key[0],
                    "track_id": key[1],
                    "hazardous": bool(authority["hazardous"]),
                    "visibility_safe": bool(authority["nonnew_visible_violation"]),
                    "query_chamfer_m": float(authority["query_chamfer_m"]),
                    "compiled_chamfer_m": float(authority["compiled_chamfer_m"]),
                    "p4_selected": bool(authority["p4_selected"]),
                    "p6c_selected": bool(authority["p6c_selected"]),
                    "provenance_selected": bool(provenance["dual_selected"]),
                    "visibility_selected": float(visibility["visibility_score"]) >= threshold,
                    "p4_visibility_selected": (
                        bool(authority["p4_selected"])
                        and float(visibility["visibility_score"]) >= threshold
                    ),
                }
            )

        selectors: dict[str, Callable[[dict[str, Any]], bool]] = {
            "query_only": lambda row: False,
            "always_repair": lambda row: True,
            "p4_defer": lambda row: bool(row["p4_selected"]),
            "p6c_defer": lambda row: bool(row["p6c_selected"]),
            "provenance_defer": lambda row: bool(row["provenance_selected"]),
            "visibility_defer": lambda row: bool(row["visibility_selected"]),
            "p4_visibility_defer": lambda row: bool(row["p4_visibility_selected"]),
        }
        requested = list(config["policies"])
        if set(requested) != set(selectors):
            raise RuntimeError("Frozen policy set differs from implementation")
        policies = {name: _policy_metrics(joined, selectors[name]) for name in requested}
        summary = {
            "schema_version": "worldsim_v7.p13_defer_to_query_composite.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "descriptive_defer_to_query_composite_frontier",
            "actor_count": len(joined),
            "log_count": len({row["log_id"] for row in joined}),
            "visibility_threshold_source": "frozen_p12_source_calibration",
            "policies": policies,
            "pareto_frontier_including_query": _pareto_frontier(policies),
            "pareto_frontier_repair_policies": _pareto_frontier(
                {name: value for name, value in policies.items() if name != "query_only"}
            ),
            "fallback_semantics": "abstained_actor_retains_original_query_surface",
            "claim_boundary": config["claim_boundary"],
            "training_executed": False,
            "dataset_read_executed": False,
            "target_fit_recalibration_or_threshold_change": False,
            "resources": {"wall_seconds": time.monotonic() - started},
        }
        for row in joined:
            for name in requested:
                row[name] = bool(selectors[name](row))
        _write_jsonl(run_dir / "COMPOSITE_POLICY_ROWS.jsonl", joined)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as exc:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
