"""Summarize frozen fresh-AV2 ray failures by hazard stratum and compiler action."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _identity(row: dict[str, Any], log_field: str) -> tuple[str, str]:
    return str(row[log_field]), str(row["track_id"])


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(
    rows: list[dict[str, Any]], scope: str, hazardous: bool, actions: list[str]
) -> dict[str, Any]:
    group = [row for row in rows if bool(row["hazardous"]) is hazardous and bool(row[scope])]
    target_rays = sum(int(row["target_ray_count"]) for row in group)
    query_early = sum(int(row["query_early_count"]) for row in group)
    compiled_early = sum(int(row["compiled_early_count"]) for row in group)
    new_early = sum(int(row["new_early_count"]) for row in group)
    new_hits = sum(int(row["new_hit_count"]) for row in group)
    resolved_early = sum(int(row["resolved_query_early_count"]) for row in group)
    contradictions = sum(
        int(row["surface_contradiction_by_provenance"][action])
        for row in group
        for action in actions
    )
    action_metrics: dict[str, dict[str, Any]] = {}
    for action in actions:
        output_points = sum(int(row["output_point_counts"][action]) for row in group)
        action_new_early = sum(
            int(row["new_early_by_provenance"][action]) for row in group
        )
        action_new_hits = sum(
            int(row["new_hit_by_provenance"][action]) for row in group
        )
        action_contradictions = sum(
            int(row["surface_contradiction_by_provenance"][action]) for row in group
        )
        action_metrics[action] = {
            "output_point_count": output_points,
            "new_early_count": action_new_early,
            "new_early_share": _ratio(action_new_early, new_early),
            "new_early_fraction_of_target_rays": _ratio(action_new_early, target_rays),
            "new_hit_count": action_new_hits,
            "new_hit_share": _ratio(action_new_hits, new_hits),
            "new_hit_to_new_early_ratio": _ratio(action_new_hits, action_new_early),
            "surface_contradiction_count": action_contradictions,
            "surface_contradiction_share": _ratio(action_contradictions, contradictions),
            "surface_contradiction_per_output_point": _ratio(
                action_contradictions, output_points
            ),
        }
    return {
        "scope": scope,
        "stratum": "hazardous" if hazardous else "clear",
        "actor_count": len(group),
        "target_ray_count": target_rays,
        "query_early_count": query_early,
        "compiled_early_count": compiled_early,
        "new_early_count": new_early,
        "new_early_fraction_of_target_rays": _ratio(new_early, target_rays),
        "resolved_query_early_count": resolved_early,
        "resolved_query_early_fraction_of_target_rays": _ratio(
            resolved_early, target_rays
        ),
        "new_hit_count": new_hits,
        "new_hit_fraction_of_target_rays": _ratio(new_hits, target_rays),
        "new_hit_to_new_early_ratio": _ratio(new_hits, new_early),
        "surface_contradiction_count": contradictions,
        "actions": action_metrics,
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    try:
        raw_dir = Path(config["raw_run"])
        score_dir = Path(config["p6c_run"])
        raw_summary = _read_json(raw_dir / "summary.json")
        raw_rows = _read_jsonl(raw_dir / "ACTOR_PROVENANCE_ATTRIBUTION.jsonl")
        score_rows = _read_jsonl(score_dir / "FRESH_AV2_SCORES.jsonl")
        raw = {_identity(row, "log_id"): row for row in raw_rows}
        scores = {_identity(row, "log_id"): row for row in score_rows}
        if len(raw) != len(raw_rows) or len(scores) != len(score_rows) or set(raw) != set(scores):
            raise RuntimeError("P15 raw attribution and frozen selector identities differ")

        joined: list[dict[str, Any]] = []
        for key in sorted(raw):
            row = dict(raw[key])
            score = scores[key]
            if bool(row["hazardous"]) != bool(score["hazardous"]):
                raise RuntimeError(f"Hazard metadata differs for {key}")
            row.update(
                {
                    "always_repair": True,
                    "p4_selected": bool(score["p4_selected"]),
                    "p4_abstained": not bool(score["p4_selected"]),
                    "p6c_selected": bool(score["candidate_selected"]),
                    "p6c_abstained": not bool(score["candidate_selected"]),
                }
            )
            joined.append(row)

        scopes = list(config["scopes"])
        actions = list(config["actions"])
        stratum_rows = [
            _summarize(joined, scope, hazardous, actions)
            for scope in scopes
            for hazardous in (True, False)
        ]
        lookup = {(row["scope"], row["stratum"]): row for row in stratum_rows}
        comparisons: dict[str, dict[str, Any]] = {}
        for scope in ("always_repair", "p4_selected", "p6c_selected"):
            hazard = lookup[(scope, "hazardous")]
            clear = lookup[(scope, "clear")]
            comparisons[scope] = {
                "hazard_to_clear_new_early_rate_ratio": _ratio(
                    hazard["new_early_fraction_of_target_rays"],
                    clear["new_early_fraction_of_target_rays"],
                ),
                "hazard_complete_new_early_share": hazard["actions"]["COMPLETE"][
                    "new_early_share"
                ],
                "clear_complete_new_early_share": clear["actions"]["COMPLETE"][
                    "new_early_share"
                ],
                "hazard_keep_contradiction_share": hazard["actions"]["KEEP"][
                    "surface_contradiction_share"
                ],
                "clear_keep_contradiction_share": clear["actions"]["KEEP"][
                    "surface_contradiction_share"
                ],
            }
        always_hazard_rate = lookup[("always_repair", "hazardous")][
            "new_early_fraction_of_target_rays"
        ]
        comparisons["p4_selected"]["hazard_new_early_rate_ratio_vs_always"] = _ratio(
            lookup[("p4_selected", "hazardous")][
                "new_early_fraction_of_target_rays"
            ],
            always_hazard_rate,
        )
        comparisons["p6c_selected"]["hazard_new_early_rate_ratio_vs_always"] = _ratio(
            lookup[("p6c_selected", "hazardous")][
                "new_early_fraction_of_target_rays"
            ],
            always_hazard_rate,
        )

        project_outputs = sum(
            row["actions"]["PROJECT"]["output_point_count"] for row in stratum_rows[:2]
        )
        summary = {
            "schema_version": "worldsim_v7.p15_fresh_hazard_action_audit.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "descriptive_fresh_hazard_action_mechanism_boundary",
            "actor_count": len(joined),
            "log_count": len({row["log_id"] for row in joined}),
            "hazard_actor_count": sum(bool(row["hazardous"]) for row in joined),
            "strata": stratum_rows,
            "comparisons": comparisons,
            "project_output_count_always": project_outputs,
            "project_deduplication_note": (
                "KEEP is concatenated before PROJECT during voxel deduplication. A zero "
                "PROJECT output count is deterministic provenance collapse, not causal zero harm."
            ),
            "raw_attribution_status": raw_summary["status"],
            "claim_boundary": config["claim_boundary"],
            "training_executed": False,
            "fit_calibration_threshold_or_policy_search": False,
            "resources": {"analysis_wall_seconds": time.monotonic() - started},
        }
        _write_jsonl(run_dir / "HAZARD_ACTION_STRATA.jsonl", stratum_rows)
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
