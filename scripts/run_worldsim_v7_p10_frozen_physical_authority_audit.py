"""Audit frozen selector authority against fresh AV2 physical visibility evidence."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import yaml


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


def _wilson_upper(failures: int, count: int, confidence: float) -> float | None:
    if count == 0:
        return None
    z = NormalDist().inv_cdf(confidence)
    rate = failures / count
    denom = 1.0 + z * z / count
    center = rate + z * z / (2.0 * count)
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    return float((center + radius) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _auroc(scores: list[float], positive: list[bool]) -> float | None:
    y = np.asarray(positive, dtype=np.bool_)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = _rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _aurc(scores: list[float], failures: list[bool]) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered_failures = np.asarray(failures, dtype=np.float64)[order]
    risks = np.cumsum(ordered_failures) / np.arange(1, len(order) + 1, dtype=np.float64)
    return float(np.mean(risks))


def _group(rows: list[dict[str, Any]], confidence: float, total: int) -> dict[str, Any]:
    count = len(rows)
    visible_failures = int(sum(not bool(row["nonnew_visible_violation"]) for row in rows))
    chamfer_failures = int(sum(bool(row["chamfer_worsened_vs_query"]) for row in rows))
    exact_zero = int(sum(bool(row["exact_no_visible_contradiction"]) for row in rows))
    gains = [float(row["query_chamfer_m"]) - float(row["compiled_chamfer_m"]) for row in rows]
    return {
        "actor_count": count,
        "coverage": count / total,
        "nonnew_visible_violation_count": visible_failures,
        "nonnew_visible_violation_rate": visible_failures / count if count else None,
        "nonnew_visible_violation_wilson_upper": _wilson_upper(visible_failures, count, confidence),
        "exact_no_visible_contradiction_count": exact_zero,
        "exact_no_visible_contradiction_rate": exact_zero / count if count else None,
        "chamfer_worsened_count": chamfer_failures,
        "chamfer_worsened_rate": chamfer_failures / count if count else None,
        "mean_chamfer_gain_m": float(np.mean(gains)) if gains else None,
        "median_chamfer_gain_m": float(np.median(gains)) if gains else None,
        "hazard_actor_count": int(sum(bool(row["hazardous"]) for row in rows)),
    }


def _method_summary(
    rows: list[dict[str, Any]], selected_field: str, score_field: str, confidence: float
) -> dict[str, Any]:
    selected = [row for row in rows if bool(row[selected_field])]
    abstained = [row for row in rows if not bool(row[selected_field])]
    total_visible_failures = int(sum(not bool(row["nonnew_visible_violation"]) for row in rows))
    abstained_visible_failures = int(sum(not bool(row["nonnew_visible_violation"]) for row in abstained))
    hazardous = [row for row in rows if bool(row["hazardous"])]
    nonhazardous = [row for row in rows if not bool(row["hazardous"])]
    scores = [float(row[score_field]) for row in rows]
    return {
        "selected": _group(selected, confidence, len(rows)),
        "abstained": _group(abstained, confidence, len(rows)),
        "visible_failure_abstention_capture": (
            abstained_visible_failures / total_visible_failures if total_visible_failures else None
        ),
        "safe_visible_auroc": _auroc(
            scores, [bool(row["nonnew_visible_violation"]) for row in rows]
        ),
        "chamfer_nonworse_auroc": _auroc(
            scores, [not bool(row["chamfer_worsened_vs_query"]) for row in rows]
        ),
        "nonnew_visible_aurc": _aurc(
            scores, [not bool(row["nonnew_visible_violation"]) for row in rows]
        ),
        "hazard_coverage": (
            sum(bool(row[selected_field]) for row in hazardous) / len(hazardous) if hazardous else None
        ),
        "nonhazard_coverage": (
            sum(bool(row[selected_field]) for row in nonhazardous) / len(nonhazardous)
            if nonhazardous
            else None
        ),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
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
        p3_rows = _read_jsonl(Path(config["p3c_fresh_run"]) / "ACTOR_VISIBILITY_CERTIFICATES.jsonl")
        score_rows = _read_jsonl(Path(config["p6c_fresh_run"]) / "FRESH_AV2_SCORES.jsonl")
        identity = lambda row: (str(row["log_id"]), str(row["track_id"]))
        p3_by_id = {identity(row): row for row in p3_rows}
        scores_by_id = {identity(row): row for row in score_rows}
        if len(p3_by_id) != len(p3_rows) or len(scores_by_id) != len(score_rows):
            raise RuntimeError("Duplicate (log_id, track_id) identity")
        if set(p3_by_id) != set(scores_by_id):
            raise RuntimeError("Frozen P3-C and selector identity sets differ")

        joined: list[dict[str, Any]] = []
        for key in sorted(p3_by_id):
            physical = p3_by_id[key]
            score = scores_by_id[key]
            if bool(physical["hazardous"]) != bool(score["hazardous"]):
                raise RuntimeError(f"Hazard metadata mismatch for {key}")
            joined.append(
                {
                    "log_id": key[0],
                    "track_id": key[1],
                    "category": physical["category"],
                    "hazardous": bool(physical["hazardous"]),
                    "p4_repair_score": float(score["p4_repair_score"]),
                    "p4_selected": bool(score["p4_selected"]),
                    "p6c_repair_score": float(score["candidate_repair_score"]),
                    "p6c_selected": bool(score["candidate_selected"]),
                    "query_chamfer_m": float(physical["query_chamfer_m"]),
                    "compiled_chamfer_m": float(physical["compiled_chamfer_m"]),
                    "chamfer_worsened_vs_query": bool(physical["chamfer_worsened_vs_query"]),
                    "nonnew_visible_violation": bool(physical["nonnew_visible_violation"]),
                    "exact_no_visible_contradiction": bool(physical["exact_no_visible_contradiction"]),
                    "compiled_visibility_fscore": float(physical["compiled"]["visibility_fscore"]),
                    "query_visibility_fscore": float(physical["query_only"]["visibility_fscore"]),
                }
            )

        confidence = float(config["confidence_level"])
        always = _group(joined, confidence, len(joined))
        p4 = _method_summary(joined, "p4_selected", "p4_repair_score", confidence)
        p6c = _method_summary(joined, "p6c_selected", "p6c_repair_score", confidence)
        p4_selected = p4["selected"]
        gates = {
            "selected_visible_risk_below_always_repair": (
                p4_selected["nonnew_visible_violation_rate"]
                < always["nonnew_visible_violation_rate"]
            ),
            "selected_visible_wilson_upper_below_always_repair_point_risk": (
                p4_selected["nonnew_visible_violation_wilson_upper"]
                < always["nonnew_visible_violation_rate"]
            ),
            "selected_chamfer_worsening_not_above_always_repair": (
                p4_selected["chamfer_worsened_rate"] <= always["chamfer_worsened_rate"]
            ),
        }
        verdict = "p4_empirically_contains_physical_failure" if all(gates.values()) else "p4_not_a_physical_safety_certificate"
        summary = {
            "schema_version": "worldsim_v7.p10_frozen_physical_authority_audit.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "actor_count": len(joined),
            "log_count": len({row["log_id"] for row in joined}),
            "primary_selector": "p4",
            "always_repair": always,
            "p4": p4,
            "p6c_context_only": p6c,
            "p4_fixed_gates": gates,
            "claim_boundary": config["claim_boundary"],
            "training_executed": False,
            "target_refit_or_threshold_change": False,
        }
        _write_jsonl(run_dir / "ACTOR_AUTHORITY_ROWS.jsonl", joined)
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
    summary = run(args.config, args.run_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
