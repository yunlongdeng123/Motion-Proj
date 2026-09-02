"""Audit a deterministic provenance witness as a second repair authority."""

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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def _group(rows: list[dict[str, Any]], total: int, confidence: float) -> dict[str, Any]:
    count = len(rows)
    visible = int(sum(not bool(row["nonnew_visible_violation"]) for row in rows))
    chamfer = int(sum(bool(row["chamfer_worsened_vs_query"]) for row in rows))
    hazards = int(sum(bool(row["hazardous"]) for row in rows))
    gains = [float(row["query_chamfer_m"]) - float(row["compiled_chamfer_m"]) for row in rows]
    return {
        "actor_count": count,
        "coverage": count / total,
        "visible_violation_count": visible,
        "visible_violation_rate": visible / count if count else None,
        "visible_violation_wilson_upper": _wilson_upper(visible, count, confidence),
        "chamfer_worsened_count": chamfer,
        "chamfer_worsened_rate": chamfer / count if count else None,
        "mean_chamfer_gain_m": float(np.mean(gains)) if gains else None,
        "hazard_actor_count": hazards,
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        physical_rows = _read_jsonl(Path(config["p3c_fresh_run"]) / "ACTOR_VISIBILITY_CERTIFICATES.jsonl")
        actor_rows = _read_jsonl(Path(config["p6c_fresh_run"]) / "FRESH_AV2_ACTORS.jsonl")
        score_rows = _read_jsonl(Path(config["p6c_fresh_run"]) / "FRESH_AV2_SCORES.jsonl")
        identity = lambda row: (str(row["log_id"]), str(row["track_id"]))
        physical = {identity(row): row for row in physical_rows}
        actors = {identity(row): row for row in actor_rows}
        scores = {identity(row): row for row in score_rows}
        if not (set(physical) == set(actors) == set(scores)):
            raise RuntimeError("Frozen identity sets differ")

        rows: list[dict[str, Any]] = []
        for key in sorted(physical):
            certificate = physical[key]
            actor = actors[key]
            score = scores[key]
            completion = int(actor["completion_decision_count"])
            if completion != int(actor["action_counts"].get("COMPLETE", 0)):
                raise RuntimeError(f"Completion provenance mismatch for {key}")
            provenance = completion == 0
            p4_selected = bool(score["p4_selected"])
            rows.append(
                {
                    "log_id": key[0],
                    "track_id": key[1],
                    "hazardous": bool(actor["hazardous"]),
                    "completion_decision_count": completion,
                    "keep_count": int(actor["action_counts"].get("KEEP", 0)),
                    "project_count": int(actor["action_counts"].get("PROJECT", 0)),
                    "unknown_count": int(actor["action_counts"].get("UNKNOWN", 0)),
                    "provenance_certified": provenance,
                    "p4_selected": p4_selected,
                    "dual_selected": p4_selected and provenance,
                    "query_chamfer_m": float(certificate["query_chamfer_m"]),
                    "compiled_chamfer_m": float(certificate["compiled_chamfer_m"]),
                    "chamfer_worsened_vs_query": bool(certificate["chamfer_worsened_vs_query"]),
                    "nonnew_visible_violation": bool(certificate["nonnew_visible_violation"]),
                }
            )

        confidence = float(config["confidence_level"])
        total = len(rows)
        hazard_total = sum(bool(row["hazardous"]) for row in rows)
        always = _group(rows, total, confidence)
        p4 = _group([row for row in rows if row["p4_selected"]], total, confidence)
        provenance = _group([row for row in rows if row["provenance_certified"]], total, confidence)
        dual_rows = [row for row in rows if row["dual_selected"]]
        dual = _group(dual_rows, total, confidence)
        dual_hazard_coverage = (
            sum(bool(row["hazardous"]) for row in dual_rows) / hazard_total if hazard_total else None
        )
        gates = {
            "dual_visible_risk_below_p4": (
                dual["visible_violation_rate"] is not None
                and dual["visible_violation_rate"] < p4["visible_violation_rate"]
            ),
            "dual_visible_upper_below_p4_point_risk": (
                dual["visible_violation_wilson_upper"] is not None
                and dual["visible_violation_wilson_upper"] < p4["visible_violation_rate"]
            ),
            "dual_chamfer_worsening_not_above_p4": (
                dual["chamfer_worsened_rate"] is not None
                and dual["chamfer_worsened_rate"] <= p4["chamfer_worsened_rate"]
            ),
            "minimum_dual_coverage": dual["coverage"] >= float(config["minimum_dual_coverage"]),
            "minimum_dual_hazard_coverage": (
                dual_hazard_coverage is not None
                and dual_hazard_coverage >= float(config["minimum_dual_hazard_coverage"])
            ),
        }
        summary = {
            "schema_version": "worldsim_v7.p11_provenance_authority_audit.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "supported_provenance_conditioned_dual_authority"
                if all(gates.values())
                else "provenance_witness_does_not_certify_future_visibility"
            ),
            "actor_count": total,
            "log_count": len({row["log_id"] for row in rows}),
            "always_repair": always,
            "p4_selected": p4,
            "provenance_only": provenance,
            "p4_and_provenance": dual,
            "dual_hazard_coverage": dual_hazard_coverage,
            "completion_count_unsafe_visible_auroc": _auroc(
                [float(row["completion_decision_count"]) for row in rows],
                [not bool(row["nonnew_visible_violation"]) for row in rows],
            ),
            "fixed_gates": gates,
            "claim_boundary": config["claim_boundary"],
            "training_executed": False,
            "target_fit_or_threshold_change": False,
        }
        _write_jsonl(run_dir / "ACTOR_PROVENANCE_AUTHORITY_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as exc:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
