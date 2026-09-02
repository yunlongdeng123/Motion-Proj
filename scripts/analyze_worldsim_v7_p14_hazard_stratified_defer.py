"""Explain frozen defer-to-query utility and failure mass by hazard stratum."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _stratum_metrics(
    rows: list[dict[str, Any]], policy: str, hazardous: bool
) -> dict[str, Any]:
    group = [row for row in rows if bool(row["hazardous"]) is hazardous]
    mask = np.asarray([bool(row[policy]) for row in group], dtype=np.bool_)
    query = np.asarray([float(row["query_chamfer_m"]) for row in group], dtype=np.float64)
    compiled = np.asarray([float(row["compiled_chamfer_m"]) for row in group], dtype=np.float64)
    visible_failure = np.asarray(
        [not bool(row["visibility_safe"]) for row in group], dtype=np.bool_
    )
    worsened = compiled > query
    selected_count = int(mask.sum())
    coverage = selected_count / len(group)
    selected_gain = float(np.mean(query[mask] - compiled[mask])) if selected_count else None
    selected_risk = float(np.mean(visible_failure[mask])) if selected_count else None
    composite_gain = float(np.mean(np.where(mask, query - compiled, 0.0)))
    introduced_mass = float(np.mean(mask & visible_failure))
    gain_product = coverage * selected_gain if selected_gain is not None else 0.0
    risk_product = coverage * selected_risk if selected_risk is not None else 0.0
    return {
        "policy": policy,
        "stratum": "hazardous" if hazardous else "clear",
        "actor_count": len(group),
        "actor_share": len(group) / len(rows),
        "selected_actor_count": selected_count,
        "repair_coverage": coverage,
        "selected_visible_failure_count": int((mask & visible_failure).sum()),
        "selected_visible_failure_rate": selected_risk,
        "population_introduced_visible_failure_rate": introduced_mass,
        "selected_chamfer_worsened_count": int((mask & worsened).sum()),
        "selected_mean_chamfer_gain_m": selected_gain,
        "composite_mean_chamfer_gain_m": composite_gain,
        "coverage_times_selected_gain_m": gain_product,
        "coverage_times_selected_risk": risk_product,
        "gain_identity_residual": composite_gain - gain_product,
        "failure_identity_residual": introduced_mass - risk_product,
    }


def _policy_summary(
    rows: list[dict[str, Any]], policy: str, strata: list[dict[str, Any]]
) -> dict[str, Any]:
    by_name = {row["stratum"]: row for row in strata}
    hazard = by_name["hazardous"]
    clear = by_name["clear"]
    total_selected = hazard["selected_actor_count"] + clear["selected_actor_count"]
    total_failures = (
        hazard["selected_visible_failure_count"] + clear["selected_visible_failure_count"]
    )
    overall_gain = sum(
        row["actor_share"] * row["composite_mean_chamfer_gain_m"] for row in strata
    )
    overall_failure = sum(
        row["actor_share"] * row["population_introduced_visible_failure_rate"]
        for row in strata
    )
    hazard_gain_contribution = (
        hazard["actor_share"] * hazard["composite_mean_chamfer_gain_m"]
    )
    hazard_failure_share = (
        hazard["selected_visible_failure_count"] / total_failures if total_failures else None
    )
    return {
        "selected_actor_count": total_selected,
        "repair_coverage": total_selected / len(rows),
        "population_introduced_visible_failure_count": total_failures,
        "population_introduced_visible_failure_rate": overall_failure,
        "composite_mean_chamfer_gain_m": overall_gain,
        "hazard_repair_coverage": hazard["repair_coverage"],
        "clear_repair_coverage": clear["repair_coverage"],
        "hazard_selected_visible_failure_rate": hazard["selected_visible_failure_rate"],
        "clear_selected_visible_failure_rate": clear["selected_visible_failure_rate"],
        "hazard_failure_share": hazard_failure_share,
        "hazard_actor_share": hazard["actor_share"],
        "hazard_failure_burden_amplification": (
            hazard_failure_share / hazard["actor_share"]
            if hazard_failure_share is not None
            else None
        ),
        "hazard_gain_contribution_m": hazard_gain_contribution,
        "hazard_gain_share": (
            hazard_gain_contribution / overall_gain if overall_gain > 0.0 else None
        ),
        "clear_gain_contribution_m": (
            clear["actor_share"] * clear["composite_mean_chamfer_gain_m"]
        ),
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
        p13_dir = Path(config["p13_run"])
        p13_summary = _read_json(p13_dir / "summary.json")
        rows = _read_jsonl(p13_dir / "COMPOSITE_POLICY_ROWS.jsonl")
        policies = list(config["policies"])
        stratum_rows: list[dict[str, Any]] = []
        policy_summaries: dict[str, dict[str, Any]] = {}
        for policy in policies:
            policy_strata = [
                _stratum_metrics(rows, policy, hazardous=True),
                _stratum_metrics(rows, policy, hazardous=False),
            ]
            stratum_rows.extend(policy_strata)
            policy_summaries[policy] = _policy_summary(rows, policy, policy_strata)

        identity_residuals = [
            abs(row["gain_identity_residual"]) for row in stratum_rows
        ] + [abs(row["failure_identity_residual"]) for row in stratum_rows]
        max_identity_residual = max(identity_residuals)
        if max_identity_residual > 1e-12:
            raise RuntimeError(f"Finite-sample accounting identity failed: {max_identity_residual}")

        always_hazard_risk = policy_summaries["always_repair"][
            "hazard_selected_visible_failure_rate"
        ]
        for metrics in policy_summaries.values():
            current = metrics["hazard_selected_visible_failure_rate"]
            metrics["hazard_selected_risk_delta_vs_always"] = (
                current - always_hazard_risk if current is not None else None
            )

        summary = {
            "schema_version": "worldsim_v7.p14_hazard_stratified_defer.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "hazard_stratified_composite_boundary",
            "actor_count": len(rows),
            "log_count": len({row["log_id"] for row in rows}),
            "hazard_actor_count": sum(bool(row["hazardous"]) for row in rows),
            "policies": policy_summaries,
            "primary_policies": config["primary_policies"],
            "accounting_proposition": {
                "group_composite_gain": "coverage * selected_mean_gain",
                "group_introduced_failure_mass": "coverage * selected_conditional_failure",
                "overall_quantity": "sum_over_strata(actor_share * group_quantity)",
                "max_absolute_identity_residual": max_identity_residual,
            },
            "interpretation": (
                "Aggregate defer-to-query utility and introduced failure are mixtures of "
                "hazard-stratum coverage and conditional outcomes. Hazard preservation of Actor "
                "state does not imply reduced hazard-stratum visibility risk."
            ),
            "claim_boundary": config["claim_boundary"],
            "p13_status": p13_summary["status"],
            "training_executed": False,
            "dataset_read_executed": False,
            "fit_calibration_threshold_or_policy_search": False,
            "resources": {"wall_seconds": time.monotonic() - started},
        }
        _write_jsonl(run_dir / "POLICY_STRATUM_ROWS.jsonl", stratum_rows)
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
