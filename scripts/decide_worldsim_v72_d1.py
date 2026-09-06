"""按冻结 route_selection.yaml 门槛机械计算 V7.2 D1 决策。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def _metric(summary: Mapping[str, Any], method: str, budget: str = "512") -> Mapping[str, Any]:
    return summary["metrics"][method][budget]["all"]


def _aggregate_gate(
    summary: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_name = str(rule["route_a_primary"])
    baselines = list(map(str, rule["comparator_pool"]))
    budget = str(rule["matched_density_budget"])
    candidate = _metric(summary, candidate_name, budget)
    by_baseline = {name: _metric(summary, name, budget) for name in baselines}
    cd_baseline_name = min(
        baselines, key=lambda name: float(by_baseline[name]["mean_symmetric_cd_l1_m"])
    )
    fscore_baseline_name = max(
        baselines, key=lambda name: float(by_baseline[name]["mean_fscore"])
    )
    paths = {}
    for name, baseline_name, value_key, direction in (
        ("cd", cd_baseline_name, "mean_symmetric_cd_l1_m", "lower"),
        ("fscore", fscore_baseline_name, "mean_fscore", "higher"),
    ):
        baseline = by_baseline[baseline_name]
        if direction == "lower":
            primary_delta = (
                float(baseline[value_key]) - float(candidate[value_key])
            ) / float(baseline[value_key])
            primary_pass = primary_delta >= float(
                rule["primary_effect_any"]["relative_cd_reduction_vs_best"]
            )
        else:
            primary_delta = float(candidate[value_key]) - float(baseline[value_key])
            primary_pass = primary_delta >= float(
                rule["primary_effect_any"]["absolute_fscore_gain_vs_best"]
            )
        early_delta = float(candidate["early_rate"]) - float(baseline["early_rate"])
        hit_delta = float(candidate["hit_recall"]) - float(baseline["hit_recall"])
        side_effect_pass = (
            early_delta
            <= float(rule["side_effect_limits"]["maximum_early_rate_increase_vs_best"])
            and hit_delta
            >= float(rule["side_effect_limits"]["minimum_hit_recall_delta_vs_best"])
        )
        paths[name] = {
            "comparator": baseline_name,
            "primary_delta": primary_delta,
            "primary_pass": primary_pass,
            "early_rate_delta": early_delta,
            "hit_recall_delta": hit_delta,
            "side_effect_pass": side_effect_pass,
            "gate_pass": primary_pass and side_effect_pass,
        }
    return {
        "candidate": candidate_name,
        "budget": budget,
        "candidate_metrics": dict(candidate),
        "baseline_metrics": {name: dict(value) for name, value in by_baseline.items()},
        "paths": paths,
        "gate_pass": any(path["gate_pass"] for path in paths.values()),
    }


def _log_direction_count(
    run_dir: Path, aggregate: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [json.loads(line) for line in (run_dir / "LOGS.jsonl").read_text().splitlines()]
    passing_paths = [name for name, value in aggregate["paths"].items() if value["gate_pass"]]
    if not passing_paths:
        return {"chosen_path": None, "nonnegative_log_count": 0, "log_count": 0, "pass": False}
    chosen_path = max(
        passing_paths, key=lambda name: float(aggregate["paths"][name]["primary_delta"])
    )
    comparator = aggregate["paths"][chosen_path]["comparator"]
    candidate = str(rule["route_a_primary"])
    budget = str(rule["matched_density_budget"])
    selected = [row for row in rows if str(row["density_budget"]) == budget]
    keyed = {(str(row["method"]), str(row["log_id"])): row for row in selected}
    log_ids = sorted({str(row["log_id"]) for row in selected if row["method"] == candidate})
    directions = []
    for log_id in log_ids:
        candidate_row = keyed[(candidate, log_id)]
        baseline_row = keyed[(comparator, log_id)]
        if chosen_path == "cd":
            delta = float(baseline_row["mean_symmetric_cd_l1_m"]) - float(candidate_row["mean_symmetric_cd_l1_m"])
        else:
            delta = float(candidate_row["mean_fscore"]) - float(baseline_row["mean_fscore"])
        directions.append({"log_id": log_id, "delta": delta, "nonnegative": delta >= 0.0})
    nonnegative = sum(int(row["nonnegative"]) for row in directions)
    required = int(rule["route_select_requirement"]["minimum_log_count_with_nonnegative_primary_direction"])
    return {
        "chosen_path": chosen_path,
        "comparator": comparator,
        "logs": directions,
        "nonnegative_log_count": nonnegative,
        "log_count": len(log_ids),
        "required": required,
        "pass": nonnegative >= required,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dev-summary", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = yaml.safe_load(args.selection.read_text(encoding="utf-8"))
    rule = selection["preregistered_d1_rule"]
    dev_summary = json.loads(args.dev_summary.read_text(encoding="utf-8"))
    dev = _aggregate_gate(dev_summary, rule)
    result: dict[str, Any] = {
        "schema_version": "worldsim_v72.d1_decision.v1",
        "selection_config": str(args.selection),
        "dev_run_uri": dev_summary["run_uri"],
        "dev": dev,
        "route": None,
        "route_a_pass": False,
        "route_b_pass": False,
        "decision": "route_evidence_pending" if dev["gate_pass"] else "close_method_claim",
    }
    if args.route_summary is not None:
        route_summary = json.loads(args.route_summary.read_text(encoding="utf-8"))
        route = _aggregate_gate(route_summary, rule)
        direction = _log_direction_count(args.route_summary.parent, route, rule)
        route["log_direction"] = direction
        route_pass = bool(route["gate_pass"] and direction["pass"])
        result.update(
            {
                "route_run_uri": route_summary["run_uri"],
                "route": route,
                "route_a_pass": bool(dev["gate_pass"] and route_pass),
                "route_b_pass": False,
                "decision": "select_route_a" if dev["gate_pass"] and route_pass else "close_method_claim",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"decision": result["decision"], "dev_pass": dev["gate_pass"], "route_a_pass": result["route_a_pass"]}))


if __name__ == "__main__":
    main()
