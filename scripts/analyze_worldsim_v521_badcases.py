#!/usr/bin/env python3
"""从 P2 Discovery census 一次性冻结 P3 taxonomy、leaderboard 与 registry。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from motion_proj.worldsim_v521.badcase_registry import (
    AXIS_CLASS,
    METRIC_PROTOCOL_VERSION,
    TAXONOMY_VERSION,
    build_leaderboards,
    build_registry,
    freeze_thresholds,
    panel_union,
    registry_summary,
)
from motion_proj.worldsim_v521.census import sha256_file
from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, atomic_text


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v521/census_protocol_v1.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run = args.census_run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("outcome") != "p2_gate_pass":
        raise RuntimeError("P3 只接受 P2 gate pass")
    base_rows = read_jsonl(run / "BASE_CENSUS_METRICS.jsonl")
    temporal_rows = read_jsonl(run / "TEMPORAL_CENSUS_METRICS.jsonl")
    if any(row.get("partition") != "discovery" for row in base_rows + temporal_rows):
        raise RuntimeError("P3 输入必须 Discovery-only")
    minimums = config["ranking_contract"]["minimum_support_pixels"]
    thresholds = freeze_thresholds(base_rows, minimums)
    if any(thresholds[base][region]["status"] != "done" for base in thresholds for region in thresholds[base]):
        raise RuntimeError("global/actor/boundary threshold denominator 不足")
    leaderboards = build_leaderboards(base_rows, temporal_rows, minimums)
    selected = panel_union(leaderboards, int(config["ranking_contract"]["panel_union_limit"]))
    registry = build_registry(base_rows, temporal_rows, thresholds, minimums, selected)
    badcase_summary = registry_summary(registry, base_rows)
    taxonomy = {
        "schema": TAXONOMY_VERSION,
        "task_id": "WS-V521-P3-BADCASE-RANKING-01",
        "status": "frozen_after_discovery_before_confirmation",
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "census_config_sha256": sha256_file(args.config),
        "p2_base_metrics_sha256": sha256_file(run / "BASE_CENSUS_METRICS.jsonl"),
        "partition": "discovery",
        "quantile_q": 0.10,
        "quantile_operator": "per_scene_numpy_q10_linear_then_equal_scene_median",
        "minimum_support_pixels": minimums,
        "metric_directions": config["ranking_contract"]["metric_directions"],
        "thresholds": thresholds,
        "predicates": config["ranking_contract"]["failure_predicates"],
        "axis_class_map": AXIS_CLASS,
        "disabled_classes": {
            "B-GEOMETRY": "undefined_no_comparable_base_depth",
            "B-OCC": "undefined_no_audited_visibility_transition",
            "B-TEMPORAL": "unwarped_temporal_proxy_cannot_trigger_label",
            "B-SPARSE-OBS": "undefined_no_audited_observability_denominator",
            "B-IDENTITY": "undefined_no_independent_identity_evidence",
        },
        "leaderboard_k": 12,
        "panel_union_limit": int(config["ranking_contract"]["panel_union_limit"]),
        "confirmation_refit": False,
        "scalar_composite_score": False,
    }
    atomic_json(run / "BADCASE_LEADERBOARDS.json", leaderboards)
    atomic_jsonl(run / "BADCASE_REGISTRY.jsonl", registry)
    atomic_json(run / "BADCASE_SUMMARY.json", badcase_summary)
    atomic_text(run / "BADCASE_TAXONOMY_FREEZE.yaml", yaml.safe_dump(taxonomy, allow_unicode=True, sort_keys=False))
    atomic_json(
        run / "P3_SUMMARY.json",
        {
            "task_id": "WS-V521-P3-BADCASE-RANKING-01",
            "status": "done",
            "outcome": "p3_gate_pass",
            "registry_rows": len(registry),
            "panel_union_cases": len(selected),
            "taxonomy_sha256": sha256_file(run / "BADCASE_TAXONOMY_FREEZE.yaml"),
            "leaderboards_sha256": sha256_file(run / "BADCASE_LEADERBOARDS.json"),
            "registry_sha256": sha256_file(run / "BADCASE_REGISTRY.jsonl"),
            "confirmation_quality_read": False,
            "scalar_composite_score": False,
        },
    )


if __name__ == "__main__":
    main()
