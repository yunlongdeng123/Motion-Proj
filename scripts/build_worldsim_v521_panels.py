#!/usr/bin/env python3
"""只对 P3 冻结的 Discovery labeled panel union 生成可追溯面板。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motion_proj.worldsim_v521.panels import build_view_panel
from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, sha256_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, type=Path)
    args = parser.parse_args()
    run = args.census_run.resolve()
    p3 = json.loads((run / "P3_SUMMARY.json").read_text(encoding="utf-8"))
    if p3.get("outcome") != "p3_gate_pass":
        raise RuntimeError("P4 只接受 frozen P3")
    registry = read_jsonl(run / "BADCASE_REGISTRY.jsonl")
    base_rows = read_jsonl(run / "BASE_CENSUS_METRICS.jsonl")
    lookup = {
        (row["base"], row["scene"], int(row["canonical_sample_index"]), int(row["camera"])): row
        for row in base_rows
    }
    panel_rows = []
    selected = [row for row in registry if row["selected_for_panel"] and row["classification_status"] == "labeled"]
    for case in selected:
        scene, frame, camera = case["scene"], int(case["canonical_sample_index"]), int(case["camera"])
        pair = {base: lookup[(base, scene, frame, camera)] for base in ("adgs", "streetgs")}
        target_hashes = {row["target_sha256"] for row in pair.values()}
        if len(target_hashes) != 1:
            raise RuntimeError(f"matched target hash 漂移：{case['case_id']}")
        output_dir = run / "panels" / case["case_id"]
        output_dir.mkdir(parents=True, exist_ok=False)
        panel_path = output_dir / "panel.png"
        panel = build_view_panel(
            target_path=pair["adgs"]["target_path"],
            prediction_paths={base: pair[base]["prediction_path"] for base in pair},
            dynamic_mask_path=pair["adgs"]["dynamic_mask_path"],
            output=panel_path,
        )
        metadata = {
            "schema": "worldsim_v521_badcase_panel_v1",
            "case_id": case["case_id"],
            "split_role": "discovery",
            "classification_status": "labeled",
            "failure_axes": case["failure_axes"],
            "metric_row_sha256": sha256_json(case),
            "inputs": {
                "target": {"path": pair["adgs"]["target_path"], "sha256": pair["adgs"]["target_sha256"]},
                "adgs": {"path": pair["adgs"]["prediction_path"], "sha256": pair["adgs"]["prediction_sha256"]},
                "streetgs": {"path": pair["streetgs"]["prediction_path"], "sha256": pair["streetgs"]["prediction_sha256"]},
                "dynamic_mask": {"path": pair["adgs"]["dynamic_mask_path"], "sha256": pair["adgs"]["dynamic_mask_sha256"]},
            },
            **panel,
        }
        atomic_json(output_dir / "metadata.json", metadata)
        case["panel_path"] = str(panel_path.resolve())
        panel_rows.append(metadata)
    atomic_jsonl(run / "BADCASE_REGISTRY.jsonl", registry)
    atomic_jsonl(run / "PANEL_REGISTRY.jsonl", panel_rows)
    atomic_json(
        run / "P4_SUMMARY.json",
        {
            "task_id": "WS-V521-P4-BADCASE-PANEL-01",
            "status": "done", "outcome": "p4_gate_pass",
            "selected_labeled_cases": len(selected), "panels": len(panel_rows),
            "manual_selection": False, "confirmation_panels": 0,
            "geometry_tiles_undefined": True,
        },
    )


if __name__ == "__main__":
    main()
