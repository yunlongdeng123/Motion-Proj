#!/usr/bin/env python3
"""Resolve per-model case eligibility without launching any model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.registry import load_registry, support_for_case
from motion_proj.cfbench.schema import validate_manifest


def build_plan(
    manifest: dict[str, Any], registry: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
    validate_manifest(manifest)
    readiness = {row["model_id"]: row["readiness"] for row in preflight["models"]}
    model_rows: list[dict[str, Any]] = []
    for model in registry["models"]:
        cases: list[dict[str, Any]] = []
        for case in manifest["cases"]:
            level = support_for_case(model, case)
            if level == "unsupported":
                continue
            cases.append(
                {
                    "case_id": case["case_id"],
                    "support": level,
                    "target_role": case["target"]["role"],
                    "family": case["intervention"]["family"],
                }
            )
        generated_case_count = sum(row["support"] != "consumer_only" for row in cases)
        consumer_case_count = sum(row["support"] == "consumer_only" for row in cases)
        model_rows.append(
            {
                "model_id": model["model_id"],
                "role": model["role"],
                "readiness": readiness.get(model["model_id"], "missing_preflight"),
                "case_count": len(cases),
                "generated_case_count": generated_case_count,
                "consumer_case_count": consumer_case_count,
                "cases": cases,
                "entrypoint": model.get("entrypoint"),
                "execution_enabled": False,
            }
        )
    return {
        "schema_version": "worldsim_v75_cfbench_run_plan_v1",
        "sequential_order": [row["model_id"] for row in model_rows],
        "execution_enabled": False,
        "reason": "GPU-free preparation turn; commands remain fail-closed until weights and case qualification pass",
        "models": model_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry = load_registry(args.registry)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    plan = build_plan(manifest, registry, preflight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                row["model_id"]: {
                    "generate": row["generated_case_count"],
                    "consume": row["consumer_case_count"],
                }
                for row in plan["models"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
