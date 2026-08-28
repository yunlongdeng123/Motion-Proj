"""运行V6.7 Actor-preserving package bake。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from motion_proj.worldsim_v66.harp_bake import PACKAGE_FILES, bake_package


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    package = bake_package(config, runs_root, run_dir / "package")
    manifest = package["manifest"]
    state_rows = package["state_rows"]
    actor_rows = package["actor_rows"]
    repair_rows = package["repair_rows"]
    runtime = manifest["runtime_contract"]
    hidden_fields = {
        "local_geometry_conflict",
        "hidden_free_count",
        "hidden_free_rate",
        "target_label",
    }
    metrics = {
        "unit_count": int(manifest["counts"]["unit_count"]),
        "unique_actor_count": len(actor_rows),
        "actor_state_count": len(state_rows),
        "actor_primitive_count": int(manifest["counts"]["actor_primitive_count"]),
        "actor_state_retention": sum(
            row["existence_state"] == "SUPPORTED_ACTOR" for row in state_rows
        )
        / len(state_rows),
        "actor_metadata_completeness": sum(
            bool(row["class"] and row["track_id"] and row["trajectory"])
            for row in actor_rows
        )
        / len(actor_rows),
        "actor_removed_count": sum(bool(row["actor_removed"]) for row in repair_rows),
        "hidden_target_field_count": sum(
            len(hidden_fields.intersection(row))
            for rows in (
                package["actor_rows"],
                package["factor_rows"],
                package["repair_rows"],
                package["hazard_rows"],
                package["provenance_rows"],
            )
            for row in rows
        ),
        "package_file_count": len(PACKAGE_FILES),
        "package_bytes": sum(
            path.stat().st_size
            for path in (run_dir / "package").iterdir()
            if path.is_file()
        ),
    }
    gates_config = config["gates"]
    gates = {
        "minimum_actor_state_retention": metrics["actor_state_retention"]
        >= float(gates_config["minimum_actor_state_retention"]),
        "minimum_actor_metadata_completeness": metrics["actor_metadata_completeness"]
        >= float(gates_config["minimum_actor_metadata_completeness"]),
        "maximum_actor_removed_count": metrics["actor_removed_count"]
        <= int(gates_config["maximum_actor_removed_count"]),
        "maximum_hidden_target_field_count": metrics["hidden_target_field_count"]
        <= int(gates_config["maximum_hidden_target_field_count"]),
        "runtime_model_loading_disabled": bool(runtime["learned_model_loaded"])
        is bool(gates_config["runtime_model_loading"]),
        "runtime_hazard_existence_coupling_disabled": bool(
            runtime["hazard_controls_actor_existence"]
        )
        is bool(gates_config["runtime_hazard_existence_coupling"]),
    }
    verdict = (
        "supported_v67_actor_preserving_package"
        if all(gates.values())
        else "rejected_v67_actor_preserving_package"
    )
    summary = {
        "schema_version": "worldsim_v67.p2_actor_package_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "metrics": metrics,
        "runtime_contract": runtime,
        "gate_results": gates,
        "failure_ledger_delta": "pending_result",
        "resources": {
            "gpu_used": False,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
