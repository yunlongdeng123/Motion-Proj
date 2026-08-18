#!/usr/bin/env python3
"""Independently audit r047 raw fresh-process Trace3D determinism evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.audit_worldsim_v51_f0b_three_view_association_parity import _load_json, _load_jsonl, _load_yaml
from scripts.audit_worldsim_v51_f0c_upstream_batch_association_repeatability import _manifest_inventory
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"
EXPECTED_RUN_NAME = "20260819T000000Z__m1-stage-g-g0b-trace3d-determinism-s20260814-r047"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _vector_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    if run_dir.name != EXPECTED_RUN_NAME:
        raise ProtocolError("r047 run identity drift")
    config = _load_yaml(config_path)
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    expected_conclusion = config["decision"]["fail_conclusion"]
    if status.get("task_id") != TASK_ID or status.get("status") != "done" or status.get("outcome") != "rejected" or status.get("conclusion") != expected_conclusion:
        raise ProtocolError("r047 terminal drift")
    source_commit = status["source_commit"]
    source_tree = _git(PROJECT, "show", "-s", "--format=%T", source_commit)
    resolved_path = run_dir / "resolved_config.yaml"
    committed = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"])
    if resolved_path.read_bytes() != committed:
        raise ProtocolError("r047 resolved config drift")
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"] or events[-1].get("outcome") != "rejected":
        raise ProtocolError("r047 event drift")

    summary_path = run_dir / "summary.json"
    forensic_path = run_dir / "artifacts/forensic.json"
    resources_path = run_dir / "artifacts/resources.json"
    hazard_path = run_dir / "artifacts/source_hazard.json"
    summary, forensic, resources, hazard = (_load_json(path) for path in (summary_path, forensic_path, resources_path, hazard_path))
    if (
        summary.get("status") != "done" or summary.get("outcome") != "rejected" or summary.get("conclusion") != expected_conclusion
        or summary.get("source_commit") != source_commit or summary.get("source_tree") != source_tree
        or summary.get("forensic") != forensic or summary.get("resources") != resources or not all(summary.get("resource_checks", {}).values())
        or summary.get("next_task") != config["decision"]["fail_next_task"] or summary.get("next_action") != config["decision"]["fail_next_action"]
    ):
        raise ProtocolError("r047 summary/decision/resource drift")

    rows = []
    process_evidence = []
    for index in range(int(config["runtime"]["fresh_process_count"])):
        path = run_dir / f"artifacts/process_{index:02d}.json"
        row = _load_json(path)
        if row.get("process_index") != index or row.get("inputs_immutable") is not True:
            raise ProtocolError(f"r047 process row drift: {index}")
        rows.append(row)
        process_evidence.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    hard_values = [row[key] for row in rows for key in ("foreground_hard_first", "foreground_hard_second")]
    alpha_values = [row[key] for row in rows for key in ("foreground_alpha_first", "foreground_alpha_second")]
    unique_hard = sorted({_vector_key(value) for value in hard_values})
    unique_alpha = sorted({_vector_key(value) for value in alpha_values})
    alpha_scalars = [float(value[0][1]) for value in alpha_values]
    replay_checks = {
        "process_count_exact": len(rows) == 8,
        "background_hard_all_exact": all(row["background_hard"] == [[1.0, 0.0]] for row in rows),
        "foreground_hard_all_exact": all(value == [[0.0, 1.0]] for value in hard_values),
        "foreground_hard_unique_count_one": len(unique_hard) == 1,
        "foreground_alpha_unique_count_exceeds_gate": len(unique_alpha) > int(config["gates"]["foreground_alpha_unique_vector_count_maximum"]),
        "alpha_all_finite_positive_bounded": all(math.isfinite(value) and 0 < value <= 1.0 for value in alpha_scalars),
        "inputs_all_immutable": all(row["inputs_immutable"] is True for row in rows),
    }
    if not all(replay_checks.values()):
        raise ProtocolError(f"r047 raw-row replay drift: {replay_checks}")
    if (
        forensic.get("outcome") != "rejected" or forensic.get("process_count") != 8
        or forensic.get("unique_hard_vectors") != unique_hard or forensic.get("unique_alpha_vectors") != unique_alpha
        or forensic.get("alpha_unique_count") != len(unique_alpha) or forensic.get("alpha_min") != min(alpha_scalars)
        or forensic.get("alpha_max") != max(alpha_scalars) or forensic["checks"].get("foreground_alpha_unique_vector_count") is not False
        or forensic.get("threshold_search") is not False or forensic.get("source_patch") is not False
    ):
        raise ProtocolError("r047 aggregate forensic drift")

    source = config["official_source_hazard"]
    repo = Path(source["repository"])
    source_path = repo / source["path"]
    if source_path.stat().st_size != int(source["bytes"]) or sha256_file(source_path) != source["sha256"] or _git(repo, "rev-parse", "HEAD") != source["commit"] or _git(repo, "status", "--porcelain"):
        raise ProtocolError("r047 official source identity drift")
    text = source_path.read_text(encoding="utf-8")
    function = text[text.index(source["function_begin"]):text.index(source["function_end"], text.index(source["function_begin"]))]
    replay_hazard = {"plain_global_weight_increment_count": function.count("weights[int(collected_id[j] * (num_class+1) + C)] +="), "atomic_add_count": function.count("atomicAdd")}
    if hazard != replay_hazard or hazard != forensic["source_hazard"] or hazard != {"plain_global_weight_increment_count": 2, "atomic_add_count": 0}:
        raise ProtocolError("r047 source hazard drift")

    false_fields = (
        "network_access", "source_patch", "official_source_mutation", "real_checkpoint_read", "camera_metadata_read",
        "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation",
    )
    if any(summary.get(field) is not False for field in false_fields):
        raise ProtocolError("r047 no-data/no-quality declaration drift")
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("task_id") != TASK_ID or manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r047 manifest drift")
    return {
        "schema_version": "worldsim_v51_stage_g_g0b_r047_audit_v1", "task_id": TASK_ID, "status": "pass",
        "outcome": "rejected", "conclusion": expected_conclusion, "run_dir": str(run_dir),
        "source_commit": source_commit, "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "forensic": {"bytes": forensic_path.stat().st_size, "sha256": sha256_file(forensic_path)},
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "entry_count": len(inventory), "logical_bytes": sum(int(row["bytes"]) for row in inventory)},
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "process_evidence": process_evidence, "replay_checks": replay_checks,
        "unique_hard_vectors": unique_hard, "unique_alpha_vectors": unique_alpha,
        "alpha_min": min(alpha_scalars), "alpha_max": max(alpha_scalars), "source_hazard": replay_hazard,
        "resources": resources, "resource_checks": summary["resource_checks"],
        "real_checkpoint_read": False, "camera_metadata_read": False, "image_pixels_read": False, "mask_pixels_read": False,
        "quality_metrics_read": False, "training": False, "gaussian_mutation": False,
        "failure_ledger_delta": "V51-F65_resolved_as_faithful_operator_rejection",
        "next_task": config["decision"]["fail_next_task"], "next_action": config["decision"]["fail_next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0b_trace3d_cross_process_determinism_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.config.resolve(), args.run_dir.resolve())
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
