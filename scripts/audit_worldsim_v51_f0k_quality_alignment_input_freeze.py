#!/usr/bin/env python3
"""Audit F0k identities and projection denominator without decoding masks."""

from __future__ import annotations

import argparse
import json
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


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    conclusion = config["decision"]["expected_conclusion"]
    if status.get("task_id") != TASK_ID or status.get("status") != "done" or status.get("conclusion") != conclusion:
        raise ProtocolError("r042 terminal drift")
    source_commit = status["source_commit"]
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    resolved_path = run_dir / "resolved_config.yaml"
    committed = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"])
    if resolved_path.read_bytes() != committed:
        raise ProtocolError("r042 resolved config drift")
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r042 event drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("status") != "done"
        or summary.get("conclusion") != conclusion
        or summary.get("source_commit") != source_commit
        or summary.get("source_tree") != source_tree
        or summary.get("view_count") != 45
        or summary.get("projection_count") != 90
    ):
        raise ProtocolError("r042 summary drift")
    input_path = run_dir / "artifacts/quality_alignment_input_manifest.json"
    inputs = _load_json(input_path)
    if summary.get("input_manifest") != inputs or inputs.get("view_count") != 45 or inputs.get("projection_count") != 90:
        raise ProtocolError("r042 embedded input manifest drift")
    for key in ("candidate_mask_pixels_read", "dynamic_mask_pixels_read", "image_pixels_read"):
        if inputs.get(key) is not False or summary.get(key) is not False:
            raise ProtocolError(f"r042 no-pixel lock drift: {key}")
    if summary.get("quality_metrics_read") is not False or summary.get("identity_training_authorized") is not False:
        raise ProtocolError("r042 quality/training lock drift")
    if inputs.get("quality_gate_preregistration") != config["quality_gate_preregistration"]:
        raise ProtocolError("r042 preregistered gate drift")

    seen = set()
    projection_count = 0
    asset_bytes = 0
    for view in inputs["views"]:
        key = (view["scene"], int(view["frame"]), int(view["camera"]))
        if key in seen:
            raise ProtocolError(f"r042 duplicate view: {key}")
        seen.add(key)
        if (int(view["width"]), int(view["height"])) != (1600, 900):
            raise ProtocolError(f"r042 view shape drift: {key}")
        for asset_name in ("source_image", "candidate_mask", "dynamic_mask", "intrinsics", "extrinsics"):
            spec = view[asset_name]
            path = Path(spec["path"])
            if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
                raise ProtocolError(f"r042 asset identity drift: {key}/{asset_name}")
            asset_bytes += int(spec["bytes"])
        for projection in view["projections"]:
            box = projection["box_xyxy"]
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1599 and 0 <= box[1] < box[3] <= 899):
                raise ProtocolError(f"r042 projected box drift: {key}")
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if abs(area - float(projection["box_area_pixels"])) > 0.25 or area < int(config["projection"]["minimum_box_area_pixels"]):
                raise ProtocolError(f"r042 projected area drift: {key}")
            if not projection.get("instance_token") or not projection.get("class_name"):
                raise ProtocolError(f"r042 projection identity drift: {key}")
            projection_count += 1
    if len(seen) != 45 or projection_count != 90:
        raise ProtocolError("r042 replay denominator drift")

    resources_path = run_dir / "artifacts/resources.json"
    resources = _load_json(resources_path)
    if summary.get("resources") != resources or not all(summary["resource_checks"].values()):
        raise ProtocolError("r042 resource drift")
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r042 inventory drift")
    return {
        "schema_version": "worldsim_v51_stage_f_f0k_r042_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "conclusion": conclusion,
        "run_dir": str(run_dir),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "input_manifest": {"bytes": input_path.stat().st_size, "sha256": sha256_file(input_path), "view_count": 45, "projection_count": projection_count, "verified_asset_logical_bytes": asset_bytes},
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "entry_count": len(inventory), "logical_bytes": sum(int(row["bytes"]) for row in inventory)},
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "resources": resources,
        "resource_checks": summary["resource_checks"],
        "candidate_mask_pixels_read": False,
        "dynamic_mask_pixels_read": False,
        "image_pixels_read": False,
        "quality_metrics_read": False,
        "identity_training_authorized": False,
        "failure_ledger_delta": "none",
        "next_action": config["decision"]["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0k_quality_alignment_input_freeze_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"refusing overwrite: {output}")
    result = audit(args.config.resolve(), args.run_dir.resolve())
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
