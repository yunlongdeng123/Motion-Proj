#!/usr/bin/env python3
"""Independently audit the rejected F0l train-only quality/alignment gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.audit_worldsim_v51_f0b_three_view_association_parity import _load_json, _load_jsonl, _load_yaml
from scripts.audit_worldsim_v51_f0c_upstream_batch_association_repeatability import _manifest_inventory
from scripts.run_worldsim_v51_f0l_train_only_quality_identity_alignment import _assignment_metrics
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    return left == right


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    expected = config["decision"]
    if (
        status.get("task_id") != TASK_ID
        or status.get("status") != "done"
        or status.get("outcome") != expected["reject_outcome"]
        or status.get("conclusion") != expected["reject_conclusion"]
    ):
        raise ProtocolError("r043 terminal drift")
    source_commit = status["source_commit"]
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    resolved_path = run_dir / "resolved_config.yaml"
    committed = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"])
    if resolved_path.read_bytes() != committed:
        raise ProtocolError("r043 resolved config drift")
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"] or events[-1].get("outcome") != "rejected":
        raise ProtocolError("r043 event drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("status") != "done"
        or summary.get("outcome") != "rejected"
        or summary.get("conclusion") != expected["reject_conclusion"]
        or summary.get("source_commit") != source_commit
        or summary.get("source_tree") != source_tree
        or summary.get("all_scenes_pass") is not False
        or summary.get("next_action") != expected["reject_next_action"]
    ):
        raise ProtocolError("r043 summary drift")
    report_path = run_dir / "artifacts/quality_alignment_report.json"
    report = _load_json(report_path)
    if (
        report.get("outcome") != "rejected"
        or report.get("conclusion") != expected["reject_conclusion"]
        or report.get("all_scenes_pass") is not False
        or report.get("scene_reports") != summary.get("scene_reports")
    ):
        raise ProtocolError("r043 report drift")

    input_spec = config["inputs"]["manifest"]
    input_manifest = _load_json(Path(input_spec["path"]))
    if len(input_manifest["views"]) != 45:
        raise ProtocolError("r043 input denominator drift")
    for view in input_manifest["views"]:
        for name in ("candidate_mask", "dynamic_mask"):
            spec = view[name]
            path = Path(spec["path"])
            if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
                raise ProtocolError(f"r043 input identity drift: {view['scene']}/{view['frame']}/{view['camera']}/{name}")
            with Image.open(path) as image:
                value = np.asarray(image)
            if value.shape != (900, 1600) or value.dtype != np.uint8:
                raise ProtocolError(f"r043 mask schema drift: {path}")

    audited_scenes = []
    for scene_report in summary["scene_reports"]:
        normalized_rows = []
        for row in scene_report["eligible_actor_views_detail"]:
            normalized = dict(row)
            normalized["label_counts"] = {int(label): int(count) for label, count in row["label_counts"].items()}
            normalized_rows.append(normalized)
        replay = _assignment_metrics(normalized_rows, config["evaluation"]["per_scene_thresholds"])
        for key, value in replay["metrics"].items():
            if not _close(value, scene_report["metrics"][key]):
                raise ProtocolError(f"r043 metric replay drift: {scene_report['scene']}/{key}")
        if replay["checks"] != scene_report["checks"] or replay["assignments"] != {token: int(label) for token, label in scene_report["assignments"].items()}:
            raise ProtocolError(f"r043 assignment replay drift: {scene_report['scene']}")
        if scene_report.get("all_checks_pass") != all(replay["checks"].values()):
            raise ProtocolError(f"r043 scene outcome drift: {scene_report['scene']}")
        audited_scenes.append({"scene": scene_report["scene"], "metrics": replay["metrics"], "checks": replay["checks"], "all_checks_pass": all(replay["checks"].values())})
    if [row["all_checks_pass"] for row in audited_scenes] != [False, True, False]:
        raise ProtocolError("r043 scene pass vector drift")

    resources_path = run_dir / "artifacts/resources.json"
    resources = _load_json(resources_path)
    if summary.get("resources") != resources or not all(summary["resource_checks"].values()):
        raise ProtocolError("r043 resource drift")
    for key, expected_value in {
        "candidate_mask_pixel_reads": 45, "dynamic_mask_pixel_reads": 45, "image_pixels_read": False,
        "threshold_search": False, "identity_training_authorized": False, "validation_quality_read": False,
        "test_quality_read": False, "m2_status": "pending", "m3_status": "pending",
    }.items():
        if summary.get(key) != expected_value:
            raise ProtocolError(f"r043 research lock drift: {key}")
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r043 inventory drift")
    return {
        "schema_version": "worldsim_v51_stage_f_f0l_r043_audit_v1", "task_id": TASK_ID, "status": "pass",
        "audited_outcome": "rejected", "conclusion": expected["reject_conclusion"], "run_dir": str(run_dir),
        "source_commit": source_commit, "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "quality_alignment_report": {"bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "entry_count": len(inventory), "logical_bytes": sum(int(row["bytes"]) for row in inventory)},
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "scene_reports": audited_scenes, "scene_pass_vector": [False, True, False],
        "candidate_mask_pixel_reads": 45, "dynamic_mask_pixel_reads": 45, "image_pixels_read": False,
        "threshold_search": False, "identity_training_authorized": False, "resources": resources,
        "resource_checks": summary["resource_checks"],
        "failure_ledger_delta": "V51-F63_faithful_gaussian_grouping_rejected_train_only_quality_identity_alignment_gate",
        "next_action": expected["reject_next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_v1.yaml")
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
