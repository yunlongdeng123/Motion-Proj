#!/usr/bin/env python3
"""独立审计 F0h r039 pre-matmul empty-cache parity。"""

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
from scripts.run_worldsim_v51_f0e_scene1087_cuda_fault_localization import _schema_record
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
ATTEMPT_NAMES = ["control_cache_1", "target_cache_1", "control_cache_2", "target_cache_2"]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if (
        config.get("task_id") != TASK_ID
        or status.get("task_id") != TASK_ID
        or status.get("status") != "done"
        or status.get("outcome") != "recovery_pass"
        or status.get("conclusion") != config["decision"]["pass_conclusion"]
    ):
        raise ProtocolError("r039 terminal identity drift")
    source_commit = str(status["source_commit"])
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    committed_config = subprocess.check_output(
        ["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"]
    )
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.read_bytes() != committed_config:
        raise ProtocolError("r039 resolved config differs from source commit")
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r039 event terminal drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("status") != "done"
        or summary.get("outcome") != "recovery_pass"
        or summary.get("source_commit") != source_commit
        or summary.get("source_tree") != source_tree
    ):
        raise ProtocolError("r039 summary identity drift")

    staged_groups = {}
    for group_name, group in config["input_groups"].items():
        records = []
        for source in group["inputs"]:
            staged = run_dir / "artifacts/inputs" / group_name / source["staging_filename"]
            if not staged.is_symlink() or staged.resolve() != Path(source["path"]).resolve():
                raise ProtocolError(f"r039 input staging drift: {group_name}/{staged.name}")
            if staged.stat().st_size != int(source["bytes"]) or sha256_file(staged) != source["sha256"]:
                raise ProtocolError(f"r039 input identity drift: {group_name}/{staged.name}")
            records.append({"filename": staged.name, "bytes": staged.stat().st_size, "sha256": sha256_file(staged)})
        staged_groups[group_name] = records

    if [row.get("name") for row in summary["attempts"]] != ATTEMPT_NAMES:
        raise ProtocolError("r039 attempt order drift")
    audited_attempts = []
    for attempt, expected in zip(summary["attempts"], config["execution"]["attempts"]):
        name = expected["name"]
        group_name = expected["input_group"]
        if attempt.get("classification") != "success" or int(attempt.get("returncode", -1)) != 0:
            raise ProtocolError(f"r039 attempt success drift: {name}")
        attempt_dir = run_dir / "artifacts/attempts" / name
        trace_path = attempt_dir / "trace.json"
        trace = _load_json(trace_path)
        if (
            trace.get("terminal") != {"status": "success"}
            or trace.get("pre_matmul_empty_cache") is not True
            or trace.get("operator_monkeypatch") is not False
            or trace.get("tensor_content_read") is not False
        ):
            raise ProtocolError(f"r039 trace terminal/intervention drift: {name}")
        pre_events = [row for row in trace["events"] if row.get("event") == "pre_matmul"]
        if len(pre_events) != 2:
            raise ProtocolError(f"r039 empty-cache denominator drift: {name}")
        deltas = []
        for row in pre_events:
            before = int(row["empty_cache"]["before"]["free_bytes"])
            after = int(row["empty_cache"]["after"]["free_bytes"])
            if after < before:
                raise ProtocolError(f"r039 empty-cache free-memory drift: {name}")
            deltas.append({"before_free_bytes": before, "after_free_bytes": after, "released_bytes": after - before})
        output_dir = attempt_dir / "output"
        masks = sorted((output_dir / "Annotations").glob("*.png"))
        expected_names = [
            f"{Path(row['staging_filename']).stem}.png"
            for row in config["input_groups"][group_name]["inputs"]
        ]
        pred_path = output_dir / "pred.json"
        if [path.name for path in masks] != expected_names or not pred_path.is_file():
            raise ProtocolError(f"r039 output denominator drift: {name}")
        computed_masks = [_schema_record(path) for path in masks]
        computed_metadata = {"bytes": pred_path.stat().st_size, "sha256": sha256_file(pred_path)}
        reference = config["input_groups"][group_name]["reference"]
        if (
            attempt.get("masks") != computed_masks
            or attempt.get("metadata") != computed_metadata
            or [row["sha256"] for row in computed_masks] != reference["mask_sha256"]
            or computed_metadata["sha256"] != reference["metadata_sha256"]
        ):
            raise ProtocolError(f"r039 reference parity drift: {name}")
        audited_attempts.append(
            {
                "name": name,
                "input_group": group_name,
                "classification": "success",
                "masks": computed_masks,
                "metadata": computed_metadata,
                "trace": {"bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path)},
                "empty_cache_calls": deltas,
            }
        )
    parity = summary.get("parity_checks")
    if (
        parity.get("classes") != ["success"] * 4
        or parity.get("empty_cache_checks") != [True] * 4
        or parity.get("reference_checks") != [True] * 4
        or parity.get("control_pair_exact") is not True
        or parity.get("target_pair_exact") is not True
    ):
        raise ProtocolError("r039 parity summary drift")

    resources_path = run_dir / "artifacts/resources.json"
    resources = _load_json(resources_path)
    samples_path = run_dir / "artifacts/resource_samples.jsonl"
    samples = _load_jsonl(samples_path)
    valid = [row for row in samples if "monitor_error" not in row]
    if (
        len(valid) != len(samples)
        or int(resources["sample_count"]) != len(samples)
        or int(resources["nvidia_peak_mib"]) != max(int(row["gpu_used_mib"]) for row in valid)
        or int(resources["cgroup_memory_peak_bytes"])
        != max(int(row["cgroup_memory_current_bytes"]) for row in valid)
    ):
        raise ProtocolError("r039 resource replay drift")
    if summary.get("resources") != resources or not all(summary["resource_checks"].values()):
        raise ProtocolError("r039 resource gate drift")
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r039 manifest inventory drift")
    for key in (
        "upstream_source_mutation",
        "operator_change",
        "tensor_content_change",
        "quality_read",
        "full_materialization",
        "identity_training_authorized",
    ):
        if summary.get(key) is not False:
            raise ProtocolError(f"r039 research lock drift: {key}")

    input_bytes = sum(int(row["bytes"]) for group in staged_groups.values() for row in group)
    return {
        "schema_version": "worldsim_v51_stage_f_f0h_r039_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "done",
        "outcome": "recovery_pass",
        "conclusion": config["decision"]["pass_conclusion"],
        "run_dir": str(run_dir),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "entry_count": len(inventory),
            "logical_bytes": sum(int(row["bytes"]) for row in inventory),
            "regular_bytes_excluding_input_symlink_targets": sum(int(row["bytes"]) for row in inventory) - input_bytes,
        },
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "attempts": audited_attempts,
        "parity_checks": parity,
        "empty_cache_call_count": sum(len(row["empty_cache_calls"]) for row in audited_attempts),
        "resources": resources,
        "resource_checks": summary["resource_checks"],
        "quality_read": False,
        "full_materialization": False,
        "identity_training_authorized": False,
        "failure_ledger_delta": "V51-F62_recovery_candidate_pass_full_materialization_not_yet_resolved",
        "next_action": "preregister_scene1087_15_view_empty_cache_recovery",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ProtocolError(f"refusing to overwrite audit: {output}")
    result = audit(args.config.resolve(), args.run_dir.resolve())
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
