#!/usr/bin/env python3
"""独立审计 F0g r038 source-neutral tensor/allocator trace。"""

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
ATTEMPT_NAMES = ["control_trace", "target_trace"]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def _matmul_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = trace.get("events", [])
    if [row.get("event") for row in events] != ["pre_matmul", "post_matmul"] * 2:
        raise ProtocolError("r038 trace event sequence drift")
    calls = []
    for index in (0, 2):
        pre = events[index]
        post = events[index + 1]
        if pre.get("source_line") != 58 or post.get("source_line") != 59:
            raise ProtocolError("r038 traced line drift")
        if pre.get("call_index") != post.get("call_index"):
            raise ProtocolError("r038 trace call pairing drift")
        value = pre["tensors"]["value"]
        affinity = pre["tensors"]["affinity"]
        readout = post["tensors"]["memory_readout"]
        if value["dtype"] != "torch.float16" or affinity["dtype"] != "torch.float32":
            raise ProtocolError("r038 matmul dtype drift")
        if value["shape"][-1] != affinity["shape"][-2]:
            raise ProtocolError("r038 matmul dimension drift")
        if readout["shape"] != [value["shape"][0], value["shape"][1], affinity["shape"][2]]:
            raise ProtocolError("r038 readout shape drift")
        if not value["contiguous"] or not affinity["contiguous"] or not readout["contiguous"]:
            raise ProtocolError("r038 matmul contiguity drift")
        calls.append(
            {
                "call_index": int(pre["call_index"]),
                "src_ti": int(pre["scalars"]["src_ti"]),
                "tar_ti": int(pre["scalars"]["tar_ti"]),
                "num_objects": int(pre["scalars"]["num_objects"]),
                "value_shape": value["shape"],
                "value_logical_bytes": int(value["logical_bytes"]),
                "affinity_shape": affinity["shape"],
                "affinity_logical_bytes": int(affinity["logical_bytes"]),
                "readout_shape": readout["shape"],
                "readout_logical_bytes": int(readout["logical_bytes"]),
                "allocator_pre": pre["allocator"],
                "allocator_post": post["allocator"],
            }
        )
    return calls


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    expected_conclusion = config["decision"]["expected_outcomes"]["both_success"]
    if (
        config.get("task_id") != TASK_ID
        or status.get("task_id") != TASK_ID
        or status.get("status") != "done"
        or status.get("outcome") != "both_success"
        or status.get("conclusion") != expected_conclusion
    ):
        raise ProtocolError("r038 terminal identity drift")
    source_commit = str(status["source_commit"])
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    committed_config = subprocess.check_output(
        ["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"]
    )
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.read_bytes() != committed_config:
        raise ProtocolError("r038 resolved config differs from source commit")
    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r038 event terminal drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("status") != "done"
        or summary.get("outcome") != "both_success"
        or summary.get("conclusion") != expected_conclusion
        or summary.get("source_commit") != source_commit
        or summary.get("source_tree") != source_tree
    ):
        raise ProtocolError("r038 summary identity drift")

    staged_groups = {}
    for group_name, group in config["input_groups"].items():
        records = []
        for source in group["inputs"]:
            staged = run_dir / "artifacts/inputs" / group_name / source["staging_filename"]
            if not staged.is_symlink() or staged.resolve() != Path(source["path"]).resolve():
                raise ProtocolError(f"r038 input staging drift: {group_name}/{staged.name}")
            if staged.stat().st_size != int(source["bytes"]) or sha256_file(staged) != source["sha256"]:
                raise ProtocolError(f"r038 input identity drift: {group_name}/{staged.name}")
            records.append(
                {"filename": staged.name, "target": str(staged.resolve()), "bytes": staged.stat().st_size, "sha256": sha256_file(staged)}
            )
        staged_groups[group_name] = records

    if [row.get("name") for row in summary["attempts"]] != ATTEMPT_NAMES:
        raise ProtocolError("r038 attempt order drift")
    audited_attempts = []
    for attempt, expected in zip(summary["attempts"], config["execution"]["attempts"]):
        name = expected["name"]
        group_name = expected["input_group"]
        if attempt.get("classification") != "success" or int(attempt.get("returncode", -1)) != 0:
            raise ProtocolError(f"r038 attempt success drift: {name}")
        attempt_dir = run_dir / "artifacts/attempts" / name
        for key in ("stdout", "stderr"):
            path = attempt_dir / f"{key}.log"
            if attempt[key] != {"bytes": path.stat().st_size, "sha256": sha256_file(path)}:
                raise ProtocolError(f"r038 {name} {key} identity drift")
        trace_path = attempt_dir / "trace.json"
        trace = _load_json(trace_path)
        trace_identity = {"bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path)}
        if (
            attempt["trace"]["bytes"] != trace_identity["bytes"]
            or attempt["trace"]["sha256"] != trace_identity["sha256"]
            or attempt["trace"]["payload"] != trace
            or trace.get("terminal") != {"status": "success"}
            or trace.get("operator_monkeypatch") is not False
            or trace.get("tensor_content_read") is not False
            or trace.get("trace_source", {}).get("sha256") != config["sources"]["traced_file"]["sha256"]
        ):
            raise ProtocolError(f"r038 trace identity/source-neutral drift: {name}")
        output_dir = attempt_dir / "output"
        mask_paths = sorted((output_dir / "Annotations").glob("*.png"))
        expected_names = [
            f"{Path(row['staging_filename']).stem}.png"
            for row in config["input_groups"][group_name]["inputs"]
        ]
        pred_path = output_dir / "pred.json"
        if [path.name for path in mask_paths] != expected_names or not pred_path.is_file():
            raise ProtocolError(f"r038 output denominator drift: {name}")
        computed_masks = [_schema_record(path) for path in mask_paths]
        if attempt.get("masks") != computed_masks:
            raise ProtocolError(f"r038 output mask identity drift: {name}")
        computed_metadata = {"bytes": pred_path.stat().st_size, "sha256": sha256_file(pred_path)}
        if attempt.get("metadata") != computed_metadata:
            raise ProtocolError(f"r038 output metadata identity drift: {name}")
        audited_attempts.append(
            {
                "name": name,
                "input_group": group_name,
                "classification": "success",
                "masks": computed_masks,
                "metadata": computed_metadata,
                "trace": trace_identity,
                "matmul_calls": _matmul_calls(trace),
            }
        )

    control_calls = audited_attempts[0]["matmul_calls"]
    target_calls = audited_attempts[1]["matmul_calls"]
    if [row["num_objects"] for row in control_calls] != [26, 36]:
        raise ProtocolError("r038 control object-count trace drift")
    if [row["num_objects"] for row in target_calls] != [3, 52]:
        raise ProtocolError("r038 target object-count trace drift")
    if any(row["allocator_pre"]["memory_stats"]["num_alloc_retries"] != 0 for row in control_calls):
        raise ProtocolError("r038 control allocator retry drift")
    if any(row["allocator_pre"]["memory_stats"]["num_alloc_retries"] != 1 for row in target_calls):
        raise ProtocolError("r038 target allocator retry drift")
    if not all(
        target["allocator_pre"]["free_bytes"] > control["allocator_pre"]["free_bytes"]
        for control, target in zip(control_calls, target_calls)
    ):
        raise ProtocolError("r038 allocator free-memory contrast drift")

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
        raise ProtocolError("r038 resource replay drift")
    if summary.get("resources") != resources or not all(summary["resource_checks"].values()):
        raise ProtocolError("r038 resource gate drift")
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r038 manifest inventory drift")
    for key in (
        "upstream_source_mutation",
        "operator_monkeypatch",
        "tensor_content_read",
        "quality_read",
        "full_materialization",
        "identity_training_authorized",
    ):
        if summary.get(key) is not False:
            raise ProtocolError(f"r038 research lock drift: {key}")

    input_bytes = sum(int(row["bytes"]) for group in staged_groups.values() for row in group)
    return {
        "schema_version": "worldsim_v51_stage_f_f0g_r038_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "done",
        "outcome": "both_success",
        "conclusion": expected_conclusion,
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
        "input_groups": staged_groups,
        "attempts": audited_attempts,
        "mechanism_observation": {
            "control_num_objects": [26, 36],
            "target_num_objects": [3, 52],
            "control_pre_matmul_free_bytes": [row["allocator_pre"]["free_bytes"] for row in control_calls],
            "target_pre_matmul_free_bytes": [row["allocator_pre"]["free_bytes"] for row in target_calls],
            "control_allocator_retries": [0, 0],
            "target_allocator_retries": [1, 1],
            "target_first_matmul_is_smaller_than_control_first": True,
            "allocator_cache_state_correlates_with_target_success": True,
            "allocator_cache_state_is_proven_root_cause": False,
        },
        "resources": resources,
        "resource_checks": summary["resource_checks"],
        "quality_read": False,
        "full_materialization": False,
        "identity_training_authorized": False,
        "failure_ledger_delta": "V51-F62_refined_allocator_cache_workspace_hypothesis_not_root_cause_proof",
        "next_action": "preregister_pre_matmul_empty_cache_execution_recovery_parity",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_v1.yaml",
    )
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
