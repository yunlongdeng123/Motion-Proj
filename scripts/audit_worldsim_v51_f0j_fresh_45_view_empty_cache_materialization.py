#!/usr/bin/env python3
"""Independently audit the F0j fresh three-scene materialization."""

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
from scripts.run_worldsim_v51_f0j_fresh_45_view_empty_cache_materialization import _record_chain
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    conclusion = config["decision"]["expected_conclusion"]
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if (
        config.get("task_id") != TASK_ID
        or status.get("task_id") != TASK_ID
        or status.get("status") != "done"
        or status.get("conclusion") != conclusion
    ):
        raise ProtocolError("r041 terminal identity drift")

    source_commit = str(status["source_commit"])
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    committed_config = subprocess.check_output(
        ["git", "-C", str(PROJECT), "show", f"{source_commit}:configs/worldsim_v51/{config_path.name}"]
    )
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.read_bytes() != committed_config:
        raise ProtocolError("r041 resolved config differs from source commit")

    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r041 event terminal drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("task_id") != TASK_ID
        or summary.get("status") != "done"
        or summary.get("conclusion") != conclusion
        or summary.get("source_commit") != source_commit
        or summary.get("source_tree") != source_tree
        or summary.get("next_action") != config["decision"]["next_action"]
    ):
        raise ProtocolError("r041 summary identity drift")

    source_manifest = _load_json(Path(config["input_manifest"]["path"]))
    source_rows = [dict(row) for row in source_manifest["records"]]
    expected_order = [
        (scene["scene"], int(scene["scene_index"]), int(frame), int(camera))
        for scene in config["scene_contracts"]
        for frame in scene["frames"]
        for camera in scene["cameras"]
    ]
    observed_order = [
        (row["scene"], int(row["scene_index"]), int(row["frame"]), int(row["camera"])) for row in source_rows
    ]
    if len(source_rows) != 45 or observed_order != expected_order:
        raise ProtocolError("r041 input denominator/order drift")
    staged_inputs = []
    for row in source_rows:
        staged = run_dir / "artifacts/inputs" / row["scene"] / Path(row["path"]).name
        if not staged.is_symlink() or staged.resolve() != Path(row["path"]).resolve():
            raise ProtocolError(f"r041 input staging drift: {row['scene']}/{staged.name}")
        if staged.stat().st_size != int(row["bytes"]) or sha256_file(staged) != row["sha256"]:
            raise ProtocolError(f"r041 input identity drift: {row['scene']}/{staged.name}")
        staged_inputs.append(
            {"scene": row["scene"], "filename": staged.name, "bytes": staged.stat().st_size, "sha256": sha256_file(staged)}
        )

    attempt_specs = config["execution"]["attempts"]
    attempts = summary["attempts"]
    if [row.get("name") for row in attempts] != [row["name"] for row in attempt_specs]:
        raise ProtocolError("r041 attempt order drift")
    audited_attempts = []
    materialized_records = []
    metadata_records = []
    for attempt, spec in zip(attempts, attempt_specs):
        scene = spec["input_group"]
        if (
            attempt.get("input_group") != scene
            or attempt.get("classification") != "success"
            or int(attempt.get("returncode", -1)) != 0
        ):
            raise ProtocolError(f"r041 attempt success drift: {scene}")
        attempt_dir = run_dir / "artifacts/attempts" / spec["name"]
        trace_path = attempt_dir / "trace.json"
        trace = _load_json(trace_path)
        if (
            trace.get("terminal") != {"status": "success"}
            or trace.get("pre_matmul_empty_cache") is not True
            or trace.get("operator_monkeypatch") is not False
            or trace.get("tensor_content_read") is not False
            or trace.get("trace_source", {}).get("sha256") != config["sources"]["traced_file"]["sha256"]
        ):
            raise ProtocolError(f"r041 trace source/intervention drift: {scene}")
        if (
            attempt.get("trace", {}).get("bytes") != trace_path.stat().st_size
            or attempt.get("trace", {}).get("sha256") != sha256_file(trace_path)
            or attempt.get("trace", {}).get("payload") != trace
        ):
            raise ProtocolError(f"r041 embedded trace drift: {scene}")
        pre_events = [row for row in trace["events"] if row.get("event") == "pre_matmul"]
        post_events = [row for row in trace["events"] if row.get("event") == "post_matmul"]
        if not pre_events or len(pre_events) != len(post_events):
            raise ProtocolError(f"r041 matmul event denominator drift: {scene}")
        empty_cache_calls = []
        for row in pre_events:
            if row.get("source_line") != 58 or "empty_cache" not in row:
                raise ProtocolError(f"r041 intervention evidence drift: {scene}")
            before = int(row["empty_cache"]["before"]["free_bytes"])
            after = int(row["empty_cache"]["after"]["free_bytes"])
            if after < before:
                raise ProtocolError(f"r041 free-memory drift: {scene}")
            empty_cache_calls.append(
                {"before_free_bytes": before, "after_free_bytes": after, "released_bytes": after - before}
            )

        scene_rows = [row for row in source_rows if row["scene"] == scene]
        output_dir = attempt_dir / "output"
        masks = sorted((output_dir / "Annotations").glob("*.png"))
        expected_names = [f"{Path(row['path']).stem}.png" for row in scene_rows]
        pred_path = output_dir / "pred.json"
        if [path.name for path in masks] != expected_names or not pred_path.is_file():
            raise ProtocolError(f"r041 output denominator drift: {scene}")
        computed_masks = [_schema_record(path) for path in masks]
        if any(row["dtype"] != "uint8" or row["shape"] != [900, 1600] for row in computed_masks):
            raise ProtocolError(f"r041 mask schema drift: {scene}")
        computed_metadata = {"bytes": pred_path.stat().st_size, "sha256": sha256_file(pred_path)}
        if attempt.get("masks") != computed_masks or attempt.get("metadata") != computed_metadata:
            raise ProtocolError(f"r041 output record drift: {scene}")
        materialized_records.extend({"scene": scene, **row} for row in computed_masks)
        metadata_records.append({"scene": scene, **computed_metadata})
        audited_attempts.append(
            {
                "name": spec["name"],
                "scene": scene,
                "classification": "success",
                "masks": computed_masks,
                "metadata": computed_metadata,
                "trace": {"bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path)},
                "empty_cache_calls": empty_cache_calls,
            }
        )

    if len(materialized_records) != 45 or len(metadata_records) != 3:
        raise ProtocolError("r041 materialized denominator drift")
    materialization_path = run_dir / "artifacts/materialization_manifest.json"
    materialization = _load_json(materialization_path)
    expected_chain = _record_chain([*materialized_records, *metadata_records])
    if (
        materialization.get("record_count") != 45
        or materialization.get("metadata_count") != 3
        or materialization.get("records") != materialized_records
        or materialization.get("metadata") != metadata_records
        or materialization.get("output_record_chain_sha256") != expected_chain
        or materialization.get("quality_read") is not False
        or materialization.get("actor_identity_alignment_read") is not False
        or summary.get("materialization_manifest") != materialization
    ):
        raise ProtocolError("r041 materialization manifest/chain drift")
    empty_cache_count = sum(len(row["empty_cache_calls"]) for row in audited_attempts)
    if int(summary.get("empty_cache_call_count", -1)) != empty_cache_count:
        raise ProtocolError("r041 empty-cache count drift")

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
        or summary.get("resources") != resources
        or not all(summary["resource_checks"].values())
    ):
        raise ProtocolError("r041 resource replay/gate drift")

    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r041 run inventory drift")
    if (
        summary.get("input_image_pixels_decoded_count") != 45
        or summary.get("output_schema_reads_count") != 45
        or summary.get("full_materialization_execution") is not True
        or summary.get("quality_read") is not False
        or summary.get("actor_identity_alignment_read") is not False
        or summary.get("identity_training_authorized") is not False
        or summary.get("validation_quality_read") is not False
        or summary.get("test_quality_read") is not False
        or summary.get("m2_status") != "pending"
        or summary.get("m3_status") != "pending"
    ):
        raise ProtocolError("r041 research lock drift")

    input_bytes = sum(int(row["bytes"]) for row in staged_inputs)
    return {
        "schema_version": "worldsim_v51_stage_f_f0j_r041_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "done",
        "conclusion": conclusion,
        "run_dir": str(run_dir),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "resolved_config": {"bytes": resolved_path.stat().st_size, "sha256": sha256_file(resolved_path)},
        "summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "materialization_manifest": {
            "bytes": materialization_path.stat().st_size,
            "sha256": sha256_file(materialization_path),
            "record_count": 45,
            "metadata_count": 3,
            "output_record_chain_sha256": expected_chain,
        },
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "entry_count": len(inventory),
            "logical_bytes": sum(int(row["bytes"]) for row in inventory),
            "regular_bytes_excluding_input_symlink_targets": sum(int(row["bytes"]) for row in inventory) - input_bytes,
        },
        "status_file": {"bytes": status_path.stat().st_size, "sha256": sha256_file(status_path)},
        "events": {"bytes": events_path.stat().st_size, "sha256": sha256_file(events_path)},
        "inputs": {"count": len(staged_inputs), "total_bytes": input_bytes},
        "attempts": audited_attempts,
        "mask_count": len(materialized_records),
        "pred_json_count": len(metadata_records),
        "empty_cache_call_count": empty_cache_count,
        "resources": resources,
        "resource_checks": summary["resource_checks"],
        "quality_read": False,
        "actor_identity_alignment_read": False,
        "identity_training_authorized": False,
        "failure_ledger_delta": "V51-F62_resolved_for_frozen_empty_cache_45_view_materialization_execution_only",
        "next_action": config["decision"]["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0j_fresh_45_view_empty_cache_materialization_v1.yaml",
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
