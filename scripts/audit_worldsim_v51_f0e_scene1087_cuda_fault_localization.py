#!/usr/bin/env python3
"""独立审计 F0e r036 scene-1087 CUDA fault localization。"""

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
from scripts.audit_worldsim_v51_f0b_three_view_association_parity import (
    _load_json,
    _load_jsonl,
    _load_yaml,
)
from scripts.audit_worldsim_v51_f0c_upstream_batch_association_repeatability import (
    _manifest_inventory,
)
from scripts.run_worldsim_v51_f0e_scene1087_cuda_fault_localization import (
    ATTEMPT_NAMES,
    _schema_record,
)
from scripts.run_worldsim_v51_h_uplift import _write_json


TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("task_id") != TASK_ID:
        raise ProtocolError("F0e config task drift")
    status_path = run_dir / "status.json"
    status = _load_json(status_path)
    if status.get("task_id") != TASK_ID or status.get("status") != "done":
        raise ProtocolError("r036 must remain done")
    if status.get("outcome") != "mixed":
        raise ProtocolError("r036 outcome drift")
    expected_conclusion = config["decision"]["expected_outcomes"]["mixed"]
    if status.get("conclusion") != expected_conclusion:
        raise ProtocolError("r036 conclusion drift")
    source_commit = str(status["source_commit"])
    source_tree = _git("show", "-s", "--format=%T", source_commit)
    committed_config = subprocess.check_output(
        [
            "git",
            "-C",
            str(PROJECT),
            "show",
            f"{source_commit}:configs/worldsim_v51/{config_path.name}",
        ]
    )
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.read_bytes() != committed_config:
        raise ProtocolError("r036 resolved config differs from source commit")

    events_path = run_dir / "events.jsonl"
    events = _load_jsonl(events_path)
    if [row.get("event") for row in events] != ["run_started", "run_completed"]:
        raise ProtocolError("r036 event terminal drift")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if (
        summary.get("status") != "done"
        or summary.get("outcome") != "mixed"
        or summary.get("conclusion") != expected_conclusion
        or summary.get("source_commit") != source_commit
        or summary.get("source_tree") != source_tree
    ):
        raise ProtocolError("r036 summary identity drift")

    input_dir = run_dir / "artifacts/input"
    staged_inputs = []
    for source in config["inputs"]:
        staged = input_dir / source["staging_filename"]
        if not staged.is_symlink() or staged.resolve() != Path(source["path"]).resolve():
            raise ProtocolError(f"r036 input staging drift: {staged}")
        if staged.stat().st_size != int(source["bytes"]) or sha256_file(staged) != source["sha256"]:
            raise ProtocolError(f"r036 input identity drift: {staged}")
        staged_inputs.append(
            {
                "filename": staged.name,
                "target": str(staged.resolve()),
                "bytes": staged.stat().st_size,
                "sha256": sha256_file(staged),
            }
        )

    if [row.get("name") for row in summary["attempts"]] != ATTEMPT_NAMES:
        raise ProtocolError("r036 attempt order drift")
    replayed_attempts = []
    for attempt, expected in zip(summary["attempts"], config["execution"]["attempts"]):
        name = expected["name"]
        attempt_dir = run_dir / "artifacts/attempts" / name
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        for key, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if attempt[key] != {"bytes": path.stat().st_size, "sha256": sha256_file(path)}:
                raise ProtocolError(f"r036 {name} {key} identity drift")
        command = attempt.get("command")
        if not isinstance(command, list):
            raise ProtocolError(f"r036 {name} command missing")
        for flag, value in (
            ("--SAM_NUM_POINTS_PER_SIDE", "32"),
            ("--SAM_NUM_POINTS_PER_BATCH", "64"),
            ("--size", "480"),
        ):
            if flag not in command or command[command.index(flag) + 1] != value:
                raise ProtocolError(f"r036 {name} command drift: {flag}")
        if attempt.get("cuda_launch_blocking") != "1":
            raise ProtocolError(f"r036 {name} CUDA launch blocking drift")
        stderr = stderr_path.read_text(encoding="utf-8")
        if "Downloading:" in stderr:
            raise ProtocolError(f"r036 {name} hidden download drift")
        output_dir = attempt_dir / "output"
        mask_dir = output_dir / "Annotations"
        masks = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
        pred_path = output_dir / "pred.json"
        if name == ATTEMPT_NAMES[0]:
            if attempt.get("classification") != "expected_cublas_internal_failure":
                raise ProtocolError("r036 first attempt class drift")
            if any(marker not in stderr for marker in config["execution"]["expected_failure_markers"]):
                raise ProtocolError("r036 first attempt failure signature drift")
            if masks or pred_path.exists() or attempt.get("explicit_pytorch_oom") is not False:
                raise ProtocolError("r036 first attempt output/OOM boundary drift")
            replayed_attempts.append(
                {
                    "name": name,
                    "classification": "expected_cublas_internal_failure",
                    "returncode": int(attempt["returncode"]),
                    "stdout": attempt["stdout"],
                    "stderr": attempt["stderr"],
                    "mask_count": 0,
                    "pred_json": False,
                    "explicit_pytorch_oom": False,
                }
            )
            continue
        if attempt.get("classification") != "success" or int(attempt.get("returncode", -1)) != 0:
            raise ProtocolError("r036 second attempt class drift")
        expected_names = [f"{Path(row['staging_filename']).stem}.png" for row in config["inputs"]]
        if [path.name for path in masks] != expected_names or not pred_path.is_file():
            raise ProtocolError("r036 second attempt output denominator drift")
        computed_masks = [_schema_record(path) for path in masks]
        if attempt.get("masks") != computed_masks:
            raise ProtocolError("r036 second attempt mask identity drift")
        pred = _load_json(pred_path)
        if [row.get("file_name") for row in pred.get("annotations", [])] != [
            row["staging_filename"] for row in config["inputs"]
        ]:
            raise ProtocolError("r036 second attempt metadata order drift")
        computed_metadata = {
            "bytes": pred_path.stat().st_size,
            "sha256": sha256_file(pred_path),
            "annotation_count": len(pred["annotations"]),
        }
        if attempt.get("metadata") != computed_metadata:
            raise ProtocolError("r036 second attempt metadata identity drift")
        replayed_attempts.append(
            {
                "name": name,
                "classification": "success",
                "returncode": 0,
                "stdout": attempt["stdout"],
                "stderr": attempt["stderr"],
                "masks": computed_masks,
                "metadata": computed_metadata,
            }
        )

    resources_path = run_dir / "artifacts/resources.json"
    resources = _load_json(resources_path)
    samples_path = run_dir / "artifacts/resource_samples.jsonl"
    samples = _load_jsonl(samples_path)
    valid = [row for row in samples if "monitor_error" not in row]
    if len(valid) != len(samples):
        raise ProtocolError("r036 resource monitor error")
    if (
        int(resources["sample_count"]) != len(samples)
        or int(resources["monitor_error_count"]) != 0
        or int(resources["nvidia_peak_mib"]) != max(int(row["gpu_used_mib"]) for row in valid)
        or int(resources["cgroup_memory_peak_bytes"])
        != max(int(row["cgroup_memory_current_bytes"]) for row in valid)
        or int(resources["nvidia_headroom_mib"])
        != int(resources["nvidia_total_mib"]) - int(resources["nvidia_peak_mib"])
    ):
        raise ProtocolError("r036 resource replay drift")
    if summary.get("resources") != resources or not all(summary["resource_checks"].values()):
        raise ProtocolError("r036 summary resource drift")

    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    inventory = _manifest_inventory(run_dir)
    if manifest.get("status") != "done" or manifest.get("inventory") != inventory:
        raise ProtocolError("r036 manifest inventory drift")
    for key in (
        "quality_read",
        "actor_identity_alignment_read",
        "full_materialization",
        "smaller_batch_retry",
        "identity_training_authorized",
        "f1_execution",
        "f2_execution",
    ):
        if summary.get(key) is not False:
            raise ProtocolError(f"r036 research lock drift: {key}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise ProtocolError("r036 milestone lock drift")

    input_bytes = sum(int(row["bytes"]) for row in staged_inputs)
    return {
        "schema_version": "worldsim_v51_stage_f_f0e_r036_audit_v1",
        "task_id": TASK_ID,
        "status": "pass",
        "audited_run_status": "done",
        "outcome": "mixed",
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
        "inputs": staged_inputs,
        "attempts": replayed_attempts,
        "runtime_probe": summary["runtime_probe"],
        "nvidia_probe": summary["nvidia_probe"],
        "resources": resources,
        "resource_checks": summary["resource_checks"],
        "quality_read": False,
        "actor_identity_alignment_read": False,
        "full_materialization": False,
        "identity_training_authorized": False,
        "f1_execution": False,
        "f2_execution": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_delta": "V51-F62_refined_as_nondeterministic_under_blocking_probe",
        "next_action": "preregister_cuda_runtime_health_and_reproducibility_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_v1.yaml",
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
