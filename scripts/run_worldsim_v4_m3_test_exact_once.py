#!/usr/bin/env python3
"""Execute or audit the committed 18-scene M3 test plan exactly once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from motion_proj.worldsim_v4.test_freeze import (
    ATTEMPT_SCHEMA,
    TASK_ID,
    TestFreezeError,
    committed_freeze,
    exclusive_json,
    load_mapping,
    sha256_file,
    validate_execution_plan,
)


COMPLETION_SCHEMA = "worldsim_v4_test_completion_v1"


def attempt_payload(
    planned: Mapping[str, Any], provenance: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": TASK_ID,
        "scene": planned["scene"],
        "ordinal": planned["ordinal"],
        "attempt_id": planned["attempt_id"],
        "run_dir": str(Path(planned["run_dir"]).resolve()),
        "freeze_sha256": provenance["freeze_sha256"],
        "freeze_commit": provenance["freeze_commit"],
        "state": "started",
    }


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TestFreezeError(f"YAML root must be mapping: {path}")
    return value


def verify_all_assets(freeze: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    if inventory.get("scene_order") != freeze.get("scene_order"):
        raise TestFreezeError("test asset scene order differs from freeze")
    for scene in freeze["scene_order"]:
        binding = inventory.get("scenes", {}).get(scene)
        if not isinstance(binding, Mapping):
            raise TestFreezeError(f"test asset binding missing: {scene}")
        for label in ("checkpoint", "drivestudio_config", "registry"):
            row = binding.get(label)
            if not isinstance(row, Mapping):
                raise TestFreezeError(f"test asset {label} missing: {scene}")
            path = Path(str(row.get("path", "")))
            if (
                not path.is_file()
                or path.stat().st_size != int(row.get("bytes", -1))
                or sha256_file(path) != row.get("sha256")
            ):
                raise TestFreezeError(f"test asset content drift: {scene}/{label}")


def runtime_preflight(
    freeze: Mapping[str, Any], project_root: Path, ledger: Path
) -> None:
    resources = freeze["resources"]
    runner_python = Path(str(freeze["runner_python"]))
    runner = project_root / "scripts/run_worldsim_v4_m3_scene.py"
    if not runner_python.is_file() or not runner.is_file():
        raise TestFreezeError("test runner Python/script is missing")
    disk_anchor = ledger.parent
    disk_anchor.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(disk_anchor)
    free_gib = stat.f_bavail * stat.f_frsize / 2**30
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise TestFreezeError("test disk-free preflight failed")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{project_root}:/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0"
    )
    import_probe = subprocess.run(
        [str(runner_python), str(runner), "--help"],
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if import_probe.returncode:
        raise TestFreezeError(
            f"test runner import preflight failed: {import_probe.stderr.strip()}"
        )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu.returncode or len(gpu.stdout.strip().splitlines()) != 1:
        raise TestFreezeError("test GPU preflight failed")
    name, separator, used = gpu.stdout.strip().rpartition(",")
    if (
        not separator
        or name.strip() != resources["required_gpu"]
        or int(used.strip())
        > int(resources["maximum_gpu_used_at_attempt_start_mib"])
    ):
        raise TestFreezeError("test GPU identity/availability preflight failed")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if processes.returncode or processes.stdout.strip():
        raise TestFreezeError("test GPU has a competing compute process")


def completion_payload(
    *,
    freeze: Mapping[str, Any],
    provenance: Mapping[str, str],
    planned: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    run = Path(str(planned["run_dir"]))
    status_path = run / "status.json"
    summary_path = run / "summary.json"
    manifest_path = run / "manifest.json"
    fingerprint_path = run / "fingerprint.json"
    for path in (status_path, summary_path, manifest_path, fingerprint_path):
        if not path.is_file():
            raise TestFreezeError(f"incomplete consumed test attempt: {path}")
    status = load_mapping(status_path)
    summary = load_mapping(summary_path)
    scene = str(planned["scene"])
    binding = inventory["scenes"][scene]
    expected_quality_read = binding["status"] == "ready"
    if (
        status.get("status") != "done"
        or status.get("scene") != scene
        or status.get("summary_sha256") != sha256_file(summary_path)
        or status.get("manifest_sha256") != sha256_file(manifest_path)
        or status.get("fingerprint_sha256") != sha256_file(fingerprint_path)
        or summary.get("scene") != scene
        or summary.get("partition") != "test"
        or summary.get("status") not in {"done", "abstain"}
        or summary.get("test_scene_attempted") is not True
        or summary.get("test_quality_read") is not expected_quality_read
        or status.get("test_quality_read") is not expected_quality_read
        or summary.get("test_attempt", {}).get("attempt_id")
        != planned["attempt_id"]
        or summary.get("test_attempt", {}).get("freeze_commit")
        != provenance["freeze_commit"]
        or summary.get("project_git_head") != provenance["freeze_commit"]
        or summary.get("project_git_dirty") is not False
    ):
        raise TestFreezeError(f"completed test run contract drift: {scene}")
    marker = run / "test_read_started.json"
    if marker.is_file() is not expected_quality_read:
        raise TestFreezeError(f"test read marker drift: {scene}")
    return {
        "schema_version": COMPLETION_SCHEMA,
        "task_id": TASK_ID,
        "scene": scene,
        "ordinal": planned["ordinal"],
        "attempt_id": planned["attempt_id"],
        "run_dir": str(run.resolve()),
        "freeze_sha256": provenance["freeze_sha256"],
        "freeze_commit": provenance["freeze_commit"],
        "state": "completed",
        "scene_status": summary["status"],
        "test_quality_read": expected_quality_read,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "status_sha256": sha256_file(status_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
    }


def run_exact_once(freeze_path: Path, project_root: Path) -> dict[str, Any]:
    freeze, provenance = committed_freeze(freeze_path, project_root)
    plan = validate_execution_plan(freeze)
    inventory_path = Path(freeze["test_asset_inventory"]["path"])
    config_path = Path(freeze["config"]["path"])
    inventory = load_yaml(inventory_path)
    ledger = Path(freeze["ledger_dir"])
    verify_all_assets(freeze, inventory)
    runtime_preflight(freeze, project_root, ledger)
    ledger_header = {
        "schema_version": "worldsim_v4_test_ledger_v1",
        "task_id": TASK_ID,
        "freeze_sha256": provenance["freeze_sha256"],
        "freeze_commit": provenance["freeze_commit"],
        "scene_order": freeze["scene_order"],
        "state": "opened",
    }
    header_path = ledger / "ledger.json"
    if ledger.exists():
        if not header_path.is_file():
            raise TestFreezeError("pre-existing test ledger lacks frozen header")
        if load_mapping(header_path) != ledger_header:
            raise TestFreezeError("test ledger header drift")
    else:
        ledger.mkdir(parents=True, exist_ok=False)
        exclusive_json(header_path, ledger_header)
    logs = ledger / "logs"
    logs.mkdir(exist_ok=True)
    for planned in plan:
        attempt_path = ledger / "attempts" / f"{planned['attempt_id']}.json"
        completion_path = ledger / "completions" / f"{planned['attempt_id']}.json"
        expected_attempt = attempt_payload(planned, provenance)
        if completion_path.exists():
            if not attempt_path.is_file() or load_mapping(attempt_path) != expected_attempt:
                raise TestFreezeError("test completion lacks its exact attempt marker")
            expected = completion_payload(
                freeze=freeze,
                provenance=provenance,
                planned=planned,
                inventory=inventory,
            )
            if load_mapping(completion_path) != expected:
                raise TestFreezeError("test completion marker drift")
            continue
        if attempt_path.exists():
            if load_mapping(attempt_path) != expected_attempt:
                raise TestFreezeError("test attempt marker drift")
            # A consumed attempt is never re-executed. Only a fully written run
            # can be recovered by completing its ledger marker.
            completion = completion_payload(
                freeze=freeze,
                provenance=provenance,
                planned=planned,
                inventory=inventory,
            )
            exclusive_json(completion_path, completion)
            continue
        run_dir = Path(planned["run_dir"])
        if run_dir.exists():
            raise TestFreezeError("planned run directory exists before attempt")
        runtime_preflight(freeze, project_root, ledger)
        exclusive_json(attempt_path, expected_attempt)
        command = [
            str(freeze["runner_python"]),
            str(project_root / "scripts/run_worldsim_v4_m3_scene.py"),
            "--config", str(config_path),
            "--inventory", str(inventory_path),
            "--scene", str(planned["scene"]),
            "--run-dir", str(run_dir),
            "--control-points", str(freeze["method_selection"]["m3_parameters"]["control_point_count"]),
            "--acceleration-regularization", str(freeze["method_selection"]["m3_parameters"]["acceleration_regularization"]),
            "--evidence-retention", str(freeze["method_selection"]["m3_parameters"]["evidence_retention"]),
            "--warp-alpha", str(freeze["method_selection"]["m3_parameters"]["warp_blend_alpha"]),
            "--test-freeze", str(freeze_path),
            "--test-attempt", str(attempt_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = f"{project_root}:/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0"
        log_path = logs / f"{planned['attempt_id']}.log"
        with log_path.open("xb") as log:
            process = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            raise TestFreezeError(
                f"consumed test attempt failed and cannot be rerun: {planned['scene']} rc={process.returncode}"
            )
        completion = completion_payload(
            freeze=freeze,
            provenance=provenance,
            planned=planned,
            inventory=inventory,
        )
        exclusive_json(completion_path, completion)
    completions = [
        load_mapping(ledger / "completions" / f"{row['attempt_id']}.json")
        for row in plan
    ]
    terminal = {
        "schema_version": "worldsim_v4_test_ledger_terminal_v1",
        "task_id": TASK_ID,
        "state": "done",
        "freeze_sha256": provenance["freeze_sha256"],
        "freeze_commit": provenance["freeze_commit"],
        "attempt_count": 18,
        "completion_count": 18,
        "quality_read_scene_count": sum(row["test_quality_read"] for row in completions),
        "scene_order": freeze["scene_order"],
        "completion_sha256": {
            row["scene"]: sha256_file(
                ledger / "completions" / f"{row['attempt_id']}.json"
            )
            for row in plan
        },
        "log_sha256": {
            row["scene"]: sha256_file(logs / f"{row['attempt_id']}.log")
            for row in plan
        },
    }
    terminal_path = ledger / "terminal.json"
    if terminal_path.exists():
        if load_mapping(terminal_path) != terminal:
            raise TestFreezeError("test ledger terminal drift")
    else:
        exclusive_json(terminal_path, terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_exact_once(args.freeze.resolve(), args.project_root.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
