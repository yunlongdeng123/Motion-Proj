"""WorldSim V6 R72: measure frozen perception changes from incremental actor edits."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R72-TWO-ACTOR-INCREMENTAL-PERCEPTION-01"


class R72ExperimentError(RuntimeError):
    """The preregistered R72 experiment contract was violated."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _compiled_rgb(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        image = archive["compiled_rgb"].astype(np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R72ExperimentError("formal R72 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R72ExperimentError("R72 task_id drift")
    sources = config["sources"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r71_run = _resolve_runs_uri(sources["r71_run"])
    r51_run = _resolve_runs_uri(sources["r51_run"])
    r36_run = _resolve_runs_uri(sources["r36_run"])
    r53_run = _resolve_runs_uri(sources["r53_run"])
    model_root = Path(sources["semantic_model_root"])
    sensors = {
        "logged": r36_run / sources["r36_sensor"],
        "actor0_only": r51_run / sources["r51_sensor"],
        "two_actor": r71_run / sources["r71_sensor"],
    }
    frozen_files = {
        r71_run / "MANIFEST.json": sources["r71_manifest_sha256"],
        r71_run / "R71_GATE.json": sources["r71_gate_sha256"],
        r71_run / "SUMMARY.json": sources["r71_summary_sha256"],
        r71_run / "JOINT_SENSOR_EFFECT.json": sources["r71_joint_sensor_effect_sha256"],
        sensors["two_actor"]: sources["r71_sensor_sha256"],
        r51_run / "MANIFEST.json": sources["r51_manifest_sha256"],
        r51_run / "R51_GATE.json": sources["r51_gate_sha256"],
        r51_run / "SUMMARY.json": sources["r51_summary_sha256"],
        sensors["actor0_only"]: sources["r51_sensor_sha256"],
        r36_run / "MANIFEST.json": sources["r36_manifest_sha256"],
        r36_run / "R36_GATE.json": sources["r36_gate_sha256"],
        r36_run / "SUMMARY.json": sources["r36_summary_sha256"],
        sensors["logged"]: sources["r36_sensor_sha256"],
        r53_run / "MANIFEST.json": sources["r53_manifest_sha256"],
        r53_run / "R53_GATE.json": sources["r53_gate_sha256"],
        r53_run / "SUMMARY.json": sources["r53_summary_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R72ExperimentError("R72 disk resource insufficient")
    gates = [
        json.loads((r71_run / "R71_GATE.json").read_text(encoding="utf-8")),
        json.loads((r51_run / "R51_GATE.json").read_text(encoding="utf-8")),
        json.loads((r36_run / "R36_GATE.json").read_text(encoding="utf-8")),
        json.loads((r53_run / "R53_GATE.json").read_text(encoding="utf-8")),
    ]
    if list(evaluation["variants"]) != ["logged", "actor0_only", "two_actor"]:
        raise R72ExperimentError("R72 variant order drift")

    rgbs = {name: _compiled_rgb(path) for name, path in sensors.items()}
    epsilon = float(evaluation["rgb_change_epsilon"])
    sensor_comparisons = {
        "actor0_increment": int(
            np.count_nonzero(
                np.mean(np.abs(rgbs["actor0_only"] - rgbs["logged"]), axis=-1) > epsilon
            )
        ),
        "actor2_increment": int(
            np.count_nonzero(
                np.mean(np.abs(rgbs["two_actor"] - rgbs["actor0_only"]), axis=-1) > epsilon
            )
        ),
        "joint_vs_logged": int(
            np.count_nonzero(
                np.mean(np.abs(rgbs["two_actor"] - rgbs["logged"]), axis=-1) > epsilon
            )
        ),
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__two-actor-perception-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    perception_dir = run_dir / "perception"
    index_rows = [
        {
            "case_id": variant,
            "repeat_index": repeat,
            "render_path": str(sensors[variant]),
        }
        for variant in evaluation["variants"]
        for repeat in range(int(evaluation["repeat_count"]))
    ]
    _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", index_rows)
    command = [
        sys.executable,
        str(repo_root / "scripts/worldsim_v6/r53_perception_worker.py"),
        "--index",
        str(run_dir / "PERCEPTION_INPUT_INDEX.jsonl"),
        "--model-root",
        str(model_root),
        "--output-dir",
        str(perception_dir),
    ]
    worker_env = os.environ.copy()
    worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with (run_dir / "perception.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            cwd=repo_root,
            env=worker_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    worker = json.loads((perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
    rows = _load_rows(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    repeat_exact = all(
        len(items) == int(evaluation["repeat_count"])
        and len({item["label_array_sha256"] for item in items}) == 1
        for items in grouped.values()
    )
    labels = {}
    label_hashes = {}
    for variant in evaluation["variants"]:
        first = sorted(grouped[variant], key=lambda item: item["repeat_index"])[0]
        labels[variant] = np.load(perception_dir / first["label_path"], allow_pickle=False)
        label_hashes[variant] = first["label_array_sha256"]
    actor0_change = labels["actor0_only"] != labels["logged"]
    actor2_increment = labels["two_actor"] != labels["actor0_only"]
    joint_change = labels["two_actor"] != labels["logged"]
    perception_comparison = {
        "schema_version": "worldsim_v6.r72_perception_comparison.v1",
        "frame_index": int(evaluation["frame_index"]),
        "sensor_changed_pixels": sensor_comparisons,
        "changed_label_pixels": {
            "actor0_increment": int(actor0_change.sum()),
            "actor2_increment": int(actor2_increment.sum()),
            "joint_vs_logged": int(joint_change.sum()),
        },
        "changed_label_fractions": {
            "actor0_increment": float(actor0_change.mean()),
            "actor2_increment": float(actor2_increment.mean()),
            "joint_vs_logged": float(joint_change.mean()),
        },
        "actor_increment_change_overlap_pixels": int((actor0_change & actor2_increment).sum()),
        "label_array_sha256": label_hashes,
        "semantic_correctness": "ABSTAIN",
    }
    _write_json(run_dir / "PERCEPTION_COMPARISON.json", perception_comparison)
    minimum_sensor = int(evaluation["minimum_sensor_changed_pixels_per_increment"])
    minimum_labels = int(evaluation["minimum_changed_label_pixels_per_increment"])
    wall_seconds = time.monotonic() - started
    checks = {
        "r71_r51_r36_and_r53_authorities_accepted": all(
            bool(gate["checks"]["passed"]) for gate in gates
        ),
        "three_case_two_repeat_denominator_exact": len(rows)
        == 3 * int(evaluation["repeat_count"]),
        "perception_repeat_exact": repeat_exact,
        "actor0_sensor_increment_nontrivial": sensor_comparisons["actor0_increment"]
        >= minimum_sensor,
        "actor2_sensor_increment_nontrivial": sensor_comparisons["actor2_increment"]
        >= minimum_sensor,
        "joint_sensor_change_nontrivial": sensor_comparisons["joint_vs_logged"]
        >= minimum_sensor,
        "actor0_perception_increment_detected": int(actor0_change.sum()) >= minimum_labels,
        "actor2_perception_increment_detected": int(actor2_increment.sum()) >= minimum_labels,
        "joint_perception_change_detected": int(joint_change.sum()) >= minimum_labels,
        "semantic_correctness_and_local_causality_abstain": True,
        "physics_planning_safety_abstain": True,
        "frozen_sources_immutable": all(
            _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
        ),
        "gpu_within_budget": int(worker["peak_gpu_memory_mib"])
        <= int(resources["maximum_peak_gpu_memory_mib"]),
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R72_GATE.json",
        {
            "schema_version": "worldsim_v6.r72_gate.v1",
            "checks": checks,
            "decision": "accept_two_actor_incremental_perception_state"
            if checks["passed"]
            else "reject_or_pivot_two_actor_incremental_perception_state",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r72_resource_audit.v1",
            "gpu_used": True,
            "peak_gpu_memory_mib": int(worker["peak_gpu_memory_mib"]),
            "perception_elapsed_seconds": float(worker["elapsed_seconds"]),
            "wall_seconds": wall_seconds,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r72_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_two_actor_incremental_perception_state"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "frame_index": int(evaluation["frame_index"]),
        "sensor_changed_pixels": sensor_comparisons,
        "changed_label_pixels": perception_comparison["changed_label_pixels"],
        "actor_increment_change_overlap_pixels": perception_comparison[
            "actor_increment_change_overlap_pixels"
        ],
        "semantic_correctness": "ABSTAIN",
        "local_causality": "ABSTAIN",
        "physical_planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R72_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
        "PERCEPTION_INPUT_INDEX.jsonl",
        "PERCEPTION_COMPARISON.json",
        "perception/PERCEPTION_OUTPUTS.jsonl",
        "perception/WORKER_RESULT.json",
        "perception.log",
    ]
    tracked.extend(
        str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy"))
    )
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r72_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        },
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {
            "schema_version": "worldsim_v6.terminal.v1",
            "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        },
    )
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r72_two_actor_incremental_perception_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
