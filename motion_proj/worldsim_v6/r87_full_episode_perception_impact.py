"""WorldSim V6 R87: quantify frozen perception impact over a full edited episode."""

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


TASK_ID = "WS-V6-R87-FULL-EPISODE-PERCEPTION-IMPACT-01"


class R87ExperimentError(RuntimeError):
    """The preregistered R87 experiment contract was violated."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _longest_positive_run(values: list[int]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value > 0 else 0
        longest = max(longest, current)
    return longest


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R87ExperimentError("formal R87 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R87ExperimentError("R87 task_id drift")
    sources = config["sources"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    r86_run = _resolve_runs_uri(sources["r86_run"])
    r53_run = _resolve_runs_uri(sources["r53_run"])
    model_root = Path(sources["semantic_model_root"])
    root_files = {
        r86_run / "MANIFEST.json": sources["r86_manifest_sha256"],
        r86_run / "R86_GATE.json": sources["r86_gate_sha256"],
        r86_run / "SUMMARY.json": sources["r86_summary_sha256"],
        r86_run / "FULL_EPISODE_SENSOR_EFFECT.json": sources[
            "r86_full_episode_sensor_effect_sha256"
        ],
        r86_run / "RESOURCE_AUDIT.json": sources["r86_resource_audit_sha256"],
        r53_run / "MANIFEST.json": sources["r53_manifest_sha256"],
        r53_run / "R53_GATE.json": sources["r53_gate_sha256"],
        r53_run / "SUMMARY.json": sources["r53_summary_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
    }
    for path, expected_sha in root_files.items():
        _verify(path, expected_sha)
    r86_manifest = json.loads((r86_run / "MANIFEST.json").read_text(encoding="utf-8"))
    frame_indices = list(
        range(
            int(evaluation["frame_start"]),
            int(evaluation["frame_stop_exclusive"]),
            int(evaluation["frame_stride"]),
        )
    )
    sensor_names = [f"worker/sensors/frame{frame:03d}.npz" for frame in frame_indices]
    if len(frame_indices) != int(evaluation["expected_frame_count"]) or any(
        name not in r86_manifest["files"] for name in sensor_names
    ):
        raise R87ExperimentError("R87 sensor frame denominator drift")
    sensors = {frame: r86_run / f"worker/sensors/frame{frame:03d}.npz" for frame in frame_indices}
    for frame, sensor in sensors.items():
        _verify(sensor, r86_manifest["files"][f"worker/sensors/frame{frame:03d}.npz"]["sha256"])
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R87ExperimentError("R87 disk resource insufficient")
    r86_gate = json.loads((r86_run / "R86_GATE.json").read_text(encoding="utf-8"))
    r53_gate = json.loads((r53_run / "R53_GATE.json").read_text(encoding="utf-8"))
    effect = json.loads(
        (r86_run / "FULL_EPISODE_SENSOR_EFFECT.json").read_text(encoding="utf-8")
    )
    variants = list(evaluation["variants"])
    if variants != ["logged", "edited"]:
        raise R87ExperimentError("R87 variant order drift")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__full-episode-perception-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    perception_dir = run_dir / "perception"
    index_rows = [
        {
            "case_id": f"frame{frame:03d}_{variant}",
            "frame_index": frame,
            "variant": variant,
            "rgb_key": evaluation["rgb_keys"][variant],
            "repeat_index": repeat,
            "render_path": str(sensors[frame]),
        }
        for frame in frame_indices
        for variant in variants
        for repeat in range(int(evaluation["repeat_count"]))
    ]
    _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", index_rows)
    command = [
        sys.executable,
        str(repo_root / "scripts/worldsim_v6/r87_full_episode_perception_worker.py"),
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
            timeout=float(resources["maximum_worker_seconds"]),
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
    changed_label_pixels = []
    label_hashes = {}
    for frame in frame_indices:
        arrays = {}
        label_hashes[str(frame)] = {}
        for variant in variants:
            case_id = f"frame{frame:03d}_{variant}"
            first = sorted(grouped[case_id], key=lambda item: item["repeat_index"])[0]
            arrays[variant] = np.load(perception_dir / first["label_path"], allow_pickle=False)
            label_hashes[str(frame)][variant] = first["label_array_sha256"]
        changed_label_pixels.append(int((arrays["edited"] != arrays["logged"]).sum()))
    sensor_changed_pixels = [
        int(effect["edited_vs_logged_changed_rgb_pixels_by_frame"][str(frame)])
        for frame in frame_indices
    ]
    visible_mask = np.asarray(sensor_changed_pixels) >= int(
        evaluation["minimum_sensor_changed_pixels_for_visible_frame"]
    )
    perception_mask = np.asarray(changed_label_pixels) >= int(
        evaluation["minimum_changed_label_pixels_for_affected_frame"]
    )
    visible_count = int(visible_mask.sum())
    affected_count = int(perception_mask.sum())
    visible_affected_count = int((visible_mask & perception_mask).sum())
    visible_detection_rate = float(visible_affected_count / visible_count) if visible_count else 0.0
    log_sensor = np.log1p(np.asarray(sensor_changed_pixels, dtype=np.float64))
    log_labels = np.log1p(np.asarray(changed_label_pixels, dtype=np.float64))
    impact_correlation = (
        float(np.corrcoef(log_sensor, log_labels)[0, 1])
        if float(log_sensor.std()) > 0 and float(log_labels.std()) > 0
        else 0.0
    )
    phase_ranges = evaluation["lifecycle_phases"]
    phase_metrics = {
        name: {
            "frame_count": int(stop - start),
            "sensor_visible_frame_count": int(visible_mask[start:stop].sum()),
            "perception_affected_frame_count": int(perception_mask[start:stop].sum()),
            "changed_label_pixels": int(sum(changed_label_pixels[start:stop])),
        }
        for name, (start, stop) in phase_ranges.items()
    }
    comparison = {
        "schema_version": "worldsim_v6.r87_full_episode_perception_impact.v1",
        "frame_count": len(frame_indices),
        "repeat_count": int(evaluation["repeat_count"]),
        "sensor_visible_frame_count": visible_count,
        "perception_affected_frame_count": affected_count,
        "visible_and_perception_affected_frame_count": visible_affected_count,
        "visible_frame_perception_detection_rate": visible_detection_rate,
        "total_changed_label_pixels": int(sum(changed_label_pixels)),
        "maximum_changed_label_pixels_in_frame": int(max(changed_label_pixels)),
        "longest_consecutive_perception_affected_run": _longest_positive_run(
            changed_label_pixels
        ),
        "log1p_sensor_label_impact_pearson": impact_correlation,
        "sensor_changed_pixels_by_frame": dict(zip(map(str, frame_indices), sensor_changed_pixels)),
        "changed_label_pixels_by_frame": dict(zip(map(str, frame_indices), changed_label_pixels)),
        "lifecycle_phase_metrics": phase_metrics,
        "label_array_sha256_by_frame": label_hashes,
        "semantic_correctness": "ABSTAIN",
        "local_causality": "ABSTAIN",
    }
    _write_json(run_dir / "FULL_EPISODE_PERCEPTION_IMPACT.json", comparison)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in perception_dir.rglob("*") if path.is_file())
    checks = {
        "r86_and_r53_authorities_accepted": bool(
            r86_gate["checks"]["passed"] and r53_gate["checks"]["passed"]
        ),
        "full_196_frame_two_variant_two_repeat_input_denominator_exact": len(index_rows)
        == int(evaluation["expected_frame_count"]) * 2 * int(evaluation["repeat_count"]),
        "full_worker_output_denominator_exact": len(rows) == len(index_rows),
        "perception_repeat_exact_every_frame_and_variant": repeat_exact,
        "r86_visible_frame_denominator_preserved": visible_count
        == int(evaluation["expected_sensor_visible_frame_count"]),
        "perception_changes_temporally_distributed": affected_count
        >= int(evaluation["minimum_perception_affected_frame_count"]),
        "visible_frame_perception_detection_rate_nontrivial": visible_detection_rate
        >= float(evaluation["minimum_visible_frame_perception_detection_rate"]),
        "total_perception_change_nontrivial": int(sum(changed_label_pixels))
        >= int(evaluation["minimum_total_changed_label_pixels"]),
        "semantic_correctness_local_causality_physics_planning_safety_abstain": True,
        "frozen_root_sources_immutable": all(
            _sha256(path) == expected_sha for path, expected_sha in root_files.items()
        ),
        "all_196_sensor_sources_immutable": all(
            _sha256(sensor)
            == r86_manifest["files"][f"worker/sensors/frame{frame:03d}.npz"]["sha256"]
            for frame, sensor in sensors.items()
        ),
        "gpu_within_budget": int(worker["peak_gpu_memory_mib"])
        <= int(resources["maximum_peak_gpu_memory_mib"]),
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "output_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R87_GATE.json",
        {
            "schema_version": "worldsim_v6.r87_gate.v1",
            "checks": checks,
            "decision": "accept_full_episode_perception_impact"
            if checks["passed"]
            else "reject_or_pivot_full_episode_perception_impact",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r87_resource_audit.v1",
            "gpu_used": True,
            "peak_gpu_memory_mib": int(worker["peak_gpu_memory_mib"]),
            "perception_elapsed_seconds": float(worker["elapsed_seconds"]),
            "wall_seconds": wall_seconds,
            "worker_output_bytes": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r87_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_full_episode_perception_impact"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "frame_count": len(frame_indices),
        "sensor_visible_frame_count": visible_count,
        "perception_affected_frame_count": affected_count,
        "visible_frame_perception_detection_rate": visible_detection_rate,
        "total_changed_label_pixels": int(sum(changed_label_pixels)),
        "maximum_changed_label_pixels_in_frame": int(max(changed_label_pixels)),
        "longest_consecutive_perception_affected_run": _longest_positive_run(
            changed_label_pixels
        ),
        "log1p_sensor_label_impact_pearson": impact_correlation,
        "lifecycle_phase_metrics": phase_metrics,
        "semantic_correctness": "ABSTAIN",
        "local_causality": "ABSTAIN",
        "physical_planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R87_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
        "PERCEPTION_INPUT_INDEX.jsonl",
        "FULL_EPISODE_PERCEPTION_IMPACT.json",
        "perception/PERCEPTION_OUTPUTS.jsonl",
        "perception/WORKER_RESULT.json",
        "perception.log",
    ]
    tracked.extend(str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy")))
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r87_manifest.v1",
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
        default=Path("configs/worldsim_v6/r87_full_episode_perception_impact_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
