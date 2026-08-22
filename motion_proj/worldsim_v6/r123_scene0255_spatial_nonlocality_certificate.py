"""WorldSim V6 R123: certify RGB-difference spatial nonlocality."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R123-SCENE0255-SPATIAL-NONLOCALITY-CERTIFICATE-01"


class R123ExperimentError(RuntimeError):
    """The preregistered R123 contract was violated."""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R123ExperimentError("formal R123 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R123ExperimentError("R123 task_id drift")
    runtime = config["runtime"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    radii = [int(value) for value in runtime["dilation_radii_pixels"]]
    if radii != sorted(set(radii)) or radii[0] != 0:
        raise R123ExperimentError("R123 dilation grid drift")

    source_specs = config["sources"]
    source_runs: dict[str, Path] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    source_gates: dict[str, dict[str, Any]] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    source_transfers: dict[str, dict[str, Any]] = {}
    for source_name in runtime["source_order"]:
        spec = source_specs[source_name]
        source_run = _resolve_runs_uri(spec["run"])
        source_runs[source_name] = source_run
        _verify(source_run / "MANIFEST.json", spec["manifest_sha256"])
        _verify(source_run / spec["gate_name"], spec["gate_sha256"])
        _verify(source_run / "SUMMARY.json", spec["summary_sha256"])
        _verify(source_run / "SELECTOR_TRANSFER.json", spec["selector_transfer_sha256"])
        source_manifests[source_name] = json.loads(
            (source_run / "MANIFEST.json").read_text(encoding="utf-8")
        )
        source_gates[source_name] = json.loads(
            (source_run / spec["gate_name"]).read_text(encoding="utf-8")
        )
        source_summaries[source_name] = json.loads(
            (source_run / "SUMMARY.json").read_text(encoding="utf-8")
        )
        source_transfers[source_name] = json.loads(
            (source_run / "SELECTOR_TRANSFER.json").read_text(encoding="utf-8")
        )

    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R123ExperimentError("R123 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__spatial-nonlocality-certificate-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)

    frame_rows: list[dict[str, Any]] = []
    required_radius_rows: list[dict[str, Any]] = []
    aggregate = {
        radius: {"covered": 0, "labels": 0, "roi_fractions": [], "frame_recalls": []}
        for radius in radii
    }
    epsilon = float(runtime["rgb_change_epsilon"])
    expected_frames = int(runtime["expected_frame_count_per_source"])
    expected_shape: tuple[int, int] | None = None
    loaded_paths: dict[Path, str] = {}

    for source_name in runtime["source_order"]:
        source_run = source_runs[source_name]
        files = source_manifests[source_name]["files"]
        transfer = source_transfers[source_name]
        if int(transfer["frame_count"]) != expected_frames:
            raise R123ExperimentError(f"{source_name} frame denominator drift")
        for frame in range(expected_frames):
            relative_sensor = f"sensor_worker/sensors/frame{frame:03d}.npz"
            relative_logged = f"perception/frame{frame:03d}_logged__repeat0.npy"
            relative_edited = f"perception/frame{frame:03d}_edited__repeat0.npy"
            for relative in (relative_sensor, relative_logged, relative_edited):
                if relative not in files:
                    raise R123ExperimentError(f"{source_name} manifest missing {relative}")
                path = source_run / relative
                _verify(path, files[relative]["sha256"])
                loaded_paths[path] = files[relative]["sha256"]
            with np.load(source_run / relative_sensor, allow_pickle=False) as sensor:
                logged_rgb = sensor["logged_rgb"]
                edited_rgb = sensor["compiled_rgb"]
            logged_labels = np.load(source_run / relative_logged, allow_pickle=False)
            edited_labels = np.load(source_run / relative_edited, allow_pickle=False)
            if logged_rgb.shape != edited_rgb.shape or logged_rgb.ndim != 3:
                raise R123ExperimentError(f"{source_name} RGB shape drift")
            if logged_labels.shape != edited_labels.shape or logged_labels.ndim != 2:
                raise R123ExperimentError(f"{source_name} label shape drift")
            if logged_rgb.shape[:2] != logged_labels.shape:
                raise R123ExperimentError(f"{source_name} RGB/label spatial drift")
            if expected_shape is None:
                expected_shape = tuple(int(value) for value in logged_labels.shape)
            if tuple(logged_labels.shape) != expected_shape:
                raise R123ExperimentError(f"{source_name} cross-frame shape drift")

            rgb_delta = np.max(
                np.abs(edited_rgb.astype(np.float32) - logged_rgb.astype(np.float32)),
                axis=-1,
            )
            rgb_mask = rgb_delta > epsilon
            label_mask = edited_labels != logged_labels
            label_count = int(label_mask.sum())
            expected_label_count = int(
                transfer["changed_label_pixels_by_frame"][str(frame)]
            )
            if label_count != expected_label_count or label_count < 1:
                raise R123ExperimentError(f"{source_name} frame{frame} label authority drift")
            distance = distance_transform_edt(~rgb_mask)
            required_radius = float(distance[label_mask].max())
            required_radius_rows.append(
                {
                    "source": source_name,
                    "frame_index": frame,
                    "required_radius_pixels": required_radius,
                }
            )
            radius_rows = []
            for radius in radii:
                roi = distance <= float(radius)
                covered = int(np.logical_and(label_mask, roi).sum())
                recall = float(covered / label_count)
                roi_fraction = float(roi.mean())
                aggregate[radius]["covered"] += covered
                aggregate[radius]["labels"] += label_count
                aggregate[radius]["roi_fractions"].append(roi_fraction)
                aggregate[radius]["frame_recalls"].append(recall)
                radius_rows.append(
                    {
                        "radius_pixels": radius,
                        "covered_label_pixels": covered,
                        "label_recall": recall,
                        "roi_fraction": roi_fraction,
                    }
                )
            frame_rows.append(
                {
                    "source": source_name,
                    "frame_index": frame,
                    "height": int(label_mask.shape[0]),
                    "width": int(label_mask.shape[1]),
                    "rgb_changed_pixels": int(rgb_mask.sum()),
                    "label_changed_pixels": label_count,
                    "required_radius_pixels": required_radius,
                    "radius_metrics": radius_rows,
                }
            )

    radius_summary = []
    selected_radius: int | None = None
    for radius in radii:
        row = aggregate[radius]
        roi = np.asarray(row["roi_fractions"], dtype=np.float64)
        recalls = np.asarray(row["frame_recalls"], dtype=np.float64)
        summary_row = {
            "radius_pixels": radius,
            "aggregate_label_recall": float(row["covered"] / row["labels"]),
            "worst_frame_label_recall": float(recalls.min()),
            "mean_roi_fraction": float(roi.mean()),
            "p95_roi_fraction": float(np.quantile(roi, 0.95)),
            "maximum_roi_fraction": float(roi.max()),
        }
        radius_summary.append(summary_row)
        if selected_radius is None and row["covered"] == row["labels"] and bool(np.all(recalls == 1.0)):
            selected_radius = radius
    selected_summary = next(
        (row for row in radius_summary if row["radius_pixels"] == selected_radius),
        None,
    )
    required_values = np.asarray(
        [row["required_radius_pixels"] for row in required_radius_rows], dtype=np.float64
    )
    required_radius_summary = {
        "minimum": float(required_values.min()),
        "median": float(np.median(required_values)),
        "p95": float(np.quantile(required_values, 0.95)),
        "maximum": float(required_values.max()),
        "frames_greater_than64": int((required_values > 64.0).sum()),
        "frames_greater_than128": int((required_values > 128.0).sum()),
        "frames_greater_than256": int((required_values > 256.0).sum()),
    }
    certificate = {
        "schema_version": "worldsim_v6.r123_spatial_nonlocality_certificate.v1",
        "source_order": runtime["source_order"],
        "frame_count_per_source": expected_frames,
        "total_frame_count": len(frame_rows),
        "image_shape": list(expected_shape or ()),
        "rgb_change_epsilon": epsilon,
        "radius_metrics": radius_summary,
        "minimum_exact_coverage_radius_pixels": selected_radius,
        "selected_radius_metrics": selected_summary,
        "required_radius_summary": required_radius_summary,
        "crop_inference_equivalence": "ABSTAIN",
        "sparse_execution_speedup": "ABSTAIN",
        "semantic_correctness_local_causality_physics_planning_safety": "ABSTAIN",
    }
    _write_jsonl(run_dir / "FRAME_LOCALITY.jsonl", frame_rows)
    _write_json(run_dir / "SPATIAL_NONLOCALITY_CERTIFICATE.json", certificate)

    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "r118_and_r121_authorities_accepted": all(
            gate["checks"]["passed"] for gate in source_gates.values()
        ),
        "both_sources_are196_of196_positive_zero_skip": all(
            int(summary["positive_target_frame_count"]) == expected_frames
            and int(summary["negative_target_frame_count"]) == 0
            and float(summary["skip_fraction"]) == 0.0
            for summary in source_summaries.values()
        ),
        "source_manifest_and_loaded_files_immutable": all(
            _sha256(path) == expected for path, expected in loaded_paths.items()
        ),
        "frame_and_shape_denominators_exact": len(frame_rows)
        == expected_frames * len(runtime["source_order"])
        and expected_shape == tuple(runtime["expected_image_shape"]),
        "every_frame_requires_more_than64_pixels": required_radius_summary[
            "frames_greater_than64"
        ] == len(frame_rows),
        "every_frame_requires_more_than128_pixels": required_radius_summary[
            "frames_greater_than128"
        ] == len(frame_rows),
        "at_least391_frames_require_more_than256_pixels": required_radius_summary[
            "frames_greater_than256"
        ] >= int(thresholds["minimum_frames_greater_than256"]),
        "required_radius_median_and_maximum_are_global_scale": required_radius_summary[
            "median"
        ] >= float(thresholds["minimum_median_required_radius_pixels"])
        and required_radius_summary["maximum"]
        >= float(thresholds["minimum_maximum_required_radius_pixels"]),
        "radius256_is_large_but_still_inexact": next(
            row for row in radius_summary if row["radius_pixels"] == 256
        )["aggregate_label_recall"] < float(thresholds["maximum_radius256_recall"])
        and next(row for row in radius_summary if row["radius_pixels"] == 256)[
            "mean_roi_fraction"
        ] >= float(thresholds["minimum_radius256_mean_roi_fraction"]),
        "fullframe_scale_grid_radius_is_exact": selected_radius is not None
        and selected_summary is not None
        and selected_summary["aggregate_label_recall"] == 1.0
        and selected_summary["worst_frame_label_recall"] == 1.0
        and selected_summary["mean_roi_fraction"] == 1.0,
        "crop_equivalence_speedup_semantics_and_safety_abstain": True,
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R123_GATE.json",
        {
            "schema_version": "worldsim_v6.r123_gate.v1",
            "checks": checks,
            "decision": "accept_rgb_diff_spatial_nonlocality_reject_sparse_roi"
            if checks["passed"]
            else "reject_spatial_nonlocality_certificate",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r123_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r123_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_rgb_diff_spatial_nonlocality"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "total_frame_count": len(frame_rows),
        "minimum_exact_coverage_radius_pixels": selected_radius,
        "selected_radius_metrics": selected_summary,
        "required_radius_summary": required_radius_summary,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R123_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
        "SPATIAL_NONLOCALITY_CERTIFICATE.json",
        "FRAME_LOCALITY.jsonl",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r123_manifest.v1",
            "files": {
                name: {
                    "bytes": (run_dir / name).stat().st_size,
                    "sha256": _sha256(run_dir / name),
                }
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
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r123_scene0255_spatial_nonlocality_certificate_v1.yaml"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/root/autodl-tmp/runs/worldsim_v6"),
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
