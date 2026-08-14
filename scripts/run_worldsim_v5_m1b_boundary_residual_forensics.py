#!/usr/bin/env python3
"""Audit whether frozen M1 residual errors are primarily boundary-localized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import yaml

from worldsim_v5_forensics_common import (
    ForensicAuditError,
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    load_json_mapping,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    verify_named_bindings,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M1B-D0-BOUNDARY-RESIDUAL-FORENSICS-01"
EXPECTED_SCENES = ("scene-0471", "scene-1087", "scene-0379")
UNARIES = ("B1", "B3")
GRAPHS = ("G0", "G3")
SUM_KEYS = (
    "pixel_count",
    "boundary_pixel_count",
    "classification_error_count",
    "boundary_classification_error_count",
    "semantic_error_mass",
    "boundary_semantic_error_mass",
    "false_positive_mass",
    "boundary_false_positive_mass",
    "false_negative_mass",
    "boundary_false_negative_mass",
    "entropy_sum",
    "boundary_entropy_sum",
)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ForensicAuditError("boundary forensic config 根节点不是 mapping")
    if payload.get("schema_version") != (
        "worldsim_v5_m1b_boundary_residual_forensics_v1"
    ):
        raise ForensicAuditError("boundary forensic config schema 漂移")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "running":
        raise ForensicAuditError("boundary forensic task/status 漂移")
    if tuple(row["scene"] for row in payload["inputs"].values()) != EXPECTED_SCENES:
        raise ForensicAuditError("boundary forensic scene 顺序或集合漂移")
    analysis = payload["analysis"]
    if tuple(analysis["unary_inputs"]) != UNARIES:
        raise ForensicAuditError("boundary forensic unary 集合漂移")
    if tuple(analysis["graph_arms"]) != GRAPHS:
        raise ForensicAuditError("boundary forensic graph 集合漂移")
    gate = payload["gate"]
    if gate.get("automatic_semantic_split_unlock") is not False:
        raise ForensicAuditError("boundary forensic 禁止自动解锁 semantic split")
    restrictions = payload["restrictions"]
    for key in (
        "development_source_image_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_quality_read",
        "parameter_search_performed",
        "training_performed",
        "graph_parameter_change",
        "semantic_split_started",
    ):
        if restrictions.get(key) is not False:
            raise ForensicAuditError(f"boundary forensic restriction violated: {key}")
    if restrictions.get("development_evaluation_artifact_read") is not True:
        raise ForensicAuditError("boundary forensic 必须显式登记 development artifact read")
    return payload


def analyze_view(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    boundary_iterations: int,
) -> dict[str, float | int]:
    prediction = np.asarray(probability, dtype=np.float64)
    label = np.asarray(target, dtype=bool)
    if prediction.shape != label.shape or prediction.ndim != 2 or prediction.size == 0:
        raise ForensicAuditError("boundary forensic probability/target shape 漂移")
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ForensicAuditError("boundary forensic probability 非有限或越界")
    if boundary_iterations <= 0:
        raise ForensicAuditError("boundary forensic band iterations 必须为正")
    target_boundary = label & ~binary_erosion(label)
    if not target_boundary.any():
        raise ForensicAuditError("accepted evaluation artifact 缺少 target boundary")
    boundary_band = binary_dilation(target_boundary, iterations=boundary_iterations)
    classified = prediction >= threshold
    classification_error = classified != label
    semantic_error = np.where(label, 1.0 - prediction, prediction)
    false_positive = np.where(~label, prediction, 0.0)
    false_negative = np.where(label, 1.0 - prediction, 0.0)
    clipped = np.clip(prediction, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return {
        "pixel_count": int(prediction.size),
        "boundary_pixel_count": int(boundary_band.sum()),
        "classification_error_count": int(classification_error.sum()),
        "boundary_classification_error_count": int(
            (classification_error & boundary_band).sum()
        ),
        "semantic_error_mass": float(semantic_error.sum()),
        "boundary_semantic_error_mass": float(semantic_error[boundary_band].sum()),
        "false_positive_mass": float(false_positive.sum()),
        "boundary_false_positive_mass": float(false_positive[boundary_band].sum()),
        "false_negative_mass": float(false_negative.sum()),
        "boundary_false_negative_mass": float(false_negative[boundary_band].sum()),
        "entropy_sum": float(entropy.sum()),
        "boundary_entropy_sum": float(entropy[boundary_band].sum()),
    }


def aggregate_view_rows(rows: list[Mapping[str, float | int]]) -> dict[str, Any]:
    if not rows:
        raise ForensicAuditError("boundary forensic 没有 evaluation rows")
    totals = {key: float(sum(float(row[key]) for row in rows)) for key in SUM_KEYS}
    pixel_count = totals["pixel_count"]
    boundary_pixels = totals["boundary_pixel_count"]
    error_count = totals["classification_error_count"]
    boundary_errors = totals["boundary_classification_error_count"]
    semantic_mass = totals["semantic_error_mass"]
    boundary_semantic_mass = totals["boundary_semantic_error_mass"]
    band_fraction = boundary_pixels / pixel_count
    error_share = boundary_errors / error_count if error_count else 0.0
    semantic_share = (
        boundary_semantic_mass / semantic_mass if semantic_mass > 0.0 else 0.0
    )
    totals.update(
        view_count=len(rows),
        boundary_pixel_fraction=band_fraction,
        boundary_classification_error_share=error_share,
        boundary_semantic_error_mass_share=semantic_share,
        boundary_error_enrichment=(error_share / band_fraction if band_fraction else 0.0),
        boundary_false_positive_mass_share=(
            totals["boundary_false_positive_mass"] / totals["false_positive_mass"]
            if totals["false_positive_mass"] > 0.0
            else 0.0
        ),
        boundary_false_negative_mass_share=(
            totals["boundary_false_negative_mass"] / totals["false_negative_mass"]
            if totals["false_negative_mass"] > 0.0
            else 0.0
        ),
        mean_entropy=totals["entropy_sum"] / pixel_count,
        boundary_mean_entropy=totals["boundary_entropy_sum"] / boundary_pixels,
        far_mean_entropy=(
            (totals["entropy_sum"] - totals["boundary_entropy_sum"])
            / (pixel_count - boundary_pixels)
            if pixel_count > boundary_pixels
            else 0.0
        ),
    )
    for key in ("pixel_count", "boundary_pixel_count", "classification_error_count", "boundary_classification_error_count"):
        totals[key] = int(totals[key])
    return totals


def _verify_run(
    binding: Mapping[str, Any], analysis: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    names = ("summary", "status", "fingerprint", "manifest", "diagnostics")
    verified = verify_named_bindings(
        binding["run"], {name: binding[name] for name in names}
    )
    summary = load_json_mapping(Path(verified["summary"]["path"]))
    status = load_json_mapping(Path(verified["status"]["path"]))
    fingerprint = load_json_mapping(Path(verified["fingerprint"]["path"]))
    manifest = load_json_mapping(Path(verified["manifest"]["path"]))
    diagnostics = load_json_mapping(Path(verified["diagnostics"]["path"]))
    replay = summary.get(
        "g0_replay_unary_float16_exact",
        summary.get("g0_replay_r037_float16_exact"),
    )
    if (
        summary.get("status") != "done"
        or summary.get("scene") != binding["scene"]
        or summary.get("parameter_search_performed") is not False
        or summary.get("validation_quality_read") is not False
        or summary.get("heldout_quality_read") is not False
        or summary.get("semantic_split_started") is not False
        or replay != {"B1": True, "B3": True}
        or status.get("status") != "done"
        or status.get("summary_sha256") != binding["summary"]["sha256"]
        or status.get("manifest_sha256") != binding["manifest"]["sha256"]
        or fingerprint.get("source_clean") is not True
        or manifest.get("status") != "done"
    ):
        raise ForensicAuditError(f"frozen graph terminal contract 漂移: {binding['scene']}")
    inventory = {row["path"]: row for row in manifest["inventory"]}
    run_root = Path(binding["run"])
    view_count = int(summary["evaluation_view_count"])
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for unary in analysis["unary_inputs"]:
        for graph in analysis["graph_arms"]:
            arm = f"{unary}_{graph}"
            rows = diagnostics.get("evaluation_rows", {}).get(arm, [])
            if len(rows) != view_count:
                raise ForensicAuditError(
                    f"{binding['scene']} {arm} evaluation denominator 漂移"
                )
            for row in rows:
                relative = str(row["path"])
                record = inventory.get(relative)
                if record is None or record.get("sha256") != row.get("sha256"):
                    raise ForensicAuditError(
                        f"{binding['scene']} {arm} inventory 漂移: {relative}"
                    )
                verify_file(run_root / relative, str(row["sha256"]))
            rows_by_arm[arm] = rows
    reference_keys = [
        (int(row["frame"]), int(row["camera_id"])) for row in rows_by_arm["B1_G0"]
    ]
    for arm, rows in rows_by_arm.items():
        keys = [(int(row["frame"]), int(row["camera_id"])) for row in rows]
        if keys != reference_keys:
            raise ForensicAuditError(
                f"{binding['scene']} evaluation view keys 漂移: {arm}"
            )
    return summary, diagnostics, rows_by_arm, verified


def audit(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    analysis = config["analysis"]
    threshold = float(analysis["probability_threshold"])
    boundary_iterations = int(analysis["boundary_band_iterations"])
    primary = analysis["boundary_primary_cell"]
    scene_results: list[dict[str, Any]] = []
    input_bindings: dict[str, Any] = {}
    for binding in config["inputs"].values():
        summary, _, rows_by_arm, verified = _verify_run(binding, analysis)
        input_bindings[binding["scene"]] = verified
        run_root = Path(binding["run"])
        arm_results: dict[str, Any] = {}
        reference_targets: dict[tuple[int, int], np.ndarray] = {}
        for arm, rows in rows_by_arm.items():
            view_rows: list[dict[str, Any]] = []
            for row in rows:
                key = (int(row["frame"]), int(row["camera_id"]))
                path = run_root / str(row["path"])
                with np.load(path, allow_pickle=False) as payload:
                    probability = np.asarray(payload["probability"])
                    target = np.asarray(payload["target"], dtype=bool)
                if key in reference_targets and not np.array_equal(
                    reference_targets[key], target
                ):
                    raise ForensicAuditError(
                        f"{binding['scene']} frozen target 漂移: {arm} {key}"
                    )
                reference_targets.setdefault(key, target)
                metrics = analyze_view(
                    probability,
                    target,
                    threshold=threshold,
                    boundary_iterations=boundary_iterations,
                )
                view_rows.append(
                    {
                        "frame": key[0],
                        "camera_id": key[1],
                        "path": str(row["path"]),
                        "sha256": str(row["sha256"]),
                        **metrics,
                    }
                )
            arm_results[arm] = {
                "aggregate": aggregate_view_rows(view_rows),
                "views": view_rows,
            }
        cells = []
        for unary in analysis["unary_inputs"]:
            g0 = arm_results[f"{unary}_G0"]["aggregate"]
            g3 = arm_results[f"{unary}_G3"]["aggregate"]
            boundary_primary = bool(
                g0["boundary_classification_error_share"]
                >= float(primary["minimum_boundary_classification_error_share"])
                and g0["boundary_semantic_error_mass_share"]
                >= float(primary["minimum_boundary_semantic_error_mass_share"])
                and g0["boundary_error_enrichment"]
                >= float(primary["minimum_boundary_error_enrichment"])
            )
            cells.append(
                {
                    "unary": unary,
                    "boundary_primary": boundary_primary,
                    "g0": g0,
                    "g3_delta_vs_g0": {
                        name: float(g3[name] - g0[name])
                        for name in (
                            "boundary_classification_error_share",
                            "boundary_semantic_error_mass_share",
                            "boundary_error_enrichment",
                            "boundary_false_positive_mass_share",
                            "boundary_false_negative_mass_share",
                            "mean_entropy",
                            "boundary_mean_entropy",
                            "far_mean_entropy",
                        )
                    },
                }
            )
        scene_results.append(
            {
                "scene": binding["scene"],
                "graph_run": binding["run"],
                "evaluation_view_count": int(summary["evaluation_view_count"]),
                "cells": cells,
                "arm_results": arm_results,
            }
        )

    cells = [cell for scene in scene_results for cell in scene["cells"]]
    if len(cells) != 6:
        raise ForensicAuditError("boundary forensic cell denominator 漂移")
    primary_count = sum(int(cell["boundary_primary"]) for cell in cells)
    mean_error_share = float(
        np.mean([cell["g0"]["boundary_classification_error_share"] for cell in cells])
    )
    mean_semantic_share = float(
        np.mean([cell["g0"]["boundary_semantic_error_mass_share"] for cell in cells])
    )
    gate = config["gate"]
    passed = bool(
        primary_count >= int(gate["minimum_boundary_primary_cells"])
        and mean_error_share
        >= float(gate["minimum_mean_boundary_classification_error_share"])
        and mean_semantic_share
        >= float(gate["minimum_mean_boundary_semantic_error_mass_share"])
    )
    conclusion = (
        "boundary_ambiguity_primary_eligible_to_freeze_semantic_split_protocol"
        if passed
        else "boundary_ambiguity_not_primary_semantic_split_remains_locked"
    )
    gate_result = {
        "boundary_primary_cell_count": primary_count,
        "cell_count": len(cells),
        "mean_boundary_classification_error_share": mean_error_share,
        "mean_boundary_semantic_error_mass_share": mean_semantic_share,
        "passed": passed,
        "conclusion": conclusion,
        "automatic_semantic_split_unlock": False,
        "semantic_split_authorized_by_this_run": False,
    }
    artifact = {
        "schema_version": "worldsim_v5_m1b_boundary_residual_audit_v1",
        "task_id": TASK_ID,
        "analysis": dict(analysis),
        "frozen_gate": dict(gate),
        "gate_result": gate_result,
        "scenes": scene_results,
    }
    summary = {
        "schema_version": "worldsim_v5_m1b_boundary_residual_summary_v1",
        "task_id": TASK_ID,
        "task_status": "done",
        "status": "done",
        "conclusion": conclusion,
        "scene_count": len(scene_results),
        "evaluation_view_count": sum(
            int(scene["evaluation_view_count"]) for scene in scene_results
        ),
        "cell_count": len(cells),
        "gate_result": gate_result,
        "development_evaluation_artifact_read": True,
        "development_source_image_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_quality_read": False,
        "parameter_search_performed": False,
        "training_performed": False,
        "semantic_split_started": False,
        "finished_at_utc": utc_now(),
    }
    return summary, artifact, input_bindings


def finalize(
    *,
    run_dir: Path,
    project_head: str,
    summary: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    events: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    atomic_json(summary_path, summary)
    fingerprint = {
        "schema_version": "worldsim_v5_m1b_boundary_forensic_fingerprint_v1",
        "task_id": TASK_ID,
        "project_git_head": project_head,
        "source_clean": True,
        "resolved_config": dict(resolved_config),
        "input_bindings": dict(input_bindings),
        "development_evaluation_artifact_read": True,
        "development_source_image_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    manifest = {
        "schema_version": "worldsim_v5_m1b_boundary_forensic_manifest_v1",
        "task_id": TASK_ID,
        "status": "done",
        "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    status = {
        "schema_version": "worldsim_v5_m1b_boundary_forensic_status_v1",
        "task_id": TASK_ID,
        "task_status": "done",
        "status": "done",
        "conclusion": str(summary["conclusion"]),
        "project_git_head": project_head,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "resolved_config_sha256": str(resolved_config["sha256"]),
        "events_sha256": str(events["sha256"]),
        "semantic_split_authorized_by_this_run": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "parameter_search_performed": False,
        "finished_at_utc": utc_now(),
    }
    atomic_json(run_dir / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    project_head = prepare_formal_run(args.run_dir, TASK_ID, project)
    config = load_config(args.config)
    resolved_config = write_resolved_config(args.run_dir, config)
    source_snapshot = copy_source_snapshot(
        args.run_dir,
        [
            Path(__file__),
            Path(__file__).with_name("worldsim_v5_forensics_common.py"),
            args.config,
            project / "tests/test_worldsim_v5_m1b_boundary_residual_forensics.py",
        ],
        project,
    )
    started = utc_now()
    summary, artifact, input_bindings = audit(config)
    artifact_path = args.run_dir / "artifacts/boundary_residual_audit.json"
    atomic_json(artifact_path, artifact)
    summary = {
        **summary,
        "project_git_head": project_head,
        "source_snapshot": source_snapshot,
        "boundary_residual_audit_path": "artifacts/boundary_residual_audit.json",
        "boundary_residual_audit_sha256": sha256_file(artifact_path),
    }
    events = write_events(
        args.run_dir,
        [
            {"event": "audit_started", "task_id": TASK_ID, "timestamp_utc": started},
            {
                "event": "audit_completed",
                "task_id": TASK_ID,
                "conclusion": summary["conclusion"],
                "timestamp_utc": utc_now(),
            },
        ],
    )
    status = finalize(
        run_dir=args.run_dir,
        project_head=project_head,
        summary=summary,
        input_bindings=input_bindings,
        resolved_config=resolved_config,
        events=events,
    )
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
