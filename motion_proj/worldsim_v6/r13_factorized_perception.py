"""WorldSim V6 R13 factorized target/static ROI 感知实验。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r13_actor_sensor_perception import (
    R13ActorSensorError,
    _dynamic,
    _git,
    _read_jsonl,
    _render_index,
    _resolve_runs_uri,
    _rgb,
    _run_worker,
    _sha256,
    _write_json,
    _write_jsonl,
)


TASK_ID = "WS-V6-R13-WORLDSIM-01"


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if tile > length:
        raise R13ActorSensorError(f"tile {tile} 超出轴长 {length}")
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _nonoverlap(left: list[int], right: list[int]) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return lx + lw <= rx or rx + rw <= lx or ly + lh <= ry or ry + rh <= ly


def _select_rois(dynamic: np.ndarray, roi: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tile = int(roi["tile_size_px"])
    stride = int(roi["candidate_stride_px"])
    actor = dynamic > float(roi["actor_opacity_threshold"])
    height, width = actor.shape
    candidates = []
    for y in _positions(height, tile, stride):
        for x in _positions(width, tile, stride):
            candidates.append(
                {
                    "crop_xywh": [x, y, tile, tile],
                    "actor_pixel_fraction": float(np.mean(actor[y : y + tile, x : x + tile])),
                }
            )
    target = max(
        candidates,
        key=lambda row: (
            row["actor_pixel_fraction"],
            -row["crop_xywh"][1],
            -row["crop_xywh"][0],
        ),
    )
    static_candidates = [
        row for row in candidates if _nonoverlap(row["crop_xywh"], target["crop_xywh"])
    ]
    if not static_candidates:
        raise R13ActorSensorError("没有与 target 不重叠的 static ROI")
    static = min(
        static_candidates,
        key=lambda row: (
            row["actor_pixel_fraction"],
            row["crop_xywh"][1],
            row["crop_xywh"][0],
        ),
    )
    return target, static


def _crop(value: np.ndarray, xywh: list[int]) -> np.ndarray:
    x, y, width, height = xywh
    return value[y : y + height, x : x + width]


def run_experiment(
    repo_root: Path, config_path: Path, run_root: Path, semantic_model_root: Path
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13ActorSensorError("正式 R13 factorized-perception run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13ActorSensorError("R13 factorized-perception task_id 漂移")
    sources = config["sources"]
    rejected_run = _resolve_runs_uri(sources["rejected_full_frame_run"])
    dynamic_run = _resolve_runs_uri(sources["r13_dynamic_run"])
    r3 = _resolve_runs_uri(sources["r3_render_run"])
    scene = config["cohort"]["scene"]
    roots = {
        frontend: r3 / "renders" / scene / frontend for frontend in config["cohort"]["frontends"]
    }
    frozen = {
        rejected_run / "MANIFEST.json": sources["rejected_manifest_sha256"],
        rejected_run / "R13_ACTOR_SENSOR_GATE.json": sources["rejected_gate_sha256"],
        dynamic_run / "R13_DYNAMIC_EDIT_GATE.json": sources["r13_dynamic_gate_sha256"],
        roots["streetgs"] / "RENDER_MAP.jsonl": sources["streetgs_render_map_sha256"],
        roots["ad_gs"] / "RENDER_MAP.jsonl": sources["ad_gs_render_map_sha256"],
        semantic_model_root / config["verifier_model"]["model_file"]: config["verifier_model"]["model_sha256"],
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13ActorSensorError(f"冻结输入漂移：{path}")
    rejected_gate = json.loads(
        (rejected_run / "R13_ACTOR_SENSOR_GATE.json").read_text(encoding="utf-8")
    )
    dynamic_gate = json.loads(
        (dynamic_run / "R13_DYNAMIC_EDIT_GATE.json").read_text(encoding="utf-8")
    )
    full_frame_failure_frozen = (
        not rejected_gate["checks"]["passed"]
        and not rejected_gate["checks"]["all_four_sensor_cases_pass"]
        and rejected_gate["decision"] == "reject_actor_sensor_perception_hypothesis"
    )
    if not full_frame_failure_frozen or not dynamic_gate["checks"]["passed"]:
        raise R13ActorSensorError("冻结的 full-frame rejection 或 typed dynamic pass 漂移")
    indexes = {frontend: _render_index(root) for frontend, root in roots.items()}
    frames = [int(value) for value in config["cohort"]["frame_indices"]]
    operation = config["cohort"]["operation"]
    selected: dict[tuple[str, int, str], Path] = {}
    roi_rows: list[dict[str, Any]] = []
    roi_index: dict[tuple[str, int, str], list[int]] = {}
    for frontend in config["cohort"]["frontends"]:
        for frame in frames:
            logged = indexes[frontend][(frame, "camera_lateral", 0.0)]
            edited = indexes[frontend][(frame, operation, 0.0)]
            selected[(frontend, frame, "logged")] = logged
            selected[(frontend, frame, "edited")] = edited
            target, static = _select_rois(_dynamic(logged), config["roi"])
            for role, row in (("target", target), ("static", static)):
                roi_index[(frontend, frame, role)] = row["crop_xywh"]
                roi_rows.append(
                    {
                        "schema_version": "worldsim_v6.r13_factorized_roi.v1",
                        "frontend": frontend,
                        "frame_index": frame,
                        "role": role,
                        **row,
                    }
                )
    for path in selected.values():
        frozen[path] = _sha256(path)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13ActorSensorError("R13 factorized-perception 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__factorized-perception-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        _write_jsonl(run_dir / "ROI_SELECTIONS.jsonl", roi_rows)
        repeat_count = int(config["cohort"]["repeat_count"])
        input_rows = []
        for frontend in config["cohort"]["frontends"]:
            for frame in frames:
                for role in ("target", "static"):
                    for state in ("logged", "edited"):
                        case_id = f"{frontend}__f{frame:03d}__{role}__{state}"
                        for repeat_index in range(1, repeat_count + 1):
                            path = selected[(frontend, frame, state)]
                            input_rows.append(
                                {
                                    "case_id": case_id,
                                    "repeat_index": repeat_index,
                                    "render_path": str(path),
                                    "render_sha256": _sha256(path),
                                    "crop_xywh": roi_index[(frontend, frame, role)],
                                }
                            )
        _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", input_rows)
        perception_dir = run_dir / "perception"
        _run_worker(
            [
                "/root/autodl-tmp/envs/motionproj/bin/python",
                str(repo_root / "scripts/worldsim_v6/r13_perception_worker.py"),
                "--index",
                str(run_dir / "PERCEPTION_INPUT_INDEX.jsonl"),
                "--model-root",
                str(semantic_model_root),
                "--output-dir",
                str(perception_dir),
            ],
            repo_root,
            run_dir / "perception.log",
        )
        worker = json.loads((perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
        perception_rows = _read_jsonl(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in perception_rows:
            by_case.setdefault(row["case_id"], []).append(row)
        repeat_exact = all(
            len(rows) == repeat_count
            and len({row["label_array_sha256"] for row in rows}) == 1
            for rows in by_case.values()
        )
        thresholds = config["metrics"]
        factor_rows = []
        for frontend in config["cohort"]["frontends"]:
            for frame in frames:
                logged = _rgb(selected[(frontend, frame, "logged")])
                edited = _rgb(selected[(frontend, frame, "edited")])
                factor_metrics: dict[str, dict[str, float]] = {}
                for role in ("target", "static"):
                    xywh = roi_index[(frontend, frame, role)]
                    logged_rgb = _crop(logged, xywh)
                    edited_rgb = _crop(edited, xywh)
                    rgb_mae = float(np.mean(np.abs(edited_rgb - logged_rgb)))
                    logged_case = f"{frontend}__f{frame:03d}__{role}__logged"
                    edited_case = f"{frontend}__f{frame:03d}__{role}__edited"
                    logged_labels = np.load(
                        perception_dir
                        / sorted(by_case[logged_case], key=lambda row: row["repeat_index"])[0]["label_path"],
                        allow_pickle=False,
                    )
                    edited_labels = np.load(
                        perception_dir
                        / sorted(by_case[edited_case], key=lambda row: row["repeat_index"])[0]["label_path"],
                        allow_pickle=False,
                    )
                    factor_metrics[role] = {
                        "rgb_mae": rgb_mae,
                        "perception_changed_fraction": float(np.mean(logged_labels != edited_labels)),
                    }
                roi_for_case = {
                    row["role"]: row
                    for row in roi_rows
                    if row["frontend"] == frontend and row["frame_index"] == frame
                }
                rgb_enrichment = factor_metrics["target"]["rgb_mae"] / max(
                    factor_metrics["static"]["rgb_mae"], 1.0e-12
                )
                perception_enrichment = factor_metrics["target"][
                    "perception_changed_fraction"
                ] / max(factor_metrics["static"]["perception_changed_fraction"], 1.0e-12)
                checks = {
                    "target_actor_fraction": roi_for_case["target"]["actor_pixel_fraction"]
                    >= float(config["roi"]["minimum_target_actor_fraction"]),
                    "static_actor_fraction": roi_for_case["static"]["actor_pixel_fraction"]
                    <= float(config["roi"]["maximum_static_actor_fraction"]),
                    "target_rgb_effect": factor_metrics["target"]["rgb_mae"]
                    >= float(thresholds["minimum_target_rgb_mae"]),
                    "static_rgb_preserved": factor_metrics["static"]["rgb_mae"]
                    <= float(thresholds["maximum_static_rgb_mae"]),
                    "rgb_locality_enrichment": rgb_enrichment
                    >= float(thresholds["minimum_rgb_locality_enrichment"]),
                    "target_perception_effect": factor_metrics["target"]["perception_changed_fraction"]
                    >= float(thresholds["minimum_target_perception_changed_fraction"]),
                    "static_perception_preserved": factor_metrics["static"]["perception_changed_fraction"]
                    <= float(thresholds["maximum_static_perception_changed_fraction"]),
                    "perception_locality_enrichment": perception_enrichment
                    >= float(thresholds["minimum_perception_locality_enrichment"]),
                }
                checks["passed"] = all(checks.values())
                factor_rows.append(
                    {
                        "schema_version": "worldsim_v6.r13_factorized_perception_metric.v1",
                        "case_id": f"{frontend}__f{frame:03d}",
                        "frontend": frontend,
                        "frame_index": frame,
                        "target_crop_xywh": roi_index[(frontend, frame, "target")],
                        "static_crop_xywh": roi_index[(frontend, frame, "static")],
                        "target_actor_fraction": roi_for_case["target"]["actor_pixel_fraction"],
                        "static_actor_fraction": roi_for_case["static"]["actor_pixel_fraction"],
                        "target_rgb_mae": factor_metrics["target"]["rgb_mae"],
                        "static_rgb_mae": factor_metrics["static"]["rgb_mae"],
                        "rgb_locality_enrichment": rgb_enrichment,
                        "target_perception_changed_fraction": factor_metrics["target"][
                            "perception_changed_fraction"
                        ],
                        "static_perception_changed_fraction": factor_metrics["static"][
                            "perception_changed_fraction"
                        ],
                        "perception_locality_enrichment": perception_enrichment,
                        "checks": checks,
                    }
                )
        _write_jsonl(run_dir / "FACTORIZED_PERCEPTION_METRICS.jsonl", factor_rows)
        unsupported = config["unsupported_metrics"]
        wall_seconds = time.monotonic() - started
        checks = {
            "all_four_roi_cases_pass": len(factor_rows) == 4
            and all(row["checks"]["passed"] for row in factor_rows),
            "perception_repeat_exact": repeat_exact,
            "full_frame_failure_reproduced_as_frozen_source": full_frame_failure_frozen,
            "typed_dynamic_gate_passed": dynamic_gate["checks"]["passed"],
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "inherited_v6_false_safe_rate_zero": float(dynamic_gate["v6_false_safe_rate"]) == 0.0,
            "inherited_naive_false_safe_rate_one": float(dynamic_gate["naive_false_safe_rate"]) == 1.0,
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "gpu_within_budget": int(worker["peak_gpu_memory_mib"])
            <= int(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_factorized_perception_gate.v1",
            "checks": checks,
            "unsupported_metrics": unsupported,
            "decision": "accept_factorized_roi_perception_verification"
            if checks["passed"]
            else "reject_factorized_roi_perception_hypothesis",
        }
        _write_json(run_dir / "R13_FACTORIZED_PERCEPTION_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_factorized_perception_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_factorized_roi_perception"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "roi_case_count": len(factor_rows),
            "perception_inference_count": len(perception_rows),
            "perception_repeat_exact": repeat_exact,
            "peak_gpu_memory_mib": int(worker["peak_gpu_memory_mib"]),
            "wall_seconds": wall_seconds,
            "unsupported_metrics": unsupported,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "ROI_SELECTIONS.jsonl",
            "PERCEPTION_INPUT_INDEX.jsonl",
            "FACTORIZED_PERCEPTION_METRICS.jsonl",
            "R13_FACTORIZED_PERCEPTION_GATE.json",
            "SUMMARY.json",
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
                "schema_version": "worldsim_v6.r13_factorized_perception_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {
                    relative: {
                        "bytes": (run_dir / relative).stat().st_size,
                        "sha256": _sha256(run_dir / relative),
                    }
                    for relative in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "blocked",
                "task_id": TASK_ID,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r13_factorized_perception_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--semantic-model-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r9_semantic_deeplab_cityscapes"),
    )
    args = parser.parse_args()
    run_dir = run_experiment(
        args.repo_root, args.config, args.run_root, args.semantic_model_root
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
