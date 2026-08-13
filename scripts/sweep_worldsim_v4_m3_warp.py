#!/usr/bin/env python3
"""复用 development Gaussian renders，只搜索 M3 warp blend alpha。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_worldsim_v4_m3_scene import (  # noqa: E402
    TASK_ID,
    atomic_json,
    build_full_warp_variant,
    effect_mask,
    flow_current_to_previous,
    git_dirty,
    git_head,
    output_manifest,
    remap_previous,
    rgb_path,
    sha256_file,
    write_jsonl,
)


class M3WarpSweepError(RuntimeError):
    pass


def candidate_warp_l1(
    *,
    base_dir: Path,
    remove_dir: Path,
    candidate_dir: Path,
    frames: list[int],
    cameras: list[int],
    minimum_effect_pixels: int,
) -> tuple[float, bool, dict[int, int]]:
    values = []
    by_frame = {frame: 0 for frame in frames}
    for camera in cameras:
        previous_base = None
        previous_candidate = None
        previous_effect = None
        for frame in frames:
            base = imageio.imread(rgb_path(base_dir, frame, camera))
            removed = imageio.imread(rgb_path(remove_dir, frame, camera))
            candidate = imageio.imread(rgb_path(candidate_dir, frame, camera))
            source = effect_mask(base, removed)
            inserted = effect_mask(candidate, removed)
            effect = source | inserted
            by_frame[frame] += int(effect.sum())
            if previous_base is not None:
                flow = flow_current_to_previous(base, previous_base)
                warped_candidate = remap_previous(previous_candidate, flow)
                warped_base = remap_previous(previous_base, flow)
                current_small = cv2.resize(candidate, (160, 90))
                base_small = cv2.resize(base, (160, 90))
                current_delta = current_small.astype(np.float32) - base_small.astype(
                    np.float32
                )
                warped_delta = warped_candidate.astype(np.float32) - warped_base.astype(
                    np.float32
                )
                current_effect = cv2.resize(
                    effect.astype(np.uint8),
                    (160, 90),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                warped_effect = remap_previous(
                    previous_effect.astype(np.uint8) * 255, flow
                ) > 127
                mask = current_effect | warped_effect
                if mask.any():
                    values.append(
                        float(
                            np.mean(np.abs(current_delta - warped_delta)[mask])
                            / 255.0
                        )
                    )
            previous_base = base
            previous_candidate = candidate
            previous_effect = effect
    if not values:
        raise M3WarpSweepError("warp candidate 没有可评时序支持")
    success = all(value >= minimum_effect_pixels for value in by_frame.values())
    return float(np.mean(values)), success, by_frame


def run(
    *,
    config_path: Path,
    source_run: Path,
    source_summary_sha256: str,
    source_manifest_sha256: str,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    summary_path = source_run / "summary.json"
    manifest_path = source_run / "manifest.json"
    if sha256_file(summary_path) != source_summary_sha256:
        raise M3WarpSweepError("source summary SHA 不匹配")
    if sha256_file(manifest_path) != source_manifest_sha256:
        raise M3WarpSweepError("source manifest SHA 不匹配")
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    if source["partition"] != "development" or source["status"] != "done":
        raise M3WarpSweepError("warp sweep 只能消费完成的 development scene")
    if source.get("test_quality_read") is not False:
        raise M3WarpSweepError("source run 触碰了 test quality")
    if "LATERAL" not in source["operations"]:
        raise M3WarpSweepError("source run 缺 LATERAL")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    alphas = [float(value) for value in config["trajectory"]["development_search"]["warp_blend_alpha"]]
    frames = [int(value) for value in source["clip"]["processed_keyframe_indices"]]
    cameras = [int(value) for value in source["clip"]["camera_ids"]]
    source_artifacts = source_run / "artifacts"
    base_dir = source_artifacts / "BASE"
    remove_dir = source_artifacts / "REMOVE_SHARED"
    evidence_dir = (
        source_artifacts / "LATERAL/CUBIC_BSPLINE_TEMPORAL_EVIDENCE"
    )
    rows = []
    baseline_dir = source_artifacts / "LATERAL/FRAME_INDEPENDENT"
    baseline_warp, baseline_success, baseline_pixels = candidate_warp_l1(
        base_dir=base_dir,
        remove_dir=remove_dir,
        candidate_dir=baseline_dir,
        frames=frames,
        cameras=cameras,
        minimum_effect_pixels=int(config["operations"]["minimum_rendered_effect_pixels"]),
    )
    for alpha in alphas:
        candidate_dir = run_dir / "artifacts" / f"alpha_{alpha:.2f}"
        build_full_warp_variant(
            base_dir=base_dir,
            evidence_dir=evidence_dir,
            output_dir=candidate_dir,
            frames=frames,
            cameras=cameras,
            alpha=alpha,
        )
        warp_l1, success, pixels = candidate_warp_l1(
            base_dir=base_dir,
            remove_dir=remove_dir,
            candidate_dir=candidate_dir,
            frames=frames,
            cameras=cameras,
            minimum_effect_pixels=int(config["operations"]["minimum_rendered_effect_pixels"]),
        )
        rows.append(
            {
                "warp_blend_alpha": alpha,
                "operation_success": success,
                "rendered_effect_pixels_by_frame": pixels,
                "warp_l1_delta": warp_l1,
                "frame_independent_warp_l1_delta": baseline_warp,
                "relative_improvement": (baseline_warp - warp_l1) / baseline_warp,
            }
        )
    eligible = [row for row in rows if row["operation_success"]]
    if not baseline_success or not eligible:
        raise M3WarpSweepError("warp sweep operation success gate 无可选项")
    selected = min(
        eligible,
        key=lambda row: (
            row["warp_l1_delta"],
            row["warp_blend_alpha"],
        ),
    )
    write_jsonl(run_dir / "metrics.jsonl", rows)
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    for path in (config_path, Path(__file__)):
        shutil.copy2(path, source_snapshot / path.name)
    summary = {
        "schema_version": "worldsim_v4_m3_warp_sweep_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "partition": "development",
        "scene": source["scene"],
        "source_run": str(source_run),
        "source_summary_sha256": source_summary_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_parameters": source["parameters"],
        "baseline": {
            "arm": "FRAME_INDEPENDENT",
            "warp_l1_delta": baseline_warp,
            "operation_success": baseline_success,
            "rendered_effect_pixels_by_frame": baseline_pixels,
        },
        "candidates": rows,
        "selected_warp_blend_alpha": selected["warp_blend_alpha"],
        "selected_relative_improvement": selected["relative_improvement"],
        "development_content_read": True,
        "development_optimization_read": True,
        "validation_content_read": False,
        "validation_optimization_read": False,
        "test_quality_read": False,
        "project_git_head": git_head(),
        "project_git_dirty": git_dirty(),
        "duration_seconds": time.monotonic() - started,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "fingerprint.json",
        {
            "task_id": TASK_ID,
            "project_git_head": git_head(),
            "project_git_dirty": git_dirty(),
            "source_summary_sha256": source_summary_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "config_sha256": sha256_file(config_path),
            "test_quality_read": False,
        },
    )
    manifest = output_manifest(run_dir)
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/worldsim_v4/m3_temporal_v1.yaml",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-summary-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        config_path=args.config.resolve(),
        source_run=args.source_run.resolve(),
        source_summary_sha256=args.source_summary_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        run_dir=args.run_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_warp_blend_alpha": summary[
                    "selected_warp_blend_alpha"
                ],
                "selected_relative_improvement": summary[
                    "selected_relative_improvement"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
