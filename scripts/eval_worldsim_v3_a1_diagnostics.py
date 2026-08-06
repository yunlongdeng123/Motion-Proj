#!/usr/bin/env python
"""对 A1 C0–C3 checkpoint 执行冻结的 ISP/位姿/速度分层诊断。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v3.calibration_diagnostics import (
    VARIANTS,
    evaluate_checkpoint_diagnostics,
    load_input_speed_contract,
    validate_diagnostic_contract,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_source_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        variant, separator, path = value.partition("=")
        if not separator or variant not in VARIANTS or not path:
            raise ValueError(f"invalid --source-run value: {value}")
        if variant in result:
            raise ValueError(f"duplicate source variant: {variant}")
        result[variant] = Path(path)
    if set(result) != set(VARIANTS):
        raise ValueError(f"source variants must be exactly {VARIANTS}")
    return result


def compact_actor_metrics(source: dict[str, Any]) -> dict[str, Any]:
    roles = source.get("actor_metrics", {}).get("roles", {})
    compact: dict[str, Any] = {}
    for role, row in roles.items():
        compact[role] = {
            "status": row.get("status"),
            "gaussian_count": (row.get("actor") or {}).get("gaussian_count"),
            "visible_effect_image_count": row.get("visible_effect_image_count"),
            "effect_pixel_coverage": row.get("effect_pixel_coverage"),
            "actor_region": row.get("actor_region"),
            "boundary_band": row.get("boundary_band"),
        }
    return compact


def compact_source(source: dict[str, Any], source_run: Path) -> dict[str, Any]:
    checkpoint = source.get("checkpoint") or {}
    metrics = source.get("heldout_metrics") or {}
    resources = source.get("train_resources") or {}
    return {
        "source_run_dir": str(source_run),
        "source_summary": str(source_run / "summary.json"),
        "scene_name": source.get("scene_name"),
        "scene_index": source.get("scene_index"),
        "variant": source.get("variant"),
        "seed": 0,
        "checkpoint": checkpoint,
        "initialization_provenance": source.get("initialization_provenance"),
        "heldout": {
            "psnr": metrics.get("image_metrics/test/psnr"),
            "ssim": metrics.get("image_metrics/test/ssim"),
            "lpips": metrics.get("image_metrics/test/lpips"),
        },
        "actors": compact_actor_metrics(source),
        "train_resources": resources,
    }


def nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def write_matrix_csv(path: Path, variants: dict[str, dict[str, Any]]) -> None:
    fields = [
        "variant",
        "global_psnr",
        "global_ssim",
        "global_lpips",
        "background_gaussians",
        "rigid_gaussians",
        "train_duration_seconds",
        "peak_gpu_memory_mib_sampled",
        "pose_translation_median_m",
        "pose_translation_p90_m",
        "pose_rotation_median_deg",
        "pose_rotation_p90_deg",
        "pose_translation_first_p90_m",
        "pose_translation_second_p90_m",
        "pose_rotation_first_p90_deg",
        "pose_rotation_second_p90_deg",
        "isp_residual_median",
        "isp_residual_p90",
        "isp_temporal_first_p90",
        "isp_temporal_second_p90",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            row = variants[variant]
            source = row["source"]
            diagnostic = row["diagnostic"]
            checkpoint = source["checkpoint"]
            writer.writerow(
                {
                    "variant": variant,
                    "global_psnr": nested(source, "heldout.psnr"),
                    "global_ssim": nested(source, "heldout.ssim"),
                    "global_lpips": nested(source, "heldout.lpips"),
                    "background_gaussians": checkpoint.get("background_gaussians"),
                    "rigid_gaussians": checkpoint.get("rigid_gaussians"),
                    "train_duration_seconds": nested(source, "train_resources.duration_seconds"),
                    "peak_gpu_memory_mib_sampled": nested(
                        source, "train_resources.peak_gpu_memory_mib_sampled"
                    ),
                    "pose_translation_median_m": nested(
                        diagnostic, "pose.overall.translation_norm_m.median"
                    ),
                    "pose_translation_p90_m": nested(
                        diagnostic, "pose.overall.translation_norm_m.p90"
                    ),
                    "pose_rotation_median_deg": nested(
                        diagnostic, "pose.overall.rotation_angle_deg.median"
                    ),
                    "pose_rotation_p90_deg": nested(
                        diagnostic, "pose.overall.rotation_angle_deg.p90"
                    ),
                    "pose_translation_first_p90_m": nested(
                        diagnostic,
                        "pose.first_difference.translation_delta_norm_m.p90",
                    ),
                    "pose_translation_second_p90_m": nested(
                        diagnostic,
                        "pose.second_difference.translation_jitter_norm_m.p90",
                    ),
                    "pose_rotation_first_p90_deg": nested(
                        diagnostic, "pose.first_difference.rotation_delta_deg.p90"
                    ),
                    "pose_rotation_second_p90_deg": nested(
                        diagnostic, "pose.second_difference.rotation_jitter_deg.p90"
                    ),
                    "isp_residual_median": nested(
                        diagnostic, "isp.overall.residual_l2.median"
                    ),
                    "isp_residual_p90": nested(
                        diagnostic, "isp.overall.residual_l2.p90"
                    ),
                    "isp_temporal_first_p90": nested(
                        diagnostic, "isp.temporal_first_difference_l2.p90"
                    ),
                    "isp_temporal_second_p90": nested(
                        diagnostic, "isp.temporal_second_difference_l2.p90"
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", action="append", required=True)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = yaml.safe_load(args.diagnostic_config.read_text(encoding="utf-8"))
    validate_diagnostic_contract(contract)
    source_runs = parse_source_runs(args.source_run)
    scene = contract["scene"]
    speed = contract["speed_tiers"]
    frame_speed, input_motion = load_input_speed_contract(
        Path(scene["processed_root"]),
        num_frames=int(scene["num_frames"]),
        source_camera_id=int(speed["source_camera_id"]),
        processed_hz=float(scene["processed_hz"]),
        near_static_upper_mps=float(speed["near_static_upper_mps"]),
        low_speed_upper_mps=float(speed["low_speed_upper_mps"]),
    )
    expected_counts = {key: int(value) for key, value in speed["expected_input_only_counts"].items()}
    if input_motion["tier_frame_counts"] != expected_counts:
        raise RuntimeError(
            "input-only speed tier count mismatch: "
            f"{input_motion['tier_frame_counts']} != {expected_counts}"
        )
    variants: dict[str, dict[str, Any]] = {}
    initialization_hashes: dict[str, str | None] = {}
    for variant in VARIANTS:
        source_run = source_runs[variant]
        terminal_path = source_run / "terminal.json"
        summary_path = source_run / "summary.json"
        if not terminal_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"source run is incomplete: {source_run}")
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("status") != "done":
            raise RuntimeError(f"source run is not done: {source_run}")
        source = json.loads(summary_path.read_text(encoding="utf-8"))
        if source.get("variant") != variant or source.get("scene_name") != scene["name"]:
            raise RuntimeError(f"source run contract mismatch: {source_run}")
        checkpoint = Path(source["checkpoint"]["checkpoint"])
        if source["checkpoint"].get("step") != 30000 or not checkpoint.is_file():
            raise RuntimeError(f"source checkpoint contract mismatch: {source_run}")
        initialization = source.get("initialization_provenance") or {}
        initialization_hashes[variant] = initialization.get("sha256")
        variants[variant] = {
            "source": compact_source(source, source_run),
            "diagnostic": evaluate_checkpoint_diagnostics(
                checkpoint,
                variant=variant,
                contract=contract,
                frame_speed_mps=frame_speed,
            ),
        }
    non_null_initialization_hashes = {value for value in initialization_hashes.values() if value}
    if len(non_null_initialization_hashes) != 1 or len(non_null_initialization_hashes) != len(
        set(initialization_hashes.values())
    ):
        raise RuntimeError(f"paired initialization hashes mismatch: {initialization_hashes}")
    if not all(row["diagnostic"]["checkpoint_unchanged"] for row in variants.values()):
        raise RuntimeError("a diagnostic modified or mismatched a checkpoint")
    summary = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A1 ISP, pose, temporal, and input-speed diagnostics",
        "diagnostic_version": contract["diagnostic_version"],
        "scene_name": scene["name"],
        "scene_index": scene["index"],
        "split": "all processed frames; held-out policy applied to ISP where defined",
        "seed": 0,
        "input_motion": input_motion,
        "paired_initialization_sha256": next(iter(non_null_initialization_hashes)),
        "variants": variants,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "diagnostics_matrix.json", summary)
    write_matrix_csv(args.output_dir / "diagnostics_matrix.csv", variants)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
