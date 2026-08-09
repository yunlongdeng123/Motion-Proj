#!/usr/bin/env python
"""Read-only heldout R0/R1 evaluation for the frozen A3 R1 contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT = Path("/root/autodl-tmp/motion_proj")
DRIVESTUDIO = Path(
    "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1"
)
EVAL_PROTOCOL = PROJECT / "configs/worldsim_v3/a3_r1_eval_protocol_v1.yaml"
MAIN_PROTOCOL = PROJECT / "configs/worldsim_v3/a3_local_refine_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


from scripts.run_worldsim_v3_a3_s_b_paired_smoke import (
    atomic_json,
    cgroup_memory_current,
    cgroup_memory_events,
    directory_bytes,
    rigid_contract_sha256,
    sha256_file,
    to_device,
)


def command_output(*command: str, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def uint8_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("A3 heldout render contains non-finite RGB")
    return np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)


def squared_rgb_error(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[int, int]:
    if predicted.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("A3 RGB comparison must use frozen uint8 encoding")
    valid = np.asarray(mask, dtype=np.bool_)
    if predicted.shape != target.shape or predicted.shape[:2] != valid.shape:
        raise ValueError("A3 RGB/mask shape drift")
    difference = predicted.astype(np.int32) - target.astype(np.int32)
    return int(np.square(difference[valid].astype(np.int64)).sum()), int(valid.sum()) * 3


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def build_resource_audit(
    *,
    duration: float,
    peak_gpu_mib: float,
    cgroup_samples: Iterable[int | None],
    run_bytes: int,
    oom_delta: int,
    oom_kill_delta: int,
    ceilings: Mapping[str, Any],
) -> dict[str, Any]:
    samples = [value for value in cgroup_samples if value is not None]
    measured = {
        "wall_time_seconds": float(duration),
        "peak_gpu_memory_mib": float(peak_gpu_mib),
        "peak_cgroup_memory_bytes_sampled": max(samples) if samples else None,
        "cgroup_memory_samples_bytes": samples,
        "run_bytes_before_resource_audit": int(run_bytes),
        "oom_events_delta": int(oom_delta),
        "oom_kill_events_delta": int(oom_kill_delta),
    }
    violations = {
        "wall_time_seconds": measured["wall_time_seconds"]
        > float(ceilings["wall_time_seconds"]),
        "peak_gpu_memory_mib": measured["peak_gpu_memory_mib"]
        > float(ceilings["peak_gpu_memory_mib"]),
        "peak_cgroup_memory_bytes": measured[
            "peak_cgroup_memory_bytes_sampled"
        ]
        is not None
        and measured["peak_cgroup_memory_bytes_sampled"]
        > int(ceilings["peak_cgroup_memory_bytes"]),
        "run_bytes": measured["run_bytes_before_resource_audit"]
        > int(ceilings["run_bytes"]),
        "oom_events_delta": measured["oom_events_delta"]
        != int(ceilings["oom_events_delta"]),
        "oom_kill_events_delta": measured["oom_kill_events_delta"]
        != int(ceilings["oom_kill_events_delta"]),
    }
    return {
        "status": "failed" if any(violations.values()) else "passed",
        "measured": measured,
        "ceilings": dict(ceilings),
        "violations": violations,
    }


def aggregate_metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    sums = {
        "view_units": len(values),
        "zero_footprint_view_units": sum(
            int(row["affected_pixels"] == 0) for row in values
        ),
        "s_b_t0_pixels": sum(int(row["s_b_t0_pixels"]) for row in values),
        "t1_valid_pixels": sum(int(row["t1_valid_pixels"]) for row in values),
        "depth_order_violations": sum(
            int(row["depth_order_violations"]) for row in values
        ),
        "common_t1_valid_pixels": sum(
            int(row["common_t1_valid_pixels"]) for row in values
        ),
        "t0_abs_error_sum_m": sum(
            float(row["t0_abs_error_sum_m"]) for row in values
        ),
        "non_target_squared_uint8_error": sum(
            int(row["non_target_squared_uint8_error"]) for row in values
        ),
        "non_target_channel_elements": sum(
            int(row["non_target_channel_elements"]) for row in values
        ),
    }
    return {
        **sums,
        "s_b_first_hit_valid_coverage": safe_ratio(
            sums["t1_valid_pixels"], sums["s_b_t0_pixels"]
        ),
        "s_b_depth_order_violation_rate": safe_ratio(
            sums["depth_order_violations"], sums["s_b_t0_pixels"]
        ),
        "s_b_t0_first_hit_mae_m": safe_ratio(
            sums["t0_abs_error_sum_m"], sums["common_t1_valid_pixels"]
        ),
        "non_target_observed_rgb_mse": safe_ratio(
            sums["non_target_squared_uint8_error"],
            sums["non_target_channel_elements"] * 65_025,
        ),
    }


def aggregate_global_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    squared = sum(int(row["squared_uint8_error"]) for row in values)
    elements = sum(int(row["channel_elements"]) for row in values)
    return {
        "unique_frame_camera_views": len(values),
        "squared_uint8_error": squared,
        "channel_elements": elements,
        "original_global_observed_rgb_mse": safe_ratio(
            squared, elements * 65_025
        ),
    }


def mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(sum(defined) / len(defined)) if defined else None


def build_variant_aggregate(
    rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    *,
    variant: str,
    group_order: list[str],
) -> dict[str, Any]:
    variant_rows = [row for row in rows if row["variant"] == variant]
    groups = {}
    for group in group_order:
        role, edit = group.split("::")
        groups[group] = aggregate_metric_rows(
            row
            for row in variant_rows
            if row["role"] == role and row["edit"] == edit
        )
    global_aggregate = aggregate_global_rows(
        row for row in global_rows if row["variant"] == variant
    )
    primary = {
        "s_b_first_hit_valid_coverage": mean_defined(
            group["s_b_first_hit_valid_coverage"] for group in groups.values()
        ),
        "s_b_depth_order_violation_rate": mean_defined(
            group["s_b_depth_order_violation_rate"] for group in groups.values()
        ),
        "non_target_observed_rgb_mse": mean_defined(
            group["non_target_observed_rgb_mse"] for group in groups.values()
        ),
        "original_global_observed_rgb_mse": global_aggregate[
            "original_global_observed_rgb_mse"
        ],
    }
    return {
        "variant": variant,
        "groups": groups,
        "group_macro": {
            **primary,
            "s_b_t0_first_hit_mae_m": mean_defined(
                group["s_b_t0_first_hit_mae_m"] for group in groups.values()
            ),
        },
        "global": global_aggregate,
        "primary_axes": primary,
    }


def classify_exact_pareto(
    r0: Mapping[str, float | None],
    r1: Mapping[str, float | None],
    directions: Mapping[str, str],
) -> dict[str, Any]:
    if set(r0) != set(r1) or set(r0) != set(directions):
        raise ValueError("A3 Pareto axes drift")
    if any(r0[name] is None or r1[name] is None for name in directions):
        return {
            "classification": "insufficient_evidence",
            "r1_non_worse": False,
            "r0_non_worse": False,
            "comparisons": {},
        }
    comparisons = {}
    for name, direction in directions.items():
        left, right = float(r0[name]), float(r1[name])
        if direction == "lower":
            r1_non_worse, r1_strict = right <= left, right < left
            r0_non_worse, r0_strict = left <= right, left < right
        elif direction == "higher":
            r1_non_worse, r1_strict = right >= left, right > left
            r0_non_worse, r0_strict = left >= right, left > right
        else:
            raise ValueError(f"unknown A3 Pareto direction: {direction}")
        comparisons[name] = {
            "direction": direction,
            "r0": left,
            "r1": right,
            "r1_non_worse": r1_non_worse,
            "r1_strict": r1_strict,
            "r0_non_worse": r0_non_worse,
            "r0_strict": r0_strict,
        }
    r1_non_worse = all(row["r1_non_worse"] for row in comparisons.values())
    r0_non_worse = all(row["r0_non_worse"] for row in comparisons.values())
    r1_strict = any(row["r1_strict"] for row in comparisons.values())
    r0_strict = any(row["r0_strict"] for row in comparisons.values())
    if r1_non_worse and r1_strict:
        classification = "r1_dominates_r0_pass"
    elif r0_non_worse and r0_strict:
        classification = "r1_rejected_dominated_by_r0"
    elif r1_non_worse and r0_non_worse:
        classification = "primary_axes_exact_equal"
    else:
        classification = "tradeoff_non_dominated"
    return {
        "classification": classification,
        "r1_non_worse": r1_non_worse,
        "r0_non_worse": r0_non_worse,
        "comparisons": comparisons,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def get_view_data(dataset: Any, frame: int, camera: int, device: torch.device):
    image_index = frame * dataset.pixel_source.num_cams + camera
    image_infos, camera_infos = dataset.full_image_set.get_image(
        image_index, camera_downscale=1.0
    )
    groundtruth = uint8_rgb(image_infos["pixels"])
    measured = image_infos["lidar_depth_map"].detach().float().cpu().numpy()
    egocar = (
        image_infos["egocar_masks"].detach().bool().cpu().numpy()
        if "egocar_masks" in image_infos
        else np.zeros(measured.shape, dtype=np.bool_)
    )
    return (
        to_device(image_infos, device),
        to_device(camera_infos, device),
        groundtruth,
        measured,
        egocar,
        image_index,
    )


def render_edit(
    trainer: Any,
    image_infos: Mapping[str, Any],
    camera_infos: Mapping[str, Any],
    *,
    model_index: int,
    edit: str,
    alpha_threshold: float,
) -> dict[str, np.ndarray]:
    from motion_proj.resim.drivestudio_adapter import gsplat_first_hit_from_info
    from scripts.run_dr_v2_m4_pilot import move_actor_local_y

    rigid = trainer.models["RigidNodes"]
    translations = rigid.instances_trans.detach().clone()
    visibility = rigid.instances_fv.detach().clone()
    try:
        if edit == "lateral":
            move_actor_local_y(rigid, model_index, 1.0)
        elif edit == "delete":
            with torch.no_grad():
                rigid.instances_fv[:, model_index] = False
        elif edit != "original":
            raise ValueError(f"unknown A3 heldout edit: {edit}")
        with torch.inference_mode():
            outputs = trainer(image_infos, camera_infos)
        first_hit, first_hit_valid = gsplat_first_hit_from_info(
            trainer.info, alpha_threshold=alpha_threshold
        )
        return {
            "rgb": uint8_rgb(outputs["rgb"]),
            "first_hit": np.asarray(first_hit, dtype=np.float32),
            "first_hit_valid": np.asarray(first_hit_valid, dtype=np.bool_),
        }
    finally:
        with torch.no_grad():
            rigid.instances_trans.copy_(translations)
            rigid.instances_fv.copy_(visibility)


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-protocol", type=Path, default=EVAL_PROTOCOL)
    parser.add_argument("--main-protocol", type=Path, default=MAIN_PROTOCOL)
    parser.add_argument("--drivestudio-root", type=Path, default=DRIVESTUDIO)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    from motion_proj.dynamic_editing_v2.pilot_metrics import (
        counterfactual_effect_mask,
    )
    from motion_proj.worldsim_v3.local_refinement import (
        affected_pixel_mask,
        measured_background_support_mask,
        validate_a3_protocol,
        validate_a3_r1_eval_protocol,
    )

    eval_protocol = OmegaConf.to_container(
        OmegaConf.load(args.eval_protocol), resolve=True
    )
    main_protocol = OmegaConf.to_container(
        OmegaConf.load(args.main_protocol), resolve=True
    )
    validate_a3_r1_eval_protocol(eval_protocol)
    validate_a3_protocol(main_protocol)
    dependencies = eval_protocol["depends_on"]
    r0_checkpoint = Path(dependencies["r0_checkpoint"])
    r1_checkpoint = Path(dependencies["r1_checkpoint"])
    source_config = r0_checkpoint.parent / "config.yaml"
    registry = Path(main_protocol["depends_on"]["selected_actor_registry"])
    input_hashes = {
        "eval_protocol": sha256_file(args.eval_protocol),
        "main_protocol": sha256_file(args.main_protocol),
        "numeric_freeze": sha256_file(
            PROJECT / dependencies["numeric_freeze"]
        ),
        "r0_checkpoint": sha256_file(r0_checkpoint),
        "r1_checkpoint": sha256_file(r1_checkpoint),
        "source_config": sha256_file(source_config),
        "actor_registry": sha256_file(registry),
    }
    if (
        input_hashes["main_protocol"] != dependencies["main_protocol_sha256"]
        or input_hashes["numeric_freeze"] != dependencies["numeric_freeze_sha256"]
        or input_hashes["r0_checkpoint"] != dependencies["r0_checkpoint_sha256"]
        or input_hashes["r1_checkpoint"] != dependencies["r1_checkpoint_sha256"]
    ):
        raise RuntimeError(f"A3 heldout immutable input drift: {input_hashes!r}")
    if (
        input_hashes["source_config"]
        != main_protocol["depends_on"]["selected_checkpoint_config_sha256"]
        or input_hashes["actor_registry"]
        != main_protocol["depends_on"]["selected_actor_registry_sha256"]
    ):
        raise RuntimeError("A3 heldout config/registry drift")

    args.run_dir.mkdir(parents=True)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "source_snapshot"):
        (args.run_dir / name).mkdir()
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "failure": None},
    )
    started = time.monotonic()
    source_files = (
        PROJECT / "scripts/eval_worldsim_v3_a3_r1_heldout.py",
        PROJECT / "motion_proj/worldsim_v3/local_refinement.py",
        args.eval_protocol,
        args.main_protocol,
    )
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)
    run_manifest = {
        "schema_version": 1,
        "task_id": eval_protocol["task_id"],
        "component": "A3 R1 heldout read-only evaluation",
        "formal_training_authorized": False,
        "quality_claim_authorized": False,
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": command_output("git", "status", "--short", cwd=PROJECT).splitlines(),
    }
    atomic_json(args.run_dir / "manifest.json", run_manifest)

    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(eval_protocol["matrix"]["seed"]))
    torch.cuda.manual_seed_all(int(eval_protocol["matrix"]["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    memory_events_before = cgroup_memory_events()
    memory_samples = [cgroup_memory_current()]
    config = OmegaConf.load(source_config)
    dataset = DrivingDataset(data_cfg=config.data)
    trainer = import_str(config.trainer.type)(
        **config.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=config.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    if hasattr(trainer, "optimizer"):
        raise RuntimeError("A3 read-only evaluator constructed an optimizer early")
    trainer.resume_from_checkpoint(str(r0_checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    rigid_sha_before = rigid_contract_sha256(rigid)
    actor_specs = main_protocol["paired_design"]["actors"]
    heldout_frames = eval_protocol["matrix"]["heldout_frames"]
    cameras = eval_protocol["matrix"]["cameras"]
    alpha_threshold = float(
        eval_protocol["typed_depth"]["depth_surface_first_hit"]["alpha_threshold"]
    )
    tolerance = float(eval_protocol["typed_depth"]["depth_order_tolerance_m"])
    global_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    r0_original: dict[tuple[int, int], np.ndarray] = {}

    print("A3 heldout phase: R0 original cache", flush=True)
    for frame in heldout_frames:
        for camera in cameras:
            image_infos, camera_infos, gt, _, egocar, image_index = get_view_data(
                dataset, frame, camera, device
            )
            rendered = render_edit(
                trainer,
                image_infos,
                camera_infos,
                model_index=0,
                edit="original",
                alpha_threshold=alpha_threshold,
            )
            r0_original[(frame, camera)] = rendered["rgb"]
            squared, elements = squared_rgb_error(
                rendered["rgb"], gt, ~egocar
            )
            global_rows.append(
                {
                    "frame": frame,
                    "camera": camera,
                    "image_index": image_index,
                    "variant": "r0-no-refine-exact-alias",
                    "squared_uint8_error": squared,
                    "channel_elements": elements,
                }
            )

    print("A3 heldout phase: R0 fixed masks and edited metrics", flush=True)
    for role in eval_protocol["matrix"]["actor_roles"]:
        model_index = int(actor_specs[role]["rigid_model_index"])
        valid_frames = {
            int(index)
            for index in torch.where(rigid.instances_fv[:, model_index].bool())[0]
            .cpu()
            .tolist()
        }
        for frame in heldout_frames:
            if frame not in valid_frames:
                continue
            for camera in cameras:
                image_infos, camera_infos, gt, measured, egocar, image_index = get_view_data(
                    dataset, frame, camera, device
                )
                deleted = render_edit(
                    trainer,
                    image_infos,
                    camera_infos,
                    model_index=model_index,
                    edit="delete",
                    alpha_threshold=alpha_threshold,
                )
                lateral = render_edit(
                    trainer,
                    image_infos,
                    camera_infos,
                    model_index=model_index,
                    edit="lateral",
                    alpha_threshold=alpha_threshold,
                )
                original_rgb = r0_original[(frame, camera)]
                source = counterfactual_effect_mask(
                    original_rgb,
                    deleted["rgb"],
                    threshold_uint8=2,
                    dilation_radius=2,
                )
                edited = counterfactual_effect_mask(
                    lateral["rgb"],
                    deleted["rgb"],
                    threshold_uint8=2,
                    dilation_radius=2,
                )
                for edit, rendered in (("lateral", lateral), ("delete", deleted)):
                    edit_footprint = edited if edit == "lateral" else np.zeros_like(source)
                    affected = affected_pixel_mask(
                        source, edit_footprint, dilation_radius=3
                    )
                    s_b = measured_background_support_mask(
                        affected_mask=affected,
                        source_actor_footprint=source,
                        edited_actor_footprint=edit_footprint,
                        depth_lidar_measured=measured,
                    )
                    non_target = ~affected & ~egocar
                    t1_valid = rendered["first_hit_valid"] & s_b
                    violations = s_b & (
                        ~rendered["first_hit_valid"]
                        | (np.abs(rendered["first_hit"] - measured) > tolerance)
                    )
                    squared, elements = squared_rgb_error(
                        rendered["rgb"], gt, non_target
                    )
                    row = {
                        "role": role,
                        "edit": edit,
                        "frame": frame,
                        "camera": camera,
                        "image_index": image_index,
                        "variant": "r0-no-refine-exact-alias",
                        "source_footprint_pixels": int(source.sum()),
                        "edited_footprint_pixels": int(edit_footprint.sum()),
                        "affected_pixels": int(affected.sum()),
                        "s_b_t0_pixels": int(s_b.sum()),
                        "t1_valid_pixels": int(t1_valid.sum()),
                        "depth_order_violations": int(violations.sum()),
                        "common_t1_valid_pixels": 0,
                        "t0_abs_error_sum_m": 0.0,
                        "non_target_squared_uint8_error": squared,
                        "non_target_channel_elements": elements,
                    }
                    metric_rows.append(row)
                    units.append(
                        {
                            "key": (role, edit, frame, camera),
                            "model_index": model_index,
                            "s_b": s_b,
                            "non_target": non_target,
                            "r0_first_hit": rendered["first_hit"],
                            "r0_first_hit_valid": rendered["first_hit_valid"],
                            "r0_row": row,
                        }
                    )
        memory_samples.append(cgroup_memory_current())

    if rigid_contract_sha256(rigid) != rigid_sha_before:
        raise RuntimeError("A3 R0 actor edit restore drift")
    print("A3 heldout phase: R1 original and edited metrics", flush=True)
    trainer.resume_from_checkpoint(str(r1_checkpoint), load_only_model=True)
    trainer.set_eval()
    if rigid_contract_sha256(rigid) != rigid_sha_before:
        raise RuntimeError("A3 R1 Rigid checkpoint drift")
    for frame in heldout_frames:
        for camera in cameras:
            image_infos, camera_infos, gt, _, egocar, image_index = get_view_data(
                dataset, frame, camera, device
            )
            rendered = render_edit(
                trainer,
                image_infos,
                camera_infos,
                model_index=0,
                edit="original",
                alpha_threshold=alpha_threshold,
            )
            squared, elements = squared_rgb_error(rendered["rgb"], gt, ~egocar)
            global_rows.append(
                {
                    "frame": frame,
                    "camera": camera,
                    "image_index": image_index,
                    "variant": "r1-reactivate",
                    "squared_uint8_error": squared,
                    "channel_elements": elements,
                }
            )

    unit_lookup = {unit["key"]: unit for unit in units}
    for role in eval_protocol["matrix"]["actor_roles"]:
        model_index = int(actor_specs[role]["rigid_model_index"])
        valid_frames = sorted(
            {
                key[2]
                for key in unit_lookup
                if key[0] == role
            }
        )
        for frame in valid_frames:
            for camera in cameras:
                image_infos, camera_infos, gt, measured, _, image_index = get_view_data(
                    dataset, frame, camera, device
                )
                for edit in eval_protocol["matrix"]["edits"]:
                    unit = unit_lookup[(role, edit, frame, camera)]
                    rendered = render_edit(
                        trainer,
                        image_infos,
                        camera_infos,
                        model_index=model_index,
                        edit=edit,
                        alpha_threshold=alpha_threshold,
                    )
                    s_b = unit["s_b"]
                    t1_valid = rendered["first_hit_valid"] & s_b
                    violations = s_b & (
                        ~rendered["first_hit_valid"]
                        | (np.abs(rendered["first_hit"] - measured) > tolerance)
                    )
                    common = (
                        s_b
                        & unit["r0_first_hit_valid"]
                        & rendered["first_hit_valid"]
                    )
                    unit["r0_row"]["common_t1_valid_pixels"] = int(common.sum())
                    unit["r0_row"]["t0_abs_error_sum_m"] = float(
                        np.abs(unit["r0_first_hit"] - measured)[common].sum(
                            dtype=np.float64
                        )
                    )
                    squared, elements = squared_rgb_error(
                        rendered["rgb"], gt, unit["non_target"]
                    )
                    metric_rows.append(
                        {
                            "role": role,
                            "edit": edit,
                            "frame": frame,
                            "camera": camera,
                            "image_index": image_index,
                            "variant": "r1-reactivate",
                            "source_footprint_pixels": unit["r0_row"]["source_footprint_pixels"],
                            "edited_footprint_pixels": unit["r0_row"]["edited_footprint_pixels"],
                            "affected_pixels": unit["r0_row"]["affected_pixels"],
                            "s_b_t0_pixels": int(s_b.sum()),
                            "t1_valid_pixels": int(t1_valid.sum()),
                            "depth_order_violations": int(violations.sum()),
                            "common_t1_valid_pixels": int(common.sum()),
                            "t0_abs_error_sum_m": float(
                                np.abs(rendered["first_hit"] - measured)[common].sum(
                                    dtype=np.float64
                                )
                            ),
                            "non_target_squared_uint8_error": squared,
                            "non_target_channel_elements": elements,
                        }
                    )
        memory_samples.append(cgroup_memory_current())

    if rigid_contract_sha256(rigid) != rigid_sha_before:
        raise RuntimeError("A3 R1 actor edit restore drift")
    if hasattr(trainer, "optimizer"):
        raise RuntimeError("A3 read-only evaluator constructed an optimizer")
    metric_rows.sort(
        key=lambda row: (
            eval_protocol["matrix"]["actor_roles"].index(row["role"]),
            eval_protocol["matrix"]["edits"].index(row["edit"]),
            row["frame"],
            row["camera"],
            eval_protocol["matrix"]["variants"].index(row["variant"]),
        )
    )
    global_rows.sort(
        key=lambda row: (
            row["frame"],
            row["camera"],
            eval_protocol["matrix"]["variants"].index(row["variant"]),
        )
    )
    raw_path = args.run_dir / "artifacts" / "heldout_metric_rows.jsonl"
    global_path = args.run_dir / "artifacts" / "heldout_global_rows.jsonl"
    write_jsonl(raw_path, metric_rows)
    write_jsonl(global_path, global_rows)
    group_order = eval_protocol["aggregation"]["primary_group_order"]
    aggregates = {
        variant: build_variant_aggregate(
            metric_rows,
            global_rows,
            variant=variant,
            group_order=group_order,
        )
        for variant in eval_protocol["matrix"]["variants"]
    }
    recomputed = {
        variant: build_variant_aggregate(
            read_jsonl(raw_path),
            read_jsonl(global_path),
            variant=variant,
            group_order=group_order,
        )
        for variant in eval_protocol["matrix"]["variants"]
    }
    if aggregates != recomputed:
        raise RuntimeError("A3 raw-row aggregate recompute drift")
    r0_name, r1_name = eval_protocol["matrix"]["variants"]
    decision = classify_exact_pareto(
        aggregates[r0_name]["primary_axes"],
        aggregates[r1_name]["primary_axes"],
        eval_protocol["decision"]["primary_axes"],
    )
    if all(
        group["s_b_t0_pixels"] == 0
        for group in aggregates[r0_name]["groups"].values()
    ):
        decision["classification"] = "insufficient_evidence"

    checkpoints_after = {
        "r0": sha256_file(r0_checkpoint),
        "r1": sha256_file(r1_checkpoint),
    }
    audits = {
        "no_optimizer_constructed_or_step_executed": not hasattr(trainer, "optimizer"),
        "no_checkpoint_written": not any(args.run_dir.rglob("*.pth")),
        "r0_checkpoint_sha_before_after_exact": checkpoints_after["r0"]
        == input_hashes["r0_checkpoint"],
        "r1_checkpoint_sha_before_after_exact": checkpoints_after["r1"]
        == input_hashes["r1_checkpoint"],
        "actor_registry_trajectory_parameters_exact": rigid_contract_sha256(rigid)
        == rigid_sha_before,
        "fixed_masks_derived_from_r0_only": True,
        "heldout_frames_only": all(
            row["frame"] in heldout_frames for row in metric_rows + global_rows
        ),
        "raw_rows_and_aggregate_recompute_match": aggregates == recomputed,
        "finite_metrics_or_explicit_null": all(
            value is None or np.isfinite(value)
            for aggregate in aggregates.values()
            for value in aggregate["group_macro"].values()
        ),
    }
    if not all(audits.values()):
        raise RuntimeError(f"A3 heldout required audit failed: {audits}")
    duration = time.monotonic() - started
    peak_gpu = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    events_after = cgroup_memory_events()
    oom_delta = events_after.get("oom", 0) - memory_events_before.get("oom", 0)
    oom_kill_delta = events_after.get("oom_kill", 0) - memory_events_before.get("oom_kill", 0)
    ceilings = eval_protocol["resource_ceilings"]
    resource_audit = build_resource_audit(
        duration=duration,
        peak_gpu_mib=peak_gpu,
        cgroup_samples=memory_samples,
        run_bytes=directory_bytes(args.run_dir),
        oom_delta=oom_delta,
        oom_kill_delta=oom_kill_delta,
        ceilings=ceilings,
    )
    resource_audit_path = args.run_dir / "artifacts" / "resource_audit.json"
    atomic_json(resource_audit_path, resource_audit)
    if resource_audit["status"] != "passed":
        failed = sorted(
            key for key, value in resource_audit["violations"].items() if value
        )
        raise RuntimeError(
            "A3 heldout resource ceiling failed: " + ", ".join(failed)
        )
    summary = {
        "status": "done",
        "task_id": eval_protocol["task_id"],
        "component": "A3 R1 heldout read-only evaluation",
        "formal_training_authorized": False,
        "quality_claim_authorized": False,
        "claim_boundary": eval_protocol["claim_boundary"],
        "input_hashes": input_hashes,
        "raw_rows": {
            "path": str(raw_path.relative_to(args.run_dir)),
            "sha256": sha256_file(raw_path),
            "count": len(metric_rows),
        },
        "global_rows": {
            "path": str(global_path.relative_to(args.run_dir)),
            "sha256": sha256_file(global_path),
            "count": len(global_rows),
        },
        "aggregates": aggregates,
        "decision": decision,
        "audits": audits,
        "resources": {
            **resource_audit["measured"],
            "audit_path": str(resource_audit_path.relative_to(args.run_dir)),
            "audit_sha256": sha256_file(resource_audit_path),
        },
        "project_commit": run_manifest["project_commit"],
    }
    atomic_json(args.run_dir / "summary.json", summary)
    run_manifest["status"] = "done"
    run_manifest["summary_sha256"] = sha256_file(args.run_dir / "summary.json")
    atomic_json(args.run_dir / "manifest.json", run_manifest, replace=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "failure": None},
        replace=True,
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and not _TERMINAL_FINAL:
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A3_R1_HELDOUT_EVAL_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
