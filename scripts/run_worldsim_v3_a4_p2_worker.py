#!/usr/bin/env python
"""执行 A4-P2 dtype 审计、候选物化、质量评估与运行时画像。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as torch_f


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml"
ARMS = ("p2-source", "p2-gs-param-fp16")
_ACTIVE_RUN_DIR: Path | None = None


from motion_proj.worldsim_v3.actor_metrics import boundary_band
from motion_proj.worldsim_v3.contribution_prune import (
    build_candidate_registry,
    compare_metric_group,
)
from motion_proj.worldsim_v3.mixed_precision import (
    RENDER_PARAMETER_FIELDS,
    apply_runtime_parameter_dtypes,
    conversion_audit,
    convert_checkpoint_state,
    install_fp32_renderer_input_adapter,
    persistent_parameter_inventory,
    renderer_adapter_summary,
    runtime_converted_field_audit,
    set_fp32_renderer_adapter_mode,
)
from scripts.eval_worldsim_v3_a0_actor_metrics import (
    RegionAccumulator,
    masked_lpips,
    ssim_map,
    to_device,
    uint8_rgb,
)
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data,
    load_model_checkpoint_read_only,
    release_trainer_render_info,
)
from scripts.run_worldsim_v3_a4_p0_profile import ResourceSampler, nearest_rank, rgb_sha256
from scripts.run_worldsim_v3_a4_p1_worker import (
    build_runtime,
    runtime_model_counts,
    runtime_point_ids,
)
from scripts.run_worldsim_v3_a4_p2_precision import (
    atomic_json,
    directory_digest,
    load_stage,
    nvidia_compute_rows,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p2_mixed_precision_protocol import (
    validate_checkpoint_state,
    validate_inputs,
    validate_schema,
)


def source_inputs_unchanged(protocol: Mapping[str, Any]) -> bool:
    selected = protocol["selected_asset"]
    return all(
        sha256_file(Path(selected[name]["path"])) == selected[name]["sha256"]
        for name in ("checkpoint", "source_config", "actor_registry")
    )


def atomic_torch_save(path: Path, value: Any) -> None:
    """一次性原子写入候选 checkpoint，禁止覆盖。"""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def stage_resource_summary(device: torch.device, sampler: ResourceSampler) -> dict[str, Any]:
    sampled = sampler.summary()
    if (
        sampled["sampling_errors"]
        or sampled["peak_nvidia_process_memory_mib_sampled"] is None
        or sampled["peak_cgroup_memory_bytes_sampled"] is None
    ):
        raise RuntimeError(f"A4-P2 resource sampling incomplete: {sampled}")
    return {
        "peak_torch_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "peak_torch_reserved_mib": float(
            torch.cuda.max_memory_reserved(device) / (1024**2)
        ),
        **sampled,
    }


def expected_model_counts(protocol: Mapping[str, Any]) -> dict[str, int]:
    inventory = protocol["selected_asset"]["inventory"]
    return {
        "Background": int(inventory["background_gaussians"]),
        "RigidNodes": int(inventory["rigid_gaussians"]),
    }


def actor_registry_exact(trainer: Any, registry_path: Path) -> bool:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    point_ids = runtime_point_ids(trainer)
    return all(
        int((point_ids == int(actor["rigid_model_index"])).sum())
        == int(actor["checkpoint_tensor_slice"]["gaussian_count"])
        for actor in registry["actors"]
    )


def arm_assets(
    run_dir: Path, protocol: Mapping[str, Any], manifest: Mapping[str, Any], arm: str
) -> tuple[Path, Path]:
    if arm == "p2-source":
        return (
            Path(protocol["selected_asset"]["checkpoint"]["path"]),
            Path(protocol["selected_asset"]["actor_registry"]["path"]),
        )
    if arm != "p2-gs-param-fp16":
        raise ValueError(f"unknown A4-P2 arm: {arm}")
    stage = load_stage(run_dir, manifest, "materialize_p2_gs_param_fp16")
    return (
        run_dir / stage["candidate_checkpoint"]["path"],
        run_dir / stage["candidate_registry"]["path"],
    )


def run_source_dtype_audit(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any]
) -> None:
    load_stage(run_dir, manifest, "input_audit")
    started = time.perf_counter()
    source_path = Path(protocol["selected_asset"]["checkpoint"]["path"])
    source_before = sha256_file(source_path)
    state = torch.load(source_path, map_location="cpu")
    validator_audit = validate_checkpoint_state(protocol, source_path)
    precision = protocol["precision_contract"]
    rows = {}
    for model_name in precision["converted_models"]:
        model = state["models"][model_name]
        fields = list(precision["converted_fields"]) + list(
            precision["preserved_float32_fields"][model_name]
        )
        for field in fields:
            tensor = model[field]
            rows[f"models.{model_name}.{field}"] = {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "bytes": int(tensor.numel() * tensor.element_size()),
                "finite": bool(torch.isfinite(tensor).all().item()),
            }
    expected_fp32 = set(
        f"models.{model}.{field}"
        for model in precision["converted_models"]
        for field in list(precision["converted_fields"])
        + list(precision["preserved_float32_fields"][model])
    )
    stage = {
        "status": "done",
        "stage": "source_dtype_audit",
        "duration_seconds": time.perf_counter() - started,
        "checkpoint": {
            "path": str(source_path),
            "sha256": source_before,
            "bytes": source_path.stat().st_size,
        },
        "validator_checkpoint_state_audit": validator_audit,
        "source_dtype_rows": rows,
        "source_checkpoint_dtype_schema_exact": set(rows) == expected_fp32
        and all(row["dtype"] == "float32" and row["finite"] for row in rows.values()),
        "source_inputs_unchanged": source_inputs_unchanged(protocol)
        and sha256_file(source_path) == source_before,
        "training_optimizer_or_source_mutation_performed": False,
        "minimum_rerun_unit": "source_dtype_audit_and_downstream",
    }
    del state
    write_stage(run_dir, manifest, "source_dtype_audit", stage)


def _registry_slice_rows(registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(actor["rigid_model_index"]): {
            "availability": actor["availability"],
            "checkpoint_tensor_slice": actor["checkpoint_tensor_slice"],
        }
        for actor in registry["actors"]
    }


def run_materialize(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any]
) -> None:
    load_stage(run_dir, manifest, "source_dtype_audit")
    started = time.perf_counter()
    source_path = Path(protocol["selected_asset"]["checkpoint"]["path"])
    source_before = sha256_file(source_path)
    source = torch.load(source_path, map_location="cpu")
    precision = protocol["precision_contract"]
    candidate = convert_checkpoint_state(
        source,
        models=precision["converted_models"],
        fields=precision["converted_fields"],
    )
    before_save_audit = conversion_audit(
        source,
        candidate,
        models=precision["converted_models"],
        fields=precision["converted_fields"],
    )
    checkpoint_path = run_dir / precision["candidate_checkpoint_output"]
    atomic_torch_save(checkpoint_path, candidate)
    checkpoint_record = {
        "path": str(checkpoint_path.relative_to(run_dir)),
        "sha256": sha256_file(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }

    source_registry = json.loads(
        Path(protocol["selected_asset"]["actor_registry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    point_ids = source["models"]["RigidNodes"]["points_ids"][:, 0].tolist()
    candidate_registry = build_candidate_registry(
        source_registry, point_ids, checkpoint_record["sha256"]
    )
    registry_path = run_dir / precision["candidate_registry_output"]
    atomic_json(registry_path, candidate_registry)
    registry_record = {
        "path": str(registry_path.relative_to(run_dir)),
        "sha256": sha256_file(registry_path),
        "bytes": registry_path.stat().st_size,
        "embedded_registry_sha256": candidate_registry["actor_registry_sha256"],
    }

    reloaded = torch.load(checkpoint_path, map_location="cpu")
    reload_audit = conversion_audit(
        source,
        reloaded,
        models=precision["converted_models"],
        fields=precision["converted_fields"],
    )
    registry_exact = (
        _registry_slice_rows(candidate_registry) == _registry_slice_rows(source_registry)
        and candidate_registry["actor_count"] == source_registry["actor_count"]
        and candidate_registry["available_actor_count"]
        == source_registry["available_actor_count"]
        and candidate_registry["empty_checkpoint_actor_count"]
        == source_registry["empty_checkpoint_actor_count"]
    )
    converted_exact = all(
        audit["converted_field_set_exact"]
        and audit["all_converted_fields_bitwise_exact"]
        and audit["preserved_tensors_exact"]
        and audit["checkpoint_schema_exact"]
        for audit in (before_save_audit, reload_audit)
    )
    stage = {
        "status": "done",
        "stage": "materialize_p2_gs_param_fp16",
        "duration_seconds": time.perf_counter() - started,
        "candidate_checkpoint": checkpoint_record,
        "candidate_registry": registry_record,
        "before_save_conversion_audit": before_save_audit,
        "reloaded_conversion_audit": reload_audit,
        "converted_field_set_and_float16_bytes_exact": converted_exact,
        "converted_field_roundtrip_error_complete_and_finite": bool(
            before_save_audit["all_converted_fields_bitwise_exact"]
            and reload_audit["all_converted_fields_bitwise_exact"]
        ),
        "preserved_fields_and_checkpoint_schema_exact": before_save_audit[
            "preserved_tensors_exact"
        ]
        and before_save_audit["checkpoint_schema_exact"]
        and reload_audit["preserved_tensors_exact"]
        and reload_audit["checkpoint_schema_exact"],
        "candidate_registry_counts_and_indices_exact": registry_exact,
        "checkpoint_bytes_strictly_less_than_source": checkpoint_path.stat().st_size
        < source_path.stat().st_size,
        "candidate_checkpoint_write_count": 1,
        "source_checkpoint_copied": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol)
        and sha256_file(source_path) == source_before,
        "training_optimizer_or_source_mutation_performed": False,
        "minimum_rerun_unit": "materialize_p2_gs_param_fp16_and_downstream",
    }
    del source, candidate, reloaded
    write_stage(run_dir, manifest, "materialize_p2_gs_param_fp16", stage)


def mean_or_minus_one(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else -1.0


def torch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((-10 * torch.log10(torch_f.mse_loss(prediction, target))).item())


def finite_metric_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_metric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_metric_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def evaluate_loaded_arm(
    *,
    protocol: Mapping[str, Any],
    dataset: Any,
    trainer: Any,
    arm: str,
    checkpoint: Path,
    registry_path: Path,
    checkpoint_load_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    """对已加载 arm 完整渲染 held-out split，并复用冻结 mask 计算区域指标。"""
    trainer.set_eval()
    if hasattr(trainer.lpips, "reset"):
        trainer.lpips.reset()
    model_counts = runtime_model_counts(trainer)
    registry_exact = actor_registry_exact(trainer, registry_path)
    point_ids = runtime_point_ids(trainer)
    quality = protocol["quality_contract"]
    test_indices = [int(value) for value in dataset.test_image_set.split_indices]
    expected_indices = [
        int(frame) * 3 + int(camera)
        for frame in quality["heldout_frames"]
        for camera in quality["cameras"]
    ]
    if test_indices != expected_indices:
        raise RuntimeError("A4-P2 heldout quality split drift")

    metric_lists: dict[str, list[float]] = {
        name: []
        for name in (
            "psnr",
            "ssim",
            "lpips",
            "occupied_psnr",
            "occupied_ssim",
            "masked_psnr",
            "masked_ssim",
            "human_psnr",
            "human_ssim",
            "vehicle_psnr",
            "vehicle_ssim",
        )
    }
    prediction_cache: dict[int, np.ndarray] = {}
    target_cache: dict[int, np.ndarray] = {}
    for position in range(len(test_indices)):
        image_infos, camera_infos = dataset.test_image_set.get_image(
            position, camera_downscale=trainer._get_downscale_factor()
        )
        target_cpu = image_infos["pixels"].detach().float().cpu().numpy()
        image_infos = to_device(image_infos, device)
        camera_infos = to_device(camera_infos, device)
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                output = trainer(image_infos, camera_infos)
            raw_rgb = output["rgb"]
            rgb = raw_rgb.clamp(0.0, 1.0)
            target = image_infos["pixels"]
            prediction = rgb.detach().float().cpu().numpy()
            metric_lists["psnr"].append(torch_psnr(rgb, target))
            metric_lists["ssim"].append(
                float(structural_similarity(target_cpu, prediction, data_range=1.0, channel_axis=-1))
            )
            metric_lists["lpips"].append(
                float(
                    trainer.lpips(
                        rgb[None].permute(0, 3, 1, 2),
                        target[None].permute(0, 3, 1, 2),
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            )
            for prefix, key, invert in (
                ("occupied", "sky_masks", True),
                ("masked", "dynamic_masks", False),
                ("human", "human_masks", False),
                ("vehicle", "vehicle_masks", False),
            ):
                if key not in image_infos:
                    continue
                selected = image_infos[key].detach().bool()
                if invert:
                    selected = ~selected
                selected_np = selected.cpu().numpy().squeeze().astype(bool)
                if int(selected_np.sum()) == 0:
                    continue
                metric_lists[f"{prefix}_psnr"].append(torch_psnr(rgb[selected], target[selected]))
                full_ssim = structural_similarity(
                    target_cpu,
                    prediction,
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1]
                metric_lists[f"{prefix}_ssim"].append(
                    float(np.asarray(full_ssim)[selected_np].mean())
                )
            prediction_cache[position] = uint8_rgb(raw_rgb.detach().float().cpu().numpy())
            target_cache[position] = target_cpu
        finally:
            release_trainer_render_info(trainer)
        print(f"A4-P2 quality arm={arm} render={position + 1}/{len(test_indices)}", flush=True)

    global_metrics = {
        f"image_metrics/test/{name}": mean_or_minus_one(values)
        for name, values in metric_lists.items()
    }
    if hasattr(trainer.lpips, "reset"):
        trainer.lpips.reset()
    frozen_source = json.loads(
        Path(protocol["baseline_quality"]["source_quality"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    masks_dir = Path(protocol["baseline_quality"]["actor_masks"]["path"])
    role_accumulators = {
        role: {"actor_region": RegionAccumulator(), "boundary_band": RegionAccumulator()}
        for role in quality["actor_endpoints"]["roles"]
    }
    union_masks = {
        position: np.zeros(target.shape[:2], dtype=bool)
        for position, target in target_cache.items()
    }
    frozen_mask_files = 0
    for role, regions in role_accumulators.items():
        for position, full_index in enumerate(test_indices):
            frame, camera = divmod(full_index, 3)
            mask_path = masks_dir / f"{role}__frame_{frame:03d}_camera_{camera}.png"
            if not mask_path.is_file():
                continue
            effect = np.asarray(imageio.imread(mask_path)) > 0
            frozen_mask_files += 1
            union_masks[position] |= effect
            boundary = boundary_band(effect, radius=3)
            prediction = prediction_cache[position].astype(np.float32) / 255.0
            target = target_cache[position]
            image_ssim = ssim_map(prediction, target)
            for region_name, selected in (("actor_region", effect), ("boundary_band", boundary)):
                lpips_value = masked_lpips(trainer.lpips, prediction, target, selected, device)
                regions[region_name].update(
                    prediction, target, selected, image_ssim, lpips_value
                )
    role_metrics = {
        role: {
            "actor": frozen_source["roles"][role]["actor"],
            **{name: accumulator.summary() for name, accumulator in regions.items()},
        }
        for role, regions in role_accumulators.items()
    }
    non_target = RegionAccumulator()
    for position in range(len(test_indices)):
        selected = ~union_masks[position]
        prediction = prediction_cache[position].astype(np.float32) / 255.0
        target = target_cache[position]
        lpips_value = masked_lpips(
            trainer.lpips, prediction, target, selected, device
        )
        non_target.update(
            prediction,
            target,
            selected,
            ssim_map(prediction, target),
            lpips_value,
        )
    non_target_quality = non_target.summary()
    result = {
        "arm": arm,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "checkpoint_reload_exact": registry_exact,
        "model_gaussian_counts": model_counts,
        "actor_counts_exact": registry_exact,
        "unavailable_actor_remains_explicitly_empty": all(
            int((point_ids == int(actor["rigid_model_index"])).sum()) == 0
            for actor in json.loads(registry_path.read_text(encoding="utf-8"))["actors"]
            if actor["availability"] == "unavailable_empty_checkpoint_slice"
        ),
        "heldout_image_count": len(test_indices),
        "heldout_full_image_indices": test_indices,
        "global_metrics": global_metrics,
        "roles": role_metrics,
        "non_target": {"quality": non_target_quality},
        "frozen_mask_file_visits": frozen_mask_files,
        "all_endpoints_complete_and_finite": finite_metric_tree(global_metrics)
        and all(
            region["status"] == "done" and finite_metric_tree(region)
            for role in role_metrics.values()
            for name, region in role.items()
            if name in {"actor_region", "boundary_band"}
        )
        and non_target_quality["status"] == "done"
        and finite_metric_tree(non_target_quality),
        "raw_render_media_written": False,
    }
    del prediction_cache, target_cache
    return result


def _metric_tolerance(tolerances: Mapping[str, Any], name: str) -> float:
    if "lpips" in name:
        return float(tolerances["lpips_absolute"])
    if "ssim" in name:
        return float(tolerances["ssim_absolute"])
    if "psnr" in name:
        return float(tolerances["psnr_absolute"])
    return float(tolerances["mean_absolute_error_absolute"])


def baseline_replay_rows(
    protocol: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    frozen = json.loads(
        Path(protocol["baseline_quality"]["source_quality"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    tolerances = protocol["quality_contract"]["baseline_p1_replay_tolerance"]
    triples = [
        (f"global.{name}", value, result["global_metrics"].get(name))
        for name, value in frozen["global_metrics"].items()
    ]
    for role in protocol["quality_contract"]["actor_endpoints"]["roles"]:
        for region in protocol["quality_contract"]["actor_endpoints"]["regions"]:
            for name in (
                "psnr",
                "ssim",
                "masked_lpips_alex_tight_crop_256px",
                "mean_absolute_error",
            ):
                triples.append(
                    (
                        f"roles.{role}.{region}.{name}",
                        frozen["roles"][role][region][name],
                        result["roles"][role][region].get(name),
                    )
                )
    for name in (
        "psnr",
        "ssim",
        "masked_lpips_alex_tight_crop_256px",
        "mean_absolute_error",
    ):
        triples.append(
            (
                f"non_target.{name}",
                frozen["non_target"]["quality"][name],
                result["non_target"]["quality"].get(name),
            )
        )
    rows = []
    for endpoint, historical, replay in triples:
        limit = _metric_tolerance(tolerances, endpoint)
        rows.append(
            {
                "endpoint": endpoint,
                "historical": historical,
                "replay": replay,
                "absolute_tolerance": limit,
                "passed": replay is not None
                and math.isfinite(float(replay))
                and abs(float(replay) - float(historical)) <= limit,
            }
        )
    return rows


def quality_safeguard_rows(
    protocol: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    quality = protocol["quality_contract"]
    rows = [
        {"group": "global", **row}
        for row in compare_metric_group(
            baseline["global_metrics"],
            candidate["global_metrics"],
            quality["global_endpoints"]["metrics"],
        )
    ]
    for role in quality["actor_endpoints"]["roles"]:
        for region in quality["actor_endpoints"]["regions"]:
            rows.extend(
                {"group": f"{role}.{region}", **row}
                for row in compare_metric_group(
                    baseline["roles"][role][region],
                    candidate["roles"][role][region],
                    quality["actor_endpoints"]["metrics"],
                )
            )
    rows.extend(
        {"group": "non_target", **row}
        for row in compare_metric_group(
            baseline["non_target"]["quality"],
            candidate["non_target"]["quality"],
            quality["non_target_endpoints"]["metrics"],
        )
    )
    return rows


def load_runtime_arm(
    trainer: Any,
    checkpoint: Path,
    *,
    candidate: bool,
    device: torch.device,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    load_model_checkpoint_read_only(trainer, checkpoint, device)
    apply_runtime_parameter_dtypes(trainer, candidate=candidate)
    set_fp32_renderer_adapter_mode(trainer, candidate=candidate)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    expected_dtype = "float16" if candidate else "float32"
    return (
        load_seconds,
        runtime_converted_field_audit(trainer, expected_dtype=expected_dtype),
        persistent_parameter_inventory(trainer),
    )


def run_evaluate(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    load_stage(run_dir, manifest, "materialize_p2_gs_param_fp16")
    if nvidia_compute_rows():
        raise RuntimeError("A4-P2 quality evaluation GPU preflight not idle")
    masks_contract = protocol["baseline_quality"]["actor_masks"]
    masks_before = directory_digest(Path(masks_contract["path"]), masks_contract["file_glob"])
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with ResourceSampler(os.getpid()) as sampler:
        _, dataset, trainer = build_runtime(protocol, device)
        install_fp32_renderer_input_adapter(trainer)
        artifacts = {}
        baseline = None
        for arm in ARMS:
            checkpoint, registry_path = arm_assets(run_dir, protocol, manifest, arm)
            is_candidate = arm == "p2-gs-param-fp16"
            load_seconds, dtype_audit, parameter_inventory = load_runtime_arm(
                trainer, checkpoint, candidate=is_candidate, device=device
            )
            result = evaluate_loaded_arm(
                protocol=protocol,
                dataset=dataset,
                trainer=trainer,
                arm=arm,
                checkpoint=checkpoint,
                registry_path=registry_path,
                checkpoint_load_seconds=load_seconds,
                device=device,
            )
            result["expected_model_counts"] = expected_model_counts(protocol)
            result["expected_model_counts_exact"] = (
                result["model_gaussian_counts"] == result["expected_model_counts"]
            )
            result["runtime_converted_field_audit"] = dtype_audit
            result["persistent_parameter_bytes_by_dtype"] = parameter_inventory
            result["renderer_input_dtypes"] = renderer_adapter_summary(trainer)
            if arm == "p2-source":
                replay = baseline_replay_rows(protocol, result)
                result["baseline_replay_rows"] = replay
                result["baseline_p1_replay_pass"] = len(replay) == 31 and all(
                    row["passed"] for row in replay
                )
                result["quality_safeguard_rows"] = []
                result["all_quality_safeguards_pass"] = result[
                    "baseline_p1_replay_pass"
                ]
                baseline = result
            else:
                if baseline is None:
                    raise RuntimeError("A4-P2 source quality baseline missing")
                safeguards = quality_safeguard_rows(protocol, baseline, result)
                result["quality_safeguard_rows"] = safeguards
                result["all_quality_safeguards_pass"] = len(safeguards) == 31 and all(
                    row["passed"] for row in safeguards
                )
            output_path = run_dir / "artifacts" / "quality" / f"{arm}.json"
            atomic_json(output_path, result)
            artifacts[arm] = {
                "path": str(output_path.relative_to(run_dir)),
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
                "all_quality_safeguards_pass": result["all_quality_safeguards_pass"],
                "all_endpoints_complete_and_finite": result[
                    "all_endpoints_complete_and_finite"
                ],
                "runtime_dtype_exact": dtype_audit["exact"],
                "renderer_inputs_float32": result["renderer_input_dtypes"][
                    "all_renderer_inputs_float32"
                ],
            }
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset
        torch.cuda.empty_cache()
    resources = stage_resource_summary(device, sampler)
    masks_after = directory_digest(Path(masks_contract["path"]), masks_contract["file_glob"])
    expected_mask = {
        "sha256": masks_contract["sha256"],
        "file_count": int(masks_contract["file_count"]),
        "total_bytes": int(masks_contract["total_bytes"]),
    }
    stage = {
        "status": "done",
        "stage": "evaluate_p2_source_and_candidate",
        "duration_seconds": time.perf_counter() - started,
        "arms": list(ARMS),
        "quality_artifacts": artifacts,
        "frozen_masks_before": masks_before,
        "frozen_masks_after": masks_after,
        "frozen_masks_reused_exactly": masks_before == masks_after == expected_mask,
        "candidate_mask_regeneration_performed": False,
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "raw_render_media_written": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "evaluate_p2_source_and_candidate_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "evaluate_p2_source_and_candidate", stage)


def render_runtime_view(
    trainer: Any,
    dataset: Any,
    *,
    frame: int,
    camera: int,
    device: torch.device,
    phase: str,
    ordinal: int,
) -> dict[str, Any]:
    image_infos, camera_infos, _, _, _, image_index = get_view_data(
        dataset, frame, camera, device
    )
    try:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            output = trainer(image_infos, camera_infos)
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        rgb = uint8_rgb(output["rgb"].detach().float().cpu().numpy())
        return {
            "phase": phase,
            "ordinal": ordinal,
            "frame": frame,
            "camera": camera,
            "image_index": image_index,
            "duration_seconds": duration,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "rgb_sha256": rgb_sha256(rgb),
        }
    finally:
        release_trainer_render_info(trainer)


def run_runtime_profile(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    load_stage(run_dir, manifest, "evaluate_p2_source_and_candidate")
    if nvidia_compute_rows():
        raise RuntimeError("A4-P2 runtime profile GPU preflight not idle")
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with ResourceSampler(os.getpid()) as sampler:
        _, dataset, trainer = build_runtime(protocol, device)
        install_fp32_renderer_input_adapter(trainer)
        contract = protocol["runtime_contract"]
        frames = [int(value) for value in contract["frames"]]
        cameras = [int(value) for value in contract["cameras"]]
        matrix = [(frame, camera) for frame in frames for camera in cameras]
        arm_rows = {}
        for arm in ARMS:
            checkpoint, registry_path = arm_assets(run_dir, protocol, manifest, arm)
            is_candidate = arm == "p2-gs-param-fp16"
            torch.cuda.reset_peak_memory_stats(device)
            load_seconds, dtype_audit, parameter_inventory = load_runtime_arm(
                trainer, checkpoint, candidate=is_candidate, device=device
            )
            trainer.set_eval()
            warmup_rows = [
                render_runtime_view(
                    trainer,
                    dataset,
                    frame=frames[0],
                    camera=cameras[0],
                    device=device,
                    phase="warmup",
                    ordinal=ordinal,
                )
                for ordinal in range(int(contract["warmup_views"]))
            ]
            measured_rows = [
                render_runtime_view(
                    trainer,
                    dataset,
                    frame=frame,
                    camera=camera,
                    device=device,
                    phase="measured",
                    ordinal=ordinal,
                )
                for ordinal, (frame, camera) in enumerate(matrix)
            ]
            durations = [row["duration_seconds"] for row in measured_rows]
            counts = runtime_model_counts(trainer)
            adapter = renderer_adapter_summary(trainer)
            arm_rows[arm] = {
                "arm": arm,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_load_seconds": load_seconds,
                "model_gaussian_counts": counts,
                "expected_model_counts": expected_model_counts(protocol),
                "checkpoint_reload_exact": counts == expected_model_counts(protocol)
                and actor_registry_exact(trainer, registry_path),
                "runtime_converted_field_audit": dtype_audit,
                "persistent_parameter_bytes_by_dtype": parameter_inventory,
                "render_input_dtypes": adapter,
                "warmup_rows": warmup_rows,
                "measured_rows": measured_rows,
                "render_p50_seconds": nearest_rank(durations, 0.50),
                "render_p95_seconds": nearest_rank(durations, 0.95),
                "aggregate_fps": len(durations) / sum(durations),
                "peak_torch_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                ),
                "peak_torch_reserved_mib": float(
                    torch.cuda.max_memory_reserved(device) / (1024**2)
                ),
                "matrix_complete_and_unique": len(measured_rows)
                == int(contract["expected_samples_per_arm"])
                and len({(row["frame"], row["camera"]) for row in measured_rows})
                == int(contract["expected_samples_per_arm"]),
                "resolution_exact": all(
                    [row["width"], row["height"]] == contract["resolution"]
                    for row in warmup_rows + measured_rows
                ),
                "synchronized_timing_complete": all(value > 0 for value in durations),
                "filesystem_cache": contract["filesystem_cache"],
            }
            print(f"A4-P2 runtime arm={arm} complete", flush=True)
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset
        torch.cuda.empty_cache()
    resources = stage_resource_summary(device, sampler)
    stage = {
        "status": "done",
        "stage": "runtime_profile_both_arms",
        "duration_seconds": time.perf_counter() - started,
        "arm_rows": arm_rows,
        "matrix": {
            "frames": frames,
            "cameras": cameras,
            "sample_count_per_arm": int(contract["expected_samples_per_arm"]),
        },
        "performance_values_report_only": True,
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "raw_render_media_written": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "runtime_profile_both_arms_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "runtime_profile_both_arms", stage)


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--operation",
        required=True,
        choices=("source-dtype-audit", "materialize", "evaluate", "runtime-profile"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256_file(args.protocol):
        raise RuntimeError("A4-P2 protocol hash drift after initialization")
    device = torch.device(args.device)
    if args.operation == "source-dtype-audit":
        run_source_dtype_audit(args.run_dir, protocol, manifest)
    elif args.operation == "materialize":
        run_materialize(args.run_dir, protocol, manifest)
    elif args.operation == "evaluate":
        run_evaluate(args.run_dir, protocol, manifest, device)
    else:
        run_runtime_profile(args.run_dir, protocol, manifest, device)
    print(json.dumps({"status": "done", "operation": args.operation, "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.exists():
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P2_WORKER_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
