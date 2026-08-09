#!/usr/bin/env python
"""执行 A4-P3 源布局、exact chunk package、质量与运行时阶段。"""

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


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml"
ARMS = ("p3-source", "p3-chunk-package")
_ACTIVE_RUN_DIR: Path | None = None


from motion_proj.worldsim_v3.actor_metrics import boundary_band
from motion_proj.worldsim_v3.chunk_package import (
    compare_checkpoint_states,
    materialize_chunk_package,
    reassemble_chunk_package,
    sha256_file as package_sha256_file,
)
from motion_proj.worldsim_v3.mixed_precision import (
    apply_runtime_parameter_dtypes,
    install_fp32_renderer_input_adapter,
    persistent_parameter_inventory,
    recursive_tensor_rows,
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
from scripts.eval_worldsim_v3_a3_r1_heldout import release_trainer_render_info
from scripts.run_worldsim_v3_a4_p0_profile import ResourceSampler, nearest_rank, rgb_sha256
from scripts.run_worldsim_v3_a4_p1_worker import (
    build_runtime,
    runtime_model_counts,
    runtime_point_ids,
)
from scripts.run_worldsim_v3_a4_p2_worker import (
    actor_registry_exact,
    finite_metric_tree,
    render_runtime_view,
    stage_resource_summary,
    torch_psnr,
)
from scripts.run_worldsim_v3_a4_p3_chunk import (
    atomic_json,
    directory_bytes,
    directory_digest,
    load_stage,
    nvidia_compute_rows,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p3_chunk_protocol import (
    validate_inputs,
    validate_schema,
    validate_source_layout,
)


def source_inputs_unchanged(protocol: Mapping[str, Any]) -> bool:
    """核对 checkpoint/config/registry 三个生产输入仍不可变。"""
    selected = protocol["selected_asset"]
    return all(
        sha256_file(Path(selected[name]["path"])) == selected[name]["sha256"]
        for name in ("checkpoint", "source_config", "actor_registry")
    )


def expected_model_counts(protocol: Mapping[str, Any]) -> dict[str, int]:
    inventory = protocol["selected_asset"]["inventory"]
    return {
        "Background": int(inventory["background_gaussians"]),
        "RigidNodes": int(inventory["rigid_gaussians"]),
    }


def cpu_stage_resource_summary(sampler: ResourceSampler) -> dict[str, Any]:
    """为 CPU-only 阶段补齐统一资源字段。"""
    sampled = sampler.summary()
    if (
        sampled["sampling_errors"]
        or sampled["peak_nvidia_process_memory_mib_sampled"] is None
        or sampled["peak_cgroup_memory_bytes_sampled"] is None
    ):
        raise RuntimeError(f"A4-P3 CPU resource sampling incomplete: {sampled}")
    return {
        "peak_torch_allocated_mib": 0.0,
        "peak_torch_reserved_mib": 0.0,
        **sampled,
    }


def artifact_record(path: Path, run_dir: Path) -> dict[str, Any]:
    """生成 run-local artifact 的相对路径、SHA 与字节数。"""
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """读取并验证 run-local JSON artifact。"""
    path = run_dir / str(record["path"])
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P3 artifact drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_source_layout_audit(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any]
) -> None:
    """只读重算 50 m static 与 24 actor 的冻结源事实。"""
    load_stage(run_dir, manifest, "input_audit")
    started = time.perf_counter()
    sampler = ResourceSampler(os.getpid())
    sampler.__enter__()
    try:
        audit = validate_source_layout(protocol)
    finally:
        sampler.__exit__(None, None, None)
    resources = cpu_stage_resource_summary(sampler)
    stage = {
        "status": "done",
        "stage": "source_layout_audit",
        "duration_seconds": time.perf_counter() - started,
        "source_layout": audit,
        "source_precision_and_row_tensor_schema_exact": True,
        "static_grid_contract_and_source_inventory_exact": True,
        "actor_inventory_exact": True,
        "training_optimizer_or_source_mutation_performed": False,
        "source_inputs_unchanged": source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "source_layout_audit_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "source_layout_audit", stage)


def run_materialize(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any]
) -> None:
    """物化 manifest、shared skeleton、133 static 与 24 actor files。"""
    layout = load_stage(run_dir, manifest, "source_layout_audit")
    started = time.perf_counter()
    checkpoint_path = Path(protocol["selected_asset"]["checkpoint"]["path"])
    source_before = sha256_file(checkpoint_path)
    sampler = ResourceSampler(os.getpid())
    sampler.__enter__()
    try:
        source = torch.load(checkpoint_path, map_location="cpu")
        package_root = run_dir / protocol["package_contract"]["root_output"]
        package_manifest = materialize_chunk_package(
            source,
            package_root=package_root,
            protocol=protocol,
            protocol_sha256=manifest["protocol_sha256"],
            project_commit=manifest["project_commit"],
        )
        manifest_path = run_dir / protocol["package_contract"]["manifest_output"]
        atomic_json(manifest_path, package_manifest)
        del source
    finally:
        sampler.__exit__(None, None, None)
    resources = cpu_stage_resource_summary(sampler)
    expected = protocol["package_contract"]
    package_files = [path for path in package_root.rglob("*") if path.is_file()]
    counts_exact = package_manifest["counts"] == {
        "static_assets": int(expected["expected_static_asset_count"]),
        "actor_assets": int(expected["expected_actor_asset_count"]),
        "data_assets": int(expected["expected_data_asset_count"]),
        "payload_files": int(expected["expected_payload_file_count"]),
    } and len(package_files) == int(expected["expected_file_count_including_manifest"])
    source_after = sha256_file(checkpoint_path)
    stage = {
        "status": "done",
        "stage": "materialize_chunk_package",
        "duration_seconds": time.perf_counter() - started,
        "package_root": str(package_root.relative_to(run_dir)),
        "package_manifest": artifact_record(manifest_path, run_dir),
        "package_counts": package_manifest["counts"],
        "package_file_count_including_manifest": len(package_files),
        "package_bytes_including_manifest": directory_bytes(package_root),
        "package_counts_exact": counts_exact,
        "source_static_inventory_sha256": layout["source_layout"]["static_inventory"][
            "inventory_sha256"
        ],
        "source_actor_inventory_sha256": layout["source_layout"]["actor_inventory"][
            "inventory_sha256"
        ],
        "source_checkpoint_before_sha256": source_before,
        "source_checkpoint_after_sha256": source_after,
        "source_checkpoint_copied": False,
        "persistent_reassembled_checkpoint_written": False,
        "training_optimizer_or_source_mutation_performed": False,
        "source_inputs_unchanged": source_before == source_after and source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "materialize_chunk_package_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "materialize_chunk_package", stage)


def package_manifest_for_run(
    run_dir: Path, materialized: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """读取并核对 package manifest artifact。"""
    record = materialized["package_manifest"]
    path = run_dir / str(record["path"])
    return path.parent, load_json_artifact(run_dir, record)


def run_reassemble(
    run_dir: Path, protocol: Mapping[str, Any], manifest: dict[str, Any]
) -> None:
    """验证全部 package files 并在内存中 bitwise 重建 checkpoint。"""
    materialized = load_stage(run_dir, manifest, "materialize_chunk_package")
    started = time.perf_counter()
    package_root, package_manifest = package_manifest_for_run(run_dir, materialized)
    checkpoint_path = Path(protocol["selected_asset"]["checkpoint"]["path"])
    source_before = sha256_file(checkpoint_path)
    sampler = ResourceSampler(os.getpid())
    sampler.__enter__()
    try:
        source = torch.load(checkpoint_path, map_location="cpu")
        candidate, package_audit = reassemble_chunk_package(
            package_root=package_root,
            manifest=package_manifest,
            protocol=protocol,
        )
        comparison = compare_checkpoint_states(source, candidate)
        skeleton = torch.load(
            package_root / package_manifest["skeleton"]["path"], map_location="cpu"
        )
        source_tensor_count = len(recursive_tensor_rows(source))
        skeleton_tensor_count = len(recursive_tensor_rows(skeleton))
        sentinel_count = int(package_manifest["skeleton"]["row_tensor_sentinel_count"])
        shared_and_no_duplication = (
            sentinel_count == 51
            and source_tensor_count - skeleton_tensor_count == sentinel_count
            and package_audit["indices_unique_disjoint_exhaustive"]
            and package_manifest["counts"]["static_assets"] == 133
            and package_manifest["counts"]["actor_assets"] == 24
        )
    finally:
        sampler.__exit__(None, None, None)
    resources = cpu_stage_resource_summary(sampler)
    source_after = sha256_file(checkpoint_path)
    stage = {
        "status": "done",
        "stage": "reassemble_and_hash_audit",
        "duration_seconds": time.perf_counter() - started,
        "package_manifest": materialized["package_manifest"],
        "package_audit": package_audit,
        "checkpoint_comparison": comparison,
        "source_tensor_count": source_tensor_count,
        "skeleton_tensor_count": skeleton_tensor_count,
        "row_tensor_sentinel_count": sentinel_count,
        "shared_state_preserved_and_source_rows_not_duplicated": shared_and_no_duplication,
        "persistent_reassembled_checkpoint_written": False,
        "source_checkpoint_before_sha256": source_before,
        "source_checkpoint_after_sha256": source_after,
        "training_optimizer_or_source_mutation_performed": False,
        "source_inputs_unchanged": source_before == source_after and source_inputs_unchanged(protocol),
        "minimum_rerun_unit": "reassemble_and_hash_audit_and_downstream",
        **resources,
    }
    del source, candidate, skeleton
    write_stage(run_dir, manifest, "reassemble_and_hash_audit", stage)


def _load_state_into_trainer(trainer: Any, state: Mapping[str, Any], device: torch.device) -> None:
    """复用只读 load-only-model 路径加载内存 checkpoint。"""
    mutable = type(state)(state.items())
    models = type(state["models"])(state["models"].items())
    if "RigidNodes" in models:
        models["RigidNodes"] = to_device(models["RigidNodes"], device)
    mutable["models"] = models
    trainer.load_state_dict(mutable, load_only_model=True, strict=True)


def load_arm(
    *,
    run_dir: Path,
    protocol: Mapping[str, Any],
    materialized: Mapping[str, Any],
    trainer: Any,
    arm: str,
    device: torch.device,
) -> tuple[float, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """读取 source 或完整 package，并加载相同 P2 mixed runtime。"""
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if arm == "p3-source":
        checkpoint = Path(protocol["selected_asset"]["checkpoint"]["path"])
        if (
            checkpoint.stat().st_size != int(protocol["selected_asset"]["checkpoint"]["bytes"])
            or sha256_file(checkpoint) != protocol["selected_asset"]["checkpoint"]["sha256"]
        ):
            raise RuntimeError("A4-P3 source checkpoint drift during load")
        state = torch.load(checkpoint, map_location="cpu")
        package_audit = {
            "source_checkpoint_verified": True,
            "indices_unique_disjoint_exhaustive": True,
        }
        asset = {
            "kind": "monolithic_checkpoint",
            "path": str(checkpoint),
            "sha256": protocol["selected_asset"]["checkpoint"]["sha256"],
            "bytes": checkpoint.stat().st_size,
            "file_count": 1,
        }
    else:
        package_root, package_manifest = package_manifest_for_run(run_dir, materialized)
        state, package_audit = reassemble_chunk_package(
            package_root=package_root,
            manifest=package_manifest,
            protocol=protocol,
        )
        manifest_path = run_dir / str(materialized["package_manifest"]["path"])
        asset = {
            "kind": "exact_chunk_package",
            "path": str(package_root),
            "manifest_sha256": materialized["package_manifest"]["sha256"],
            "bytes": directory_bytes(package_root),
            "file_count": len([path for path in package_root.rglob("*") if path.is_file()]),
            "payload_sha256": package_manifest["payload_sha256"],
            "manifest_bytes": manifest_path.stat().st_size,
        }
    _load_state_into_trainer(trainer, state, device)
    del state
    apply_runtime_parameter_dtypes(trainer, candidate=True)
    set_fp32_renderer_adapter_mode(trainer, candidate=True)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    dtype_audit = runtime_converted_field_audit(trainer, expected_dtype="float16")
    parameter_inventory = persistent_parameter_inventory(trainer)
    reload_audit = {
        "model_counts": runtime_model_counts(trainer),
        "expected_model_counts": expected_model_counts(protocol),
        "actor_registry_exact": actor_registry_exact(
            trainer, Path(protocol["selected_asset"]["actor_registry"]["path"])
        ),
    }
    reload_audit["exact"] = (
        reload_audit["model_counts"] == reload_audit["expected_model_counts"]
        and reload_audit["actor_registry_exact"]
        and dtype_audit["exact"]
    )
    return load_seconds, dtype_audit, parameter_inventory, package_audit, {
        "asset": asset,
        "reload": reload_audit,
    }


def mean_or_minus_one(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else -1.0


def evaluate_loaded_arm(
    *,
    protocol: Mapping[str, Any],
    dataset: Any,
    trainer: Any,
    arm: str,
    asset: Mapping[str, Any],
    load_seconds: float,
    reload_audit: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """完整渲染 57-view split，并记录逐 view RGB SHA 与 31 个端点。"""
    trainer.set_eval()
    if hasattr(trainer.lpips, "reset"):
        trainer.lpips.reset()
    model_counts = runtime_model_counts(trainer)
    point_ids = runtime_point_ids(trainer)
    quality = protocol["quality_contract"]
    test_indices = [int(value) for value in dataset.test_image_set.split_indices]
    expected_indices = [
        int(frame) * 3 + int(camera)
        for frame in quality["heldout_frames"]
        for camera in quality["cameras"]
    ]
    if test_indices != expected_indices:
        raise RuntimeError("A4-P3 heldout quality split drift")

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
    view_hashes = []
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
                float(
                    structural_similarity(
                        target_cpu, prediction, data_range=1.0, channel_axis=-1
                    )
                )
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
            prediction_u8 = uint8_rgb(raw_rgb.detach().float().cpu().numpy())
            prediction_cache[position] = prediction_u8
            target_cache[position] = target_cpu
            frame, camera = divmod(test_indices[position], 3)
            view_hashes.append(
                {
                    "position": position,
                    "full_image_index": test_indices[position],
                    "frame": frame,
                    "camera": camera,
                    "rgb_sha256": rgb_sha256(prediction_u8),
                }
            )
        finally:
            release_trainer_render_info(trainer)
        print(f"A4-P3 quality arm={arm} render={position + 1}/{len(test_indices)}", flush=True)

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
            for region_name, selected in (
                ("actor_region", effect),
                ("boundary_band", boundary),
            ):
                lpips_value = masked_lpips(
                    trainer.lpips, prediction, target, selected, device
                )
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
        lpips_value = masked_lpips(trainer.lpips, prediction, target, selected, device)
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
        "asset": dict(asset),
        "asset_load_reassembly_seconds": load_seconds,
        "checkpoint_reload_exact": bool(reload_audit["exact"]),
        "model_gaussian_counts": model_counts,
        "actor_counts_exact": bool(reload_audit["actor_registry_exact"]),
        "unavailable_actor_remains_explicitly_empty": all(
            int((point_ids == int(actor["rigid_model_index"])).sum()) == 0
            for actor in json.loads(
                Path(protocol["selected_asset"]["actor_registry"]["path"]).read_text(
                    encoding="utf-8"
                )
            )["actors"]
            if actor["availability"] == "unavailable_empty_checkpoint_slice"
        ),
        "heldout_image_count": len(test_indices),
        "heldout_full_image_indices": test_indices,
        "per_view_rgb_sha256": view_hashes,
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


def endpoint_triples(
    protocol: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """按冻结顺序展开 11+16+4=31 个质量端点。"""
    quality = protocol["quality_contract"]
    triples = [
        (f"global.{name}", baseline["global_metrics"].get(name), candidate["global_metrics"].get(name))
        for name in quality["global_endpoints"]["metrics"]
    ]
    for role in quality["actor_endpoints"]["roles"]:
        for region in quality["actor_endpoints"]["regions"]:
            for name in quality["actor_endpoints"]["metrics"]:
                triples.append(
                    (
                        f"roles.{role}.{region}.{name}",
                        baseline["roles"][role][region].get(name),
                        candidate["roles"][role][region].get(name),
                    )
                )
    for name in quality["non_target_endpoints"]["metrics"]:
        triples.append(
            (
                f"non_target.{name}",
                baseline["non_target"]["quality"].get(name),
                candidate["non_target"]["quality"].get(name),
            )
        )
    return triples


def exact_replay_rows(
    protocol: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tolerance_key: str,
) -> list[dict[str, Any]]:
    """按冻结绝对容差比较 31 个端点。"""
    tolerances = protocol["quality_contract"][tolerance_key]
    rows = []
    for endpoint, expected, actual in endpoint_triples(protocol, baseline, candidate):
        limit = _metric_tolerance(tolerances, endpoint)
        passed = (
            expected is not None
            and actual is not None
            and math.isfinite(float(expected))
            and math.isfinite(float(actual))
            and abs(float(actual) - float(expected)) <= limit
        )
        rows.append(
            {
                "endpoint": endpoint,
                "expected": expected,
                "actual": actual,
                "absolute_difference": (
                    abs(float(actual) - float(expected)) if passed or (expected is not None and actual is not None) else None
                ),
                "absolute_tolerance": limit,
                "passed": bool(passed),
            }
        )
    return rows


def per_view_rgb_exact(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    """逐 view 比较 57 个 uint8 RGB SHA。"""
    baseline_rows = baseline["per_view_rgb_sha256"]
    candidate_rows = candidate["per_view_rgb_sha256"]
    rows = []
    for expected, actual in zip(baseline_rows, candidate_rows):
        passed = (
            expected["full_image_index"] == actual["full_image_index"]
            and expected["rgb_sha256"] == actual["rgb_sha256"]
        )
        rows.append(
            {
                "full_image_index": expected["full_image_index"],
                "source_rgb_sha256": expected["rgb_sha256"],
                "candidate_rgb_sha256": actual["rgb_sha256"],
                "passed": passed,
            }
        )
    exact = len(baseline_rows) == len(candidate_rows) == 57 and all(
        row["passed"] for row in rows
    )
    return exact, rows


def run_evaluate(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    """执行两臂 57-view exact replay。"""
    load_stage(run_dir, manifest, "reassemble_and_hash_audit")
    materialized = load_stage(run_dir, manifest, "materialize_chunk_package")
    if nvidia_compute_rows():
        raise RuntimeError("A4-P3 quality evaluation GPU preflight not idle")
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
        results = {}
        artifacts = {}
        for arm in ARMS:
            load_seconds, dtype_audit, parameter_inventory, package_audit, load_audit = load_arm(
                run_dir=run_dir,
                protocol=protocol,
                materialized=materialized,
                trainer=trainer,
                arm=arm,
                device=device,
            )
            result = evaluate_loaded_arm(
                protocol=protocol,
                dataset=dataset,
                trainer=trainer,
                arm=arm,
                asset=load_audit["asset"],
                load_seconds=load_seconds,
                reload_audit=load_audit["reload"],
                device=device,
            )
            result["expected_model_counts"] = expected_model_counts(protocol)
            result["expected_model_counts_exact"] = (
                result["model_gaussian_counts"] == result["expected_model_counts"]
            )
            result["runtime_converted_field_audit"] = dtype_audit
            result["persistent_parameter_bytes_by_dtype"] = parameter_inventory
            result["renderer_input_dtypes"] = renderer_adapter_summary(trainer)
            result["package_load_audit"] = package_audit
            results[arm] = result

        frozen = json.loads(
            Path(protocol["baseline_quality"]["source_quality"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        source_rows = exact_replay_rows(
            protocol,
            frozen,
            results["p3-source"],
            tolerance_key="p2_baseline_replay_tolerance",
        )
        results["p3-source"]["baseline_p2_replay_rows"] = source_rows
        results["p3-source"]["baseline_p2_replay_pass"] = len(source_rows) == 31 and all(
            row["passed"] for row in source_rows
        )
        candidate_rows = exact_replay_rows(
            protocol,
            results["p3-source"],
            results["p3-chunk-package"],
            tolerance_key="candidate_source_replay_tolerance",
        )
        rgb_exact, rgb_rows = per_view_rgb_exact(
            results["p3-source"], results["p3-chunk-package"]
        )
        results["p3-chunk-package"]["candidate_source_replay_rows"] = candidate_rows
        results["p3-chunk-package"]["candidate_source_endpoint_replay_pass"] = (
            len(candidate_rows) == 31 and all(row["passed"] for row in candidate_rows)
        )
        results["p3-chunk-package"]["candidate_source_rgb_rows"] = rgb_rows
        results["p3-chunk-package"]["candidate_per_view_rgb_sha256_exact"] = rgb_exact
        results["p3-chunk-package"]["all_exact_quality_gates_pass"] = bool(
            rgb_exact
            and results["p3-chunk-package"]["candidate_source_endpoint_replay_pass"]
        )
        for arm, result in results.items():
            path = run_dir / "artifacts" / "quality" / f"{arm}.json"
            atomic_json(path, result)
            artifacts[arm] = artifact_record(path, run_dir)
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset, results
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
        "stage": "evaluate_source_and_chunk",
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
        "minimum_rerun_unit": "evaluate_source_and_chunk_and_downstream",
        **resources,
    }
    write_stage(run_dir, manifest, "evaluate_source_and_chunk", stage)


def run_runtime_profile(
    run_dir: Path,
    protocol: Mapping[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    """执行完整 source/package load 与两臂 9-view report-only runtime。"""
    load_stage(run_dir, manifest, "evaluate_source_and_chunk")
    materialized = load_stage(run_dir, manifest, "materialize_chunk_package")
    if nvidia_compute_rows():
        raise RuntimeError("A4-P3 runtime profile GPU preflight not idle")
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
            torch.cuda.reset_peak_memory_stats(device)
            load_seconds, dtype_audit, parameter_inventory, package_audit, load_audit = load_arm(
                run_dir=run_dir,
                protocol=protocol,
                materialized=materialized,
                trainer=trainer,
                arm=arm,
                device=device,
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
            adapter = renderer_adapter_summary(trainer)
            arm_rows[arm] = {
                "arm": arm,
                "asset": load_audit["asset"],
                "checkpoint_or_package_load_reassembly_seconds": load_seconds,
                "model_gaussian_counts": runtime_model_counts(trainer),
                "expected_model_counts": expected_model_counts(protocol),
                "checkpoint_reload_exact": load_audit["reload"]["exact"],
                "package_load_audit": package_audit,
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
            print(f"A4-P3 runtime arm={arm} complete", flush=True)
        source_runtime = {
            (row["frame"], row["camera"]): row["rgb_sha256"]
            for row in arm_rows["p3-source"]["measured_rows"]
        }
        chunk_runtime = {
            (row["frame"], row["camera"]): row["rgb_sha256"]
            for row in arm_rows["p3-chunk-package"]["measured_rows"]
        }
        runtime_rgb_exact = source_runtime == chunk_runtime
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
        "runtime_rgb_sha256_exact_between_arms": runtime_rgb_exact,
        "performance_values_report_only": True,
        "selective_loading_or_view_culling_performed": False,
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
        choices=("source-layout-audit", "materialize", "reassemble", "evaluate", "runtime-profile"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256_file(args.protocol):
        raise RuntimeError("A4-P3 protocol hash drift after initialization")
    device = torch.device(args.device)
    if args.operation == "source-layout-audit":
        run_source_layout_audit(args.run_dir, protocol, manifest)
    elif args.operation == "materialize":
        run_materialize(args.run_dir, protocol, manifest)
    elif args.operation == "reassemble":
        run_reassemble(args.run_dir, protocol, manifest)
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
                        "code": "A4_P3_WORKER_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
