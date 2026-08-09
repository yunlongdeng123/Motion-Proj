#!/usr/bin/env python
"""校验 A4-P3 chunk 协议、P2 选择证据与只读源布局事实。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml"

COMMON_ROW_SCHEMA = [
    ("_means", "float32", [3]),
    ("_scales", "float16", [3]),
    ("_quats", "float16", [4]),
    ("_features_dc", "float16", [3]),
    ("_features_rest", "float16", [15, 3]),
    ("_opacities", "float16", [1]),
    ("worldsim_a2_ancestry.fields.gaussian_id", "int64", []),
    ("worldsim_a2_ancestry.fields.actor_id", "int64", []),
    ("worldsim_a2_ancestry.fields.init_source", "int64", []),
    ("worldsim_a2_ancestry.fields.parent_id", "int64", []),
    ("worldsim_a2_ancestry.fields.lineage_root_id", "int64", []),
    ("worldsim_a2_ancestry.fields.birth_step", "int64", []),
    ("worldsim_a2_ancestry.fields.generation", "int64", []),
    ("worldsim_a2_ancestry.fields.visibility_count", "int64", []),
    ("worldsim_a2_ancestry.fields.screen_grad", "float32", []),
    ("worldsim_a2_ancestry.fields.screen_grad_count", "int64", []),
    ("worldsim_a2_ancestry.fields.boundary_contribution", "float32", []),
    ("worldsim_a2_ancestry.fields.boundary_contribution_count", "int64", []),
    ("worldsim_a2_ancestry.fields.photometric_residual", "float32", []),
    ("worldsim_a2_ancestry.fields.photometric_residual_count", "int64", []),
    ("worldsim_a2_ancestry.fields.depth_residual", "float32", []),
    ("worldsim_a2_ancestry.fields.depth_residual_count", "int64", []),
    ("worldsim_a2_ancestry.fields.normal_residual", "float32", []),
    ("worldsim_a2_ancestry.fields.normal_residual_count", "int64", []),
    ("worldsim_a2_ancestry.fields.nearest_lidar_distance", "float32", []),
]


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256，避免把 checkpoint 整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(directory: Path, pattern: str) -> dict[str, Any]:
    """按冻结的 sha256sum 文本清单算法计算目录摘要。"""
    paths = sorted(directory.glob(pattern), key=lambda path: path.name)
    manifest = "".join(
        f"{sha256_file(path)}  ./{path.name}\n" for path in paths if path.is_file()
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(manifest).hexdigest(),
        "file_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"A4-P3 protocol invalid: {message}")


def schema_tuples(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, list[int]]]:
    """把 YAML row schema 规范化为稳定比较元组。"""
    return [
        (str(row["path"]), str(row["dtype"]), [int(value) for value in row["shape_tail"]])
        for row in rows
    ]


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-A4-DEPLOYMENT-01", "task_id")
    require(protocol["profile_id"] == "A4-P3-CHUNK-v1", "profile_id")
    require(
        protocol["protocol_status"] == "frozen_before_new_p3_measurements",
        "protocol_status",
    )
    require(protocol["seed"] == 0 and protocol["scene"] == "scene-0230", "scene/seed")

    authorization = protocol["authorization"]
    require(authorization["p3_chunk_materialization_authorized"], "P3 materialization")
    require(
        authorization["p3_reassembled_candidate_render_authorized"],
        "P3 candidate render",
    )
    for name in (
        "source_checkpoint_mutation_authorized",
        "training_authorized",
        "optimizer_authorized",
        "selective_chunk_render_authorized",
        "view_dependent_chunk_culling_authorized",
        "chunk_size_search_authorized",
        "p4_lod_authorized",
    ):
        require(not authorization[name], f"authorization {name}")

    inventory = protocol["selected_asset"]["inventory"]
    require(
        inventory
        == {
            "background_gaussians": 1_205_164,
            "rigid_gaussians": 104_704,
            "total_gaussians": 1_309_868,
            "actor_count": 24,
            "available_actor_count": 23,
            "unavailable_actor_count": 1,
        },
        "selected inventory",
    )
    precision = protocol["source_precision_contract"]
    require(precision["models"] == ["Background", "RigidNodes"], "precision models")
    require(precision["float32_row_fields"] == ["_means"], "FP32 row fields")
    require(
        precision["float16_row_fields"]
        == ["_scales", "_quats", "_features_dc", "_features_rest", "_opacities"],
        "FP16 row fields",
    )
    adapter = precision["runtime_adapter"]
    require(adapter["renderer_input_dtype_required"] == "float32", "renderer dtype")
    require(
        adapter["spherical_harmonics_input_dtype_required"] == "float32",
        "SH dtype",
    )
    require(not adapter["autocast_enabled"], "autocast disabled")
    require(not adapter["fp16_renderer_kernel_claim_allowed"], "FP16 renderer claim")

    row_schema = protocol["row_tensor_schema"]
    require(
        schema_tuples(row_schema["common_gaussian_row_tensors"]) == COMMON_ROW_SCHEMA,
        "common row tensor schema",
    )
    require(
        row_schema["models"]["Background"]
        == {"row_count": 1_205_164, "additional_row_tensors": []},
        "Background row schema",
    )
    rigid = row_schema["models"]["RigidNodes"]
    require(rigid["row_count"] == 104_704, "Rigid row count")
    require(
        schema_tuples(rigid["additional_row_tensors"])
        == [("points_ids", "int64", [1])],
        "Rigid additional row schema",
    )

    static = protocol["static_chunk_contract"]
    require(static["model"] == "Background", "static model")
    require(static["axes"] == ["x", "y"], "static axes")
    require(static["origin_xy_m"] == [0.0, 0.0], "static origin")
    require(float(static["cell_size_m"]) == 50.0, "static cell size")
    require(static["coordinate_precision_for_membership"] == "float64", "cell precision")
    require(static["cell_bounds"] == "half_open_lower_inclusive_upper_exclusive", "cell bounds")
    require(static["cell_order"] == "ascending_integer_ix_then_iy", "cell order")
    require(not static["empty_cells_emitted"], "empty static cells")
    require(static["sparse_chunks_preserved"], "sparse chunks")
    require(static["outlier_chunks_preserved"], "outlier chunks")
    require(static["minimum_chunk_count"] is None, "minimum chunk count")
    require(static["merge_policy"] == "none", "merge policy")
    require(static["post_hoc_cell_size_or_merge_forbidden"], "post-hoc grid")
    expected_static = static["expected_source_inventory"]
    require(expected_static["occupied_chunk_count"] == 133, "static chunk count")
    require(expected_static["background_count"] == 1_205_164, "static row count")
    require(expected_static["count_min"] == 1, "static minimum count")
    require(expected_static["count_max"] == 330_169, "static maximum count")
    require(expected_static["chunks_below_100"] == 98, "sparse inventory")
    require(expected_static["chunks_at_least_10000"] == 7, "dense inventory")
    require(expected_static["boundary_band_m"] == 0.25, "boundary band")
    require(expected_static["boundary_band_count"] == 69_393, "boundary count")
    require(
        expected_static["inventory_sha256"]
        == "d78fa6e1046a365f6606b4e04a692351b0668af5f27b5d0a401d69d0dec27cae",
        "static inventory digest",
    )

    actor = protocol["actor_chunk_contract"]
    require(actor["model"] == "RigidNodes", "actor model")
    require(actor["assignment_tensor_path"] == "points_ids", "actor assignment")
    require(actor["actor_index_domain_inclusive"] == [0, 23], "actor domain")
    require(not actor["contiguous_slice_assumption_allowed"], "actor contiguous assumption")
    require(actor["expected_asset_count"] == 24, "actor asset count")
    require(actor["expected_available_count"] == 23, "available actors")
    require(actor["expected_unavailable_count"] == 1, "unavailable actors")
    require(actor["empty_actor_indices"] == [14], "empty actor")
    require(
        actor["empty_actor_asset_policy"]
        == "emit_zero_row_tensor_asset_with_all_row_fields",
        "empty actor policy",
    )
    expected_actors = actor["expected_actors"]
    require([row["actor_index"] for row in expected_actors] == list(range(24)), "actor order")
    require(sum(int(row["count"]) for row in expected_actors) == 104_704, "actor row total")
    require(expected_actors[14]["count"] == 0, "actor 14 count")
    require(
        actor["expected_inventory_sha256"]
        == "384870e6773c639be26c21e1e6067c28999acbb8785d52faff5b43792d24f23a",
        "actor inventory digest",
    )

    package = protocol["package_contract"]
    require(
        [row["id"] for row in package["arms"]] == ["p3-source", "p3-chunk-package"],
        "arm grid",
    )
    require(package["package_format"] == "worldsim_v3_chunk_package_v1", "package format")
    require(package["expected_static_asset_count"] == 133, "package static count")
    require(package["expected_actor_asset_count"] == 24, "package actor count")
    require(package["expected_data_asset_count"] == 157, "package data count")
    require(package["expected_payload_file_count"] == 158, "payload file count")
    require(package["expected_file_count_including_manifest"] == 159, "package file count")
    require(package["source_checkpoint_copy_forbidden"], "source copy forbidden")
    require(
        package["persistent_reassembled_checkpoint_forbidden"],
        "persistent reassembled checkpoint",
    )
    require(package["reassembly"]["mode"] == "in_memory_only", "reassembly mode")
    require(
        package["reassembly"]["source_flat_indices_must_be_unique_disjoint_exhaustive"],
        "reassembly indices",
    )
    require(package["reassembly"]["every_tensor_shape_dtype_and_value_sha256_must_match_source"], "tensor exactness")
    require(package["raw_render_media_forbidden"], "raw media")

    quality = protocol["quality_contract"]
    require(quality["expected_views"] == 57, "quality views")
    require(quality["original_resolution"] == [800, 450], "quality resolution")
    require(quality["heldout_frames"] == list(range(10, 200, 10)), "quality frames")
    require(quality["cameras"] == [0, 1, 2], "quality cameras")
    require(len(quality["global_endpoints"]["metrics"]) == 11, "global endpoints")
    require(quality["actor_endpoints"]["roles"] == ["high-support", "boundary-support"], "actor roles")
    require(quality["actor_endpoints"]["regions"] == ["actor_region", "boundary_band"], "actor regions")
    require(len(quality["actor_endpoints"]["metrics"]) == 4, "actor metrics")
    require(len(quality["non_target_endpoints"]["metrics"]) == 4, "non-target metrics")
    require(quality["endpoint_count"] == 31, "endpoint count")
    expected_tolerance = {
        "psnr_absolute": 0.000001,
        "ssim_absolute": 0.00000001,
        "lpips_absolute": 0.00000001,
        "mean_absolute_error_absolute": 0.00000001,
    }
    require(quality["p2_baseline_replay_tolerance"] == expected_tolerance, "P2 replay tolerance")
    require(quality["candidate_source_replay_tolerance"] == expected_tolerance, "candidate tolerance")
    require(quality["candidate_per_view_rgb_sha256_must_equal_source"], "RGB exactness")
    require(
        quality["candidate_pass_rule"]
        == "all_57_rgb_hashes_and_all_31_quality_endpoints_exact",
        "quality pass rule",
    )

    runtime = protocol["runtime_contract"]
    require(runtime["frames"] == [10, 100, 190], "runtime frames")
    require(runtime["cameras"] == [0, 1, 2], "runtime cameras")
    require(runtime["expected_samples_per_arm"] == 9, "runtime samples")
    require(runtime["warmup_views"] == 2, "runtime warmup")
    require(runtime["resolution"] == [800, 450], "runtime resolution")
    require(runtime["percentile"] == "nearest_rank", "runtime percentile")
    require(runtime["selective_loading_or_view_culling_forbidden"], "runtime full package")
    require(runtime["performance_values_are_report_only_not_quality_selection"], "runtime report only")

    selection = protocol["selection_contract"]
    require(selection["candidate_pass"]["selected_arm"] == "p3-chunk-package", "candidate selection")
    require(selection["candidate_pass"]["method_state"] == "selected_exact_chunk_package", "candidate method")
    require(selection["candidate_fail"]["selected_arm"] == "p3-source", "source fallback")
    require(
        selection["candidate_fail"]["fallback"]
        == "immutable_p2_selected_checkpoint_exact_alias",
        "fallback alias",
    )
    require(not selection["runtime_values_used_for_selection"], "runtime selection")
    require(selection["no_result_dependent_chunk_size_merge_field_policy_or_threshold"], "post-hoc policy")

    require(
        protocol["recovery_contract"]["stage_order"]
        == [
            "input_audit",
            "source_layout_audit",
            "materialize_chunk_package",
            "reassemble_and_hash_audit",
            "evaluate_source_and_chunk",
            "runtime_profile_both_arms",
            "aggregate",
            "resume_audit",
        ],
        "stage order",
    )
    require(
        protocol["resource_ceilings"]
        == {
            "wall_time_seconds": 900,
            "peak_torch_allocated_mib": 16_384,
            "peak_torch_reserved_mib": 24_576,
            "peak_nvidia_process_memory_mib_sampled": 24_000,
            "peak_cgroup_memory_bytes": 51_539_607_552,
            "run_bytes": 1_000_000_000,
            "disk_free_floor_bytes": 30_000_000_000,
            "oom_events_delta": 0,
            "oom_kill_events_delta": 0,
        },
        "resource ceilings",
    )
    require(len(protocol["required_audits"]) == 21, "required audit count")
    require(all(protocol["required_audits"].values()), "required audit values")
    require(all(protocol["claim_boundary"].values()), "claim boundary values")
    require("PIVOT-F28" in protocol["failure_precedents"], "PIVOT-F28 precedent")


def iter_fingerprinted_inputs(protocol: Mapping[str, Any]):
    """按协议顺序产出九个 exact file 输入。"""
    for name in ("checkpoint", "source_config", "actor_registry"):
        yield f"selected_asset.{name}", protocol["selected_asset"][name]
    for name in (
        "summary",
        "manifest",
        "resource_audit",
        "selected_quality",
        "resume_audit",
        "terminal",
    ):
        yield f"p2_canonical_evidence.{name}", protocol["p2_canonical_evidence"][name]


def validate_inputs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """核对 P2 canonical 选择、九个文件及冻结 mask 目录。"""
    audits: dict[str, dict[str, Any]] = {}
    for name, spec in iter_fingerprinted_inputs(protocol):
        path = Path(spec["path"])
        require(path.is_file(), f"missing {name}: {path}")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        require(actual_sha == spec["sha256"], f"hash drift {name}")
        require(actual_bytes == int(spec["bytes"]), f"byte drift {name}")
        audits[name] = {"path": str(path), "sha256": actual_sha, "bytes": actual_bytes}

    require(
        protocol["baseline_quality"]["source_quality"]
        == protocol["p2_canonical_evidence"]["selected_quality"],
        "baseline quality must alias P2 selected quality",
    )
    mask_spec = protocol["baseline_quality"]["actor_masks"]
    mask_dir = Path(mask_spec["path"])
    require(mask_dir.is_dir(), f"missing actor masks: {mask_dir}")
    mask_audit = directory_digest(mask_dir, mask_spec["file_glob"])
    require(mask_audit["sha256"] == mask_spec["sha256"], "actor mask hash drift")
    require(mask_audit["file_count"] == int(mask_spec["file_count"]), "actor mask count drift")
    require(mask_audit["total_bytes"] == int(mask_spec["total_bytes"]), "actor mask bytes drift")
    audits["baseline_quality.actor_masks"] = {"path": str(mask_dir), **mask_audit}

    selected = protocol["selected_asset"]
    registry = json.loads(Path(selected["actor_registry"]["path"]).read_text())
    require(
        registry["actor_registry_sha256"]
        == selected["actor_registry"]["embedded_registry_sha256"],
        "embedded registry hash",
    )
    evidence = protocol["p2_canonical_evidence"]
    summary = json.loads(Path(evidence["summary"]["path"]).read_text())
    require(summary["status"] == "done" and all(summary["audits"].values()), "P2 summary")
    require(len(summary["audits"]) == 19, "P2 audit count")
    require(summary["project_commit"] == evidence["source_commit"], "P2 source commit")
    require(summary["p2_experiment_terminal"] == "done", "P2 experiment terminal")
    require(summary["selection"]["selected_arm"] == "p2-gs-param-fp16", "P2 selected arm")
    require(
        summary["method_state"]
        == "selected_mixed_precision_parameter_storage_fp32_render",
        "P2 method state",
    )
    require(
        summary["selected_asset"]["checkpoint"]["sha256"]
        == selected["checkpoint"]["sha256"],
        "P2 selected checkpoint",
    )
    require(
        summary["selected_asset"]["actor_registry"]["sha256"]
        == selected["actor_registry"]["sha256"],
        "P2 selected registry",
    )
    manifest = json.loads(Path(evidence["manifest"]["path"]).read_text())
    require(manifest["status"] == "done", "P2 manifest")
    require(manifest["summary_sha256"] == evidence["summary"]["sha256"], "P2 manifest summary")
    resources = json.loads(Path(evidence["resource_audit"]["path"]).read_text())
    require(resources["status"] == "passed", "P2 resources")
    quality = json.loads(Path(evidence["selected_quality"]["path"]).read_text())
    require(quality["arm"] == "p2-gs-param-fp16", "P2 quality arm")
    require(quality["heldout_image_count"] == 57, "P2 quality views")
    require(quality["all_endpoints_complete_and_finite"], "P2 quality endpoints")
    require(quality["all_quality_safeguards_pass"], "P2 quality safeguards")
    resume = json.loads(Path(evidence["resume_audit"]["path"]).read_text())
    require(not resume["torch_imported"] and not resume["gpu_launch_observed"], "P2 resume")
    require(resume["all_completed_stages_reused"], "P2 completed stage reuse")
    require(len(resume["actions"]) == 6, "P2 resume actions")
    terminal = json.loads(Path(evidence["terminal"]["path"]).read_text())
    require(terminal == {"failure": None, "status": "done"}, "P2 terminal")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["p2_closeout_commit"], "HEAD"],
        cwd=PROJECT,
        check=False,
    ).returncode == 0
    require(ancestor, "P2 closeout commit is not an ancestor of HEAD")
    return audits


def chunk_id(ix: int, iy: int) -> str:
    """按冻结格式生成静态块 ID。"""
    encode = lambda value: f"{'n' if value < 0 else 'p'}{abs(value):04d}"
    return f"static-x-{encode(ix)}-y-{encode(iy)}"


def index_sha256(indices: Any) -> str:
    """计算排序源扁平索引的冻结 little-endian int64 摘要。"""
    values = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def discover_row_tensor_schema(
    value: Mapping[str, Any], row_count: int, prefix: str = ""
) -> list[tuple[str, str, list[int]]]:
    """递归发现第一维等于模型行数的 tensor。"""
    import torch

    rows: list[tuple[str, str, list[int]]] = []
    for name, child in value.items():
        path = f"{prefix}.{name}" if prefix else str(name)
        if torch.is_tensor(child) and child.ndim > 0 and int(child.shape[0]) == row_count:
            rows.append(
                (
                    path,
                    str(child.dtype).removeprefix("torch."),
                    [int(item) for item in child.shape[1:]],
                )
            )
        elif isinstance(child, Mapping):
            rows.extend(discover_row_tensor_schema(child, row_count, path))
    return rows


def build_static_inventory(means: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """按 50 m 半开 XY 网格重算静态块事实。"""
    import torch

    cell_size = float(contract["cell_size_m"])
    origin = torch.tensor(contract["origin_xy_m"], dtype=torch.float64)
    xy = means[:, :2].double()
    cells = torch.floor((xy - origin) / cell_size).to(torch.int64)
    unique_cells = sorted(tuple(map(int, row)) for row in torch.unique(cells, dim=0).tolist())
    rows = []
    lines = []
    for ix, iy in unique_cells:
        indices = torch.nonzero(
            (cells[:, 0] == ix) & (cells[:, 1] == iy), as_tuple=False
        ).reshape(-1)
        row = {
            "id": chunk_id(ix, iy),
            "ix": ix,
            "iy": iy,
            "count": int(indices.numel()),
            "source_flat_indices_sha256": index_sha256(indices),
        }
        rows.append(row)
        lines.append(
            f"{row['id']}\t{ix}\t{iy}\t{row['count']}\t"
            f"{row['source_flat_indices_sha256']}\n"
        )
    remainder = torch.remainder(xy - origin, cell_size)
    distance = torch.minimum(remainder, cell_size - remainder).amin(dim=1)
    counts = sorted(row["count"] for row in rows)
    return {
        "rows": rows,
        "occupied_chunk_count": len(rows),
        "background_count": int(means.shape[0]),
        "xyz_min": [float(value) for value in means.amin(dim=0)],
        "xyz_max": [float(value) for value in means.amax(dim=0)],
        "count_min": min(counts),
        "count_max": max(counts),
        "chunks_below_100": sum(value < 100 for value in counts),
        "chunks_at_least_10000": sum(value >= 10_000 for value in counts),
        "boundary_band_count": int(
            (distance <= float(contract["expected_source_inventory"]["boundary_band_m"])).sum()
        ),
        "inventory_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
    }


def build_actor_inventory(points_ids: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """按显式 points_ids 重算 24 个 actor 的索引清单。"""
    import torch

    values = points_ids.reshape(-1).to(torch.int64)
    first, last = map(int, contract["actor_index_domain_inclusive"])
    rows = []
    lines = []
    for actor_index in range(first, last + 1):
        indices = torch.nonzero(values == actor_index, as_tuple=False).reshape(-1)
        available = bool(indices.numel())
        contiguous = bool(
            not available
            or torch.equal(indices, torch.arange(indices[0], indices[0] + indices.numel()))
        )
        availability = "available" if available else "unavailable_empty"
        row = {
            "id": f"actor-{actor_index:04d}",
            "actor_index": actor_index,
            "availability": availability,
            "count": int(indices.numel()),
            "source_flat_indices_sha256": index_sha256(indices),
            "contiguous": contiguous,
        }
        rows.append(row)
        lines.append(
            f"{row['id']}\t{actor_index}\t{availability}\t{row['count']}\t"
            f"{row['source_flat_indices_sha256']}\n"
        )
    return {
        "rows": rows,
        "available_count": sum(row["availability"] == "available" for row in rows),
        "unavailable_count": sum(row["availability"] == "unavailable_empty" for row in rows),
        "row_count": sum(row["count"] for row in rows),
        "inventory_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
    }


def validate_source_layout(
    protocol: Mapping[str, Any], checkpoint_path: Path | None = None
) -> dict[str, Any]:
    """只读加载 P2-selected checkpoint 并核对冻结分块事实。"""
    import torch

    path = checkpoint_path or Path(protocol["selected_asset"]["checkpoint"]["path"])
    before = sha256_file(path)
    state = torch.load(path, map_location="cpu")
    models = state["models"]
    row_contract = protocol["row_tensor_schema"]
    schema_audit = {}
    for model_name in ("Background", "RigidNodes"):
        expected = COMMON_ROW_SCHEMA.copy()
        expected.extend(
            schema_tuples(row_contract["models"][model_name]["additional_row_tensors"])
        )
        row_count = int(row_contract["models"][model_name]["row_count"])
        actual = discover_row_tensor_schema(models[model_name], row_count)
        require(actual == expected, f"{model_name} row tensor schema drift")
        schema_audit[model_name] = {
            "row_count": row_count,
            "row_tensor_count": len(actual),
            "row_tensors": [row[0] for row in actual],
        }

    for model_name in protocol["source_precision_contract"]["models"]:
        model = models[model_name]
        require(model["_means"].dtype == torch.float32, f"{model_name} means dtype")
        for field in protocol["source_precision_contract"]["float16_row_fields"]:
            require(model[field].dtype == torch.float16, f"{model_name}.{field} dtype")

    static = build_static_inventory(
        models["Background"]["_means"], protocol["static_chunk_contract"]
    )
    expected_static = protocol["static_chunk_contract"]["expected_source_inventory"]
    for name in (
        "occupied_chunk_count",
        "background_count",
        "xyz_min",
        "xyz_max",
        "count_min",
        "count_max",
        "chunks_below_100",
        "chunks_at_least_10000",
        "boundary_band_count",
        "inventory_sha256",
    ):
        require(static[name] == expected_static[name], f"static source inventory {name}")

    actor = build_actor_inventory(
        models["RigidNodes"]["points_ids"], protocol["actor_chunk_contract"]
    )
    actor_contract = protocol["actor_chunk_contract"]
    require(actor["available_count"] == actor_contract["expected_available_count"], "actor available count")
    require(actor["unavailable_count"] == actor_contract["expected_unavailable_count"], "actor unavailable count")
    require(actor["row_count"] == protocol["selected_asset"]["inventory"]["rigid_gaussians"], "actor row total")
    require(actor["inventory_sha256"] == actor_contract["expected_inventory_sha256"], "actor inventory digest")
    expected_actor_rows = actor_contract["expected_actors"]
    for actual, expected in zip(actor["rows"], expected_actor_rows):
        require(actual["actor_index"] == expected["actor_index"], "actor index")
        require(actual["count"] == expected["count"], f"actor {actual['actor_index']} count")
        require(
            actual["source_flat_indices_sha256"] == expected["source_flat_indices_sha256"],
            f"actor {actual['actor_index']} index digest",
        )
        require(
            actual["contiguous"] == (actual["count"] == 0),
            f"actor {actual['actor_index']} interleaving fact",
        )
    require(sha256_file(path) == before, "source checkpoint changed during layout audit")
    return {
        "checkpoint_sha256": before,
        "row_tensor_schema": schema_audit,
        "static_inventory": {key: value for key, value in static.items() if key != "rows"},
        "actor_inventory": {
            "available_count": actor["available_count"],
            "unavailable_count": actor["unavailable_count"],
            "row_count": actor["row_count"],
            "inventory_sha256": actor["inventory_sha256"],
            "actors": actor["rows"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--skip-source-layout", action="store_true")
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    inputs = validate_inputs(protocol)
    source_layout = None if args.skip_source_layout else validate_source_layout(protocol)
    print(
        json.dumps(
            {
                "status": "passed",
                "protocol": str(args.protocol),
                "protocol_sha256": sha256_file(args.protocol),
                "input_audits": inputs,
                "source_layout_audit": source_layout,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
