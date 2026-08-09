#!/usr/bin/env python
"""校验 A4-P2 mixed-precision 协议、不可变输入与 source dtype 事实。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml"


def sha256_file(path: Path) -> str:
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
        raise RuntimeError(f"A4-P2 protocol invalid: {message}")


def endpoint_rows(
    contract: Mapping[str, Any], group: str
) -> list[Mapping[str, Any]]:
    return list(contract[group]["metrics"])


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-A4-DEPLOYMENT-01", "task_id")
    require(protocol["profile_id"] == "A4-P2-MIXED-PRECISION-v1", "profile_id")
    require(
        protocol["protocol_status"] == "frozen_before_new_p2_measurements",
        "protocol_status",
    )
    require(protocol["seed"] == 0 and protocol["scene"] == "scene-0230", "scene/seed")
    authorization = protocol["authorization"]
    require(authorization["p2_checkpoint_conversion_authorized"], "P2 conversion authorization")
    require(authorization["p2_candidate_render_authorized"], "P2 render authorization")
    for name in (
        "source_checkpoint_mutation_authorized",
        "training_authorized",
        "optimizer_authorized",
        "fp16_renderer_compute_authorized",
        "p3_chunk_authorized",
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
    precision = protocol["precision_contract"]
    require(
        [row["id"] for row in precision["arms"]]
        == ["p2-source", "p2-gs-param-fp16"],
        "arm grid",
    )
    require(
        precision["converted_models"] == ["Background", "RigidNodes"],
        "converted models",
    )
    require(
        precision["converted_fields"]
        == ["_scales", "_quats", "_features_dc", "_features_rest", "_opacities"],
        "converted field set/order",
    )
    require(precision["source_dtype_required"] == "float32", "source dtype")
    require(
        precision["checkpoint_candidate_dtype_required"] == "float16"
        and precision["runtime_candidate_dtype_required"] == "float16",
        "candidate dtype",
    )
    require(precision["renderer_input_dtype_required"] == "float32", "renderer dtype")
    require(
        precision["preserved_float32_fields"]
        == {
            "Background": ["_means"],
            "RigidNodes": ["_means", "instances_quats", "instances_trans", "instances_size"],
        },
        "preserved FP32 fields",
    )
    require(
        precision["means_fp16_exclusion"]["background_fp16_roundtrip_max_absolute_error"]
        == 0.999267578125,
        "means exclusion diagnostic",
    )
    adapter = precision["runtime_adapter"]
    require(not adapter["autocast_enabled"], "autocast disabled")
    require(not adapter["fp16_renderer_kernel_claim_allowed"], "renderer claim disabled")
    require(precision["candidate_checkpoint_write_count"] == 1, "candidate write count")
    require(precision["source_checkpoint_copy_forbidden"], "source copy forbidden")
    require(precision["checkpoint_schema_must_match_source"], "checkpoint schema")
    require(precision["candidate_registry_required"], "candidate registry")
    require(precision["raw_render_media_forbidden"], "raw media")

    quality = protocol["quality_contract"]
    require(quality["expected_views"] == 57, "quality views")
    require(quality["original_resolution"] == [800, 450], "quality resolution")
    require(quality["heldout_frames"] == list(range(10, 200, 10)), "quality frames")
    require(quality["cameras"] == [0, 1, 2], "quality cameras")
    require(len(endpoint_rows(quality, "global_endpoints")) == 11, "global endpoints")
    require(len(endpoint_rows(quality, "actor_endpoints")) == 4, "actor endpoints")
    require(len(endpoint_rows(quality, "non_target_endpoints")) == 4, "non-target endpoints")
    global_expected = {
        "image_metrics/test/human_psnr": ("higher", 0.05),
        "image_metrics/test/human_ssim": ("higher", 0.001),
        "image_metrics/test/lpips": ("lower", 0.001),
        "image_metrics/test/masked_psnr": ("higher", 0.05),
        "image_metrics/test/masked_ssim": ("higher", 0.001),
        "image_metrics/test/occupied_psnr": ("higher", 0.05),
        "image_metrics/test/occupied_ssim": ("higher", 0.001),
        "image_metrics/test/psnr": ("higher", 0.05),
        "image_metrics/test/ssim": ("higher", 0.001),
        "image_metrics/test/vehicle_psnr": ("higher", 0.05),
        "image_metrics/test/vehicle_ssim": ("higher", 0.001),
    }
    require(
        {
            row["name"]: (row["direction"], float(row["maximum_regression"]))
            for row in endpoint_rows(quality, "global_endpoints")
        }
        == global_expected,
        "global thresholds",
    )
    actor_expected = {
        "psnr": ("higher", 0.10),
        "ssim": ("higher", 0.0025),
        "masked_lpips_alex_tight_crop_256px": ("lower", 0.0025),
        "mean_absolute_error": ("lower", 0.001),
    }
    non_target_expected = {
        "psnr": ("higher", 0.05),
        "ssim": ("higher", 0.001),
        "masked_lpips_alex_tight_crop_256px": ("lower", 0.001),
        "mean_absolute_error": ("lower", 0.0005),
    }
    require(
        {
            row["name"]: (row["direction"], float(row["maximum_regression"]))
            for row in endpoint_rows(quality, "actor_endpoints")
        }
        == actor_expected,
        "actor thresholds",
    )
    require(
        {
            row["name"]: (row["direction"], float(row["maximum_regression"]))
            for row in endpoint_rows(quality, "non_target_endpoints")
        }
        == non_target_expected,
        "non-target thresholds",
    )
    require(quality["candidate_pass_rule"] == "all_31_quality_safeguards_pass", "quality pass rule")

    runtime = protocol["runtime_contract"]
    require(runtime["frames"] == [10, 100, 190], "runtime frames")
    require(runtime["cameras"] == [0, 1, 2], "runtime cameras")
    require(runtime["expected_samples_per_arm"] == 9, "runtime samples")
    require(runtime["warmup_views"] == 2, "runtime warmup")
    require(runtime["resolution"] == [800, 450], "runtime resolution")
    require(runtime["percentile"] == "nearest_rank", "runtime percentile")
    require(runtime["performance_values_are_report_only_not_quality_selection"], "runtime report only")

    selection = protocol["selection_contract"]
    require(selection["candidate_pass"]["selected_arm"] == "p2-gs-param-fp16", "candidate selection")
    require(selection["candidate_fail"]["selected_arm"] == "p2-source", "source fallback")
    require(selection["candidate_fail"]["fallback"] == "immutable_source_exact_alias", "fallback alias")
    require(selection["no_result_dependent_field_policy_or_threshold"], "no post-hoc policy")

    require(
        protocol["recovery_contract"]["stage_order"]
        == [
            "input_audit",
            "source_dtype_audit",
            "materialize_p2_gs_param_fp16",
            "evaluate_p2_source_and_candidate",
            "runtime_profile_both_arms",
            "aggregate",
            "resume_audit",
        ],
        "stage order",
    )
    ceilings = protocol["resource_ceilings"]
    require(
        ceilings
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
    require(len(protocol["required_audits"]) == 19, "required audit count")
    require(all(protocol["required_audits"].values()), "required audit values")
    require(all(protocol["claim_boundary"].values()), "claim boundary values")
    require("PIVOT-F27" in protocol["failure_precedents"], "PIVOT-F27 precedent")


def iter_fingerprinted_inputs(protocol: Mapping[str, Any]):
    for name in ("checkpoint", "source_config", "actor_registry"):
        yield f"selected_asset.{name}", protocol["selected_asset"][name]
    for name in (
        "summary",
        "manifest",
        "resource_audit",
        "source_quality",
        "resume_audit",
        "terminal",
    ):
        yield f"p1_canonical_evidence.{name}", protocol["p1_canonical_evidence"][name]


def validate_inputs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
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
        == protocol["p1_canonical_evidence"]["source_quality"],
        "baseline quality must alias P1 source quality",
    )
    mask_spec = protocol["baseline_quality"]["actor_masks"]
    mask_dir = Path(mask_spec["path"])
    require(mask_dir.is_dir(), f"missing actor masks: {mask_dir}")
    mask_audit = directory_digest(mask_dir, mask_spec["file_glob"])
    require(mask_audit["sha256"] == mask_spec["sha256"], "actor mask hash drift")
    require(mask_audit["file_count"] == int(mask_spec["file_count"]), "actor mask count drift")
    require(mask_audit["total_bytes"] == int(mask_spec["total_bytes"]), "actor mask bytes drift")
    audits["baseline_quality.actor_masks"] = {"path": str(mask_dir), **mask_audit}

    registry = json.loads(Path(protocol["selected_asset"]["actor_registry"]["path"]).read_text())
    require(
        registry["actor_registry_sha256"]
        == protocol["selected_asset"]["actor_registry"]["embedded_registry_sha256"],
        "embedded registry hash",
    )
    evidence = protocol["p1_canonical_evidence"]
    summary = json.loads(Path(evidence["summary"]["path"]).read_text())
    require(summary["status"] == "done" and all(summary["audits"].values()), "P1 summary")
    require(summary["project_commit"] == evidence["source_commit"], "P1 source commit")
    require(summary["selection"]["selected_arm"] == "p1-source", "P1 selected source")
    require(summary["selection"]["fallback_exact_alias"], "P1 exact fallback")
    require(
        summary["method_state"] == "rejected_quality_or_integrity_gate",
        "P1 method state",
    )
    require(
        summary["selected_asset"]["checkpoint"]["sha256"]
        == protocol["selected_asset"]["checkpoint"]["sha256"],
        "P1 selected checkpoint",
    )
    manifest = json.loads(Path(evidence["manifest"]["path"]).read_text())
    require(manifest["status"] == "done", "P1 manifest")
    require(manifest["summary_sha256"] == evidence["summary"]["sha256"], "P1 manifest summary")
    resources = json.loads(Path(evidence["resource_audit"]["path"]).read_text())
    require(resources["status"] == "passed", "P1 resources")
    source_quality = json.loads(Path(evidence["source_quality"]["path"]).read_text())
    require(source_quality["all_quality_safeguards_pass"], "P1 source quality")
    require(source_quality["baseline_historical_replay_pass"], "P1 source replay")
    resume = json.loads(Path(evidence["resume_audit"]["path"]).read_text())
    require(not resume["torch_imported"] and not resume["gpu_launch_observed"], "P1 resume")
    require(resume["all_completed_stages_reused"], "P1 completed stage reuse")
    terminal = json.loads(Path(evidence["terminal"]["path"]).read_text())
    require(terminal == {"failure": None, "status": "done"}, "P1 terminal")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["p1_closeout_commit"], "HEAD"],
        cwd=PROJECT,
        check=False,
    ).returncode == 0
    require(ancestor, "P1 closeout commit is not an ancestor of HEAD")
    return audits


def validate_checkpoint_state(
    protocol: Mapping[str, Any], checkpoint_path: Path | None = None
) -> dict[str, Any]:
    """加载 source checkpoint，核对转换字段与保留 means 的 dtype/量化事实。"""
    import torch

    path = checkpoint_path or Path(protocol["selected_asset"]["checkpoint"]["path"])
    before = sha256_file(path)
    state = torch.load(path, map_location="cpu")
    precision = protocol["precision_contract"]
    inventory = protocol["selected_asset"]["inventory"]
    expected_counts = {
        "Background": int(inventory["background_gaussians"]),
        "RigidNodes": int(inventory["rigid_gaussians"]),
    }
    rows = {}
    means_errors = {}
    for model_name in precision["converted_models"]:
        model = state["models"][model_name]
        require(int(model["_means"].shape[0]) == expected_counts[model_name], f"{model_name} count")
        require(model["_means"].dtype == torch.float32, f"{model_name} means dtype")
        require(torch.isfinite(model["_means"]).all().item(), f"{model_name} means finite")
        means_roundtrip = model["_means"].half().float()
        means_errors[model_name] = float((means_roundtrip - model["_means"]).abs().max())
        for field in precision["converted_fields"]:
            tensor = model[field]
            require(tensor.dtype == torch.float32, f"{model_name}.{field} dtype")
            require(torch.isfinite(tensor).all().item(), f"{model_name}.{field} finite")
            half = tensor.half()
            require(torch.isfinite(half).all().item(), f"{model_name}.{field} half finite")
            rows[f"models.{model_name}.{field}"] = {
                "shape": list(tensor.shape),
                "source_dtype": str(tensor.dtype).removeprefix("torch."),
                "candidate_dtype": str(half.dtype).removeprefix("torch."),
                "source_min": float(tensor.min()),
                "source_max": float(tensor.max()),
                "roundtrip_max_absolute_error": float((half.float() - tensor).abs().max()),
            }
    exclusion = precision["means_fp16_exclusion"]
    require(
        means_errors["Background"]
        == float(exclusion["background_fp16_roundtrip_max_absolute_error"]),
        "Background means exclusion drift",
    )
    require(
        means_errors["RigidNodes"]
        == float(exclusion["rigid_fp16_roundtrip_max_absolute_error"]),
        "Rigid means exclusion drift",
    )
    require(sha256_file(path) == before, "source checkpoint changed during dtype audit")
    return {
        "checkpoint_sha256": before,
        "converted_field_count": len(rows),
        "converted_fields": rows,
        "means_fp16_roundtrip_max_absolute_error": means_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--skip-checkpoint-state", action="store_true")
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    inputs = validate_inputs(protocol)
    checkpoint = None if args.skip_checkpoint_state else validate_checkpoint_state(protocol)
    print(
        json.dumps(
            {
                "status": "passed",
                "protocol": str(args.protocol),
                "protocol_sha256": sha256_file(args.protocol),
                "input_audits": inputs,
                "checkpoint_state_audit": checkpoint,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
