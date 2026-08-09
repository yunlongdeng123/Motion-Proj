#!/usr/bin/env python
"""校验 A4-P1 contribution-prune 结果前冻结协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
DEFAULT_PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml"


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
        raise RuntimeError(f"A4-P1 protocol invalid: {message}")


def endpoint_rows(
    names: Iterable[str], direction: str, maximum_regression: float
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "direction": direction,
            "maximum_regression": maximum_regression,
        }
        for name in names
    ]


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-A4-DEPLOYMENT-01", "task_id")
    require(
        protocol["profile_id"] == "A4-P1-CONTRIBUTION-PRUNE-v1", "profile_id"
    )
    require(
        protocol["protocol_status"] == "frozen_before_new_p1_measurements",
        "protocol_status",
    )
    require(protocol["seed"] == 0 and protocol["scene"] == "scene-0230", "split")
    require(
        protocol["authorization"]
        == {
            "p1_contribution_prune_execution_authorized": True,
            "source_checkpoint_mutation_authorized": False,
            "candidate_checkpoint_output_authorized": True,
            "candidate_render_authorized": True,
            "training_authorized": False,
            "optimizer_authorized": False,
            "p2_fp16_authorized": False,
            "p3_chunk_authorized": False,
            "p4_lod_authorized": False,
        },
        "authorization",
    )
    selected = protocol["selected_asset"]
    require(selected["role"] == "A3-star-R0-off-D2-immutable-exact-alias", "role")
    require(
        selected["inventory"]
        == {
            "background_gaussians": 1_205_164,
            "rigid_gaussians": 104_704,
            "total_gaussians": 1_309_868,
            "actor_count": 24,
            "available_actor_count": 23,
            "unavailable_actor_count": 1,
        },
        "source inventory",
    )
    masks = protocol["baseline_quality"]["actor_masks"]
    require(
        masks["digest_algorithm"]
        == "sha256_of_sorted_lines_sha256_two_spaces_dot_slash_relative_path_newline",
        "mask digest algorithm",
    )
    require(
        masks["file_glob"] == "*.png"
        and masks["file_count"] == 33
        and masks["total_bytes"] == 25_714,
        "frozen masks",
    )
    require(
        protocol["p5_canonical_evidence"]["source_commit"]
        == "0e899b2e6dcf7d5a091a0a4092ea99767c982357",
        "P5 source commit",
    )

    contribution = protocol["contribution_contract"]
    require(contribution["ranking_partition"] == "train_only", "ranking partition")
    require(not contribution["heldout_may_influence_ranking"], "heldout leakage")
    require(contribution["cameras"] == [0, 1, 2], "contribution cameras")
    require(
        contribution["training_discovery_frames"] == [5, 45, 85, 125, 165, 195],
        "training discovery frames",
    )
    require(
        contribution["heldout_audit_frames"] == [10, 50, 90, 130, 170, 190],
        "heldout audit frames",
    )
    require(
        contribution["full_quality_heldout_frames"]
        == list(range(10, 191, 10)),
        "full heldout frames",
    )
    require(
        set(contribution["training_discovery_frames"])
        .isdisjoint(contribution["full_quality_heldout_frames"]),
        "train/heldout overlap",
    )
    require(
        contribution["per_view_algorithm"]
        == {
            "intersections": "gsplat.cuda._wrapper.rasterize_to_indices_in_range_full_near_to_far",
            "grouping": "stable_sort_by_camera_then_pixel_preserve_near_to_far",
            "alpha": "clamp_max(opacity_times_exp_negative_sigma, 0.999)",
            "weight": "transmittance_before_intersection_times_alpha",
            "accumulation": "per_gaussian_cpu_float64_stable_order",
        },
        "contribution algorithm",
    )
    require(
        contribution["score_quantization"]
        == {"decimal_places": 12, "rounding": "round_half_to_even"},
        "score quantization",
    )
    require(
        contribution["rank_key"]
        == [
            "quantized_train_alpha_weight_sum_ascending",
            "train_visible_view_count_ascending",
            "learned_opacity_ascending",
            "gaussian_id_ascending",
            "model_flat_index_ascending",
        ],
        "rank key",
    )
    require(contribution["raw_render_media_forbidden"], "raw media")

    candidates = protocol["candidate_contract"]
    require(
        candidates["arms"]
        == [
            {"id": "p1-source", "prune_fraction": 0.0, "storage": "immutable_source_reference"},
            {"id": "p1-b05", "prune_fraction": 0.05, "storage": "atomic_candidate_checkpoint"},
            {"id": "p1-b10", "prune_fraction": 0.10, "storage": "atomic_candidate_checkpoint"},
            {"id": "p1-b20", "prune_fraction": 0.20, "storage": "atomic_candidate_checkpoint"},
        ],
        "candidate arms",
    )
    require(
        candidates["ranking_unit"]
        == "background_and_each_available_actor_independently",
        "ranking unit",
    )
    require(
        candidates["removal_count"]
        == "floor_source_asset_count_times_prune_fraction",
        "removal count",
    )
    require(candidates["no_actor_role_special_casing"], "actor role special casing")
    require(candidates["source_arm_checkpoint_copy_forbidden"], "source copy")
    require(candidates["candidate_checkpoint_write_count"] == 1, "checkpoint writes")
    require(
        candidates["checkpoint_schema"] == "source_schema_no_new_checkpoint_keys",
        "checkpoint schema",
    )
    require(
        candidates["unavailable_actor_policy"] == "preserve_explicit_empty_slice",
        "unavailable actor",
    )

    quality = protocol["quality_contract"]
    require(quality["baseline_arm"] == "p1-source", "baseline arm")
    require(quality["baseline_replay_required"], "baseline replay")
    require(
        quality["candidate_actor_masks"] == "reuse_frozen_baseline_mask_bytes_exact",
        "frozen mask reuse",
    )
    require(quality["candidate_mask_regeneration_forbidden"], "mask regeneration")
    require(quality["heldout_ranking_forbidden"], "heldout ranking")
    global_expected = (
        endpoint_rows(
            ["image_metrics/test/human_psnr"], "higher", 0.10
        )
        + endpoint_rows(
            ["image_metrics/test/human_ssim"], "higher", 0.002
        )
        + endpoint_rows(["image_metrics/test/lpips"], "lower", 0.002)
        + endpoint_rows(
            [
                "image_metrics/test/masked_psnr",
                "image_metrics/test/occupied_psnr",
                "image_metrics/test/psnr",
                "image_metrics/test/vehicle_psnr",
            ],
            "higher",
            0.10,
        )
        + endpoint_rows(
            [
                "image_metrics/test/masked_ssim",
                "image_metrics/test/occupied_ssim",
                "image_metrics/test/ssim",
                "image_metrics/test/vehicle_ssim",
            ],
            "higher",
            0.002,
        )
    )
    global_by_name = {row["name"]: row for row in quality["global_endpoints"]["metrics"]}
    expected_by_name = {row["name"]: row for row in global_expected}
    require(global_by_name == expected_by_name, "global quality thresholds")
    require(
        quality["actor_endpoints"]["roles"]
        == ["high-support", "boundary-support"],
        "actor roles",
    )
    require(
        quality["actor_endpoints"]["regions"] == ["actor_region", "boundary_band"],
        "actor regions",
    )
    require(
        quality["actor_endpoints"]["metrics"]
        == [
            {"name": "psnr", "direction": "higher", "maximum_regression": 0.20},
            {"name": "ssim", "direction": "higher", "maximum_regression": 0.005},
            {
                "name": "masked_lpips_alex_tight_crop_256px",
                "direction": "lower",
                "maximum_regression": 0.005,
            },
            {
                "name": "mean_absolute_error",
                "direction": "lower",
                "maximum_regression": 0.002,
            },
        ],
        "actor quality thresholds",
    )
    require(
        quality["non_target_endpoints"]["metrics"]
        == [
            {"name": "psnr", "direction": "higher", "maximum_regression": 0.10},
            {"name": "ssim", "direction": "higher", "maximum_regression": 0.002},
            {
                "name": "masked_lpips_alex_tight_crop_256px",
                "direction": "lower",
                "maximum_regression": 0.002,
            },
            {
                "name": "mean_absolute_error",
                "direction": "lower",
                "maximum_regression": 0.001,
            },
        ],
        "non-target quality thresholds",
    )
    require(
        quality["arm_pass_rule"]
        == "all_global_actor_boundary_non_target_safeguards_pass",
        "quality pass rule",
    )
    require(quality["missing_or_nonfinite_metric_policy"] == "arm_rejected", "missing metric")

    runtime = protocol["runtime_contract"]
    require(
        runtime["frames"] == [10, 100, 190]
        and runtime["cameras"] == [0, 1, 2]
        and runtime["warmup_views"] == 2,
        "runtime matrix",
    )
    require(runtime["resolution"] == [800, 450], "runtime resolution")
    require(runtime["cuda_synchronize_each_measurement"], "runtime synchronization")
    require(runtime["percentile"] == "nearest_rank", "runtime percentile")
    require(
        runtime["performance_values_are_report_only_not_quality_selection"],
        "runtime selection leakage",
    )

    selection = protocol["selection_contract"]
    require(
        selection["order"] == "largest_prune_fraction_then_arm_id"
        and selection["selected_arm"] == "largest_eligible_non_source_arm",
        "selection order",
    )
    require(
        selection["no_eligible_candidate"]
        == {
            "selected_asset": "p1-source-immutable-exact-alias",
            "method_state": "rejected_quality_or_integrity_gate",
            "p1_experiment_terminal": "done",
        },
        "fallback",
    )
    require(selection["no_result_dependent_new_arm_or_threshold"], "post-hoc arms")

    recovery = protocol["recovery_contract"]
    require(
        recovery["stage_order"]
        == [
            "input_audit",
            "contribution_scan",
            "materialize_p1_b05",
            "evaluate_p1_source_and_b05",
            "materialize_p1_b10",
            "evaluate_p1_b10",
            "materialize_p1_b20",
            "evaluate_p1_b20",
            "runtime_profile_all_arms",
            "aggregate",
            "resume_audit",
        ],
        "stage order",
    )
    require(recovery["completed_stage_policy"] == "never_overwrite", "overwrite policy")
    require(recovery["resume_probe_process"] == "no_torch_no_gpu", "resume process")

    require(
        protocol["resource_ceilings"]
        == {
            "wall_time_seconds": 1800,
            "peak_torch_allocated_mib": 20_480,
            "peak_torch_reserved_mib": 24_576,
            "peak_nvidia_process_memory_mib_sampled": 24_000,
            "peak_cgroup_memory_bytes": 51_539_607_552,
            "run_bytes": 2_500_000_000,
            "disk_free_floor_bytes": 30_000_000_000,
            "oom_events_delta": 0,
            "oom_kill_events_delta": 0,
        },
        "resource ceilings",
    )
    require(
        set(protocol["required_audits"])
        == {
            "all_fingerprinted_inputs_exact",
            "p0_and_p5_canonical_evidence_done",
            "source_checkpoint_registry_and_config_unchanged",
            "contribution_ranking_uses_train_only",
            "contribution_score_schema_and_hashes_exact",
            "heldout_contribution_is_audit_only",
            "candidate_grid_and_removal_counts_exact",
            "every_row_aligned_tensor_and_ancestry_field_pruned_by_same_mask",
            "invariant_checkpoint_fields_exact",
            "candidate_checkpoints_reload_and_counts_exact",
            "unavailable_actor_remains_explicitly_empty",
            "frozen_baseline_masks_reused_exactly",
            "baseline_replay_matches_frozen_metrics",
            "all_quality_endpoints_complete_and_finite",
            "candidate_quality_safeguards_applied_exactly",
            "checkpoint_and_gaussian_reduction_exact",
            "runtime_matrix_and_statistics_exact",
            "no_training_optimizer_or_raw_render_media",
            "resources_within_frozen_ceilings",
            "selection_and_fallback_rule_exact",
            "dry_run_resume_reuses_completed_stages_without_gpu_launch",
        },
        "audit schema",
    )
    require(all(bool(value) for value in protocol["required_audits"].values()), "audits")
    require(all(bool(value) for value in protocol["claim_boundary"].values()), "claims")
    require(
        {"PIVOT-F05", "PIVOT-F22", "PIVOT-F24", "PIVOT-F25", "PIVOT-F26"}
        <= set(protocol["failure_precedents"]),
        "failure precedents",
    )


def iter_fingerprinted_inputs(protocol: Mapping[str, Any]):
    for name in ("checkpoint", "source_config", "actor_registry"):
        yield f"selected_asset.{name}", protocol["selected_asset"][name]
    for name in ("d2_evaluation_summary", "actor_metrics_summary", "actor_metric_rows"):
        yield f"baseline_quality.{name}", protocol["baseline_quality"][name]
    for name in ("summary", "resource_audit", "runtime_rows"):
        yield f"p0_canonical_evidence.{name}", protocol["p0_canonical_evidence"][name]
    for name in ("summary", "deployment_registry", "resume_audit", "terminal"):
        yield f"p5_canonical_evidence.{name}", protocol["p5_canonical_evidence"][name]


def validate_inputs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for name, spec in iter_fingerprinted_inputs(protocol):
        path = Path(spec["path"])
        require(path.is_file(), f"missing {name}: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == spec["sha256"], f"hash drift {name}")
        actual_bytes = path.stat().st_size
        require(actual_bytes == int(spec["bytes"]), f"byte drift {name}")
        audits[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }

    mask_spec = protocol["baseline_quality"]["actor_masks"]
    mask_dir = Path(mask_spec["path"])
    require(mask_dir.is_dir(), f"missing actor masks: {mask_dir}")
    mask_audit = directory_digest(mask_dir, mask_spec["file_glob"])
    require(mask_audit["sha256"] == mask_spec["sha256"], "actor mask hash drift")
    require(mask_audit["file_count"] == mask_spec["file_count"], "actor mask count drift")
    require(mask_audit["total_bytes"] == mask_spec["total_bytes"], "actor mask bytes drift")
    audits["baseline_quality.actor_masks"] = {"path": str(mask_dir), **mask_audit}

    registry = json.loads(Path(protocol["selected_asset"]["actor_registry"]["path"]).read_text())
    require(
        registry["actor_registry_sha256"]
        == protocol["selected_asset"]["actor_registry"]["embedded_registry_sha256"],
        "embedded actor registry hash",
    )
    d2 = json.loads(Path(protocol["baseline_quality"]["d2_evaluation_summary"]["path"]).read_text())
    require(d2["checkpoint"]["sha256"] == protocol["selected_asset"]["checkpoint"]["sha256"], "D2 checkpoint")
    require(
        d2["checkpoint"]["background_gaussians"] == 1_205_164
        and d2["checkpoint"]["rigid_gaussians"] == 104_704,
        "D2 counts",
    )
    require(len(d2["heldout_metrics"]) == 11, "D2 heldout metrics")
    actor = json.loads(Path(protocol["baseline_quality"]["actor_metrics_summary"]["path"]).read_text())
    require(actor["status"] == "done", "actor metrics status")
    require(actor["heldout_split"]["test_image_count"] == 57, "actor heldout count")
    require(
        set(actor["roles"]) == {"high-support", "boundary-support"}
        and all(row["status"] == "done" for row in actor["roles"].values()),
        "actor roles",
    )
    require(actor["non_target"]["status"] == "done", "non-target status")

    p0 = json.loads(Path(protocol["p0_canonical_evidence"]["summary"]["path"]).read_text())
    require(p0["status"] == "done" and all(p0["audits"].values()), "P0 summary")
    p5 = json.loads(Path(protocol["p5_canonical_evidence"]["summary"]["path"]).read_text())
    require(p5["status"] == "done" and all(p5["audits"].values()), "P5 summary")
    require(p5["project_commit"] == protocol["p5_canonical_evidence"]["source_commit"], "P5 commit")
    require(p5["deployment_registry"]["bytes"] == 14_729, "P5 registry bytes")
    resume = json.loads(Path(protocol["p5_canonical_evidence"]["resume_audit"]["path"]).read_text())
    require(not resume["torch_imported"] and not resume["gpu_launch_observed"], "P5 resume")
    terminal = json.loads(Path(protocol["p5_canonical_evidence"]["terminal"]["path"]).read_text())
    require(terminal == {"failure": None, "status": "done"}, "P5 terminal")

    closeout = protocol["p5_closeout_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", closeout, "HEAD"],
        cwd=PROJECT,
        check=False,
    ).returncode == 0
    require(ancestor, "P5 closeout commit is not an ancestor of HEAD")
    return audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    inputs = {} if args.schema_only else validate_inputs(protocol)
    print(
        json.dumps(
            {
                "status": "done",
                "protocol": str(args.protocol),
                "protocol_sha256": sha256_file(args.protocol),
                "schema_only": args.schema_only,
                "input_audits": inputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
