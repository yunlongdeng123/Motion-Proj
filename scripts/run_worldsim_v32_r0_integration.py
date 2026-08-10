#!/usr/bin/env python
"""执行 WorldSim V3.2 R0 最终资产集成与单卡可复现验证。"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT = Path("/root/autodl-tmp/motion_proj")
DEFAULT_CONFIG = PROJECT / "configs/worldsim_v32/r0_final_integration_v1.yaml"
ACTIVE_RUN_DIR: Path | None = None

sys.path.insert(0, str(PROJECT))


from motion_proj.worldsim_v3.chunk_package import (
    compare_checkpoint_states,
    materialize_chunk_package,
    reassemble_chunk_package,
)
from motion_proj.worldsim_v3.mixed_precision import (
    apply_runtime_parameter_dtypes,
    conversion_audit,
    convert_checkpoint_state,
    install_fp32_renderer_input_adapter,
    renderer_adapter_summary,
    runtime_converted_field_audit,
    set_fp32_renderer_adapter_mode,
)
from motion_proj.worldsim_v32.integration import (
    build_chunk_protocol,
    extend_semantic_sidecar,
    validate_extended_semantic_sidecar,
)
from scripts.eval_worldsim_v3_a0_actor_metrics import to_device
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data,
    load_model_checkpoint_read_only,
    release_trainer_render_info,
)
from scripts.run_worldsim_v3_a4_p0_profile import ResourceSampler, rgb_sha256
from scripts.run_worldsim_v3_a4_p1_prune import cgroup_memory_events, nvidia_compute_rows
from scripts.run_worldsim_v3_a4_p1_worker import build_runtime


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def artifact_record(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def verify_record(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    actual = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    actual["exact"] = (
        actual["bytes"] == int(record["bytes"])
        and actual["sha256"] == str(record["sha256"])
    )
    if not actual["exact"]:
        raise RuntimeError(f"输入资产漂移：{path}")
    return actual


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def create_run(config: Mapping[str, Any], config_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(str(config["output_root"])) / f"{timestamp}__{config['run_label']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": config["task_id"],
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
        },
    )
    snapshot_root = run_dir / "source_snapshot"
    sources = [
        config_path,
        PROJECT / "scripts/run_worldsim_v32_r0_integration.py",
        PROJECT / "motion_proj/worldsim_v32/integration.py",
        PROJECT / "motion_proj/worldsim_v3/mixed_precision.py",
        PROJECT / "motion_proj/worldsim_v3/chunk_package.py",
    ]
    snapshots = {}
    for source in sources:
        relative = source.relative_to(PROJECT)
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[str(relative)] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v32_r0_run_manifest_v1",
            "task_id": config["task_id"],
            "status": "running",
            "project_commit": git_output("rev-parse", "HEAD"),
            "project_branch": git_output("branch", "--show-current"),
            "project_dirty": bool(git_output("status", "--porcelain")),
            "config_sha256": sha256_file(config_path),
            "source_snapshots": snapshots,
            "artifacts": {},
        },
    )
    return run_dir


def load_config(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def input_records(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {
        "source_config": config["source_config"],
        "source_checkpoint": config["source_checkpoint"],
        "base_actor_registry": config["base_actor_registry"],
        "generated_background_provenance": config["generated_background_provenance"],
        "selected_actor_asset": config["selected_actor_asset"],
        "selected_actor_manifest": config["selected_actor_manifest"],
        "harmonizer_diagnostic": config["harmonizer_diagnostic"],
    }
    for role, record in config["semantic_sidecars"].items():
        records[f"semantic_sidecar_{role}"] = record
    return records


def validate_generated_background(config: Mapping[str, Any]) -> dict[str, Any]:
    counts = config["counts"]
    with np.load(
        config["generated_background_provenance"]["path"], allow_pickle=False
    ) as payload:
        row_index = np.asarray(payload["background_row_index"])
        provenance = np.asarray(payload["provenance_code"])
        observed = np.asarray(payload["observed_cross_view"])
        target_code = np.asarray(payload["target_code"])
        confidence = np.asarray(payload["confidence"])
        finite_payload = all(
            bool(np.isfinite(payload[name]).all())
            for name in ("means", "rgb", "scales", "confidence")
        )
    expected = np.arange(
        int(counts["old_background"]),
        int(counts["final_background"]),
        dtype=np.int64,
    )
    audit = {
        "row_count": int(row_index.size),
        "expected_row_count": int(counts["generated_background"]),
        "row_indices_contiguous_exact": bool(np.array_equal(row_index, expected)),
        "provenance_code_all_generated_background": bool(np.all(provenance == 1)),
        "cross_view_true_count": int(observed.sum()),
        "cross_view_false_count": int((~observed).sum()),
        "cross_view_counts_exact": bool(
            int(observed.sum())
            == int(config["generated_background_provenance"]["expected_cross_view_true"])
            and int((~observed).sum())
            == int(config["generated_background_provenance"]["expected_cross_view_false"])
        ),
        "target_code_0_count": int((target_code == 0).sum()),
        "target_code_1_count": int((target_code == 1).sum()),
        "target_code_counts_exact": bool(
            int((target_code == 0).sum())
            == int(config["generated_background_provenance"]["expected_target_code_0"])
            and int((target_code == 1).sum())
            == int(config["generated_background_provenance"]["expected_target_code_1"])
            and bool(np.all((target_code == 0) | (target_code == 1)))
        ),
        "confidence_matches_cross_view_contract": bool(
            np.all(confidence[observed] == np.float32(0.9))
            and np.all(confidence[~observed] == np.float32(0.5))
        ),
        "numeric_payload_finite": finite_payload,
    }
    audit["all_exact"] = all(
        bool(value)
        for key, value in audit.items()
        if key
        not in {
            "row_count",
            "expected_row_count",
            "cross_view_true_count",
            "cross_view_false_count",
            "target_code_0_count",
            "target_code_1_count",
        }
    ) and audit["row_count"] == audit["expected_row_count"]
    return audit


def extend_semantics(
    run_dir: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    counts = config["counts"]
    audits = {}
    records = {}
    for role, source in config["semantic_sidecars"].items():
        output = run_dir / str(config["semantic_outputs"][role])
        extension = extend_semantic_sidecar(
            Path(str(source["path"])),
            output,
            old_background_count=int(counts["old_background"]),
            generated_background_count=int(counts["generated_background"]),
            rigid_count=int(counts["rigid"]),
        )
        validation = validate_extended_semantic_sidecar(
            Path(str(source["path"])),
            output,
            old_background_count=int(counts["old_background"]),
            generated_background_count=int(counts["generated_background"]),
            rigid_count=int(counts["rigid"]),
        )
        audits[role] = {"extension": extension, "validation": validation}
        records[role] = artifact_record(output, run_dir)
    return audits, records


def conversion_exact(audit: Mapping[str, Any]) -> bool:
    return bool(
        audit["converted_field_set_exact"]
        and audit["all_converted_fields_bitwise_exact"]
        and audit["preserved_tensors_exact"]
        and audit["checkpoint_schema_exact"]
    )


def convert_checkpoint(
    run_dir: Path, config: Mapping[str, Any]
) -> tuple[OrderedDict, dict[str, Any], dict[str, Any]]:
    source_path = Path(str(config["source_checkpoint"]["path"]))
    source = torch.load(source_path, map_location="cpu")
    counts = config["counts"]
    model_counts = {
        name: int(source["models"][name]["_means"].shape[0])
        for name in ("Background", "RigidNodes")
    }
    if model_counts != {
        "Background": int(counts["final_background"]),
        "RigidNodes": int(counts["rigid"]),
    }:
        raise RuntimeError(f"S2 checkpoint Gaussian 计数漂移：{model_counts}")
    precision = config["precision"]
    candidate = convert_checkpoint_state(
        source,
        models=precision["converted_models"],
        fields=precision["converted_fields"],
    )
    before_save = conversion_audit(
        source,
        candidate,
        models=precision["converted_models"],
        fields=precision["converted_fields"],
    )
    output = run_dir / str(precision["output"])
    atomic_torch_save(output, candidate)
    del candidate
    reloaded = torch.load(output, map_location="cpu")
    after_reload = conversion_audit(
        source,
        reloaded,
        models=precision["converted_models"],
        fields=precision["converted_fields"],
    )
    record = artifact_record(output, run_dir)
    audit = {
        "source_checkpoint": dict(config["source_checkpoint"]),
        "candidate_checkpoint": record,
        "model_counts": model_counts,
        "before_save": before_save,
        "after_reload": after_reload,
        "before_save_exact": conversion_exact(before_save),
        "after_reload_exact": conversion_exact(after_reload),
        "checkpoint_bytes_strictly_less_than_source": record["bytes"]
        < int(config["source_checkpoint"]["bytes"]),
        "source_checkpoint_unchanged": sha256_file(source_path)
        == config["source_checkpoint"]["sha256"],
    }
    audit["all_exact"] = bool(
        audit["before_save_exact"]
        and audit["after_reload_exact"]
        and audit["checkpoint_bytes_strictly_less_than_source"]
        and audit["source_checkpoint_unchanged"]
    )
    del source
    return reloaded, audit, record


def build_registry(
    run_dir: Path,
    config: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    semantic_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor_manifest = json.loads(
        Path(str(config["selected_actor_manifest"]["path"])).read_text(encoding="utf-8")
    )
    with np.load(config["selected_actor_asset"]["path"], allow_pickle=False) as asset:
        asset_count = int(asset["means"].shape[0])
    selected = config["selected_actor_asset"]
    identity_exact = bool(
        actor_manifest["instance_token"] == selected["instance_token"]
        and int(actor_manifest["dataset_instance_id"]) == int(selected["dataset_instance_id"])
        and int(actor_manifest["rigid_model_index"]) == int(selected["rigid_model_index"])
        and actor_manifest["generation_provenance"] == "GENERATED_ACTOR"
        and asset_count == int(selected["gaussian_count"])
    )
    if not identity_exact:
        raise RuntimeError("S3 actor asset 身份或点数合同失败")
    registry = {
        "schema_version": "worldsim_v32_asset_registry_v1",
        "status": "done",
        "production_chain": {
            "scene": "S2_GENERATED_BACKGROUND_MIXED_PRECISION",
            "semantics": "S1_EXTENDED_WITH_GENERATED_BACKGROUND_NEGATIVES",
            "actor_override": "S3_GENERATED_ACTOR_HIGH_SUPPORT_2VIEW",
            "delivery": "R0_EXACT_CHUNK_PACKAGE",
        },
        "scene_checkpoint": dict(checkpoint_record),
        "base_actor_registry": dict(config["base_actor_registry"]),
        "semantic_sidecars": {
            role: dict(record) for role, record in semantic_records.items()
        },
        "generated_background": {
            "provenance": "GENERATED_BACKGROUND",
            "row_range_half_open": [
                int(config["counts"]["old_background"]),
                int(config["counts"]["final_background"]),
            ],
            "source": dict(config["generated_background_provenance"]),
        },
        "actor_overrides": {
            str(selected["rigid_model_index"]): {
                "instance_token": selected["instance_token"],
                "dataset_instance_id": int(selected["dataset_instance_id"]),
                "rigid_model_index": int(selected["rigid_model_index"]),
                "provenance": "GENERATED_ACTOR",
                "gaussian_count": asset_count,
                "asset": {
                    key: selected[key] for key in ("path", "sha256", "bytes")
                },
                "manifest": dict(config["selected_actor_manifest"]),
            }
        },
        "fallback_policy": {
            "boundary_support_actor": "V3.1_NATIVE_RIGIDNODES",
            "other_actors": "V3.1_BASE_REGISTRY",
        },
        "excluded_from_production": {
            "S4_HARMONIZER_NONTEMPORAL": {
                "reason": config["harmonizer_diagnostic"]["reason"],
                "diagnostic": dict(config["harmonizer_diagnostic"]),
            },
            "S5_TEMPORAL_HARMONIZER": {
                "status": "blocked",
                "reason": "external_gated_model_authorization_and_official_runtime_unavailable",
            },
        },
        "no_source_checkpoint_or_d2_mutation": True,
    }
    output = run_dir / str(config["registry_output"])
    atomic_json(output, registry)
    record = artifact_record(output, run_dir)
    return {
        "actor_identity_exact": identity_exact,
        "actor_asset_gaussian_count": asset_count,
        "registry_sha256": record["sha256"],
        "all_exact": identity_exact,
    }, record


def build_and_validate_chunks(
    run_dir: Path,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    project_commit: str,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_external = {
        **dict(checkpoint_record),
        "path": str(run_dir / str(checkpoint_record["path"])),
    }
    registry_external = {
        **dict(registry_record),
        "path": str(run_dir / str(registry_record["path"])),
    }
    protocol = build_chunk_protocol(
        state,
        checkpoint_record=checkpoint_external,
        registry_record=registry_external,
    )
    protocol_path = run_dir / "artifacts/chunk_protocol.json"
    atomic_json(protocol_path, protocol)
    package_root = run_dir / str(config["chunk_package_root"])
    package_manifest = materialize_chunk_package(
        state,
        package_root=package_root,
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_path),
        project_commit=project_commit,
    )
    manifest_path = run_dir / str(config["chunk_manifest_output"])
    atomic_json(manifest_path, package_manifest)
    reassembled, package_audit = reassemble_chunk_package(
        package_root=package_root,
        manifest=package_manifest,
        protocol=protocol,
    )
    comparison = compare_checkpoint_states(state, reassembled)
    audit = {
        "protocol": artifact_record(protocol_path, run_dir),
        "package_manifest": artifact_record(manifest_path, run_dir),
        "package_counts": package_manifest["counts"],
        "package_payload_bytes": package_manifest["payload_bytes"],
        "package_payload_sha256": package_manifest["payload_sha256"],
        "package_audit": package_audit,
        "checkpoint_comparison": comparison,
        "all_exact": bool(
            package_audit["manifest_records_exact"]
            and package_audit["row_fields_exact"]
            and package_audit["static_cell_membership_exact"]
            and package_audit["actor_membership_exact"]
            and package_audit["indices_unique_disjoint_exhaustive"]
            and comparison["all_exact"]
        ),
    }
    return reassembled, audit, protocol, artifact_record(manifest_path, run_dir)


def uint8_render(output: Mapping[str, torch.Tensor]) -> np.ndarray:
    value = output["rgb"].detach().float().cpu().numpy()
    if not np.isfinite(value).all():
        raise RuntimeError("渲染结果包含非有限值")
    return np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_views(
    trainer: Any,
    dataset: Any,
    views: list[Mapping[str, Any]],
    device: torch.device,
    arm: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    images = []
    rows = []
    for ordinal, view in enumerate(views):
        image_infos, camera_infos, _, _, _, image_index = get_view_data(
            dataset, int(view["frame"]), int(view["camera"]), device
        )
        try:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                output = trainer(image_infos, camera_infos)
            torch.cuda.synchronize(device)
            rgb = uint8_render(output)
            images.append(rgb)
            rows.append(
                {
                    "arm": arm,
                    "ordinal": ordinal,
                    "frame": int(view["frame"]),
                    "camera": int(view["camera"]),
                    "camera_name": str(view["camera_name"]),
                    "image_index": int(image_index),
                    "height": int(rgb.shape[0]),
                    "width": int(rgb.shape[1]),
                    "duration_seconds": time.perf_counter() - started,
                    "rgb_sha256": rgb_sha256(rgb),
                }
            )
        finally:
            release_trainer_render_info(trainer)
    return images, rows


def load_memory_state(trainer: Any, state: Mapping[str, Any], device: torch.device) -> None:
    mutable = type(state)(state.items())
    models = type(state["models"])(state["models"].items())
    models["RigidNodes"] = to_device(models["RigidNodes"], device)
    mutable["models"] = models
    trainer.load_state_dict(mutable, load_only_model=True, strict=True)


def compare_images(source: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = candidate.astype(np.int16) - source.astype(np.int16)
    squared = np.square(difference.astype(np.float64))
    mse = float(squared.mean())
    psnr = 99.0 if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse))
    return {
        "mean_absolute_error_uint8": float(np.abs(difference).mean()),
        "max_absolute_error_uint8": int(np.abs(difference).max()),
        "mse_uint8": mse,
        "psnr_db": psnr,
        "changed_value_fraction": float(np.count_nonzero(difference) / difference.size),
    }


def validate_runtime(
    run_dir: Path,
    config: Mapping[str, Any],
    mixed_checkpoint: Path,
    reassembled: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    mini_protocol = {"selected_asset": {"source_config": config["source_config"]}}
    _, dataset, trainer = build_runtime(mini_protocol, device)
    if hasattr(trainer, "optimizer"):
        raise RuntimeError("R0 只读验证意外构造 optimizer")
    install_fp32_renderer_input_adapter(trainer)
    trainer.set_eval()
    views = list(config["render_validation"]["views"])

    load_model_checkpoint_read_only(
        trainer, Path(str(config["source_checkpoint"]["path"])), device
    )
    apply_runtime_parameter_dtypes(trainer, candidate=False)
    set_fp32_renderer_adapter_mode(trainer, candidate=False)
    trainer.set_eval()
    source_images, source_rows = render_views(trainer, dataset, views, device, "s2-fp32")
    source_dtype = runtime_converted_field_audit(trainer, expected_dtype="float32")

    load_model_checkpoint_read_only(trainer, mixed_checkpoint, device)
    apply_runtime_parameter_dtypes(trainer, candidate=True)
    set_fp32_renderer_adapter_mode(trainer, candidate=True)
    trainer.set_eval()
    candidate_images, candidate_rows = render_views(
        trainer, dataset, views, device, "v32-mixed"
    )
    candidate_dtype = runtime_converted_field_audit(trainer, expected_dtype="float16")
    candidate_adapter = renderer_adapter_summary(trainer)

    load_memory_state(trainer, reassembled, device)
    apply_runtime_parameter_dtypes(trainer, candidate=True)
    set_fp32_renderer_adapter_mode(trainer, candidate=True)
    trainer.set_eval()
    reassembled_images, reassembled_rows = render_views(
        trainer, dataset, views, device, "v32-reassembled"
    )
    reassembled_dtype = runtime_converted_field_audit(trainer, expected_dtype="float16")
    reassembled_adapter = renderer_adapter_summary(trainer)

    comparisons = []
    media_records = []
    thresholds = config["render_validation"]
    for index, view in enumerate(views):
        quality = compare_images(source_images[index], candidate_images[index])
        candidate_reassembled_exact = bool(
            np.array_equal(candidate_images[index], reassembled_images[index])
        )
        quality["source_candidate_gate_pass"] = bool(
            quality["psnr_db"] >= float(thresholds["min_source_candidate_psnr_db"])
            and quality["mean_absolute_error_uint8"]
            <= float(thresholds["max_source_candidate_mae_uint8"])
        )
        quality["candidate_reassembled_rgb_exact"] = candidate_reassembled_exact
        quality["candidate_rgb_sha256"] = candidate_rows[index]["rgb_sha256"]
        quality["reassembled_rgb_sha256"] = reassembled_rows[index]["rgb_sha256"]
        comparisons.append({"view": dict(view), **quality})
        difference = np.abs(
            candidate_images[index].astype(np.int16) - source_images[index].astype(np.int16)
        ).astype(np.uint8)
        panel = np.concatenate(
            [
                source_images[index],
                candidate_images[index],
                reassembled_images[index],
                np.clip(difference.astype(np.uint16) * 32, 0, 255).astype(np.uint8),
            ],
            axis=1,
        )
        media_path = run_dir / "artifacts/render_comparisons" / (
            f"f{int(view['frame']):03d}_c{int(view['camera'])}.png"
        )
        media_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(media_path, panel)
        media_records.append(artifact_record(media_path, run_dir))

    audit = {
        "views": views,
        "source": source_rows,
        "candidate": candidate_rows,
        "reassembled": reassembled_rows,
        "comparisons": comparisons,
        "source_runtime_dtype_exact": source_dtype["exact"],
        "candidate_runtime_dtype_exact": candidate_dtype["exact"],
        "reassembled_runtime_dtype_exact": reassembled_dtype["exact"],
        "candidate_renderer_adapter": candidate_adapter,
        "reassembled_renderer_adapter": reassembled_adapter,
        "comparison_media": media_records,
        "no_optimizer_constructed_or_step_executed": not hasattr(trainer, "optimizer"),
    }
    audit["all_exact"] = bool(
        audit["source_runtime_dtype_exact"]
        and audit["candidate_runtime_dtype_exact"]
        and audit["reassembled_runtime_dtype_exact"]
        and candidate_adapter["all_renderer_inputs_float32"]
        and reassembled_adapter["all_renderer_inputs_float32"]
        and audit["no_optimizer_constructed_or_step_executed"]
        and all(row["source_candidate_gate_pass"] for row in comparisons)
        and all(row["candidate_reassembled_rgb_exact"] for row in comparisons)
    )
    del trainer, dataset
    torch.cuda.empty_cache()
    return audit


def write_report(
    run_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    rows = summary["render_audit"]["comparisons"]
    report = [
        "# WorldSim V3.2 R0 最终集成报告",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 正式 run：`{run_dir}`",
        "- 生产 3D 场景：S2 generated-background 检查点的混合精度版本",
        "- 语义：S1 两份 sidecar 已为 S2 新增 Background 行插入 actor-negative/zero evidence",
        "- actor：高支持目标使用 S3 `GENERATED_ACTOR` 外部资产覆盖；其余 actor 沿用 V3.1 registry",
        "- 交付：mixed checkpoint + exact static/actor chunk package",
        "- S4：仅诊断，因语义重引入门失败而未进入生产链",
        "- S5：blocked（外部 gated 权重授权与官方 runtime 条件）",
        "",
        "## 固定视角验证",
        "",
        "| frame/camera | source→mixed PSNR | 最大 uint8 误差 | mixed↔reassembled RGB |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        view = row["view"]
        report.append(
            f"| {view['frame']}/{view['camera']} | {row['psnr_db']:.4f} dB | "
            f"{row['max_absolute_error_uint8']} | "
            f"{'exact' if row['candidate_reassembled_rgb_exact'] else 'mismatch'} |"
        )
    report.extend(
        [
            "",
            "## 裁决",
            "",
            f"- 语义扩展：`{summary['gates']['semantic_extension_exact']}`",
            f"- 混合精度转换：`{summary['gates']['mixed_precision_exact']}`",
            f"- chunk 精确重组：`{summary['gates']['chunk_reassembly_exact']}`",
            f"- 固定视角：`{summary['gates']['render_validation_exact']}`",
            f"- 输入不变：`{summary['gates']['all_inputs_unchanged']}`",
            f"- 资源门：`{summary['gates']['resources_within_ceilings']}`",
            "",
            "本 run 未训练、未执行 optimizer step，也未改写 D2、S1、S2、S3、S4 的正式输入资产。",
            "",
        ]
    )
    path = run_dir / "FINAL_INTEGRATION_REPORT.md"
    atomic_text(path, "\n".join(report))
    return artifact_record(path, run_dir)


def run(config_path: Path) -> int:
    global ACTIVE_RUN_DIR
    config_path = config_path.resolve()
    config = load_config(config_path)
    run_dir = create_run(config, config_path)
    ACTIVE_RUN_DIR = run_dir
    started = time.perf_counter()
    input_before = {name: verify_record(record) for name, record in input_records(config).items()}
    if nvidia_compute_rows():
        raise RuntimeError("R0 GPU preflight 非空，拒绝与其他进程共享正式测量")
    if not torch.cuda.is_available():
        raise RuntimeError("R0 需要单卡 CUDA 做只读推理验证")
    device = torch.device(str(config["runtime"]["device"]))
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if str(config["runtime"]["expected_gpu"]) not in gpu_name:
        raise RuntimeError(f"GPU 型号不符：{gpu_name}")
    torch.empty((), device=device)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    oom_before = cgroup_memory_events()

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with ResourceSampler(os.getpid()) as sampler:
        generated_background_audit = validate_generated_background(config)
        semantic_audit, semantic_records = extend_semantics(run_dir, config)
        state, conversion_audit_payload, checkpoint_record = convert_checkpoint(
            run_dir, config
        )
        registry_audit, registry_record = build_registry(
            run_dir, config, checkpoint_record, semantic_records
        )
        reassembled, chunk_audit, protocol, chunk_manifest_record = (
            build_and_validate_chunks(
                run_dir,
                config,
                state,
                checkpoint_record,
                registry_record,
                manifest["project_commit"],
            )
        )
        render_audit = validate_runtime(
            run_dir,
            config,
            run_dir / str(checkpoint_record["path"]),
            reassembled,
            device,
        )
        del state, reassembled
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    sampled = sampler.summary()
    oom_after = cgroup_memory_events()
    oom_delta = int(oom_after.get("oom", 0) - oom_before.get("oom", 0))
    oom_kill_delta = int(
        oom_after.get("oom_kill", 0) - oom_before.get("oom_kill", 0)
    )
    input_after = {name: verify_record(record) for name, record in input_records(config).items()}
    inputs_unchanged = input_before == input_after
    ceilings = config["resource_ceilings"]
    peak_nvidia = sampled["peak_nvidia_process_memory_mib_sampled"]
    peak_cgroup = sampled["peak_cgroup_memory_bytes_sampled"]
    resource_audit = {
        **sampled,
        "gpu_name": gpu_name,
        "peak_torch_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "peak_torch_reserved_mib": float(
            torch.cuda.max_memory_reserved(device) / (1024**2)
        ),
        "wall_seconds": elapsed,
        "oom_delta": oom_delta,
        "oom_kill_delta": oom_kill_delta,
        "ceilings": dict(ceilings),
    }
    preliminary_bytes = directory_bytes(run_dir)
    resource_audit["run_bytes_at_audit"] = preliminary_bytes
    resource_audit["within_ceilings"] = bool(
        not sampled["sampling_errors"]
        and peak_nvidia is not None
        and peak_nvidia <= int(ceilings["max_peak_nvidia_process_memory_mib"])
        and peak_cgroup is not None
        and peak_cgroup <= int(ceilings["max_peak_cgroup_memory_bytes"])
        and elapsed <= float(ceilings["max_wall_seconds"])
        and preliminary_bytes <= int(ceilings["max_run_bytes"])
        and (
            not bool(ceilings["require_zero_oom_delta"])
            or (oom_delta == 0 and oom_kill_delta == 0)
        )
    )

    reports_dir = run_dir / "reports"
    atomic_json(reports_dir / "semantic_audit.json", {
        "generated_background": generated_background_audit,
        "sidecars": semantic_audit,
    })
    atomic_json(reports_dir / "conversion_audit.json", conversion_audit_payload)
    atomic_json(reports_dir / "registry_audit.json", registry_audit)
    atomic_json(reports_dir / "chunk_audit.json", chunk_audit)
    atomic_json(reports_dir / "render_audit.json", render_audit)
    atomic_json(reports_dir / "resource_audit.json", resource_audit)
    atomic_json(reports_dir / "input_immutability.json", {
        "before": input_before,
        "after": input_after,
        "all_inputs_unchanged": inputs_unchanged,
    })

    gates = {
        "generated_background_provenance_exact": generated_background_audit["all_exact"],
        "semantic_extension_exact": all(
            row["validation"]["all_exact"] for row in semantic_audit.values()
        ),
        "mixed_precision_exact": conversion_audit_payload["all_exact"],
        "actor_registry_exact": registry_audit["all_exact"],
        "chunk_reassembly_exact": chunk_audit["all_exact"],
        "render_validation_exact": render_audit["all_exact"],
        "all_inputs_unchanged": inputs_unchanged,
        "resources_within_ceilings": resource_audit["within_ceilings"],
    }
    final_status = "done" if all(gates.values()) else "rejected"
    summary = {
        "schema_version": "worldsim_v32_r0_summary_v1",
        "task_id": config["task_id"],
        "status": final_status,
        "run_dir": str(run_dir),
        "production_candidate_selected": final_status == "done",
        "production_chain": [
            "S1_EXTENDED_SEMANTIC_SIDECARS",
            "S2_GENERATED_BACKGROUND_MIXED_PRECISION_SCENE",
            "S3_GENERATED_ACTOR_OVERRIDE",
            "R0_EXACT_CHUNK_PACKAGE",
        ],
        "s4_disposition": "optional_diagnostic_excluded_semantic_gate_failed",
        "s5_disposition": "blocked_external_gated_model_authorization_and_runtime",
        "gates": gates,
        "semantic_records": semantic_records,
        "checkpoint_record": checkpoint_record,
        "registry_record": registry_record,
        "chunk_manifest_record": chunk_manifest_record,
        "chunk_protocol_sha256": canonical_sha256(protocol),
        "render_audit": render_audit,
        "resource_audit": resource_audit,
        "training_or_optimizer_step_performed": False,
    }
    report_record = write_report(run_dir, config, summary)
    summary["final_report"] = report_record
    summary_path = run_dir / "summary.json"
    atomic_json(summary_path, summary)

    manifest["status"] = final_status
    manifest["artifacts"] = {
        "summary": artifact_record(summary_path, run_dir),
        "final_report": report_record,
        "mixed_checkpoint": checkpoint_record,
        "asset_registry": registry_record,
        "semantic_sidecars": semantic_records,
        "chunk_manifest": chunk_manifest_record,
        "reports": {
            path.stem: artifact_record(path, run_dir)
            for path in sorted(reports_dir.glob("*.json"))
        },
    }
    atomic_json(manifest_path, manifest, replace=True)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": config["task_id"],
            "status": final_status,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary_sha256": sha256_file(summary_path),
            "all_gates_pass": all(gates.values()),
        },
        replace=True,
    )
    print(json.dumps({
        "run_dir": str(run_dir),
        "status": final_status,
        "gates": gates,
        "summary_sha256": sha256_file(summary_path),
    }, indent=2, ensure_ascii=False))
    return 0 if final_status == "done" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.config))
    except SystemExit:
        raise
    except Exception as error:
        if ACTIVE_RUN_DIR is not None:
            failure = {
                "status": "rejected",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(ACTIVE_RUN_DIR / "failure.json", failure)
            atomic_json(
                ACTIVE_RUN_DIR / "status.json",
                failure,
                replace=True,
            )
        raise


if __name__ == "__main__":
    main()
