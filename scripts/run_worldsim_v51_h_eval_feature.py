#!/usr/bin/env python3
"""Extract frozen H heldout DINO features and transform with the frozen H PCA only."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any

import numpy as np
import scipy
import torch
import xformers
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.view_selection import atomic_save_deterministic_npz
from motion_proj.worldsim_v51.feature_sidecar import (
    array_sha256,
    pca_patch_grid,
    record_chain_sha256,
    select_h_uplift_records,
    sidecar_relative_path,
    standardize_in_place,
    validate_sidecar_identity,
)
from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
)
from scripts.run_worldsim_v51_h_feature_pca import (
    ResourceMonitor,
    _git,
    _gpu_name_and_total,
    _inventory,
    _nvidia_used_mib,
    _predict,
    _preprocess,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


def _load_pca_state(config: dict[str, Any]) -> dict[str, np.ndarray]:
    spec = config["feature_freeze"]
    path = Path(spec["run_path"]) / spec["pca_state_path"]
    if not path.is_file() or sha256_file(path) != spec["pca_state_sha256"]:
        raise ProtocolError("H eval PCA state identity 漂移")
    with np.load(path, allow_pickle=False) as archive:
        state = {name: np.asarray(archive[name]) for name in archive.files}
    expected = {
        "feature_mean": ((1536,), "float64"),
        "feature_std": ((1536,), "float64"),
        "pca_mean": ((1536,), "float32"),
        "components": ((40, 1536), "float32"),
        "singular_values": ((40,), "float32"),
    }
    if set(state) != set(expected):
        raise ProtocolError("H eval PCA state fields 漂移")
    for name, (shape, dtype) in expected.items():
        if state[name].shape != shape or str(state[name].dtype) != dtype:
            raise ProtocolError(f"H eval PCA state shape/dtype 漂移: {name}")
        if not np.isfinite(state[name]).all():
            raise ProtocolError(f"H eval PCA state non-finite: {name}")
    if np.any(state["feature_std"] <= 0):
        raise ProtocolError("H eval PCA feature_std 非正")
    return state


def validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_b_h_eval_feature_v1":
        raise ProtocolError("H eval feature schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("H eval feature task 漂移")
    if config.get("status") != "running" or config.get("seed") != 20260814:
        raise ProtocolError("H eval feature status/seed 漂移")
    for name in ("input_freeze", "dino_resource_freeze", "feature_freeze", "uplift_freeze"):
        spec = config[name]
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"H eval feature freeze identity 漂移: {name}")
        if load_yaml(path).get("status") != spec["required_status"]:
            raise ProtocolError(f"H eval feature freeze status 漂移: {name}")

    source = config["source"]
    source_path = Path(source["path"])
    checks = {
        "repository": _git(source_path, "remote", "get-url", "origin"),
        "commit": _git(source_path, "rev-parse", "HEAD"),
        "tree": _git(source_path, "rev-parse", "HEAD^{tree}"),
        "expected_git_status": _git(source_path, "status", "--short"),
    }
    for name, observed in checks.items():
        if observed != source[name]:
            raise ProtocolError(f"H eval DINO source 漂移: {name}")
    if sha256_file(source_path / "LICENSE") != source["license_sha256"]:
        raise ProtocolError("H eval DINO LICENSE 漂移")
    if sha256_file(source_path / "hubconf.py") != source["hubconf_sha256"]:
        raise ProtocolError("H eval DINO hubconf 漂移")

    checkpoint = Path(config["checkpoint"]["path"])
    if not checkpoint.is_file() or checkpoint.stat().st_size != int(config["checkpoint"]["bytes"]):
        raise ProtocolError("H eval DINO checkpoint bytes 漂移")
    if sha256_file(checkpoint) != config["checkpoint"]["sha256"]:
        raise ProtocolError("H eval DINO checkpoint SHA 漂移")

    input_spec = config["input_freeze"]
    manifest_path = Path(input_spec["image_manifest_path"])
    if not manifest_path.is_file() or sha256_file(manifest_path) != input_spec["image_manifest_sha256"]:
        raise ProtocolError("H eval image manifest identity 漂移")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("record_chain_sha256") != input_spec["image_manifest_record_chain_sha256"]:
        raise ProtocolError("H eval image manifest chain 漂移")
    views = config["views"]
    records = select_h_uplift_records(
        manifest,
        scenes=views["scenes"],
        frames=views["frames"],
        cameras=views["cameras"],
    )
    if len(records) != int(views["expected_view_count"]):
        raise ProtocolError("H eval view denominator 漂移")
    if any(int(record["frame"]) % 5 != int(views["evaluation_remainder"]) for record in records):
        raise ProtocolError("H eval remainder 漂移")
    if any(int(record["frame"]) % 5 == int(views["forbidden_heldout_remainder"]) for record in records):
        raise ProtocolError("H eval 触碰 forbidden remainder")

    expected_model = {
        "entrypoint": "dinov2_vitg14_reg",
        "architecture": "vit_giant2",
        "patch_size": 14,
        "register_tokens": 4,
        "raw_dimension": 1536,
        "intermediate_layers": 4,
        "selected_layer": "last_of_four",
        "reshape": True,
        "norm": True,
        "parameter_dtype": "float32",
        "inference_autocast": "float16",
        "expected_output_shape": [1, 1536, 64, 114],
        "strict_state_dict_load": True,
        "weights_only_load": True,
        "checkpoint_mmap": False,
        "repeat_first_image_exact": True,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "resize_mode": "bilinear",
        "resize_align_corners": False,
    }
    for name, expected in expected_model.items():
        if config["model"].get(name) != expected:
            raise ProtocolError(f"H eval model contract 漂移: {name}")
    transform = config["pca_transform"]
    expected_transform = {
        "fit": False,
        "components": 40,
        "feature_mean_dtype": "float64",
        "feature_std_dtype": "float64",
        "pca_mean_dtype": "float32",
        "components_dtype": "float32",
        "raw_input_dtype": "float32",
        "standardized_storage_dtype": "float32",
        "output_dtype": "float32",
        "transform_repeat_first_image_exact": True,
    }
    for name, expected in expected_transform.items():
        if transform.get(name) != expected:
            raise ProtocolError(f"H eval transform contract 漂移: {name}")

    environment = config["environment"]
    observed = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "xformers": xformers.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected_environment = {name: environment[name] for name in observed}
    if observed != expected_environment:
        raise ProtocolError(f"H eval feature environment 漂移: {observed}")
    locks = config["locks"]
    if locks.get("h_heldout_pixels_read_for_feature_extraction") is not True:
        raise ProtocolError("H eval heldout pixel authorization 漂移")
    for name in (
        "h_evidence_pixels_read",
        "pca_fit",
        "membership_proxy_read",
        "renderer_start",
        "uplift_feature_read",
        "method_quality_read",
        "screening_pixels_read",
        "screening_quality_read",
        "confirmation_pixels_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"H eval feature lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("H eval feature M2/M3 必须 pending")
    return config, records, manifest, _load_pca_state(config)


def _transform(raw: np.ndarray, pca_state: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    standardized = np.asarray(raw, dtype=np.float32).copy()
    standardize_in_place(
        standardized,
        mean=pca_state["feature_mean"],
        std=pca_state["feature_std"],
        chunk_rows=2048,
    )
    return pca_patch_grid(
        standardized,
        pca_mean=pca_state["pca_mean"],
        components=pca_state["components"],
        grid_hw=config["views"]["patch_grid_hw"],
    )


def execute(
    config: dict[str, Any], records: list[dict[str, Any]], pca_state: dict[str, np.ndarray], run_dir: Path
) -> dict[str, Any]:
    source_path = Path(config["source"]["path"])
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    backbones = importlib.import_module("dinov2.hub.backbones")
    torch.set_num_threads(int(config["environment"]["cpu_threads"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    constructor = getattr(backbones, config["model"]["entrypoint"])
    model = constructor(pretrained=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint = Path(config["checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=bool(config["model"]["weights_only_load"]),
        mmap=bool(config["model"]["checkpoint_mmap"]),
    )
    state_dict_key_count = len(state_dict)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ProtocolError("H eval DINO strict load 漂移")
    del state_dict
    model.eval().cuda()
    torch.cuda.synchronize()

    started = time.monotonic()
    first_raw_repeat_exact = False
    first_transform_repeat_exact = False
    sidecar_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        input_cpu = _preprocess(record, config)
        raw = _predict(model, input_cpu, config)
        expected_shape = (int(config["views"]["patches_per_view"]), int(config["model"]["raw_dimension"]))
        if raw.shape != expected_shape or str(raw.dtype) != "float32":
            raise ProtocolError(f"H eval raw feature shape/dtype 漂移: {raw.shape}/{raw.dtype}")
        if index == 0:
            repeated_raw = _predict(model, input_cpu, config)
            first_raw_repeat_exact = bool(np.array_equal(raw, repeated_raw))
            if not first_raw_repeat_exact:
                raise ProtocolError("H eval first raw feature repeat 非 bit-exact")
            del repeated_raw
        feature = _transform(raw, pca_state, config)
        if index == 0:
            repeated_feature = _transform(raw, pca_state, config)
            first_transform_repeat_exact = bool(np.array_equal(feature, repeated_feature))
            if not first_transform_repeat_exact:
                raise ProtocolError("H eval first PCA transform repeat 非 bit-exact")
            del repeated_feature
        relative = sidecar_relative_path(record)
        path = run_dir / relative
        atomic_save_deterministic_npz(path, {config["sidecars"]["feature_key"]: feature})
        sidecar_record = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "source_image_path": record["path"],
            "source_image_sha256": record["sha256"],
            "raw_feature_sha256": array_sha256(raw),
            "model_commit": config["source"]["commit"],
            "model_checkpoint_sha256": config["checkpoint"]["sha256"],
            "pca_state_sha256": config["feature_freeze"]["pca_state_sha256"],
            "scene": record["scene"],
            "scene_index": record["scene_index"],
            "frame": record["frame"],
            "camera": record["camera"],
            "shape": list(feature.shape),
            "dtype": str(feature.dtype),
            "content_sha256": array_sha256(feature),
        }
        if not set(config["sidecars"]["required_identity_fields"]).issubset(sidecar_record):
            raise ProtocolError("H eval feature identity fields 不完整")
        with np.load(path, allow_pickle=False) as archive:
            validate_sidecar_identity(sidecar_record, archive[config["sidecars"]["feature_key"]])
        sidecar_records.append(sidecar_record)
        _write_json(
            run_dir / "artifacts/extraction_progress.json",
            {
                "schema_version": "worldsim_v51_h_eval_feature_progress_v1",
                "completed_view_count": index + 1,
                "expected_view_count": len(records),
                "last_view": {"scene": record["scene"], "frame": record["frame"], "camera": record["camera"]},
            },
        )
        del raw, feature, input_cpu
    torch_peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    torch_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise ProtocolError("H eval DINO checkpoint 前后 SHA 漂移")
    manifest = {
        "schema_version": "worldsim_v51_h_eval_feature_manifest_v1",
        "task_id": config["task_id"],
        "record_count": len(sidecar_records),
        "record_chain_sha256": record_chain_sha256(sidecar_records),
        "pca_state_sha256": config["feature_freeze"]["pca_state_sha256"],
        "records": sidecar_records,
    }
    manifest_path = run_dir / config["sidecars"]["manifest_path"]
    _write_json(manifest_path, manifest)
    return {
        "view_count": len(sidecar_records),
        "patch_count": len(sidecar_records) * int(config["views"]["patches_per_view"]),
        "parameter_count": parameter_count,
        "state_dict_key_count": state_dict_key_count,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "pca_fit": False,
        "pca_state_sha256": config["feature_freeze"]["pca_state_sha256"],
        "selected_record_chain_sha256": record_chain_sha256(records),
        "first_raw_repeat_bit_exact": first_raw_repeat_exact,
        "first_transform_repeat_bit_exact": first_transform_repeat_exact,
        "feature_manifest_sha256": sha256_file(manifest_path),
        "feature_record_chain_sha256": manifest["record_chain_sha256"],
        "sidecar_total_bytes": sum(record["bytes"] for record in sidecar_records),
        "torch_peak_allocated_mib": torch_peak_allocated,
        "torch_peak_reserved_mib": torch_peak_reserved,
        "duration_feature_seconds": time.monotonic() - started,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    if branch != V51_BRANCH or _git(PROJECT, "status", "--short"):
        raise ProtocolError("H eval feature formal run 要求 V5.1 clean worktree")
    config, records, input_manifest, pca_state = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {
                "h_eval_feature": config,
                "selected_h_eval_records": records,
                "input_manifest_identity": {
                    "schema_version": input_manifest["schema_version"],
                    "record_count": input_manifest["record_count"],
                    "record_chain_sha256": input_manifest["record_chain_sha256"],
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    gpu_name, gpu_total = _gpu_name_and_total()
    gpu_start = _nvidia_used_mib()
    disk_available = shutil.disk_usage("/root/autodl-tmp").free
    resources = config["resources"]
    if gpu_start > int(resources["maximum_gpu_used_at_start_mib"]):
        raise ProtocolError("H eval feature GPU start 非空闲")
    if disk_available < int(resources["minimum_disk_available_bytes"]):
        raise ProtocolError("H eval feature disk available 不足")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(float(resources["sample_interval_seconds"]))
    monitor.start()
    timeout_seconds = int(resources["timeout_seconds"])

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"H eval feature 超过 {timeout_seconds} s")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    started = time.monotonic()
    try:
        report = execute(config, records, pca_state, run_dir)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        monitor.stop()
        _write_jsonl(run_dir / "resource_samples.jsonl", monitor.samples)
    valid = [row for row in monitor.samples if "gpu_used_mib" in row]
    if not valid:
        raise ProtocolError("H eval feature resource monitor 无 sample")
    resource = {
        "gpu_name": gpu_name,
        "gpu_total_mib": gpu_total,
        "gpu_used_at_start_mib": gpu_start,
        "nvidia_smi_peak_used_mib": max(row["gpu_used_mib"] for row in valid),
        "torch_peak_allocated_mib": report["torch_peak_allocated_mib"],
        "torch_peak_reserved_mib": report["torch_peak_reserved_mib"],
        "cgroup_memory_peak_bytes": max(row["cgroup_memory_current_bytes"] for row in valid),
        "disk_available_at_start_bytes": disk_available,
        "sample_count": len(valid),
        "monitor_error_count": len(monitor.samples) - len(valid),
        "duration_seconds": time.monotonic() - started,
    }
    _write_json(run_dir / "artifacts/resources.json", resource)
    _write_json(run_dir / "artifacts/h_eval_feature_report.json", report)
    if resource["nvidia_smi_peak_used_mib"] > int(resources["maximum_gpu_peak_mib"]):
        raise ProtocolError("H eval feature NVIDIA peak 超限")
    if resource["torch_peak_reserved_mib"] > int(resources["maximum_gpu_peak_mib"]):
        raise ProtocolError("H eval feature Torch reserved peak 超限")
    if resource["cgroup_memory_peak_bytes"] > int(resources["maximum_cgroup_peak_bytes"]):
        raise ProtocolError("H eval feature cgroup peak 超限")
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"metric": "view_count", "value": report["view_count"]},
            {"metric": "patch_count", "value": report["patch_count"]},
            {"metric": "sidecar_total_bytes", "value": report["sidecar_total_bytes"]},
            {"metric": "resource_terminal", **resource},
        ],
    )
    summary = {
        "schema_version": "worldsim_v51_h_eval_feature_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "h_heldout_dinov2_features_transformed_with_frozen_pca_without_quality_read",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "report": report,
        "resource": resource,
        "h_heldout_pixels_read_for_feature_extraction": True,
        "pca_fit": False,
        "membership_proxy_read": False,
        "renderer_started": False,
        "uplift_feature_read": False,
        "method_quality_read": False,
        "screening_quality_read": False,
        "confirmation_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "fingerprint.json",
        {
            "schema_version": "worldsim_v51_h_eval_feature_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "pca_state_sha256": report["pca_state_sha256"],
            "selected_record_chain_sha256": report["selected_record_chain_sha256"],
            "feature_record_chain_sha256": report["feature_record_chain_sha256"],
            "seed": int(config["seed"]),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_b_h_eval_feature_v1.yaml"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    try:
        summary = run(args.config.resolve(), run_dir)
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_h_eval_feature_manifest_terminal_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_h_eval_feature_status_v1",
                "task_id": summary["task_id"],
                "status": "done",
                "source_commit": summary["source_commit"],
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "finished_at_utc": _utc_now(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        events.append({"event": "run_blocked", "at_utc": _utc_now(), "reason": f"{type(error).__name__}: {error}"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_h_eval_feature_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
