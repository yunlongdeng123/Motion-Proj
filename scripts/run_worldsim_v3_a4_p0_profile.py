#!/usr/bin/env python
"""执行 A4-P0 inventory、prepare/load 与 runtime render profile。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping

import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT = Path("/root/autodl-tmp/motion_proj")
DRIVESTUDIO = Path("/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p0_profile_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None


from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data,
    release_trainer_render_info,
    uint8_rgb,
)
from scripts.run_worldsim_v3_a3_s_b_paired_smoke import (
    atomic_json,
    cgroup_memory_current,
    cgroup_memory_events,
    directory_bytes,
    sha256_file,
)
from scripts.validate_worldsim_v3_a4_p0_profile_protocol import (
    validate_inputs,
    validate_schema,
)


def command_output(*command: str, cwd: Path = PROJECT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 < percentile <= 1:
        raise ValueError("A4-P0 nearest-rank 输入非法")
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def rgb_sha256(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.dtype).encode())
    digest.update(json.dumps(list(rgb.shape)).encode())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def nvidia_compute_rows() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi 失败: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, used = (part.strip() for part in line.split(",", 1))
        rows.append({"pid": int(pid), "used_memory_mib": int(used)})
    return rows


class ResourceSampler:
    def __init__(self, pid: int, interval_seconds: float = 0.2) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        rows = nvidia_compute_rows()
        process_memory = next(
            (row["used_memory_mib"] for row in rows if row["pid"] == self.pid),
            0,
        )
        self.samples.append(
            {
                "elapsed_seconds": time.monotonic() - self.started_at,
                "nvidia_process_memory_mib": process_memory,
                "cgroup_memory_bytes": cgroup_memory_current(),
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as error:  # 采样失败必须进入结果，而不是静默丢弃。
                self.samples.append({"sampling_error": f"{type(error).__name__}: {error}"})
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self._sample()
        except Exception as error:  # 最终采样与后台采样遵循同一显式失败语义。
            self.samples.append({"sampling_error": f"{type(error).__name__}: {error}"})

    def summary(self) -> dict[str, Any]:
        errors = [sample["sampling_error"] for sample in self.samples if "sampling_error" in sample]
        nvidia = [sample["nvidia_process_memory_mib"] for sample in self.samples if "nvidia_process_memory_mib" in sample]
        cgroup = [sample["cgroup_memory_bytes"] for sample in self.samples if sample.get("cgroup_memory_bytes") is not None]
        return {
            "sample_count": len(self.samples),
            "sampling_errors": errors,
            "peak_nvidia_process_memory_mib_sampled": max(nvidia) if nvidia else None,
            "peak_cgroup_memory_bytes_sampled": max(cgroup) if cgroup else None,
        }


def snapshot_sources(run_dir: Path, paths: Iterable[Path]) -> dict[str, str]:
    root = run_dir / "source_snapshot"
    hashes = {}
    for source in paths:
        relative = source.relative_to(PROJECT)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256_file(target)
    return hashes


def write_stage(run_dir: Path, manifest: dict[str, Any], name: str, payload: dict[str, Any]) -> None:
    path = run_dir / "stages" / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"A4-P0 completed stage overwrite forbidden: {path}")
    atomic_json(path, payload)
    manifest.setdefault("stage_hashes", {})[name] = sha256_file(path)
    atomic_json(run_dir / "manifest.json", manifest, replace=True)


def output_validation_bytes(value: Any) -> int:
    """统计历史 stage 已声明且仍存在的文件输出字节。"""
    if isinstance(value, Mapping):
        return sum(output_validation_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(output_validation_bytes(item) for item in value)
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return directory_bytes(path)
    return 0


def build_inventory(protocol: Mapping[str, Any], input_audits: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = Path(protocol["selected_asset"]["actor_registry"]["path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    historical_stages = {}
    for name, spec in protocol["historical_evidence"]["stages"].items():
        stage = json.loads(Path(spec["path"]).read_text(encoding="utf-8"))
        historical_stages[name] = {
            "status": stage["status"],
            "terminal_status": stage["status"],
            "duration_seconds": stage["duration_seconds"],
            "peak_gpu_memory_mib_sampled": stage.get("peak_gpu_memory_mib_sampled"),
            "peak_gpu_memory_mib_torch_log": stage.get("peak_gpu_memory_mib_torch_log"),
            "peak_cgroup_memory_bytes": stage.get("peak_cgroup_memory_bytes"),
            "source_path": spec["path"],
            "source_sha256": spec["sha256"],
            "input_bytes": None,
            "input_bytes_missing_reason": (
                "immutable_historical_stage_did_not_persist_exact_input_byte_count"
            ),
            "output_bytes": output_validation_bytes(stage.get("output_validation", {})),
            "filesystem_cache": None,
            "filesystem_cache_missing_reason": (
                "immutable_historical_stage_did_not_persist_cache_state"
            ),
        }
    selected = protocol["selected_asset"]
    return {
        "status": "done",
        "stage": "inventory",
        "selected_asset_role": selected["role"],
        "input_audits": input_audits,
        "checkpoint_bytes": int(selected["checkpoint"]["bytes"]),
        "source_config_bytes": int(selected["source_config"]["bytes"]),
        "actor_registry_bytes": int(selected["actor_registry"]["bytes"]),
        "checkpoint_and_registry_inventory_bytes": sum(
            int(selected[name]["bytes"])
            for name in ("checkpoint", "actor_registry")
        ),
        "static_block_count": 1,
        "static_block_policy": "monolithic_background_not_yet_chunked",
        "actor_asset_count": int(registry["actor_count"]),
        "available_actor_asset_count": int(registry["available_actor_count"]),
        "unavailable_actor_asset_count": int(registry["empty_checkpoint_actor_count"]),
        "conversion_status": "inventory_only_no_parameter_conversion",
        "historical_stages": historical_stages,
    }


def render_view(
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
        with torch.inference_mode():
            outputs = trainer(image_infos, camera_infos)
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        rgb = uint8_rgb(outputs["rgb"])
        return {
            "phase": phase,
            "ordinal": ordinal,
            "frame": frame,
            "camera": camera,
            "image_index": image_index,
            "duration_seconds": duration,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "rgb_bytes": int(rgb.nbytes),
            "rgb_sha256": rgb_sha256(rgb),
        }
    finally:
        release_trainer_render_info(trainer)


def build_runtime_probe(protocol: Mapping[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(DRIVESTUDIO))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    selected = protocol["selected_asset"]
    checkpoint = Path(selected["checkpoint"]["path"])
    config = OmegaConf.load(selected["source_config"]["path"])
    device = torch.device(protocol["new_probe"]["device"])
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    with ResourceSampler(os.getpid()) as sampler:
        started = time.perf_counter()
        dataset_started = time.perf_counter()
        dataset = DrivingDataset(data_cfg=config.data)
        prepare_seconds = time.perf_counter() - dataset_started
        trainer_started = time.perf_counter()
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
        trainer_construction_seconds = time.perf_counter() - trainer_started
        if hasattr(trainer, "optimizer"):
            raise RuntimeError("A4-P0 read-only profile constructed optimizer")
        torch.cuda.synchronize(device)
        cold_started = time.perf_counter()
        trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
        torch.cuda.synchronize(device)
        cold_load_seconds = time.perf_counter() - cold_started
        trainer.set_eval()
        torch.cuda.synchronize(device)
        warm_started = time.perf_counter()
        trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
        torch.cuda.synchronize(device)
        warm_load_seconds = time.perf_counter() - warm_started
        trainer.set_eval()
        warmup = protocol["new_probe"]["warmup"]
        warmup_rows = [
            render_view(
                trainer,
                dataset,
                frame=int(warmup["frame"]),
                camera=int(warmup["camera"]),
                device=device,
                phase="warmup",
                ordinal=ordinal,
            )
            for ordinal in range(int(warmup["repeats"]))
        ]
        measured = protocol["new_probe"]["measured"]
        measured_views = [
            (int(frame), int(camera))
            for frame in measured["frames"]
            for camera in measured["cameras"]
        ]
        measured_rows = [
            render_view(
                trainer,
                dataset,
                frame=frame,
                camera=camera,
                device=device,
                phase="measured",
                ordinal=ordinal,
            )
            for ordinal, (frame, camera) in enumerate(measured_views)
        ]
        total_seconds = time.perf_counter() - started
        peak_allocated = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024**2))
        no_optimizer = not hasattr(trainer, "optimizer")
        model_counts = {
            name: int(model._means.shape[0])
            for name, model in trainer.models.items()
            if hasattr(model, "_means")
        }
        del trainer, dataset
        torch.cuda.empty_cache()
    sampler_summary = sampler.summary()
    if (
        sampler_summary["sampling_errors"]
        or sampler_summary["peak_nvidia_process_memory_mib_sampled"] is None
        or sampler_summary["peak_cgroup_memory_bytes_sampled"] is None
    ):
        raise RuntimeError(f"A4-P0 resource sampling incomplete: {sampler_summary}")
    durations = [row["duration_seconds"] for row in measured_rows]
    expected_width = int(protocol["new_probe"]["resolution"]["width"])
    expected_height = int(protocol["new_probe"]["resolution"]["height"])
    return {
        "status": "done",
        "stage": "runtime_probe",
        "prepare_dataset_seconds": prepare_seconds,
        "trainer_construction_seconds": trainer_construction_seconds,
        "process_cold_checkpoint_load_seconds": cold_load_seconds,
        "process_warm_checkpoint_reload_seconds": warm_load_seconds,
        "filesystem_cache": protocol["new_probe"]["load_semantics"]["filesystem_cache"],
        "runtime_render_warmup_seconds": [row["duration_seconds"] for row in warmup_rows],
        "runtime_render_sample_seconds": durations,
        "runtime_render_p50_seconds": nearest_rank(durations, 0.50),
        "runtime_render_p95_seconds": nearest_rank(durations, 0.95),
        "runtime_render_fps": len(durations) / sum(durations),
        "runtime_probe_total_seconds": total_seconds,
        "warmup_rows": warmup_rows,
        "measured_rows": measured_rows,
        "warmup_rgb_hash_repeat_exact": len({row["rgb_sha256"] for row in warmup_rows}) == 1,
        "measured_matrix_complete_and_unique": len(measured_rows)
        == int(measured["expected_samples"])
        and len({(row["frame"], row["camera"]) for row in measured_rows})
        == len(measured_rows),
        "native_resolution_exact": all(
            row["width"] == expected_width and row["height"] == expected_height
            for row in warmup_rows + measured_rows
        ),
        "synchronized_timing_complete": all(value > 0 for value in durations),
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "model_gaussian_counts": model_counts,
        "peak_torch_allocated_mib": peak_allocated,
        "peak_torch_reserved_mib": peak_reserved,
        **sampler_summary,
    }


def build_aggregate(inventory: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    historical = inventory["historical_stages"]
    return {
        "status": "done",
        "stage": "aggregate",
        "stage_ledger": {
            "prepare": {
                "source": "new_process_probe",
                "status": "done",
                "wall_time_seconds": runtime["prepare_dataset_seconds"],
                "input_bytes": inventory["source_config_bytes"],
                "output_bytes": None,
                "output_bytes_missing_reason": "dataset_is_in_memory_and_not_persisted",
                "filesystem_cache": "uncontrolled_report_explicitly",
                "minimum_rerun_unit": "runtime_probe_and_downstream",
            },
            "train": {
                "source": "immutable_historical_stage",
                **historical["train"],
                "minimum_rerun_unit": "not_rerunnable_in_p0",
            },
            "render_eval": {
                "source": "immutable_historical_stage",
                "eval": historical["render_eval"],
                "actor_metrics": historical["actor_metrics"],
                "status": "done",
                "input_bytes": None,
                "input_bytes_missing_reason": (
                    "immutable_historical_stages_did_not_persist_exact_input_byte_count"
                ),
                "output_bytes": historical["render_eval"]["output_bytes"]
                + historical["actor_metrics"]["output_bytes"],
                "filesystem_cache": None,
                "filesystem_cache_missing_reason": (
                    "immutable_historical_stages_did_not_persist_cache_state"
                ),
                "minimum_rerun_unit": "not_rerunnable_in_p0",
            },
            "convert": {
                "source": "inventory_only",
                "status": "done",
                "operation": inventory["conversion_status"],
                "registry": historical["registry"],
                "input_bytes": inventory["checkpoint_and_registry_inventory_bytes"],
                "output_bytes": inventory["checkpoint_and_registry_inventory_bytes"],
                "filesystem_cache": "not_applicable_no_parameter_conversion",
                "minimum_rerun_unit": "inventory_and_downstream",
            },
            "load": {
                "source": "new_process_probe",
                "status": "done",
                "process_cold_seconds": runtime["process_cold_checkpoint_load_seconds"],
                "process_warm_seconds": runtime["process_warm_checkpoint_reload_seconds"],
                "input_bytes": inventory["checkpoint_bytes"],
                "output_bytes": None,
                "output_bytes_missing_reason": "loaded_parameters_are_in_memory_not_persisted",
                "filesystem_cache": runtime["filesystem_cache"],
                "minimum_rerun_unit": "runtime_probe_and_downstream",
            },
            "runtime_render": {
                "source": "new_process_probe",
                "status": "done",
                "sample_count": len(runtime["runtime_render_sample_seconds"]),
                "p50_seconds": runtime["runtime_render_p50_seconds"],
                "p95_seconds": runtime["runtime_render_p95_seconds"],
                "fps": runtime["runtime_render_fps"],
                "input_bytes": inventory["checkpoint_bytes"],
                "output_bytes": runtime["runtime_rgb_bytes_in_memory"],
                "output_persistence": "hashes_only_no_media",
                "filesystem_cache": "not_separately_controlled_after_load",
                "minimum_rerun_unit": "runtime_probe_and_downstream",
            },
            "failure_recovery": {
                "source": "pending_resume_audit",
                "status": None,
                "missing_reason": "resume_audit_runs_in_a_separate_no_torch_process",
                "minimum_rerun_unit": "resume_audit_only",
            },
        },
    }


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite A4-P0 run: {args.run_dir}")
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    input_audits = validate_inputs(protocol)
    gpu_rows = nvidia_compute_rows()
    if gpu_rows:
        raise RuntimeError(f"A4-P0 GPU preflight not idle: {gpu_rows}")
    disk_free = shutil.disk_usage(args.run_dir.parent).free
    if disk_free < int(protocol["resource_ceilings"]["disk_free_floor_bytes"]):
        raise RuntimeError(f"A4-P0 disk preflight failed: {disk_free}")
    args.run_dir.mkdir(parents=True)
    _ACTIVE_RUN_DIR = args.run_dir
    (args.run_dir / "stages").mkdir()
    (args.run_dir / "artifacts").mkdir()
    source_hashes = snapshot_sources(
        args.run_dir,
        [
            args.protocol,
            PROJECT / "scripts/run_worldsim_v3_a4_p0_profile.py",
            PROJECT / "scripts/audit_worldsim_v3_a4_p0_resume.py",
            PROJECT / "scripts/finalize_worldsim_v3_a4_p0.py",
            PROJECT / "scripts/run_worldsim_v3_a4_p0_profile.sh",
            PROJECT / "scripts/validate_worldsim_v3_a4_p0_profile_protocol.py",
        ],
    )
    events_before = cgroup_memory_events()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.protocol),
        "project_commit": command_output("git", "rev-parse", "HEAD"),
        "project_status": command_output("git", "status", "--short").splitlines(),
        "source_hashes": source_hashes,
        "input_audits": input_audits,
        "stage_hashes": {},
        "preflight": {
            "gpu_compute_rows": gpu_rows,
            "disk_free_bytes": disk_free,
            "cgroup_memory_bytes": cgroup_memory_current(),
            "cgroup_memory_events": events_before,
        },
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    inventory = build_inventory(protocol, input_audits)
    write_stage(args.run_dir, manifest, "inventory", inventory)
    runtime = build_runtime_probe(protocol)
    runtime_rows = runtime.pop("warmup_rows") + runtime.pop("measured_rows")
    runtime["runtime_rgb_bytes_in_memory"] = sum(
        int(row["rgb_bytes"]) for row in runtime_rows
    )
    rows_path = args.run_dir / "artifacts" / "runtime_rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in runtime_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    runtime["runtime_rows"] = {
        "path": str(rows_path.relative_to(args.run_dir)),
        "count": len(runtime_rows),
        "sha256": sha256_file(rows_path),
    }
    write_stage(args.run_dir, manifest, "runtime_probe", runtime)
    aggregate = build_aggregate(inventory, runtime)
    write_stage(args.run_dir, manifest, "aggregate", aggregate)
    manifest["probe_complete"] = True
    manifest["checkpoint_sha256_after_probe"] = sha256_file(
        Path(protocol["selected_asset"]["checkpoint"]["path"])
    )
    manifest["actor_registry_sha256_after_probe"] = sha256_file(
        Path(protocol["selected_asset"]["actor_registry"]["path"])
    )
    manifest["cgroup_memory_events_after_probe"] = cgroup_memory_events()
    manifest["run_bytes_after_probe"] = directory_bytes(args.run_dir)
    atomic_json(args.run_dir / "manifest.json", manifest, replace=True)
    print(json.dumps({"status": "probe_complete", "run_dir": str(args.run_dir)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None:
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P0_PROBE_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
