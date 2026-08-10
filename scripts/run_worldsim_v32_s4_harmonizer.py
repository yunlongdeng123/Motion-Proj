#!/usr/bin/env python
"""运行 V3.2 S4 Harmonizer 非时序正式评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v32.harmonizer_adapter import (
    HarmonizerJITAdapter,
    validate_rmsnorm_operator,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 不匹配：{path}: {actual} != {expected_sha256}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def git_output(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args],
        text=True,
    ).strip()


def read_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.exists():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    return values


def read_int_file(path: str) -> int | str | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    value = candidate.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else value


class ResourceSampler:
    """轻量采样 NVIDIA 显存与 cgroup memory.current。"""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.peak_nvidia_memory_mib = 0
        self.peak_cgroup_current_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            self.peak_nvidia_memory_mib = max(
                self.peak_nvidia_memory_mib,
                max(int(line.strip()) for line in output.splitlines() if line.strip()),
            )
        except Exception:
            pass
        current = read_int_file("/sys/fs/cgroup/memory.current")
        if isinstance(current, int):
            self.peak_cgroup_current_bytes = max(self.peak_cgroup_current_bytes, current)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()


def image_metrics(source: Image.Image, output: Image.Image, mask: Image.Image, threshold: int) -> dict[str, Any]:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    output_array = np.asarray(output.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if source_array.shape != output_array.shape:
        raise ValueError(f"输出尺寸变化：{source_array.shape} -> {output_array.shape}")
    diff = np.abs(source_array - output_array)
    squared = np.square(source_array - output_array)
    mse = float(squared.mean())
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    changed = np.max(diff, axis=2) > threshold
    outside = ~mask_array
    inside_l1 = float(diff[mask_array].mean()) if bool(mask_array.any()) else None
    outside_l1 = float(diff[outside].mean()) if bool(outside.any()) else None

    def edge_energy(array: np.ndarray) -> float:
        gray = array.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1)).mean()
        dy = np.abs(np.diff(gray, axis=0)).mean()
        return float((dx + dy) / 2.0)

    source_edge = edge_energy(source_array)
    output_edge = edge_energy(output_array)
    return {
        "input_output_psnr_db": psnr,
        "global_l1_uint8": float(diff.mean()),
        "outside_mask_l1_uint8": outside_l1,
        "inside_mask_l1_uint8": inside_l1,
        "target_mask_fraction": float(mask_array.mean()),
        "changed_pixel_fraction_gt8": float(changed.mean()),
        "inside_changed_pixel_fraction_gt8": float(changed[mask_array].mean()) if bool(mask_array.any()) else None,
        "outside_changed_pixel_fraction_gt8": float(changed[outside].mean()) if bool(outside.any()) else None,
        "channel_mean_shift_uint8": (output_array.mean(axis=(0, 1)) - source_array.mean(axis=(0, 1))).tolist(),
        "source_edge_energy": source_edge,
        "output_edge_energy": output_edge,
        "edge_energy_ratio": output_edge / max(source_edge, 1e-12),
        "output_min_uint8": int(output_array.min()),
        "output_max_uint8": int(output_array.max()),
    }


def save_panel(source: Image.Image, output: Image.Image, path: Path, title: str) -> None:
    source = source.convert("RGB")
    output = output.convert("RGB")
    source_array = np.asarray(source, dtype=np.int16)
    output_array = np.asarray(output, dtype=np.int16)
    diff = np.clip(np.abs(source_array - output_array) * 4, 0, 255).astype(np.uint8)
    diff_image = Image.fromarray(diff)
    header = 34
    panel = Image.new("RGB", (source.width * 3, source.height + header), "white")
    panel.paste(source, (0, header))
    panel.paste(output, (source.width, header))
    panel.paste(diff_image, (source.width * 2, header))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{title} | source", fill="black")
    draw.text((source.width + 8, 8), "harmonized", fill="black")
    draw.text((source.width * 2 + 8, 8), "abs diff x4", fill="black")
    panel.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output_root"]) / f"{timestamp}__{config['run_label']}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "artifacts" / "inputs").mkdir(parents=True)
    (run_dir / "artifacts" / "outputs").mkdir(parents=True)
    (run_dir / "artifacts" / "panels").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    status_path = run_dir / "status.json"
    write_json(
        status_path,
        {
            "schema_version": "worldsim_v32_s4_status_v1",
            "task_id": config["task_id"],
            "status": "running",
            "run_dir": str(run_dir),
            "started_at_utc": timestamp,
        },
    )

    sampler = ResourceSampler()
    sampler.start()
    memory_before = read_memory_events()
    wall_start = time.perf_counter()
    immutable_before: dict[str, Any] = {}
    try:
        config_audit = verify_file(args.config, sha256_file(args.config))
        source = config["official_source"]
        checkout = Path(source["checkout"])
        source_commit = git_output(checkout, "rev-parse", "HEAD")
        if source_commit != source["commit"]:
            raise RuntimeError(f"Harmonizer commit 漂移：{source_commit}")
        source_clean = git_output(checkout, "status", "--porcelain") == ""
        if not source_clean:
            raise RuntimeError("Harmonizer 官方 checkout 非 clean")
        source_license = verify_file(Path(source["license_path"]), source["license_sha256"])

        model_cfg = config["model"]
        model_path = Path(model_cfg["path"])
        model_before = verify_file(model_path, model_cfg["sha256"])
        if model_before["bytes"] != model_cfg["bytes"]:
            raise RuntimeError("Harmonizer 模型字节数漂移")
        model_card = verify_file(Path(model_cfg["model_card"]), model_cfg["model_card_sha256"])

        verified_inputs = []
        for item in config["inputs"]:
            image_audit = verify_file(Path(item["path"]), item["sha256"])
            mask_audit = verify_file(Path(item["target_mask"]), item["target_mask_sha256"])
            verified_inputs.append({"id": item["id"], "image": image_audit, "mask": mask_audit})

        for item in config["immutable_assets"]:
            immutable_before[item["name"]] = verify_file(Path(item["path"]), item["sha256"])

        formal_device = torch.device(config["runtime"]["device"])
        torch.cuda.set_device(formal_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(formal_device)
        rmsnorm_audit = validate_rmsnorm_operator(config["runtime"]["device"])
        if not rmsnorm_audit["exact"]:
            raise RuntimeError(f"RMSNorm 回退验证失败：{rmsnorm_audit}")
        adapter = HarmonizerJITAdapter(str(model_path), config["runtime"]["device"])

        output_records = []
        panel_paths: list[Path] = []
        evaluation_cfg = config["evaluation"]
        for item in config["inputs"]:
            source_path = Path(item["path"])
            mask_path = Path(item["target_mask"])
            source_image = Image.open(source_path).convert("RGB")
            if list(source_image.size) != config["runtime"]["source_size"]:
                raise RuntimeError(f"输入尺寸不符合冻结合同：{item['id']} {source_image.size}")
            copied_input = run_dir / "artifacts" / "inputs" / f"{item['id']}.png"
            copied_mask = run_dir / "artifacts" / "inputs" / f"{item['id']}__target_mask.png"
            shutil.copy2(source_path, copied_input)
            shutil.copy2(mask_path, copied_mask)
            output_image, runtime = adapter.infer(source_image)
            output_path = run_dir / "artifacts" / "outputs" / f"{item['id']}__harmonized.png"
            output_image.save(output_path)
            mask_image = Image.open(mask_path)
            metrics = image_metrics(
                source_image,
                output_image,
                mask_image,
                evaluation_cfg["changed_pixel_threshold_uint8"],
            )
            gates = {
                "shape_preserved": output_image.size == source_image.size,
                "outside_mask_drift": metrics["outside_mask_l1_uint8"] <= evaluation_cfg["max_outside_mask_l1_uint8"],
                "global_psnr": metrics["input_output_psnr_db"] >= evaluation_cfg["min_input_output_psnr_db"],
                "non_identity_effect": metrics["changed_pixel_fraction_gt8"] >= evaluation_cfg["min_changed_pixel_fraction_gt8"],
                "dynamic_range": metrics["output_max_uint8"] > metrics["output_min_uint8"],
            }
            if item["group"] == "G1_semantic_remove_inpaint":
                gates["semantic_delete_inside_l1"] = (
                    metrics["inside_mask_l1_uint8"]
                    <= evaluation_cfg["max_semantic_delete_inside_l1_uint8"]
                )
                gates["semantic_delete_changed_fraction"] = (
                    metrics["inside_changed_pixel_fraction_gt8"]
                    <= evaluation_cfg["max_semantic_delete_inside_changed_fraction_gt8"]
                )
            panel_path = run_dir / "artifacts" / "panels" / f"{item['id']}__panel.png"
            save_panel(source_image, output_image, panel_path, item["id"])
            panel_paths.append(panel_path)
            output_records.append(
                {
                    **{key: item[key] for key in ("id", "group", "frame", "camera_id", "camera_name", "variant")},
                    "input": {"path": str(copied_input), "sha256": sha256_file(copied_input)},
                    "target_mask": {"path": str(copied_mask), "sha256": sha256_file(copied_mask)},
                    "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
                    "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
                    "runtime": runtime,
                    "metrics": metrics,
                    "gates": gates,
                    "all_gates_passed": all(gates.values()),
                    "provenance": evaluation_cfg["provenance"],
                    "provenance_alias": evaluation_cfg["provenance_alias"],
                }
            )

        grid_width = max(Image.open(path).width for path in panel_paths)
        panels = [Image.open(path).convert("RGB") for path in panel_paths]
        grid = Image.new("RGB", (grid_width, sum(panel.height for panel in panels)), "white")
        y = 0
        for panel in panels:
            grid.paste(panel, (0, y))
            y += panel.height
        grid_path = run_dir / "artifacts" / "comparison_grid.png"
        grid.save(grid_path)

        temporal_block = {
            **config["temporal_arm"],
            "disposition": "external_access_blocked_no_circumvention",
            "quality_claim": "none",
        }
        write_json(run_dir / "artifacts" / "temporal_arm_blocked.json", temporal_block)

        immutable_after = {
            item["name"]: verify_file(Path(item["path"]), item["sha256"])
            for item in config["immutable_assets"]
        }
        model_after = verify_file(model_path, model_cfg["sha256"])
        immutable_exact = immutable_before == immutable_after and model_before == model_after
        if not immutable_exact:
            raise RuntimeError("S4 只读合同失败：输入模型或三维资产发生变化")

        all_frame_gates = all(record["all_gates_passed"] for record in output_records)
        semantic_delete_preserved = all(
            record["all_gates_passed"]
            for record in output_records
            if record["group"] == "G1_semantic_remove_inpaint"
        )
        inference_seconds = [record["runtime"]["inference_seconds"] for record in output_records]
        group_metrics: dict[str, Any] = {}
        for group in sorted({record["group"] for record in output_records}):
            records = [record for record in output_records if record["group"] == group]
            group_metrics[group] = {
                "count": len(records),
                "mean_outside_mask_l1_uint8": statistics.fmean(
                    record["metrics"]["outside_mask_l1_uint8"] for record in records
                ),
                "mean_input_output_psnr_db": statistics.fmean(
                    record["metrics"]["input_output_psnr_db"] for record in records
                ),
                "all_gates_passed": all(record["all_gates_passed"] for record in records),
            }

        sampler.stop()
        wall_seconds = time.perf_counter() - wall_start
        memory_after = read_memory_events()
        memory_delta = {
            key: memory_after.get(key, 0) - memory_before.get(key, 0)
            for key in sorted(set(memory_before) | set(memory_after))
        }
        summary = {
            "schema_version": "worldsim_v32_s4_summary_v1",
            "task_id": config["task_id"],
            "status": "done",
            "run_dir": str(run_dir),
            "mode_completed": "exported_jit_nontemporal",
            "final_disposition": "optional_diagnostic",
            "candidate_selected": False,
            "non_temporal_candidate_selected": all_frame_gates,
            "candidate_selection_reason": (
                "temporal production arm remains externally blocked; non-temporal output "
                "fails semantic-delete preservation by reintroducing actor-like appearance"
            ),
            "temporal_arm": temporal_block,
            "temporal_consistency": {
                "status": "not_evaluated",
                "reason": "released JIT checkpoint is explicitly per-image and input frames are not consecutive temporal ground truth",
            },
            "quality_claim_scope": evaluation_cfg["claim_scope"],
            "provenance": {
                "label": evaluation_cfg["provenance"],
                "alias": evaluation_cfg["provenance_alias"],
                "scope": "2D output only",
                "written_back_to_gaussian_checkpoint": False,
            },
            "implementation": {
                "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
                "adapter": {
                    "path": str(Path(sys.modules[HarmonizerJITAdapter.__module__].__file__).resolve()),
                    "sha256": sha256_file(Path(sys.modules[HarmonizerJITAdapter.__module__].__file__).resolve()),
                },
                "config": config_audit,
                "source_commit": source_commit,
                "source_clean": source_clean,
                "source_license": source_license,
                "model": model_before,
                "model_card": model_card,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "operator_runtime": adapter.load_audit.operator_runtime,
                "rmsnorm_validation": rmsnorm_audit,
                "constant_device_patch": adapter.load_audit.constant_device_patch,
                "resize_rule": config["runtime"]["resize_rule"],
            },
            "verified_inputs": verified_inputs,
            "outputs": output_records,
            "group_metrics": group_metrics,
            "gates": {
                "all_outputs_produced": len(output_records) == len(config["inputs"]),
                "non_temporal_candidate_gates": all_frame_gates,
                "semantic_delete_preserved": semantic_delete_preserved,
                "three_input_groups_present": set(group_metrics) == {
                    "G0_original_render",
                    "G1_semantic_remove_inpaint",
                    "G2_asset_harvester_lateral",
                },
                "single_camera_identity": {record["camera_name"] for record in output_records} == {"CAM_FRONT_LEFT"},
                "immutable_3d_assets": immutable_exact,
                "no_gaussian_checkpoint_write": immutable_exact,
                "single_gpu": torch.cuda.device_count() == 1,
                "no_oom": memory_delta.get("oom", 0) == 0 and memory_delta.get("oom_kill", 0) == 0,
            },
            "immutable_assets_before": immutable_before,
            "immutable_assets_after": immutable_after,
            "resource_audit": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "model_load_seconds": adapter.load_audit.load_seconds,
                "inference_seconds_per_frame": inference_seconds,
                "median_inference_seconds": statistics.median(inference_seconds),
                "mean_inference_seconds": statistics.fmean(inference_seconds),
                "wall_seconds": wall_seconds,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_nvidia_memory_mib_sampled": sampler.peak_nvidia_memory_mib,
                "peak_cgroup_current_bytes_sampled": sampler.peak_cgroup_current_bytes,
                "cgroup_memory_max": read_int_file("/sys/fs/cgroup/memory.max"),
                "memory_events_before": memory_before,
                "memory_events_after": memory_after,
                "memory_events_delta": memory_delta,
            },
            "comparison_grid": {"path": str(grid_path), "sha256": sha256_file(grid_path)},
        }
        execution_gate_names = [
            "all_outputs_produced",
            "three_input_groups_present",
            "single_camera_identity",
            "immutable_3d_assets",
            "no_gaussian_checkpoint_write",
            "single_gpu",
            "no_oom",
        ]
        summary["gates"]["all_required_execution_gates"] = all(
            summary["gates"][name] for name in execution_gate_names
        )
        if not summary["gates"]["all_required_execution_gates"]:
            raise RuntimeError(f"S4 执行门失败：{summary['gates']}")
        summary_path = run_dir / "summary.json"
        write_json(summary_path, summary)
        write_json(
            status_path,
            {
                "schema_version": "worldsim_v32_s4_status_v1",
                "task_id": config["task_id"],
                "status": "done",
                "run_dir": str(run_dir),
                "finished_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
            },
        )
        print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
    except Exception as exc:
        sampler.stop()
        write_json(
            status_path,
            {
                "schema_version": "worldsim_v32_s4_status_v1",
                "task_id": config["task_id"],
                "status": "rejected",
                "run_dir": str(run_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            },
        )
        raise


if __name__ == "__main__":
    main()
