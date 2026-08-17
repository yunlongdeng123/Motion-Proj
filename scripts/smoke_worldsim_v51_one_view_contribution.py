#!/usr/bin/env python3
"""只统计一个 H view 的真实 renderer contribution denominator。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.renderer_intersections import renderer_intersections
from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
)


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), *args], text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    ]


def _nvidia_used_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if len(values) != 1:
        raise ProtocolError("one-view contribution 预期单 GPU")
    return values[0]


class ResourceMonitor:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(
                    {
                        "at_utc": _utc_now(),
                        "gpu_used_mib": _nvidia_used_mib(),
                        "cgroup_memory_current_bytes": int(
                            Path("/sys/fs/cgroup/memory.current")
                            .read_text(encoding="utf-8")
                            .strip()
                        ),
                    }
                )
            except Exception as error:
                self.samples.append(
                    {
                        "at_utc": _utc_now(),
                        "monitor_error": f"{type(error).__name__}: {error}",
                    }
                )
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != (
        "worldsim_v51_stage_b_one_view_contribution_v1"
    ):
        raise ProtocolError("one-view contribution schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("one-view contribution task 漂移")
    if config.get("status") != "running" or config.get("seed") != 20260814:
        raise ProtocolError("one-view contribution status/seed 漂移")

    freeze_spec = config["operator_freeze"]
    freeze_path = PROJECT / freeze_spec["path"]
    if sha256_file(freeze_path) != freeze_spec["sha256"]:
        raise ProtocolError("operator freeze binding 漂移")
    operator_freeze = load_yaml(freeze_path)
    if operator_freeze.get("status") != freeze_spec["required_status"]:
        raise ProtocolError("operator freeze status 漂移")

    for name, spec in config["inputs"].items():
        path = Path(spec["path"])
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"one-view contribution input 漂移: {name}")
    image = Path(config["scene"]["image_path"])
    if not image.is_file() or sha256_file(image) != config["scene"]["image_sha256"]:
        raise ProtocolError("one-view contribution image identity 漂移")

    scene = config["scene"]
    expected_scene = {
        "name": "scene-0471",
        "index": 382,
        "role": "H",
        "frame": 0,
        "camera": 0,
        "image_index": 0,
        "image_size_wh": [1600, 900],
        "expected_background_gaussians": 809902,
        "expected_rigid_gaussians": 49711,
        "expected_total_gaussians": 859613,
    }
    for name, expected in expected_scene.items():
        if scene.get(name) != expected:
            raise ProtocolError(f"one-view scene contract 漂移: {name}")

    contribution = config["contribution"]
    expected_contribution = {
        "source": "motion_proj.worldsim_v5.renderer_intersections",
        "formula": "alpha_times_transmittance_before_alpha",
        "minimum_intersection_contribution": 1e-4,
        "minimum_gaussian_view_mass": 1e-3,
        "expected_n_cameras": 1,
        "report_quantiles": [0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
        "persist_intersection_rows": False,
        "consume_pixel_rgb_values": False,
        "consume_lidar_values": False,
        "consume_membership_proxy": False,
        "compute_feature_or_quality_metrics": False,
    }
    for name, expected in expected_contribution.items():
        if contribution.get(name) != expected:
            raise ProtocolError(f"one-view contribution contract 漂移: {name}")

    runtime = config["runtime"]
    upstream = Path(runtime["drivestudio_checkout"])
    if _git(upstream, "rev-parse", "HEAD") != runtime["drivestudio_commit"]:
        raise ProtocolError("DriveStudio source commit 漂移")
    if _git(upstream, "status", "--short") != runtime["drivestudio_expected_status"]:
        raise ProtocolError("DriveStudio frozen patch status 漂移")

    locks = config["locks"]
    for name in (
        "dino_model_load",
        "pca_fit",
        "feature_sidecar_persist",
        "uplift_feature_compute",
        "membership_proxy_read",
        "method_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"one-view contribution lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, operator_freeze


def summarize_contributions(
    *,
    gaussian_id: np.ndarray,
    pixel_id: np.ndarray,
    contribution_weight: np.ndarray,
    gaussian_count: int,
    pixel_count: int,
    minimum_intersection_contribution: float,
    minimum_gaussian_view_mass: float,
    quantiles: list[float],
) -> dict[str, Any]:
    gids = np.asarray(gaussian_id, dtype=np.int64)
    pixels = np.asarray(pixel_id, dtype=np.int64)
    weights = np.asarray(contribution_weight, dtype=np.float64)
    if gids.ndim != 1 or pixels.shape != gids.shape or weights.shape != gids.shape:
        raise ValueError("contribution arrays 必须是一一对齐的一维数组")
    if np.any((gids < 0) | (gids >= int(gaussian_count))):
        raise ValueError("gaussian_id 越界")
    if np.any((pixels < 0) | (pixels >= int(pixel_count))):
        raise ValueError("pixel_id 越界")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("contribution weight 必须为 finite 非负")
    selected = weights >= float(minimum_intersection_contribution)
    selected_gids = gids[selected]
    selected_pixels = pixels[selected]
    selected_weights = weights[selected]
    mass = np.bincount(
        selected_gids,
        weights=selected_weights,
        minlength=int(gaussian_count),
    )
    intersection_count_by_gaussian = np.bincount(
        selected_gids, minlength=int(gaussian_count)
    )
    intersection_count_by_pixel = np.bincount(
        selected_pixels, minlength=int(pixel_count)
    )
    before_mass = mass > 0.0
    after_mass = mass >= float(minimum_gaussian_view_mass)

    def quantile_map(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {str(value): 0.0 for value in quantiles}
        observed = np.quantile(values, quantiles)
        return {
            str(key): float(value) for key, value in zip(quantiles, observed)
        }

    return {
        "raw_intersection_count": int(gids.size),
        "supported_intersection_count": int(selected.sum()),
        "dropped_intersection_count": int((~selected).sum()),
        "raw_contribution_mass": float(weights.sum()),
        "supported_contribution_mass": float(selected_weights.sum()),
        "dropped_contribution_mass": float(weights[~selected].sum()),
        "gaussian_count": int(gaussian_count),
        "gaussian_with_intersection_support": int(before_mass.sum()),
        "gaussian_after_view_mass_floor": int(after_mass.sum()),
        "gaussian_dropped_by_view_mass_floor": int((before_mass & ~after_mass).sum()),
        "gaussian_coverage_before_mass_floor": float(before_mass.mean()),
        "gaussian_coverage_after_mass_floor": float(after_mass.mean()),
        "pixel_count": int(pixel_count),
        "pixel_with_intersection_support": int(np.count_nonzero(intersection_count_by_pixel)),
        "maximum_supported_intersections_per_pixel": int(
            intersection_count_by_pixel.max(initial=0)
        ),
        "maximum_supported_intersections_per_gaussian": int(
            intersection_count_by_gaussian.max(initial=0)
        ),
        "supported_intersection_weight_quantiles": quantile_map(selected_weights),
        "supported_gaussian_mass_quantiles": quantile_map(mass[before_mass]),
        "minimum_intersection_contribution": float(
            minimum_intersection_contribution
        ),
        "minimum_gaussian_view_mass": float(minimum_gaussian_view_mass),
        "intersection_rows_persisted": False,
    }


def _build_and_render(config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    from scripts.run_worldsim_v3_a3_s_b_paired_smoke import to_device
    from scripts.run_worldsim_v5_m1_unary_diagnostic import (
        _build_runtime,
        _collect_render_state,
        _global_layout,
        _runtime_helpers,
    )

    device = torch.device("cuda:0")
    _, dataset, trainer = _build_runtime(config, device)
    _, _, background_count, rigid_count = _global_layout(trainer)
    scene = config["scene"]
    if background_count != int(scene["expected_background_gaussians"]):
        raise ProtocolError("Background Gaussian count 漂移")
    if rigid_count != int(scene["expected_rigid_gaussians"]):
        raise ProtocolError("Rigid Gaussian count 漂移")
    gaussian_count = background_count + rigid_count
    if gaussian_count != int(scene["expected_total_gaussians"]):
        raise ProtocolError("total Gaussian count 漂移")

    image_infos_all, camera_infos = dataset.full_image_set.get_image(
        int(scene["image_index"]), camera_downscale=1.0
    )
    image_infos = {
        name: image_infos_all[name] for name in ("normed_time", "img_idx")
    }
    image_infos = to_device(image_infos, device)
    camera_infos = to_device(camera_infos, device)
    del image_infos_all
    _, _, release_trainer_render_info = _runtime_helpers()
    try:
        with torch.inference_mode():
            processed_camera, _ = _collect_render_state(
                trainer, image_infos, camera_infos
            )
            gids, pixels, projected, weights, depths = renderer_intersections(
                trainer.info
            )
        height = int(trainer.info["height"])
        width = int(trainer.info["width"])
        n_cameras = int(trainer.info["n_cameras"])
        if n_cameras != int(config["contribution"]["expected_n_cameras"]):
            raise ProtocolError("renderer n_cameras 漂移")
        if [width, height] != list(scene["image_size_wh"]):
            raise ProtocolError("renderer width/height 漂移")
        del projected, depths
        report = summarize_contributions(
            gaussian_id=gids,
            pixel_id=pixels,
            contribution_weight=weights,
            gaussian_count=gaussian_count,
            pixel_count=height * width,
            minimum_intersection_contribution=float(
                config["contribution"]["minimum_intersection_contribution"]
            ),
            minimum_gaussian_view_mass=float(
                config["contribution"]["minimum_gaussian_view_mass"]
            ),
            quantiles=[
                float(value) for value in config["contribution"]["report_quantiles"]
            ],
        )
        report.update(
            {
                "height": height,
                "width": width,
                "n_cameras": n_cameras,
                "background_gaussian_count": background_count,
                "rigid_gaussian_count": rigid_count,
                "dataset_image_tensor_materialized": True,
                "pixel_rgb_values_consumed": False,
                "lidar_values_consumed": False,
                "membership_proxy_consumed": False,
                "feature_or_quality_metrics_computed": False,
            }
        )
        return report, trainer
    finally:
        release_trainer_render_info(trainer)


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    if branch != V51_BRANCH or _git(PROJECT, "status", "--short"):
        raise ProtocolError("one-view contribution 要求 V5.1 branch clean worktree")
    config, operator_freeze = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {"contribution_smoke": config, "operator_freeze": operator_freeze},
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    checkpoint = Path(config["inputs"]["formal_checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    gpu_start = _nvidia_used_mib()
    if gpu_start > int(config["resources"]["maximum_nvidia_used_at_start_mib"]):
        raise ProtocolError("one-view contribution GPU start 非空闲")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(float(config["resources"]["sample_interval_seconds"]))
    monitor.start()
    timeout_seconds = int(config["resources"]["timeout_seconds"])

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"one-view contribution 超过 {timeout_seconds} s")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    started = time.monotonic()
    trainer = None
    try:
        report, trainer = _build_and_render(config)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        monitor.stop()
        _write_jsonl(run_dir / "resource_samples.jsonl", monitor.samples)
    duration = time.monotonic() - started
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise ProtocolError("one-view contribution 前后 checkpoint SHA 不一致")
    if trainer is not None:
        del trainer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    valid_samples = [sample for sample in monitor.samples if "gpu_used_mib" in sample]
    if not valid_samples:
        raise ProtocolError("one-view contribution resource monitor 无有效 sample")
    resource = {
        "gpu_used_at_start_mib": gpu_start,
        "nvidia_smi_peak_used_mib": max(
            int(sample["gpu_used_mib"]) for sample in valid_samples
        ),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "cgroup_memory_peak_bytes": max(
            int(sample["cgroup_memory_current_bytes"]) for sample in valid_samples
        ),
        "sample_count": len(valid_samples),
        "monitor_error_count": len(monitor.samples) - len(valid_samples),
        "duration_seconds": duration,
    }
    if resource["nvidia_smi_peak_used_mib"] > int(
        config["resources"]["maximum_nvidia_peak_mib"]
    ):
        raise ProtocolError("one-view contribution NVIDIA peak 超限")
    if resource["torch_peak_reserved_mib"] > int(
        config["resources"]["maximum_torch_reserved_peak_mib"]
    ):
        raise ProtocolError("one-view contribution Torch reserved peak 超限")
    if resource["cgroup_memory_peak_bytes"] > int(
        config["resources"]["maximum_cgroup_peak_bytes"]
    ):
        raise ProtocolError("one-view contribution cgroup peak 超限")
    _write_json(run_dir / "artifacts/contribution_inventory.json", report)
    _write_json(run_dir / "artifacts/resources.json", resource)
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"metric": name, "value": value}
            for name, value in (
                ("raw_intersection_count", report["raw_intersection_count"]),
                (
                    "supported_intersection_count",
                    report["supported_intersection_count"],
                ),
                (
                    "gaussian_after_view_mass_floor",
                    report["gaussian_after_view_mass_floor"],
                ),
                (
                    "gaussian_coverage_after_mass_floor",
                    report["gaussian_coverage_after_mass_floor"],
                ),
                ("duration_seconds", duration),
            )
        ],
    )
    summary = {
        "schema_version": "worldsim_v51_one_view_contribution_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "one_h_view_renderer_contribution_denominator_ready",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "scene": config["scene"],
        "contribution_inventory": report,
        "resource": resource,
        "dino_model_load": False,
        "pca_fit": False,
        "feature_sidecar_persisted": False,
        "uplift_feature_computed": False,
        "pixel_rgb_values_consumed": False,
        "lidar_values_consumed": False,
        "membership_proxy_read": False,
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
            "schema_version": "worldsim_v51_one_view_contribution_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "checkpoint_sha256": checkpoint_before,
            "operator_freeze_sha256": config["operator_freeze"]["sha256"],
            "scene": config["scene"],
            "seed": int(config["seed"]),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml",
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
            "schema_version": "worldsim_v51_one_view_contribution_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_one_view_contribution_status_v1",
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
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_one_view_contribution_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
