#!/usr/bin/env python3
"""在冻结 H 视图上流式计算 matched B0/B1 uplift；不读取质量或 proxy。"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
import scipy
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.view_selection import atomic_save_deterministic_npz
from motion_proj.worldsim_v5.renderer_intersections import renderer_intersections
from motion_proj.worldsim_v51.feature_sidecar import array_sha256, record_chain_sha256
from motion_proj.worldsim_v51.feature_uplift import (
    accumulate_streaming_uplift_view,
    finalize_streaming_uplift,
    initialize_streaming_uplift,
)
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
        raise ProtocolError("H uplift 预期单 GPU")
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


def validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_b_h_uplift_v1":
        raise ProtocolError("H uplift schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("H uplift task 漂移")
    if config.get("status") != "running" or config.get("seed") != 20260814:
        raise ProtocolError("H uplift status/seed 漂移")
    for name in ("operator_freeze", "feature_freeze"):
        spec = config[name]
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"H uplift freeze identity 漂移: {name}")
        if load_yaml(path).get("status") != spec["required_status"]:
            raise ProtocolError(f"H uplift freeze status 漂移: {name}")

    feature_spec = config["feature_freeze"]
    feature_manifest_path = Path(feature_spec["run_path"]) / feature_spec[
        "manifest_path"
    ]
    if not feature_manifest_path.is_file() or sha256_file(
        feature_manifest_path
    ) != feature_spec["manifest_sha256"]:
        raise ProtocolError("H uplift feature manifest identity 漂移")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("pca_state_sha256") != feature_spec["pca_state_sha256"]:
        raise ProtocolError("H uplift PCA state identity 漂移")
    if feature_manifest.get("record_chain_sha256") != feature_spec[
        "record_chain_sha256"
    ]:
        raise ProtocolError("H uplift feature record chain 漂移")
    records = list(feature_manifest["records"])
    if len(records) != 45:
        raise ProtocolError("H uplift feature record denominator 漂移")
    record_by_view: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record['scene']}:{int(record['frame'])}:{int(record['camera'])}"
        if key in record_by_view:
            raise ProtocolError(f"H uplift duplicate feature view: {key}")
        sidecar = Path(feature_spec["run_path"]) / record["path"]
        if not sidecar.is_file() or sha256_file(sidecar) != record["file_sha256"]:
            raise ProtocolError(f"H uplift feature sidecar identity 漂移: {key}")
        if record["pca_state_sha256"] != feature_spec["pca_state_sha256"]:
            raise ProtocolError(f"H uplift sidecar PCA identity 漂移: {key}")
        record_by_view[key] = record

    expected_scene_names = ["scene-0471", "scene-1087", "scene-0379"]
    if [scene["name"] for scene in config["scenes"]] != expected_scene_names:
        raise ProtocolError("H uplift scene order 漂移")
    for scene in config["scenes"]:
        for name in ("formal_summary", "formal_checkpoint", "source_config"):
            spec = scene[name]
            path = Path(spec["path"])
            if not path.is_file():
                raise ProtocolError(f"H uplift scene input 缺失: {scene['name']}/{name}")
            if name == "formal_checkpoint" and path.stat().st_size != int(spec["bytes"]):
                raise ProtocolError(f"H uplift checkpoint bytes 漂移: {scene['name']}")
            if sha256_file(path) != spec["sha256"]:
                raise ProtocolError(f"H uplift scene input SHA 漂移: {scene['name']}/{name}")
        source = load_yaml(Path(scene["source_config"]["path"]))
        if source["data"]["scene_idx"] != int(scene["index"]):
            raise ProtocolError(f"H uplift source scene_idx 漂移: {scene['name']}")
        if source["data"]["pixel_source"]["downscale_when_loading"] != [2, 2, 2]:
            raise ProtocolError(f"H uplift source downscale 漂移: {scene['name']}")
        total = int(scene["background_gaussians"]) + int(scene["rigid_gaussians"])
        if total != int(scene["total_gaussians"]):
            raise ProtocolError(f"H uplift Gaussian denominator 漂移: {scene['name']}")

    view = config["view_contract"]
    expected_view = {
        "frames": [0, 40, 80, 120, 160],
        "cameras": [0, 1, 2],
        "views_per_scene": 15,
        "total_views": 45,
        "image_index_formula": "frame_times_3_plus_camera",
        "sensor_image_size_wh": [1600, 900],
        "source_downscale_when_loading": [2, 2, 2],
        "model_native_renderer_size_wh": [800, 450],
        "patch_grid_shape": [40, 64, 114],
        "feature_dimension": 40,
    }
    for name, expected in expected_view.items():
        if view.get(name) != expected:
            raise ProtocolError(f"H uplift view contract 漂移: {name}")
    expected_keys = {
        f"{scene}:{frame}:{camera}"
        for scene in expected_scene_names
        for frame in view["frames"]
        for camera in view["cameras"]
    }
    if set(record_by_view) != expected_keys:
        raise ProtocolError("H uplift feature view grid 不完整")

    operator = config["operator"]
    expected_operator = {
        "contribution_source": "motion_proj.worldsim_v5.renderer_intersections",
        "formula": "alpha_times_transmittance_before_alpha",
        "minimum_intersection_contribution": 1e-4,
        "minimum_gaussian_view_mass": 1e-3,
        "epsilon": 1e-8,
        "accumulator_dtype": "float64",
        "output_dtype": "float32",
        "sparse_transpose": "scipy_csr_float64",
        "dense_intersection_feature_materialization": False,
        "pixel_feature_sampling": "lazy_bilinear_align_corners_false",
        "b0": "view_saturated_intersection",
        "b1": "normalized_renderer_transpose",
        "optional_pruning": False,
    }
    for name, expected in expected_operator.items():
        if operator.get(name) != expected:
            raise ProtocolError(f"H uplift operator contract 漂移: {name}")

    runtime = config["runtime"]
    observed_runtime = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected_runtime = {name: runtime[name] for name in observed_runtime}
    if observed_runtime != expected_runtime:
        raise ProtocolError(f"H uplift runtime 漂移: {observed_runtime}")
    upstream = Path(runtime["drivestudio_checkout"])
    if _git(upstream, "rev-parse", "HEAD") != runtime["drivestudio_commit"]:
        raise ProtocolError("H uplift DriveStudio commit 漂移")
    if _git(upstream, "status", "--short") != runtime["drivestudio_expected_status"]:
        raise ProtocolError("H uplift DriveStudio patch status 漂移")
    patch = Path(runtime["compatibility_patch"]["path"])
    if not patch.is_file() or sha256_file(patch) != runtime["compatibility_patch"]["sha256"]:
        raise ProtocolError("H uplift compatibility patch 漂移")

    locks = config["locks"]
    for name in ("h_feature_sidecar_read", "h_renderer_start", "uplift_feature_compute"):
        if locks.get(name) is not True:
            raise ProtocolError(f"H uplift authorization 漂移: {name}")
    for name in (
        "pixel_rgb_values_consumed",
        "lidar_values_consumed",
        "membership_proxy_read",
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
            raise ProtocolError(f"H uplift lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, records, record_by_view


def _build_scene_runtime(config: dict[str, Any], scene: dict[str, Any]):
    from scripts.run_worldsim_v5_m1_unary_diagnostic import (
        _build_runtime,
        _global_layout,
    )

    runtime_config = {
        "runtime": config["runtime"],
        "inputs": {
            "source_config": scene["source_config"],
            "formal_checkpoint": scene["formal_checkpoint"],
        },
    }
    _, dataset, trainer = _build_runtime(runtime_config, torch.device("cuda:0"))
    _, _, background_count, rigid_count = _global_layout(trainer)
    if background_count != int(scene["background_gaussians"]):
        raise ProtocolError(f"Background Gaussian count 漂移: {scene['name']}")
    if rigid_count != int(scene["rigid_gaussians"]):
        raise ProtocolError(f"Rigid Gaussian count 漂移: {scene['name']}")
    return dataset, trainer


def _render_intersections(
    config: dict[str, Any], dataset: Any, trainer: Any, *, frame: int, camera: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from scripts.run_worldsim_v3_a3_s_b_paired_smoke import to_device
    from scripts.run_worldsim_v5_m1_unary_diagnostic import (
        _collect_render_state,
        _runtime_helpers,
    )

    image_index = int(frame) * 3 + int(camera)
    image_infos_all, camera_infos = dataset.full_image_set.get_image(
        image_index, camera_downscale=1.0
    )
    observed_image_index = int(image_infos_all["img_idx"].flatten()[0].item())
    if observed_image_index != image_index:
        raise ProtocolError(
            f"H uplift image index 漂移: observed={observed_image_index}, expected={image_index}"
        )
    image_infos = {
        name: image_infos_all[name] for name in ("normed_time", "img_idx")
    }
    image_infos = to_device(image_infos, torch.device("cuda:0"))
    camera_infos = to_device(camera_infos, torch.device("cuda:0"))
    del image_infos_all
    _, _, release_trainer_render_info = _runtime_helpers()
    try:
        with torch.inference_mode():
            _collect_render_state(trainer, image_infos, camera_infos)
            gids, pixels, projected, weights, depths = renderer_intersections(trainer.info)
        height = int(trainer.info["height"])
        width = int(trainer.info["width"])
        n_cameras = int(trainer.info["n_cameras"])
        if [width, height] != config["view_contract"][
            "model_native_renderer_size_wh"
        ]:
            raise ProtocolError(f"H uplift renderer size 漂移: {[width, height]}")
        if n_cameras != 1:
            raise ProtocolError("H uplift renderer n_cameras 漂移")
        del projected, depths
        return gids, pixels, weights, {
            "image_index": image_index,
            "height": height,
            "width": width,
            "n_cameras": n_cameras,
        }
    finally:
        release_trainer_render_info(trainer)


def _difference_l2(first: np.ndarray, second: np.ndarray, chunk_rows: int = 8192) -> float:
    squared = 0.0
    for start in range(0, first.shape[0], chunk_rows):
        delta = (
            np.asarray(first[start : start + chunk_rows], dtype=np.float64)
            - np.asarray(second[start : start + chunk_rows], dtype=np.float64)
        )
        squared += float(np.einsum("ij,ij->", delta, delta, dtype=np.float64))
    return float(np.sqrt(squared))


def execute(
    config: dict[str, Any], record_by_view: dict[str, dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    from motion_proj.worldsim_v51.feature_sidecar import validate_sidecar_identity

    feature_root = Path(config["feature_freeze"]["run_path"])
    scene_reports: list[dict[str, Any]] = []
    sidecar_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    total_processed_views = 0
    for scene_index, scene in enumerate(config["scenes"]):
        scene_started = time.monotonic()
        checkpoint_path = Path(scene["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint_path)
        dataset, trainer = _build_scene_runtime(config, scene)
        state = initialize_streaming_uplift(
            gaussian_count=int(scene["total_gaussians"]),
            feature_dimension=int(config["view_contract"]["feature_dimension"]),
        )
        view_reports: list[dict[str, Any]] = []
        for frame in config["view_contract"]["frames"]:
            for camera in config["view_contract"]["cameras"]:
                key = f"{scene['name']}:{int(frame)}:{int(camera)}"
                feature_record = record_by_view[key]
                feature_path = feature_root / feature_record["path"]
                with np.load(feature_path, allow_pickle=False) as archive:
                    patch_grid = np.asarray(archive["feature"], dtype=np.float32)
                validate_sidecar_identity(feature_record, patch_grid)
                gids, pixels, weights, render_report = _render_intersections(
                    config,
                    dataset,
                    trainer,
                    frame=int(frame),
                    camera=int(camera),
                )
                torch.cuda.empty_cache()
                view_started = time.monotonic()
                uplift_report = accumulate_streaming_uplift_view(
                    state,
                    gaussian_id=gids,
                    pixel_id=pixels,
                    contribution_weight=weights,
                    patch_grid=patch_grid,
                    image_height=render_report["height"],
                    image_width=render_report["width"],
                    minimum_intersection_contribution=float(
                        config["operator"]["minimum_intersection_contribution"]
                    ),
                    minimum_gaussian_view_mass=float(
                        config["operator"]["minimum_gaussian_view_mass"]
                    ),
                    epsilon=float(config["operator"]["epsilon"]),
                )
                view_report = {
                    "scene": scene["name"],
                    "frame": int(frame),
                    "camera": int(camera),
                    "feature_file_sha256": feature_record["file_sha256"],
                    "feature_content_sha256": feature_record["content_sha256"],
                    "seconds_cpu_uplift": time.monotonic() - view_started,
                    **render_report,
                    **uplift_report,
                }
                view_reports.append(view_report)
                total_processed_views += 1
                _write_json(
                    run_dir / "artifacts/uplift_progress.json",
                    {
                        "schema_version": "worldsim_v51_h_uplift_progress_v1",
                        "completed_scene_count": scene_index,
                        "completed_view_count": total_processed_views,
                        "expected_view_count": int(
                            config["view_contract"]["total_views"]
                        ),
                        "last_view": view_report,
                    },
                )
                del gids, pixels, weights, patch_grid
                gc.collect()
        final = finalize_streaming_uplift(
            state, epsilon=float(config["operator"]["epsilon"])
        )
        if int(final["report"]["processed_view_count"]) != int(
            config["view_contract"]["views_per_scene"]
        ):
            raise ProtocolError(f"H uplift processed view count 漂移: {scene['name']}")
        if not np.array_equal(
            final["b0_denominator"] > 0.0, final["b1_denominator"] > 0.0
        ):
            raise ProtocolError(f"H uplift B0/B1 coverage alias 漂移: {scene['name']}")
        difference_l2 = _difference_l2(final["b0_feature"], final["b1_feature"])
        if difference_l2 <= 0.0:
            raise ProtocolError(f"H uplift B0/B1 意外完全相同: {scene['name']}")
        for arm, feature_key, weight_key in (
            ("B0", "b0_feature", "b0_denominator"),
            ("B1", "b1_feature", "b1_denominator"),
        ):
            relative = Path("artifacts/gaussian_features") / (
                f"{scene['name']}__{arm.lower()}.npz"
            )
            path = run_dir / relative
            arrays = {
                "feature": final[feature_key],
                "weight": final[weight_key],
                "supported_view_count": final["supported_view_count"],
            }
            atomic_save_deterministic_npz(path, arrays)
            sidecar_records.append(
                {
                    "scene": scene["name"],
                    "scene_index": int(scene["index"]),
                    "arm": arm,
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "file_sha256": sha256_file(path),
                    "feature_shape": list(arrays["feature"].shape),
                    "feature_dtype": str(arrays["feature"].dtype),
                    "feature_content_sha256": array_sha256(arrays["feature"]),
                    "weight_shape": list(arrays["weight"].shape),
                    "weight_dtype": str(arrays["weight"].dtype),
                    "weight_content_sha256": array_sha256(arrays["weight"]),
                    "supported_view_count_content_sha256": array_sha256(
                        arrays["supported_view_count"]
                    ),
                    "base_checkpoint_sha256": scene["formal_checkpoint"]["sha256"],
                    "pca_state_sha256": config["feature_freeze"][
                        "pca_state_sha256"
                    ],
                    "operator_freeze_sha256": config["operator_freeze"]["sha256"],
                }
            )
        checkpoint_after = sha256_file(checkpoint_path)
        if checkpoint_before != checkpoint_after:
            raise ProtocolError(f"H uplift checkpoint 前后 SHA 漂移: {scene['name']}")
        checkpoint_records.append(
            {
                "scene": scene["name"],
                "before": checkpoint_before,
                "after": checkpoint_after,
                "immutable": True,
            }
        )
        scene_report = {
            "scene": scene["name"],
            "scene_index": int(scene["index"]),
            "gaussian_count": int(scene["total_gaussians"]),
            "background_gaussian_count": int(scene["background_gaussians"]),
            "rigid_gaussian_count": int(scene["rigid_gaussians"]),
            "b0_b1_difference_l2": difference_l2,
            "gaussian_coverage": float(
                final["report"]["covered_gaussian_count"]
                / int(scene["total_gaussians"])
            ),
            "supported_view_count_min_covered": int(
                final["supported_view_count"][
                    final["supported_view_count"] > 0
                ].min()
            ),
            "supported_view_count_max": int(final["supported_view_count"].max(initial=0)),
            "report": final["report"],
            "view_reports": view_reports,
            "seconds": time.monotonic() - scene_started,
        }
        _write_json(
            run_dir / f"artifacts/scene_reports/{scene['name']}.json", scene_report
        )
        scene_reports.append(scene_report)
        del final, state, trainer, dataset
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    sidecar_manifest = {
        "schema_version": "worldsim_v51_h_gaussian_feature_manifest_v1",
        "task_id": config["task_id"],
        "record_count": len(sidecar_records),
        "record_chain_sha256": record_chain_sha256(sidecar_records),
        "records": sidecar_records,
    }
    _write_json(run_dir / "artifacts/gaussian_feature_manifest.json", sidecar_manifest)
    return {
        "processed_scene_count": len(scene_reports),
        "processed_view_count": total_processed_views,
        "scene_reports": scene_reports,
        "checkpoint_records": checkpoint_records,
        "gaussian_sidecar_count": len(sidecar_records),
        "gaussian_sidecar_total_bytes": sum(row["bytes"] for row in sidecar_records),
        "gaussian_feature_manifest_sha256": sha256_file(
            run_dir / "artifacts/gaussian_feature_manifest.json"
        ),
        "gaussian_feature_record_chain_sha256": sidecar_manifest[
            "record_chain_sha256"
        ],
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    if branch != V51_BRANCH or _git(PROJECT, "status", "--short"):
        raise ProtocolError("H uplift formal run 要求 V5.1 clean worktree")
    config, records, record_by_view = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {
                "h_uplift": config,
                "feature_record_count": len(records),
                "feature_record_chain_sha256": config["feature_freeze"][
                    "record_chain_sha256"
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    gpu_start = _nvidia_used_mib()
    resources = config["resources"]
    disk_available = shutil.disk_usage("/root/autodl-tmp").free
    if gpu_start > int(resources["maximum_nvidia_used_at_start_mib"]):
        raise ProtocolError("H uplift GPU start 非空闲")
    if disk_available < int(resources["minimum_disk_available_bytes"]):
        raise ProtocolError("H uplift disk available 不足")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(float(resources["sample_interval_seconds"]))
    monitor.start()
    timeout_seconds = int(resources["timeout_seconds"])

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"H uplift 超过 {timeout_seconds} s")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    started = time.monotonic()
    try:
        report = execute(config, record_by_view, run_dir)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        monitor.stop()
        _write_jsonl(run_dir / "resource_samples.jsonl", monitor.samples)
    duration = time.monotonic() - started
    valid = [row for row in monitor.samples if "gpu_used_mib" in row]
    if not valid:
        raise ProtocolError("H uplift resource monitor 无有效 sample")
    resource = {
        "gpu_used_at_start_mib": gpu_start,
        "nvidia_smi_peak_used_mib": max(row["gpu_used_mib"] for row in valid),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "cgroup_memory_peak_bytes": max(row["cgroup_memory_current_bytes"] for row in valid),
        "disk_available_at_start_bytes": disk_available,
        "sample_count": len(valid),
        "monitor_error_count": len(monitor.samples) - len(valid),
        "duration_seconds": duration,
    }
    _write_json(run_dir / "artifacts/resources.json", resource)
    _write_json(run_dir / "artifacts/uplift_report.json", report)
    if resource["nvidia_smi_peak_used_mib"] > int(resources["maximum_nvidia_peak_mib"]):
        raise ProtocolError("H uplift NVIDIA peak 超限")
    if resource["torch_peak_reserved_mib"] > int(
        resources["maximum_torch_reserved_peak_mib"]
    ):
        raise ProtocolError("H uplift Torch reserved peak 超限")
    if resource["cgroup_memory_peak_bytes"] > int(resources["maximum_cgroup_peak_bytes"]):
        raise ProtocolError("H uplift cgroup peak 超限")
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {
                "metric": "scene_denominator",
                "scene": row["scene"],
                "gaussian_count": row["gaussian_count"],
                "covered_gaussian_count": row["report"]["covered_gaussian_count"],
                "gaussian_coverage": row["gaussian_coverage"],
                "b0_b1_difference_l2": row["b0_b1_difference_l2"],
            }
            for row in report["scene_reports"]
        ]
        + [{"metric": "resource_terminal", **resource}],
    )
    summary = {
        "schema_version": "worldsim_v51_h_uplift_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "h_45_view_b0_b1_gaussian_sidecars_ready_without_quality_read",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "report": report,
        "resource": resource,
        "h_feature_sidecar_read": True,
        "h_renderer_started": True,
        "uplift_feature_computed": True,
        "pixel_rgb_values_consumed": False,
        "lidar_values_consumed": False,
        "membership_proxy_read": False,
        "method_quality_read": False,
        "screening_pixels_read": False,
        "screening_quality_read": False,
        "confirmation_pixels_read": False,
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
            "schema_version": "worldsim_v51_h_uplift_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "operator_freeze_sha256": config["operator_freeze"]["sha256"],
            "feature_freeze_sha256": config["feature_freeze"]["sha256"],
            "pca_state_sha256": config["feature_freeze"]["pca_state_sha256"],
            "gaussian_feature_record_chain_sha256": report[
                "gaussian_feature_record_chain_sha256"
            ],
            "seed": int(config["seed"]),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_b_h_uplift_v1.yaml",
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
            "schema_version": "worldsim_v51_h_uplift_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_h_uplift_status_v1",
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
                "schema_version": "worldsim_v51_h_uplift_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
