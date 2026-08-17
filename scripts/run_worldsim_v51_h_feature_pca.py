#!/usr/bin/env python3
"""提取冻结 H 视图的 DINOv2 features 并确定性拟合 40-D PCA；不读质量。"""

from __future__ import annotations

import argparse
import gc
import importlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import sklearn
import scipy
import torch
from torch.nn import functional
import xformers
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.view_selection import atomic_save_deterministic_npz
from motion_proj.worldsim_v51.feature_sidecar import (
    array_sha256,
    feature_mean_std_correction1,
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
        raise ProtocolError("H feature/PCA 预期单 GPU")
    return values[0]


def _gpu_name_and_total() -> tuple[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, total = output.rsplit(",", 1)
    return name.strip(), int(total.strip())


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
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_b_h_feature_pca_v1":
        raise ProtocolError("H feature/PCA schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("H feature/PCA task 漂移")
    if config.get("status") != "running" or config.get("seed") != 20260814:
        raise ProtocolError("H feature/PCA status/seed 漂移")

    for name in ("input_freeze", "dino_resource_freeze", "contribution_freeze"):
        spec = config[name]
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"H feature/PCA freeze binding 漂移: {name}")
        freeze = load_yaml(path)
        if freeze.get("status") != spec["required_status"]:
            raise ProtocolError(f"H feature/PCA freeze status 漂移: {name}")

    source = config["source"]
    source_path = Path(source["path"])
    if _git(source_path, "remote", "get-url", "origin") != source["repository"]:
        raise ProtocolError("DINOv2 source origin 漂移")
    if _git(source_path, "rev-parse", "HEAD") != source["commit"]:
        raise ProtocolError("DINOv2 source commit 漂移")
    if _git(source_path, "rev-parse", "HEAD^{tree}") != source["tree"]:
        raise ProtocolError("DINOv2 source tree 漂移")
    if _git(source_path, "status", "--short") != source["expected_git_status"]:
        raise ProtocolError("DINOv2 source worktree 非 clean")
    if sha256_file(source_path / "LICENSE") != source["license_sha256"]:
        raise ProtocolError("DINOv2 LICENSE SHA 漂移")
    if sha256_file(source_path / "hubconf.py") != source["hubconf_sha256"]:
        raise ProtocolError("DINOv2 hubconf SHA 漂移")

    checkpoint = Path(config["checkpoint"]["path"])
    if not checkpoint.is_file():
        raise ProtocolError("DINOv2 checkpoint 缺失")
    if checkpoint.stat().st_size != int(config["checkpoint"]["bytes"]):
        raise ProtocolError("DINOv2 checkpoint bytes 漂移")
    if sha256_file(checkpoint) != config["checkpoint"]["sha256"]:
        raise ProtocolError("DINOv2 checkpoint SHA 漂移")

    input_spec = config["input_freeze"]
    manifest_path = Path(input_spec["image_manifest_path"])
    if not manifest_path.is_file() or sha256_file(manifest_path) != input_spec[
        "image_manifest_sha256"
    ]:
        raise ProtocolError("H image manifest identity 漂移")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("record_chain_sha256") != input_spec[
        "image_manifest_record_chain_sha256"
    ]:
        raise ProtocolError("H image manifest record chain 漂移")
    views = config["views"]
    records = select_h_uplift_records(
        manifest,
        scenes=views["scenes"],
        frames=views["frames"],
        cameras=views["cameras"],
    )
    if len(records) != int(views["expected_view_count"]):
        raise ProtocolError("H feature view denominator 漂移")
    expected_patch_count = (
        len(records) * int(views["patches_per_view"])
    )
    if expected_patch_count != int(views["expected_patch_count"]):
        raise ProtocolError("H feature patch denominator 漂移")
    expected_raw_bytes = (
        expected_patch_count * int(config["model"]["raw_dimension"]) * 4
    )
    if expected_raw_bytes != int(views["expected_raw_memmap_bytes"]):
        raise ProtocolError("H raw memmap bytes 漂移")

    model = config["model"]
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
        if model.get(name) != expected:
            raise ProtocolError(f"H feature model contract 漂移: {name}")

    pca = config["pca"]
    expected_pca = {
        "standardization_std_correction": 1,
        "statistics_accumulator_dtype": "float64",
        "standardized_storage_dtype": "float32",
        "statistics_chunk_rows": 2048,
        "components": 40,
        "solver": "randomized",
        "random_state": 20260814,
        "whiten": False,
        "sklearn_version": "1.7.2",
        "numpy_version": "1.26.4",
        "scipy_version": "1.15.3",
        "subsample_cap": 500000,
        "subsampling_applied": False,
        "output_dtype": "float32",
        "deterministic_npz_writer": True,
        "pca_fit_on_screening_or_confirmation": False,
    }
    for name, expected in expected_pca.items():
        if pca.get(name) != expected:
            raise ProtocolError(f"H PCA contract 漂移: {name}")
    if list(pca["persisted_state"]) != [
        "feature_mean",
        "feature_std",
        "pca_mean",
        "components",
        "singular_values",
    ]:
        raise ProtocolError("H PCA persisted state 漂移")
    if int(views["expected_patch_count"]) > int(pca["subsample_cap"]):
        raise ProtocolError("H PCA 意外触发 subsample")

    environment = config["environment"]
    observed_environment = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "xformers": xformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected_environment = {
        "python": environment["python"],
        "torch": environment["torch"],
        "torch_cuda": environment["torch_cuda"],
        "xformers": environment["xformers"],
        "sklearn": pca["sklearn_version"],
        "numpy": pca["numpy_version"],
        "scipy": pca["scipy_version"],
    }
    if observed_environment != expected_environment:
        raise ProtocolError(
            f"H feature/PCA environment 漂移: {observed_environment}"
        )

    locks = config["locks"]
    if locks.get("h_pixels_read_for_feature_extraction") is not True:
        raise ProtocolError("H pixel extraction authorization 漂移")
    for name in (
        "screening_pixels_read",
        "confirmation_pixels_read",
        "membership_proxy_read",
        "renderer_start",
        "uplift_feature_compute",
        "method_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"H feature/PCA lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, records, manifest


def _preprocess(record: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ProtocolError(f"H image bytes 漂移: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ProtocolError(f"H image SHA 漂移: {path}")
    with Image.open(path) as image:
        if list(image.size) != config["views"]["expected_image_size_wh"]:
            raise ProtocolError(f"H image size 漂移: {path}")
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    width, height = [int(value) for value in config["views"]["model_size_wh"]]
    tensor = functional.interpolate(
        tensor,
        size=(height, width),
        mode=config["model"]["resize_mode"],
        align_corners=bool(config["model"]["resize_align_corners"]),
    )
    mean = torch.tensor(
        config["model"]["normalization_mean"], dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        config["model"]["normalization_std"], dtype=torch.float32
    ).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _predict(model: torch.nn.Module, tensor: torch.Tensor, config: dict[str, Any]) -> np.ndarray:
    input_cuda = tensor.cuda(non_blocking=False)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model.get_intermediate_layers(
            input_cuda,
            n=int(config["model"]["intermediate_layers"]),
            reshape=bool(config["model"]["reshape"]),
            norm=bool(config["model"]["norm"]),
        )
    if len(outputs) != 4 or list(outputs[-1].shape) != config["model"][
        "expected_output_shape"
    ]:
        raise ProtocolError("H DINO intermediate output shape 漂移")
    selected = outputs[-1].float()
    if not bool(torch.isfinite(selected).all().item()):
        raise ProtocolError("H DINO selected feature 非 finite")
    rows = (
        selected.squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    result = np.ascontiguousarray(rows.reshape(-1, rows.shape[-1]))
    del input_cuda, outputs, selected
    return result


def execute(config: dict[str, Any], records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []

    def phase(name: str, started: float) -> None:
        phases.append({"phase": name, "seconds": time.monotonic() - started})

    source_path = Path(config["source"]["path"])
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    import_started = time.monotonic()
    backbones = importlib.import_module("dinov2.hub.backbones")
    phase("import_official_source", import_started)

    torch.set_num_threads(int(config["environment"]["cpu_threads"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    constructor_started = time.monotonic()
    constructor = getattr(backbones, config["model"]["entrypoint"])
    model = constructor(pretrained=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    phase("construct_vitg14_reg_cpu", constructor_started)

    checkpoint = Path(config["checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    load_started = time.monotonic()
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=bool(config["model"]["weights_only_load"]),
        mmap=bool(config["model"]["checkpoint_mmap"]),
    )
    state_dict_key_count = len(state_dict)
    load_result = model.load_state_dict(
        state_dict, strict=bool(config["model"]["strict_state_dict_load"])
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ProtocolError("H DINO strict state_dict missing/unexpected")
    del state_dict
    model.eval().cuda()
    torch.cuda.synchronize()
    phase("load_checkpoint_and_move_model", load_started)

    views = config["views"]
    row_count = int(views["expected_patch_count"])
    dimension = int(config["model"]["raw_dimension"])
    scratch = run_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=False)
    raw_path = scratch / "h_raw_1536d.float32"
    raw = np.memmap(raw_path, mode="w+", dtype=np.float32, shape=(row_count, dimension))
    extraction_records: list[dict[str, Any]] = []
    patches_per_view = int(views["patches_per_view"])
    extraction_started = time.monotonic()
    first_repeat_exact = False
    for index, record in enumerate(records):
        input_cpu = _preprocess(record, config)
        feature = _predict(model, input_cpu, config)
        if feature.shape != (patches_per_view, dimension):
            raise ProtocolError(f"H raw feature shape 漂移: {feature.shape}")
        if index == 0:
            repeated = _predict(model, input_cpu, config)
            first_repeat_exact = bool(np.array_equal(feature, repeated))
            if not first_repeat_exact:
                raise ProtocolError("H DINO first-image repeat 非 bit-exact")
            del repeated
        start = index * patches_per_view
        stop = start + patches_per_view
        raw[start:stop] = feature
        raw.flush()
        extraction_records.append(
            {
                **record,
                "view_order": index,
                "raw_row_start": start,
                "raw_row_stop": stop,
                "raw_feature_sha256": array_sha256(feature),
            }
        )
        _write_json(
            run_dir / "artifacts/extraction_progress.json",
            {
                "schema_version": "worldsim_v51_h_feature_extraction_progress_v1",
                "completed_view_count": index + 1,
                "expected_view_count": len(records),
                "last_view": {
                    "scene": record["scene"],
                    "frame": record["frame"],
                    "camera": record["camera"],
                    "raw_feature_sha256": extraction_records[-1][
                        "raw_feature_sha256"
                    ],
                },
            },
        )
        del feature, input_cpu
    phase("extract_45_h_views_to_cpu_memmap", extraction_started)
    raw.flush()
    if raw_path.stat().st_size != int(views["expected_raw_memmap_bytes"]):
        raise ProtocolError("H raw memmap terminal bytes 漂移")

    torch_peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    torch_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    statistics_started = time.monotonic()
    feature_mean, feature_std = feature_mean_std_correction1(
        raw, chunk_rows=int(config["pca"]["statistics_chunk_rows"])
    )
    standardize_in_place(
        raw,
        mean=feature_mean,
        std=feature_std,
        chunk_rows=int(config["pca"]["statistics_chunk_rows"]),
    )
    phase("deterministic_correction1_standardization", statistics_started)

    pca_started = time.monotonic()
    estimator = PCA(
        n_components=int(config["pca"]["components"]),
        svd_solver=config["pca"]["solver"],
        random_state=int(config["pca"]["random_state"]),
        whiten=bool(config["pca"]["whiten"]),
    )
    estimator.fit(raw)
    phase("fit_seeded_randomized_pca_on_h", pca_started)
    pca_state = {
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "feature_std": np.asarray(feature_std, dtype=np.float64),
        "pca_mean": np.asarray(estimator.mean_, dtype=np.float32),
        "components": np.asarray(estimator.components_, dtype=np.float32),
        "singular_values": np.asarray(estimator.singular_values_, dtype=np.float32),
    }
    state_path = run_dir / config["sidecars"]["state_path"]
    repeat_state_path = state_path.with_name("pca_state.repeat.npz")
    atomic_save_deterministic_npz(state_path, pca_state)
    atomic_save_deterministic_npz(repeat_state_path, dict(reversed(list(pca_state.items()))))
    if state_path.read_bytes() != repeat_state_path.read_bytes():
        raise ProtocolError("H PCA deterministic NPZ repeat 非 byte-exact")
    repeat_state_path.unlink()
    state_sha256 = sha256_file(state_path)

    sidecar_started = time.monotonic()
    sidecar_records: list[dict[str, Any]] = []
    for index, source_record in enumerate(extraction_records):
        start = index * patches_per_view
        stop = start + patches_per_view
        feature = pca_patch_grid(
            raw[start:stop],
            pca_mean=pca_state["pca_mean"],
            components=pca_state["components"],
            grid_hw=views["patch_grid_hw"],
        )
        relative_path = sidecar_relative_path(source_record)
        output_path = run_dir / relative_path
        atomic_save_deterministic_npz(
            output_path, {config["sidecars"]["feature_key"]: feature}
        )
        sidecar_record = {
            "path": relative_path.as_posix(),
            "bytes": output_path.stat().st_size,
            "file_sha256": sha256_file(output_path),
            "source_image_path": source_record["path"],
            "source_image_sha256": source_record["sha256"],
            "raw_feature_sha256": source_record["raw_feature_sha256"],
            "model_commit": config["source"]["commit"],
            "model_checkpoint_sha256": config["checkpoint"]["sha256"],
            "pca_state_sha256": state_sha256,
            "scene": source_record["scene"],
            "scene_index": source_record["scene_index"],
            "frame": source_record["frame"],
            "camera": source_record["camera"],
            "shape": list(feature.shape),
            "dtype": str(feature.dtype),
            "content_sha256": array_sha256(feature),
        }
        required = set(config["sidecars"]["required_identity_fields"])
        if not required.issubset(sidecar_record):
            raise ProtocolError("H feature sidecar identity fields 不完整")
        with np.load(output_path, allow_pickle=False) as archive:
            replay = archive[config["sidecars"]["feature_key"]]
            validate_sidecar_identity(sidecar_record, replay)
        sidecar_records.append(sidecar_record)
    phase("persist_and_replay_45_pca_sidecars", sidecar_started)
    feature_manifest = {
        "schema_version": "worldsim_v51_h_feature_manifest_v1",
        "task_id": config["task_id"],
        "record_count": len(sidecar_records),
        "record_chain_sha256": record_chain_sha256(sidecar_records),
        "pca_state_path": state_path.relative_to(run_dir).as_posix(),
        "pca_state_sha256": state_sha256,
        "records": sidecar_records,
    }
    _write_json(run_dir / config["sidecars"]["manifest_path"], feature_manifest)
    _write_json(
        run_dir / "artifacts/raw_extraction_manifest.json",
        {
            "schema_version": "worldsim_v51_h_raw_extraction_manifest_v1",
            "record_count": len(extraction_records),
            "record_chain_sha256": record_chain_sha256(extraction_records),
            "raw_memmap_bytes_before_cleanup": raw_path.stat().st_size,
            "raw_memmap_persisted_after_success": False,
            "records": extraction_records,
        },
    )

    del raw
    gc.collect()
    raw_path.unlink()
    scratch.rmdir()
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise ProtocolError("H feature/PCA checkpoint SHA 前后不一致")

    return {
        "phases": phases,
        "parameter_count": parameter_count,
        "state_dict_key_count": state_dict_key_count,
        "strict_missing_key_count": len(load_result.missing_keys),
        "strict_unexpected_key_count": len(load_result.unexpected_keys),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "view_count": len(records),
        "patch_count": row_count,
        "raw_dimension": dimension,
        "first_image_repeat_bit_exact": first_repeat_exact,
        "h_input_record_chain_sha256": record_chain_sha256(records),
        "raw_extraction_record_chain_sha256": record_chain_sha256(extraction_records),
        "pca_state_sha256": state_sha256,
        "feature_manifest_sha256": sha256_file(
            run_dir / config["sidecars"]["manifest_path"]
        ),
        "feature_record_chain_sha256": feature_manifest["record_chain_sha256"],
        "feature_mean_min": float(feature_mean.min()),
        "feature_mean_max": float(feature_mean.max()),
        "feature_std_min": float(feature_std.min()),
        "feature_std_max": float(feature_std.max()),
        "singular_value_min": float(pca_state["singular_values"].min()),
        "singular_value_max": float(pca_state["singular_values"].max()),
        "sidecar_total_bytes": sum(row["bytes"] for row in sidecar_records),
        "torch_peak_allocated_mib": torch_peak_allocated,
        "torch_peak_reserved_mib": torch_peak_reserved,
        "raw_memmap_persisted_after_success": False,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    status = _git(PROJECT, "status", "--short")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行")
    if status:
        raise ProtocolError("H feature/PCA formal run 要求 clean worktree")
    config, records, input_manifest = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {
                "h_feature_pca": config,
                "selected_h_records": records,
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
    cgroup_max = int(
        Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
    )
    disk_available = shutil.disk_usage("/root/autodl-tmp").free
    environment = config["environment"]
    resources = config["resources"]
    if gpu_name != environment["gpu"] or gpu_total != int(environment["gpu_total_mib"]):
        raise ProtocolError("H feature/PCA GPU identity 漂移")
    if cgroup_max != int(environment["cgroup_memory_max_bytes"]):
        raise ProtocolError("H feature/PCA cgroup max 漂移")
    if gpu_start > int(resources["maximum_gpu_used_at_start_mib"]):
        raise ProtocolError("H feature/PCA GPU start 非空闲")
    if disk_available < int(resources["minimum_disk_available_bytes"]):
        raise ProtocolError("H feature/PCA disk available 不足")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(float(resources["sample_interval_seconds"]))
    monitor.start()
    timeout_seconds = int(resources["timeout_seconds"])

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"H feature/PCA 超过 {timeout_seconds} s")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    started = time.monotonic()
    try:
        report = execute(config, records, run_dir)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        monitor.stop()
        _write_jsonl(run_dir / "resource_samples.jsonl", monitor.samples)
    duration = time.monotonic() - started
    valid_samples = [row for row in monitor.samples if "gpu_used_mib" in row]
    if not valid_samples:
        raise ProtocolError("H feature/PCA resource monitor 无有效 sample")
    resource = {
        "gpu_name": gpu_name,
        "gpu_total_mib": gpu_total,
        "gpu_used_at_start_mib": gpu_start,
        "nvidia_smi_peak_used_mib": max(row["gpu_used_mib"] for row in valid_samples),
        "torch_peak_allocated_mib": report["torch_peak_allocated_mib"],
        "torch_peak_reserved_mib": report["torch_peak_reserved_mib"],
        "cgroup_memory_max_bytes": cgroup_max,
        "cgroup_memory_peak_bytes": max(
            row["cgroup_memory_current_bytes"] for row in valid_samples
        ),
        "disk_available_at_start_bytes": disk_available,
        "sample_count": len(valid_samples),
        "monitor_error_count": len(monitor.samples) - len(valid_samples),
        "duration_seconds": duration,
    }
    _write_json(run_dir / "artifacts/resources.json", resource)
    _write_json(run_dir / "artifacts/feature_pca_report.json", report)
    if resource["nvidia_smi_peak_used_mib"] > int(resources["maximum_gpu_peak_mib"]):
        raise ProtocolError("H feature/PCA NVIDIA peak 超限")
    if resource["torch_peak_reserved_mib"] > int(resources["maximum_gpu_peak_mib"]):
        raise ProtocolError("H feature/PCA Torch reserved peak 超限")
    if resource["cgroup_memory_peak_bytes"] > int(resources["maximum_cgroup_peak_bytes"]):
        raise ProtocolError("H feature/PCA cgroup peak 超限")

    _write_jsonl(
        run_dir / "metrics.jsonl",
        [{"metric": "phase_duration", **row} for row in report["phases"]]
        + [
            {"metric": "view_count", "value": report["view_count"]},
            {"metric": "patch_count", "value": report["patch_count"]},
            {"metric": "sidecar_total_bytes", "value": report["sidecar_total_bytes"]},
            {"metric": "resource_terminal", **resource},
        ],
    )
    summary = {
        "schema_version": "worldsim_v51_h_feature_pca_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "h_dinov2_feature_sidecars_and_seeded_pca_ready_without_quality_read",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "report": report,
        "resource": resource,
        "h_pixels_read_for_feature_extraction": True,
        "screening_pixels_read": False,
        "confirmation_pixels_read": False,
        "feature_sidecar_persisted": True,
        "pca_fit": True,
        "pca_fit_role": "historical_diagnostic_only",
        "membership_proxy_read": False,
        "renderer_started": False,
        "uplift_feature_computed": False,
        "method_quality_read": False,
        "screening_quality_read": False,
        "confirmation_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": ["V51-F14"],
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "fingerprint.json",
        {
            "schema_version": "worldsim_v51_h_feature_pca_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "official_source_commit": config["source"]["commit"],
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "input_manifest_sha256": config["input_freeze"]["image_manifest_sha256"],
            "selected_h_record_chain_sha256": report[
                "h_input_record_chain_sha256"
            ],
            "pca_state_sha256": report["pca_state_sha256"],
            "feature_record_chain_sha256": report[
                "feature_record_chain_sha256"
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
        default=PROJECT / "configs/worldsim_v51/stage_b_h_feature_pca_v1.yaml",
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
            "schema_version": "worldsim_v51_h_feature_pca_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_h_feature_pca_status_v1",
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
                "schema_version": "worldsim_v51_h_feature_pca_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
