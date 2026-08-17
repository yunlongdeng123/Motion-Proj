#!/usr/bin/env python3
"""在单张冻结图像上实测 DINOv2 ViT-g/14 registers 资源与输出合同。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

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
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


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
        raise ProtocolError(f"预期单 GPU，实际 nvidia-smi rows={len(values)}")
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
                sample = {
                    "at_utc": _utc_now(),
                    "monotonic_seconds": time.monotonic(),
                    "gpu_used_mib": _nvidia_used_mib(),
                    "cgroup_memory_current_bytes": int(
                        Path("/sys/fs/cgroup/memory.current")
                        .read_text(encoding="utf-8")
                        .strip()
                    ),
                }
                self.samples.append(sample)
            except Exception as error:  # 监控失败也必须留在正式证据中。
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
        "worldsim_v51_stage_b_dinov2_resource_smoke_v1"
    ):
        raise ProtocolError("DINOv2 resource smoke schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("DINOv2 resource smoke task 漂移")
    if config.get("status") != "running":
        raise ProtocolError("DINOv2 resource smoke status 漂移")

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

    freeze_spec = config["asset_freeze"]
    freeze_path = PROJECT / freeze_spec["path"]
    if not freeze_path.is_file() or sha256_file(freeze_path) != freeze_spec["sha256"]:
        raise ProtocolError("DINOv2 asset freeze binding 漂移")
    asset_freeze = load_yaml(freeze_path)
    checkpoint = Path(freeze_spec["checkpoint_path"])
    if asset_freeze.get("status") != "done" or not checkpoint.is_file():
        raise ProtocolError("DINOv2 frozen checkpoint 缺失")
    if checkpoint.stat().st_size != int(freeze_spec["checkpoint_bytes"]):
        raise ProtocolError("DINOv2 checkpoint bytes 漂移")
    if sha256_file(checkpoint) != freeze_spec["checkpoint_sha256"]:
        raise ProtocolError("DINOv2 checkpoint SHA 漂移")

    input_spec = config["input"]
    image = Path(input_spec["path"])
    if not image.is_file() or image.stat().st_size != int(input_spec["bytes"]):
        raise ProtocolError("DINOv2 smoke image bytes 漂移")
    if sha256_file(image) != input_spec["sha256"]:
        raise ProtocolError("DINOv2 smoke image SHA 漂移")
    with Image.open(image) as opened:
        observed_size = list(opened.size)
        opened.verify()
    if observed_size != list(input_spec["original_size_wh"]):
        raise ProtocolError("DINOv2 smoke image size 漂移")

    expected_input_contract = {
        "scene": "scene-0471",
        "scene_index": 382,
        "frame": 0,
        "camera": 0,
        "original_size_wh": [1600, 900],
        "model_size_wh": [1596, 896],
        "resize_mode": "bilinear",
        "resize_align_corners": False,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }
    for name, expected in expected_input_contract.items():
        if input_spec.get(name) != expected:
            raise ProtocolError(f"DINOv2 smoke input contract 漂移: {name}")

    model = config["model"]
    expected_model_contract = {
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
        "expected_output_shapes": [[1, 1536, 64, 114]] * 4,
        "strict_state_dict_load": True,
        "weights_only_load": True,
        "checkpoint_mmap": False,
    }
    for name, expected in expected_model_contract.items():
        if model.get(name) != expected:
            raise ProtocolError(f"DINOv2 smoke model contract 漂移: {name}")

    resources = config["resources"]
    expected_resource_contract = {
        "maximum_gpu_used_at_start_mib": 2048,
        "maximum_gpu_peak_mib": 22528,
        "maximum_cgroup_peak_bytes": 85899345920,
        "sample_interval_seconds": 0.5,
        "timeout_seconds": 900,
        "single_dino_process_only": True,
        "renderer_concurrent": False,
        "no_smaller_model_or_resolution_fallback": True,
    }
    for name, expected in expected_resource_contract.items():
        if resources.get(name) != expected:
            raise ProtocolError(f"DINOv2 smoke resource contract 漂移: {name}")

    locks = config["locks"]
    for name in (
        "pca_fit",
        "feature_sidecar_persist",
        "renderer_start",
        "method_inference",
        "quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"DINOv2 resource smoke lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, asset_freeze


def _preprocess(config: dict[str, Any]) -> torch.Tensor:
    spec = config["input"]
    with Image.open(spec["path"]) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    target_width, target_height = [int(value) for value in spec["model_size_wh"]]
    tensor = functional.interpolate(
        tensor,
        size=(target_height, target_width),
        mode=spec["resize_mode"],
        align_corners=bool(spec["resize_align_corners"]),
    )
    mean = torch.tensor(spec["normalization_mean"], dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std = torch.tensor(spec["normalization_std"], dtype=torch.float32).view(
        1, 3, 1, 1
    )
    return (tensor - mean) / std


def execute_smoke(
    config: dict[str, Any], monitor: ResourceMonitor
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phases = []

    def phase(name: str, started: float) -> None:
        phases.append({"phase": name, "seconds": time.monotonic() - started})

    source_path = Path(config["source"]["path"])
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    import_started = time.monotonic()
    backbones = importlib.import_module("dinov2.hub.backbones")
    xformers = importlib.import_module("xformers")
    phase("import_official_source", import_started)

    if torch.__version__ != config["environment"]["torch"]:
        raise ProtocolError("torch version 漂移")
    if torch.version.cuda != config["environment"]["torch_cuda"]:
        raise ProtocolError("torch CUDA version 漂移")
    if xformers.__version__ != config["environment"]["xformers"]:
        raise ProtocolError("xformers version 漂移")

    preprocess_started = time.monotonic()
    input_cpu = _preprocess(config)
    phase("preprocess_one_image", preprocess_started)
    if list(input_cpu.shape) != [1, 3, 896, 1596]:
        raise ProtocolError(f"DINOv2 model input shape 漂移: {list(input_cpu.shape)}")

    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    constructor_started = time.monotonic()
    constructor = getattr(backbones, config["model"]["entrypoint"])
    model = constructor(pretrained=False)
    phase("construct_vitg14_reg_cpu", constructor_started)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    load_started = time.monotonic()
    state_dict = torch.load(
        config["asset_freeze"]["checkpoint_path"],
        map_location="cpu",
        weights_only=bool(config["model"]["weights_only_load"]),
        mmap=bool(config["model"]["checkpoint_mmap"]),
    )
    state_dict_key_count = len(state_dict)
    load_result = model.load_state_dict(
        state_dict, strict=bool(config["model"]["strict_state_dict_load"])
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ProtocolError("DINOv2 strict state_dict 出现 missing/unexpected keys")
    del state_dict
    phase("load_checkpoint_strict_cpu", load_started)

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_started = time.monotonic()
    model = model.cuda()
    input_cuda = input_cpu.cuda(non_blocking=False)
    torch.cuda.synchronize()
    phase("move_model_and_input_to_cuda", cuda_started)

    inference_started = time.monotonic()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        outputs = model.get_intermediate_layers(
            input_cuda,
            n=int(config["model"]["intermediate_layers"]),
            reshape=bool(config["model"]["reshape"]),
            norm=bool(config["model"]["norm"]),
        )
    torch.cuda.synchronize()
    phase("get_last_four_intermediate_layers", inference_started)

    shapes = [list(output.shape) for output in outputs]
    if shapes != config["model"]["expected_output_shapes"]:
        raise ProtocolError(f"DINOv2 intermediate output shape 漂移: {shapes}")
    selected = outputs[-1].float()
    finite = bool(torch.isfinite(selected).all().item())
    if not finite:
        raise ProtocolError("DINOv2 selected feature 非 finite")
    flattened = selected.flatten()
    diagnostics = {
        "output_shapes": shapes,
        "output_dtypes": [str(output.dtype) for output in outputs],
        "selected_finite": finite,
        "selected_mean": float(selected.mean().item()),
        "selected_std_correction_1": float(selected.std(correction=1).item()),
        "selected_min": float(selected.min().item()),
        "selected_max": float(selected.max().item()),
        "selected_first16": [float(value) for value in flattened[:16].cpu().tolist()],
        "parameter_count": parameter_count,
        "state_dict_key_count": state_dict_key_count,
        "strict_missing_key_count": len(load_result.missing_keys),
        "strict_unexpected_key_count": len(load_result.unexpected_keys),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }

    del outputs, selected, flattened, input_cuda, input_cpu, model
    cleanup_started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    phase("cuda_cleanup", cleanup_started)
    diagnostics["torch_allocated_after_cleanup_mib"] = (
        torch.cuda.memory_allocated() / (1024**2)
    )
    diagnostics["torch_reserved_after_cleanup_mib"] = (
        torch.cuda.memory_reserved() / (1024**2)
    )
    return diagnostics, phases


def run(
    config_path: Path, run_dir: Path, events: list[dict[str, Any]]
) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    status = _git(PROJECT, "status", "--short")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行，当前为 {branch}")
    if status:
        raise ProtocolError("DINOv2 resource smoke 要求 clean project worktree")
    config, asset_freeze = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {"resource_smoke": config, "asset_freeze": asset_freeze},
            allow_unicode=True,
            sort_keys=False,
        ),
    )

    gpu_name, gpu_total = _gpu_name_and_total()
    gpu_start = _nvidia_used_mib()
    cgroup_max = int(
        Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
    )
    if gpu_name != config["environment"]["gpu"]:
        raise ProtocolError("resource smoke GPU model 漂移")
    if gpu_total != int(config["environment"]["gpu_total_mib"]):
        raise ProtocolError("resource smoke GPU total 漂移")
    if gpu_start > int(config["resources"]["maximum_gpu_used_at_start_mib"]):
        raise ProtocolError("resource smoke GPU start 非空闲")
    if cgroup_max != int(config["environment"]["cgroup_memory_max_bytes"]):
        raise ProtocolError("resource smoke cgroup max 漂移")

    monitor = ResourceMonitor(float(config["resources"]["sample_interval_seconds"]))
    monitor.start()
    events.append({"event": "model_smoke_started", "at_utc": _utc_now()})
    _write_jsonl(run_dir / "events.jsonl", events)
    timeout_seconds = int(config["resources"]["timeout_seconds"])

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"DINOv2 resource smoke 超过 {timeout_seconds} s")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        diagnostics, phases = execute_smoke(config, monitor)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        monitor.stop()
        _write_jsonl(run_dir / "resource_samples.jsonl", monitor.samples)

    valid_samples = [sample for sample in monitor.samples if "gpu_used_mib" in sample]
    if not valid_samples:
        raise ProtocolError("resource monitor 没有有效 sample")
    nvidia_peak = max(int(sample["gpu_used_mib"]) for sample in valid_samples)
    cgroup_peak = max(
        int(sample["cgroup_memory_current_bytes"]) for sample in valid_samples
    )
    if nvidia_peak > int(config["resources"]["maximum_gpu_peak_mib"]):
        raise ProtocolError("DINOv2 smoke GPU peak 超限")
    if cgroup_peak > int(config["resources"]["maximum_cgroup_peak_bytes"]):
        raise ProtocolError("DINOv2 smoke cgroup peak 超限")
    if diagnostics["torch_peak_reserved_mib"] > int(
        config["resources"]["maximum_gpu_peak_mib"]
    ):
        raise ProtocolError("DINOv2 smoke torch reserved peak 超限")

    resource = {
        "gpu_name": gpu_name,
        "gpu_total_mib": gpu_total,
        "gpu_used_at_start_mib": gpu_start,
        "nvidia_smi_peak_used_mib": nvidia_peak,
        "cgroup_memory_max_bytes": cgroup_max,
        "cgroup_memory_peak_bytes": cgroup_peak,
        "sample_count": len(valid_samples),
        "monitor_error_count": len(monitor.samples) - len(valid_samples),
        **diagnostics,
    }
    _write_json(run_dir / "artifacts/diagnostics.json", diagnostics)
    _write_json(run_dir / "artifacts/resources.json", resource)
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"metric": "phase_duration", **phase} for phase in phases
        ]
        + [{"metric": "resource_terminal", **resource}],
    )
    summary = {
        "schema_version": "worldsim_v51_dinov2_resource_smoke_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "official_dinov2_vitg14_reg4_one_image_resource_and_shape_gate_passed",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "official_source_commit": config["source"]["commit"],
        "official_source_tree": config["source"]["tree"],
        "checkpoint_sha256": config["asset_freeze"]["checkpoint_sha256"],
        "input_sha256": config["input"]["sha256"],
        "resource": resource,
        "phases": phases,
        "pca_fit": False,
        "feature_sidecar_persisted": False,
        "renderer_started": False,
        "method_inference_started": False,
        "quality_read": False,
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
            "schema_version": "worldsim_v51_dinov2_resource_smoke_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "official_source_commit": summary["official_source_commit"],
            "official_source_tree": summary["official_source_tree"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "input_sha256": summary["input_sha256"],
            "environment": config["environment"],
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT
        / "configs/worldsim_v51/stage_b_dinov2_resource_smoke_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    try:
        summary = run(args.config.resolve(), run_dir, events)
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_dinov2_resource_smoke_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_dinov2_resource_smoke_status_v1",
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
                "schema_version": "worldsim_v51_dinov2_resource_smoke_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
