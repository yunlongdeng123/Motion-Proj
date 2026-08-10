#!/usr/bin/env python3
"""用冻结 SAM2 检测删除路径中的 actor 语义回灌。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torchvision
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.semantic_gate import (  # noqa: E402
    semantic_mass,
    semantic_reintroduction_decision,
)
from motion_proj.worldsim_v33.spatial_delta import (  # noqa: E402
    atomic_json,
    atomic_save_npz,
    sha256_file,
)


def verify(path: str | Path, expected: str, role: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{role} 不存在: {target}")
    actual = sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"{role} SHA 漂移: expected={expected} actual={actual}")
    return {"path": str(target), "sha256": actual, "bytes": target.stat().st_size}


def git_output(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *arguments], text=True
    ).strip()


def read_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    return {
        key: int(value)
        for key, value in (
            line.split() for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


class ResourceSampler:
    def __init__(self) -> None:
        self.peak_nvidia_memory_mib = 0
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
            values = [int(row) for row in output.splitlines() if row.strip()]
            if values:
                self.peak_nvidia_memory_mib = max(
                    self.peak_nvidia_memory_mib, max(values)
                )
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(0.2):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        # Pillow 可能返回只读 view；SAM2/torchvision 会把它包装为 Tensor，显式
        # 拷贝可避免未定义写行为警告，同时保持输入字节完全不变。
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def infer_logits(
    predictor: Any, image: np.ndarray, box: list[int], output: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    predictor.set_image(image)
    logits, scores, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=False,
        return_logits=True,
    )
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"SAM2 image logits shape 非法: {logits.shape}")
    value = np.asarray(logits[0], dtype=np.float16)
    if value.shape != image.shape[:2] or not np.isfinite(value).all():
        raise RuntimeError("SAM2 logits shape/有限性漂移")
    atomic_save_npz(output, {"logits": value})
    return value.astype(np.float32), {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "score": float(scores[0]),
        "inference_seconds": time.perf_counter() - started,
        "dtype": "float16",
        "shape": list(value.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha", required=True)
    parser.add_argument("--harmonizer-manifest", type=Path, required=True)
    parser.add_argument("--harmonizer-manifest-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verify(args.input_manifest, args.input_manifest_sha, "S5 input manifest")
    verify(
        args.harmonizer_manifest,
        args.harmonizer_manifest_sha,
        "S5 Harmonizer manifest",
    )
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    harmonizer = json.loads(args.harmonizer_manifest.read_text(encoding="utf-8"))
    if inputs.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("S5 input manifest config SHA 漂移")
    if harmonizer.get("input_manifest_sha256") != args.input_manifest_sha:
        raise RuntimeError("S5 Harmonizer manifest 链接漂移")

    sam_cfg = config["sam2"]
    checkout = Path(sam_cfg["source_checkout"])
    if git_output(checkout, "rev-parse", "HEAD") != sam_cfg["source_commit"]:
        raise RuntimeError("SAM2 source commit 漂移")
    if git_output(checkout, "status", "--porcelain"):
        raise RuntimeError("SAM2 source checkout 非 clean")
    source = {
        "checkout": str(checkout),
        "commit": sam_cfg["source_commit"],
        "tree_sha": git_output(checkout, "rev-parse", "HEAD^{tree}"),
        "clean": True,
        "license": verify(
            sam_cfg["license_path"], sam_cfg["license_sha256"], "SAM2 license"
        ),
    }
    checkpoint_before = verify(
        sam_cfg["checkpoint"], sam_cfg["checkpoint_sha256"], "SAM2 checkpoint"
    )
    environment = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "conda_explicit": verify(
            sam_cfg["conda_explicit"],
            sam_cfg["conda_explicit_sha256"],
            "SAM2 conda explicit",
        ),
        "pip_freeze": verify(
            sam_cfg["pip_freeze"], sam_cfg["pip_freeze_sha256"], "SAM2 pip freeze"
        ),
    }
    for name in ("python", "torch", "torchvision", "numpy"):
        if environment[name] != str(sam_cfg["runtime"][name]):
            raise RuntimeError(
                f"SAM2 runtime {name} 漂移: {environment[name]} != {sam_cfg['runtime'][name]}"
            )
    immutable_before = {
        name: verify(spec["path"], spec["sha256"], name)
        for name, spec in config["inputs"].items()
        if name in {"checkpoint", "actor_registry"}
    }
    if not torch.cuda.is_available():
        raise RuntimeError("S5 SAM2 正式路径需要 CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated(device) > 512 * 1024**2:
        raise RuntimeError("S5 SAM2 GPU preflight 非空闲")
    torch.cuda.reset_peak_memory_stats(device)

    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from sam2.build_sam import build_sam2  # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

    memory_before = read_memory_events()
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        model = build_sam2(
            sam_cfg["model_config"],
            sam_cfg["checkpoint"],
            device=str(device),
        )
        predictor = SAM2ImagePredictor(model)
        input_by_view = {
            (int(row["frame"]), int(row["camera_id"])): row
            for row in inputs["records"]
        }
        rows = []
        thresholds = config["semantic_reintroduction"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for harmonizer_row in harmonizer["rows"]:
                frame = int(harmonizer_row["frame"])
                camera = int(harmonizer_row["camera_id"])
                view = (frame, camera)
                input_row = input_by_view[view]
                view_name = f"f{frame:03d}_c{camera}"
                mask_spec = input_row["target_mask"]
                verify(mask_spec["path"], mask_spec["sha256"], f"{view_name} mask")
                with Image.open(mask_spec["path"]) as mask_image:
                    reference_mask = np.asarray(mask_image.convert("L")) > 0
                output_specs = harmonizer_row["outputs"]
                for variant in (
                    "delete_raw",
                    "delete_unconstrained",
                    "delete_production",
                ):
                    verify(
                        output_specs[variant]["path"],
                        output_specs[variant]["sha256"],
                        f"{view_name} {variant}",
                    )
                if (
                    output_specs["delete_raw"]["sha256"]
                    != output_specs["delete_production"]["sha256"]
                ):
                    raise RuntimeError("delete production 非 raw 的精确副本")
                raw_image = load_rgb(output_specs["delete_raw"]["path"])
                candidate_image = load_rgb(
                    output_specs["delete_unconstrained"]["path"]
                )
                output_root = args.output_dir / view_name
                output_root.mkdir(parents=True)
                raw_logits, raw_record = infer_logits(
                    predictor,
                    raw_image,
                    input_row["sam_prompt_box_xyxy"],
                    output_root / "delete_raw_logits.npz",
                )
                candidate_logits, candidate_record = infer_logits(
                    predictor,
                    candidate_image,
                    input_row["sam_prompt_box_xyxy"],
                    output_root / "delete_unconstrained_logits.npz",
                )
                # production 图像与 raw 内容寻址相同，复用同一 logits，消除重复
                # 非确定性推理，使语义安全证明与像素级 fallback 合同一致。
                production_path = output_root / "delete_production_logits.npz"
                shutil.copy2(raw_record["path"], production_path)
                production_record = {
                    "path": str(production_path),
                    "sha256": sha256_file(production_path),
                    "bytes": production_path.stat().st_size,
                    "score": raw_record["score"],
                    "inference_seconds": 0.0,
                    "dtype": raw_record["dtype"],
                    "shape": raw_record["shape"],
                    "reused_from": "delete_raw",
                    "reason": "byte_identical_input",
                }
                if production_record["sha256"] != raw_record["sha256"]:
                    raise RuntimeError("SAM2 production logits 未精确复用 raw")
                masses = {
                    "raw": semantic_mass(raw_logits, reference_mask),
                    "unconstrained": semantic_mass(
                        candidate_logits, reference_mask
                    ),
                    "production": semantic_mass(raw_logits, reference_mask),
                }
                decision = semantic_reintroduction_decision(
                    raw=masses["raw"],
                    unconstrained=masses["unconstrained"],
                    production=masses["production"],
                    **{
                        name: thresholds[name]
                        for name in (
                            "minimum_candidate_mass_increase",
                            "minimum_candidate_positive_fraction_increase",
                            "maximum_production_mass_increase",
                            "maximum_production_positive_fraction_increase",
                        )
                    },
                )
                if not decision["production_safe"]:
                    raise RuntimeError(f"{view_name} delete production 语义不安全")
                rows.append(
                    {
                        "frame": frame,
                        "camera_id": camera,
                        "phase": harmonizer_row["phase"],
                        "prompt_box_xyxy": input_row["sam_prompt_box_xyxy"],
                        "reference_mask_pixels": int(reference_mask.sum()),
                        "logits": {
                            "raw": raw_record,
                            "unconstrained": candidate_record,
                            "production": production_record,
                        },
                        "semantic_mass": masses,
                        "semantic_reintroduction": decision,
                    }
                )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        torch_peak_mib = int(torch.cuda.max_memory_reserved(device) / (1024**2))
    finally:
        sampler.stop()

    memory_after = read_memory_events()
    checkpoint_after = verify(
        sam_cfg["checkpoint"],
        sam_cfg["checkpoint_sha256"],
        "SAM2 checkpoint after inference",
    )
    immutable_after = {
        name: verify(spec["path"], spec["sha256"], f"{name} after SAM2")
        for name, spec in config["inputs"].items()
        if name in {"checkpoint", "actor_registry"}
    }
    if immutable_before != immutable_after or checkpoint_before != checkpoint_after:
        raise RuntimeError("S5 SAM2 改写了 immutable asset")
    resource = {
        "wall_seconds": wall_seconds,
        "peak_torch_reserved_mib": torch_peak_mib,
        "peak_nvidia_memory_mib": sampler.peak_nvidia_memory_mib,
        "memory_events_before": memory_before,
        "memory_events_after": memory_after,
        "oom_events_delta": memory_after.get("oom", 0) - memory_before.get("oom", 0),
        "oom_kill_events_delta": memory_after.get("oom_kill", 0)
        - memory_before.get("oom_kill", 0),
    }
    manifest = {
        "schema_version": "worldsim_v33_s5_sam2_detector_manifest_v1",
        "task_id": config["task_id"],
        "config_sha256": sha256_file(args.config),
        "input_manifest_sha256": args.input_manifest_sha,
        "harmonizer_manifest_sha256": args.harmonizer_manifest_sha,
        "source": source,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "environment": environment,
        "immutable_before": immutable_before,
        "immutable_after": immutable_after,
        "rows": rows,
        "resource": resource,
        "model_training_performed": False,
        "checkpoint_written": False,
        "production_inference_reuse_policy": "byte_identical_content_addressed",
    }
    manifest_path = args.output_dir / "sam2_detector_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "done",
                "manifest_sha256": sha256_file(manifest_path),
                "views": len(rows),
                "candidate_flagged_views": sum(
                    int(row["semantic_reintroduction"]["unconstrained_candidate_flagged"])
                    for row in rows
                ),
                "wall_seconds": wall_seconds,
                "peak_nvidia_memory_mib": sampler.peak_nvidia_memory_mib,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
