#!/usr/bin/env python3
"""运行 S5 Harmonizer，并只把受限 residual 应用到插入路径。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.harmonizer_adapter import (  # noqa: E402
    HarmonizerJITAdapter,
    validate_rmsnorm_operator,
)
from motion_proj.worldsim_v33.semantic_gate import (  # noqa: E402
    apply_gated_residual,
    evaluate_semantic_gate,
    validate_semantic_gate,
)
from motion_proj.worldsim_v33.spatial_delta import (  # noqa: E402
    atomic_json,
    load_npz,
    sha256_file,
)


def verify_file(path: str | Path, expected: str, role: str) -> dict[str, Any]:
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


def verify_clean_checkout(
    checkout: str | Path, *, commit: str, tree: str | None = None
) -> dict[str, Any]:
    root = Path(checkout)
    if git_output(root, "rev-parse", "HEAD") != commit:
        raise RuntimeError(f"第三方 checkout commit 漂移: {root}")
    actual_tree = git_output(root, "rev-parse", "HEAD^{tree}")
    if tree is not None and actual_tree != tree:
        raise RuntimeError(f"第三方 checkout tree 漂移: {root}")
    status = git_output(root, "status", "--porcelain")
    if status:
        raise RuntimeError(f"第三方 checkout 非 clean: {root}: {status[:200]}")
    return {
        "checkout": str(root),
        "commit": commit,
        "tree_sha": actual_tree,
        "clean": True,
    }


def read_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, value = line.split()
        values[name] = int(value)
    return values


class ResourceSampler:
    """采样整卡 NVIDIA 显存，避免只报告 torch allocator。"""

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
            values = [int(row.strip()) for row in output.splitlines() if row.strip()]
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


def save_rgb(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def verify_input_manifest(
    config_path: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    expected_sha: str,
) -> dict[str, Any]:
    verify_file(manifest_path, expected_sha, "S5 input manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "worldsim_v33_s5_input_manifest_v1":
        raise RuntimeError("S5 input manifest schema 漂移")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("S5 input manifest 与 config 不一致")
    if manifest.get("heldout_read_for_selection") is not False:
        raise RuntimeError("S5 input manifest 未冻结 heldout selection 边界")
    expected_views = 1 + len(config["views"]["development"]) + len(
        config["views"]["heldout_confirmation"]
    )
    if len(manifest.get("records", [])) != expected_views:
        raise RuntimeError("S5 input manifest view 数漂移")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s5_semantic_gate_v1":
        raise RuntimeError("S5 config schema 漂移")
    input_manifest = verify_input_manifest(
        args.config, config, args.input_manifest, args.input_manifest_sha
    )

    harmonizer_cfg = config["harmonizer"]
    harmonizer_source = verify_clean_checkout(
        harmonizer_cfg["source_checkout"], commit=harmonizer_cfg["source_commit"]
    )
    harmonizer_source["license"] = verify_file(
        harmonizer_cfg["license_path"],
        harmonizer_cfg["license_sha256"],
        "Harmonizer license",
    )
    model_before = verify_file(
        harmonizer_cfg["model_path"],
        harmonizer_cfg["model_sha256"],
        "Harmonizer exported model",
    )
    if model_before["bytes"] != int(harmonizer_cfg["model_bytes"]):
        raise RuntimeError("Harmonizer exported model 字节数漂移")
    model_card = verify_file(
        harmonizer_cfg["model_card"],
        harmonizer_cfg["model_card_sha256"],
        "Harmonizer model card",
    )

    r3d2_cfg = config["r3d2"]
    r3d2_source = verify_clean_checkout(
        r3d2_cfg["checkout"],
        commit=r3d2_cfg["commit"],
        tree=r3d2_cfg["tree_sha"],
    )
    r3d2_source["license"] = verify_file(
        r3d2_cfg["license_path"], r3d2_cfg["license_sha256"], "R3D2 license"
    )
    if r3d2_cfg.get("exported_author_model") is not None:
        raise RuntimeError("R3D2 contract 要求显式冻结 author model 缺失")
    if not r3d2_cfg.get("from_scratch_training_forbidden"):
        raise RuntimeError("R3D2 from-scratch 禁令缺失")
    r3d2_audit = {
        **r3d2_source,
        "disposition": "blocked_pretrained_model_unavailable",
        "model_loaded": False,
        "training_performed": False,
    }

    immutable_before = {
        name: verify_file(spec["path"], spec["sha256"], name)
        for name, spec in config["inputs"].items()
        if name in {"checkpoint", "actor_registry"}
    }
    if not torch.cuda.is_available():
        raise RuntimeError("S5 Harmonizer 正式路径需要 CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated(device) > 512 * 1024**2:
        raise RuntimeError("S5 Harmonizer GPU preflight 非空闲")
    torch.cuda.reset_peak_memory_stats(device)

    memory_before = read_memory_events()
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        rmsnorm = validate_rmsnorm_operator(str(device))
        if not rmsnorm["exact"]:
            raise RuntimeError(f"Harmonizer RMSNorm fallback 不等价: {rmsnorm}")
        adapter = HarmonizerJITAdapter(harmonizer_cfg["model_path"], str(device))
        rows = []
        for record in input_manifest["records"]:
            frame = int(record["frame"])
            camera = int(record["camera_id"])
            view_name = f"f{frame:03d}_c{camera}"
            for stack, spec in record["images"].items():
                verify_file(spec["path"], spec["sha256"], f"{view_name} {stack}")
            verify_file(
                record["reference_rgb"]["path"],
                record["reference_rgb"]["sha256"],
                f"{view_name} reference",
            )
            verify_file(
                record["semantic_gate"]["path"],
                record["semantic_gate"]["sha256"],
                f"{view_name} semantic gate",
            )
            regions = load_npz(record["semantic_gate"]["path"])
            validate_semantic_gate(regions)
            full_raw = load_rgb(record["images"]["full"]["path"])
            delete_raw = load_rgb(record["images"]["erase_background"]["path"])
            reference = load_rgb(record["reference_rgb"]["path"])

            full_image, full_runtime = adapter.infer(Image.fromarray(full_raw))
            delete_image, delete_runtime = adapter.infer(Image.fromarray(delete_raw))
            full_unconstrained = np.asarray(full_image, dtype=np.uint8)
            delete_unconstrained = np.asarray(delete_image, dtype=np.uint8)
            full_gated, blend_audit = apply_gated_residual(
                full_raw,
                full_unconstrained,
                np.asarray(regions["gate"], dtype=np.float32),
                residual_cap_uint8=float(
                    config["semantic_gate"]["residual_cap_uint8"]
                ),
            )
            # 删除路径不允许 2D 网络写回；生产产物是逐像素相同的 3D 输出。
            delete_production = delete_raw.copy()
            delete_exact = bool(np.array_equal(delete_raw, delete_production))
            if not delete_exact:
                raise RuntimeError("delete production 未精确回退 raw 3D render")

            output_root = args.output_dir / view_name
            outputs = {
                "full_raw": save_rgb(output_root / "full_raw.png", full_raw),
                "full_unconstrained": save_rgb(
                    output_root / "full_unconstrained.png", full_unconstrained
                ),
                "full_gated": save_rgb(output_root / "full_gated.png", full_gated),
                "delete_raw": save_rgb(output_root / "delete_raw.png", delete_raw),
                "delete_unconstrained": save_rgb(
                    output_root / "delete_unconstrained.png", delete_unconstrained
                ),
                "delete_production": save_rgb(
                    output_root / "delete_production.png", delete_production
                ),
            }
            if outputs["delete_raw"]["sha256"] != outputs["delete_production"]["sha256"]:
                raise RuntimeError("delete raw/production 文件 SHA 不一致")
            metrics = evaluate_semantic_gate(
                raw=full_raw,
                gated=full_gated,
                reference=reference,
                regions=regions,
            )
            rows.append(
                {
                    "frame": frame,
                    "camera_id": camera,
                    "phase": record["phase"],
                    "input_gate_sha256": record["semantic_gate"]["sha256"],
                    "outputs": outputs,
                    "metrics": metrics,
                    "blend_audit": blend_audit,
                    "delete_raw_production_exact": delete_exact,
                    "runtime": {
                        "full": full_runtime,
                        "delete": delete_runtime,
                    },
                }
            )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        torch_peak_mib = int(torch.cuda.max_memory_reserved(device) / (1024**2))
    finally:
        sampler.stop()

    memory_after = read_memory_events()
    model_after = verify_file(
        harmonizer_cfg["model_path"],
        harmonizer_cfg["model_sha256"],
        "Harmonizer exported model after inference",
    )
    immutable_after = {
        name: verify_file(spec["path"], spec["sha256"], f"{name} after inference")
        for name, spec in config["inputs"].items()
        if name in {"checkpoint", "actor_registry"}
    }
    if immutable_before != immutable_after:
        raise RuntimeError("S5 Harmonizer 改写了 3D immutable inputs")
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
        "schema_version": "worldsim_v33_s5_harmonizer_manifest_v1",
        "task_id": config["task_id"],
        "config_sha256": sha256_file(args.config),
        "input_manifest_sha256": args.input_manifest_sha,
        "provenance": harmonizer_cfg["provenance"],
        "harmonizer": {
            "source": harmonizer_source,
            "model_before": model_before,
            "model_after": model_after,
            "model_card": model_card,
            "load_audit": adapter.load_audit.__dict__,
            "rmsnorm": rmsnorm,
        },
        "r3d2": r3d2_audit,
        "immutable_before": immutable_before,
        "immutable_after": immutable_after,
        "rows": rows,
        "resource": resource,
        "training_performed": False,
        "checkpoint_written": False,
        "delete_policy": "raw_3d_render_only",
    }
    manifest_path = args.output_dir / "harmonizer_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "done",
                "manifest_sha256": sha256_file(manifest_path),
                "views": len(rows),
                "wall_seconds": wall_seconds,
                "peak_nvidia_memory_mib": sampler.peak_nvidia_memory_mib,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
