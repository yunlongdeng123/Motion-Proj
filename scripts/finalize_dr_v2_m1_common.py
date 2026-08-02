#!/usr/bin/env python3
"""复用冻结 AD-GS evaluator 完成 M1 common-observation 诊断并封存主 run。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT = Path("/root/autodl-tmp/motion_proj")
LEGACY_PATH = PROJECT / "scripts/finalize_dr_m5_common_diagnostic.py"
TASK_ID = "DR-V2-M1-DGGT-REPAIR-01"
COMPONENT = "common-observation-diagnostic-v2"
LPIPS_ALEXNET = Path(
    "/root/autodl-tmp/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
)


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def load_legacy():
    spec = importlib.util.spec_from_file_location("dr_v2_common_core", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载冻结 common diagnostic core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TASK_ID = TASK_ID
    module.COMPONENT = COMPONENT
    return module


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"缺少必需文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def resource_sample(phase: str) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "timestamp": now(),
        "phase": phase,
        "disk_free_bytes": shutil.disk_usage("/root/autodl-tmp").free,
    }
    cgroup = Path("/sys/fs/cgroup")
    for name in ("memory.current", "memory.max", "memory.events"):
        path = cgroup / name
        if path.is_file():
            sample[name.replace(".", "_")] = path.read_text().strip()
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        sample["gpu_memory_used_total_utilization"] = gpu
    except (OSError, subprocess.CalledProcessError):
        sample["gpu_memory_used_total_utilization"] = None
    return sample


def validate_native_source(native_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证可复用的原生结果，不依赖后续 common 阶段写入的主 terminal。"""

    native_summary = load_json(native_run / "native_summary.json")
    if native_summary.get("status") != "native_done":
        raise RuntimeError("native inference 尚未完成")
    native_metrics = load_json(native_run / "metrics.json")
    for key in ("native_1view_rows", "native_3view_rows"):
        rows = native_metrics.get(key, [])
        if len(rows) != 18 or len({row.get("pseudo_scene") for row in rows}) != 18:
            raise RuntimeError(f"native 覆盖不完整: {key}={len(rows)}/18")
        for row in rows:
            stage_path = native_run / "stages" / f"{row['stage']}.json"
            stage = load_json(stage_path)
            if stage.get("status") != "done":
                raise RuntimeError(f"native 阶段未完成: {stage_path}")
    return native_summary, native_metrics


def initialize_common(
    core, common_dir: Path, native_run: Path, result_run: Path
) -> dict[str, Any]:
    if result_run.exists() and any(result_run.iterdir()):
        raise RuntimeError(f"结果 run 目录非空，禁止覆盖: {result_run}")
    native_summary, native_metrics = validate_native_source(native_run)
    m4_terminal = core.load_json(core.M4_AGGREGATE / "terminal.json")
    m4_summary = core.load_json(core.M4_AGGREGATE / "summary.json")
    if m4_terminal.get("status") != "done" or not m4_summary.get(
        "all_gates_passed"
    ):
        raise RuntimeError("冻结 AD-GS M4 aggregate 未通过")
    evaluator = core.verify_frozen_evaluator()

    result_run.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (result_run / name).mkdir()
    common_dir.mkdir()
    snapshot_dir = common_dir / "source_snapshot"
    snapshot_dir.mkdir()
    for source in (Path(__file__).resolve(), LEGACY_PATH):
        destination = snapshot_dir / source.name
        shutil.copy2(source, destination)
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": result_run.name,
        "native_run": str(native_run),
        "m4_aggregate_run": str(core.M4_AGGREGATE),
        "scenes": core.SCENES,
        "raw_windows": core.WINDOWS,
        "processed_frame_rule": "raw_frame - 10",
        "camera_count": core.CAMERA_COUNT,
        "expected_target_pairs": len(core.SCENES)
        * len(core.WINDOWS)
        * 4
        * core.CAMERA_COUNT,
        "metric_source": "frozen 8-bit AD-GS render/gt PNG pairs",
        "metrics": ["PSNR", "SSIM", "LPIPS(ALEX)"],
        "metric_lpips_backbone": "Alex",
        "adgs_commit": core.ADGS_COMMIT,
        "frozen_evaluator": evaluator,
        "protocol_boundary": {
            "dggt": {
                "observations_per_window": "4 for 1-view; 12 for 3-view",
                "poses_as_input": False,
                "per_scene_optimization": False,
                "model_input_hw": [294, 518],
            },
            "adgs": {
                "training_observations_per_scene": 138,
                "held_out_targets_per_scene": 42,
                "poses_as_input": True,
                "per_scene_optimization": True,
                "iterations": 60000,
                "render_hw": [900, 1600],
            },
            "claim": "failure characterization only; not a matched leaderboard",
        },
    }
    resolved["config_fingerprint"] = core.canonical_sha256(resolved)
    core.atomic_json(common_dir / "resolved.yaml", resolved)
    project_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=PROJECT, text=True
    )
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": result_run.name,
        "started_at": core.now(),
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "project_git_status": project_status.splitlines(),
        "project_git_status_sha256": hashlib.sha256(
            project_status.encode()
        ).hexdigest(),
        "config_fingerprint": resolved["config_fingerprint"],
        "native_summary_sha256": sha256_file(native_run / "native_summary.json"),
        "native_metrics_sha256": sha256_file(native_run / "metrics.json"),
        "m4_summary_sha256": sha256_file(core.M4_AGGREGATE / "summary.json"),
        "frozen_evaluator": evaluator,
        "native_status_at_retry": load_json(native_run / "terminal.json"),
    }
    core.atomic_json(common_dir / "manifest.json", manifest)
    core.atomic_json(
        common_dir / "terminal.json",
        {"status": "running", "updated_at": core.now(), "failure": None},
    )
    for source in (Path(__file__).resolve(), LEGACY_PATH):
        shutil.copy2(source, result_run / "source_snapshot" / source.name)
    pip_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    (result_run / "environment" / "pip-freeze.txt").write_text(
        pip_freeze, encoding="utf-8"
    )
    result_manifest = {
        **manifest,
        "component": "common-observation-retry-v2",
        "status": "running",
        "source_native_run": str(native_run),
        "source_native_terminal_sha256": sha256_file(native_run / "terminal.json"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "flow_vis": next(
                (line for line in pip_freeze.splitlines() if line.lower().startswith("flow-vis==")),
                None,
            ),
        },
    }
    write_json(result_run / "manifest.json", result_manifest)
    write_json(result_run / "resolved.yaml", resolved)
    write_json(
        result_run / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    append_jsonl(result_run / "resource.jsonl", resource_sample("start"))
    append_jsonl(
        result_run / "logs" / "finalizer.jsonl",
        {"timestamp": now(), "event": "initialized", "native_run": str(native_run)},
    )
    write_json(
        result_run / "stages" / "validate_native_source.json",
        {
            "status": "done",
            "native_status": native_summary["status"],
            "native_1view_count": len(native_metrics["native_1view_rows"]),
            "native_3view_count": len(native_metrics["native_3view_rows"]),
            "native_summary_sha256": sha256_file(native_run / "native_summary.json"),
            "native_metrics_sha256": sha256_file(native_run / "metrics.json"),
        },
    )
    return resolved


def index_result_artifacts(result_run: Path, native_run: Path) -> None:
    names = [
        "manifest.json",
        "resolved.yaml",
        "metrics.jsonl",
        "summary.json",
        "summary.md",
        "terminal.json",
        "resource.jsonl",
        "logs/finalizer.jsonl",
        "source_snapshot/finalize_dr_v2_m1_common.py",
        "source_snapshot/finalize_dr_m5_common_diagnostic.py",
        "environment/pip-freeze.txt",
        "stages/validate_native_source.json",
        "common_observation/manifest.json",
        "common_observation/resolved.yaml",
        "common_observation/mapping_audit.json",
        "common_observation/comparison.json",
        "common_observation/metrics.jsonl",
        "common_observation/summary.json",
        "common_observation/terminal.json",
    ]
    artifacts = []
    for name in names:
        path = result_run / name
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in sorted((native_run / "outputs").rglob("*")):
        if path.is_file() and not path.name.endswith(".partial"):
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(result_run / "artifacts.json", {"artifacts": artifacts})


def finalize(native_run: Path, result_run: Path) -> dict[str, Any]:
    if native_run == result_run:
        raise RuntimeError("result-run 必须与 native-run 分离")
    core = load_legacy()
    common_dir = result_run / "common_observation"
    resolved = initialize_common(core, common_dir, native_run, result_run)
    try:
        common_summary = core.evaluate(common_dir, native_run, resolved)
    except Exception as exc:
        core.atomic_json(
            common_dir / "summary.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "component": COMPONENT,
                "status": "blocked",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        core.atomic_json(
            common_dir / "terminal.json",
            {
                "status": "blocked",
                "updated_at": core.now(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        core.write_artifacts(common_dir)
        write_json(
            result_run / "terminal.json",
            {
                "status": "blocked",
                "updated_at": now(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    core.atomic_json(
        common_dir / "terminal.json",
        {"status": "done", "updated_at": core.now(), "failure": None},
    )
    core.write_artifacts(common_dir)

    native_summary = load_json(native_run / "native_summary.json")
    if not LPIPS_ALEXNET.is_file() or LPIPS_ALEXNET.stat().st_size <= 0:
        raise RuntimeError(f"DGGT LPIPS AlexNet 权重缺失: {LPIPS_ALEXNET}")
    final = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": result_run.name,
        "status": "done",
        "completed_at": core.now(),
        "pointops2": "upstream setup.py install + CUDA forward/backward PASS",
        "checkpoint": {
            "path": "/root/autodl-tmp/checkpoints/dggt-v2/model_latest_nuscenes.pt",
            "sha256": load_json(native_run / "resolved.yaml")["checkpoint_sha256"],
            "storage": "same-filesystem hardlink to verified preload",
        },
        "untouched_failure": native_summary["untouched_failure"],
        "compatibility_patch": native_summary["compatibility_patch"],
        "lpips_alexnet": {
            "source": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path": str(LPIPS_ALEXNET),
            "bytes": LPIPS_ALEXNET.stat().st_size,
            "sha256": sha256_file(LPIPS_ALEXNET),
        },
        "native_1view": native_summary["native_1view"],
        "native_3view": native_summary["native_3view"],
        "common_observation": common_summary,
        "claim": "DGGT 与 AD-GS 输入、pose、分辨率和优化预算不同；仅作 failure characterization",
        "next": "DR-V2-M2-ACTOR-EVAL-01",
        "source_native_run": str(native_run),
    }
    write_json(result_run / "summary.json", final)
    (result_run / "summary.md").write_text(
        "# DR-V2 M1 summary\n\n"
        f"- status: `done`\n"
        f"- 1-view coverage: `{final['native_1view']['count']}/18`\n"
        f"- 3-view status: `{final['native_3view']['status']}`\n"
        f"- untouched failure matched: `{final['untouched_failure']['expected_failure_matched']}`\n"
        f"- common target coverage: `{common_summary['target_mapping_count']}/{common_summary['expected_target_mapping_count']}`\n"
        "- next: `DR-V2-M2-ACTOR-EVAL-01`\n",
        encoding="utf-8",
    )
    write_json(
        result_run / "terminal.json",
        {"status": "done", "updated_at": core.now(), "failure": None},
    )
    manifest = load_json(result_run / "manifest.json")
    manifest["status"] = "done"
    manifest["completed_at"] = core.now()
    write_json(result_run / "manifest.json", manifest)
    shutil.copy2(common_dir / "metrics.jsonl", result_run / "metrics.jsonl")
    append_jsonl(result_run / "resource.jsonl", resource_sample("done"))
    append_jsonl(
        result_run / "logs" / "finalizer.jsonl",
        {"timestamp": now(), "event": "done", "metric_rows": 216},
    )
    index_result_artifacts(result_run, native_run)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--result-run", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = finalize(args.native_run.resolve(), args.result_run.resolve())
    except Exception as exc:
        if args.result_run.exists():
            append_jsonl(
                args.result_run / "resource.jsonl", resource_sample("blocked")
            )
            append_jsonl(
                args.result_run / "logs" / "finalizer.jsonl",
                {
                    "timestamp": now(),
                    "event": "blocked",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            common_terminal = args.result_run / "common_observation" / "terminal.json"
            if common_terminal.is_file():
                common_state = load_json(common_terminal)
                if common_state.get("status") == "running":
                    write_json(
                        common_terminal,
                        {
                            "status": "blocked",
                            "updated_at": now(),
                            "failure": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    )
            write_json(
                args.result_run / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
