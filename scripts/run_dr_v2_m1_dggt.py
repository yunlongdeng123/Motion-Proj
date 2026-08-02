#!/usr/bin/env python3
"""运行 DR-V2 M1 的 DGGT untouched/patch/1-view/3-view 协议。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT = Path("/root/autodl-tmp/motion_proj")
DGGT = Path("/root/autodl-tmp/third_party/dggt")
ENV = Path("/root/autodl-tmp/envs/dggt-v2")
PYTHON = str(ENV / "bin/python")
CHECKPOINT = Path("/root/autodl-tmp/checkpoints/dggt-v2/model_latest_nuscenes.pt")
INPUT_ROOT = Path("/root/autodl-tmp/data/dynamic_editing_v2/dggt_nuscenes_v1")
ADAPTER = PROJECT / "scripts/prepare_dr_m5_dggt_inputs.py"
PATCH = PROJECT / "compatibility/DGGT-2026-07-28.patch"
TASK_ID = "DR-V2-M1-DGGT-REPAIR-01"
DGGT_COMMIT = "a3276d2bbe4cbb03bcc117830b1836110a27adeb"
PSEUDO_SCENES = [f"{index:03d}" for index in range(18)]
EXPECTED_UNTOUCHED_FAILURE = (
    "AttributeError: 'Namespace' object has no attribute 'difix'"
)
METRICS = {
    "PSNR": re.compile(r"^PSNR:\s+([-+0-9.eE]+)\s*$", re.MULTILINE),
    "SSIM": re.compile(r"^SSIM:\s+([-+0-9.eE]+)\s*$", re.MULTILINE),
    "LPIPS(ALEX)": re.compile(
        r"^LPIPS:\s+([-+0-9.eE]+)\s*$", re.MULTILINE
    ),
    "inference_time_seconds": re.compile(
        r"^Avg Inference Time \(s\):\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    ),
}


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"缺少必需文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_resource() -> dict[str, Any]:
    events = {}
    for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines():
        key, value = line.split()
        events[key] = int(value)
    maximum_raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total, utilization = [int(value.strip()) for value in gpu.split(",")]
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": now(),
        "memory_current_bytes": int(
            Path("/sys/fs/cgroup/memory.current").read_text()
        ),
        "memory_max_bytes": None if maximum_raw == "max" else int(maximum_raw),
        "memory_events": events,
        "disk_free_bytes": disk.free,
        "gpu_memory_used_mb": used,
        "gpu_memory_total_mb": total,
        "gpu_utilization_percent": utilization,
    }


def stop_reason(
    sample: dict[str, Any], initial_events: dict[str, int], high_count: int
) -> tuple[str | None, int]:
    maximum = sample["memory_max_bytes"]
    if maximum and sample["memory_current_bytes"] / maximum >= 0.90:
        high_count += 1
    else:
        high_count = 0
    if high_count >= 2:
        return "cgroup memory 连续两个采样达到 90%", high_count
    events = sample["memory_events"]
    if events.get("oom", 0) > initial_events.get("oom", 0):
        return "memory.events oom 增加", high_count
    if events.get("oom_kill", 0) > initial_events.get("oom_kill", 0):
        return "memory.events oom_kill 增加", high_count
    if sample["disk_free_bytes"] < 20 * 1024**3:
        return "数据盘空闲低于 20 GiB", high_count
    return None, high_count


def terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def run_stage(
    run_dir: Path,
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    require_gpu_free: bool = False,
    expected_failure: str | None = None,
    allow_cuda_oom: bool = False,
) -> dict[str, Any]:
    marker = run_dir / f"stages/{name}.json"
    if marker.exists():
        raise RuntimeError(f"stage 已存在，禁止覆盖: {marker}")
    initial = read_resource()
    maximum = initial["memory_max_bytes"]
    if maximum is not None and maximum < 32 * 1024**3:
        raise RuntimeError("cgroup memory.max 低于 32 GiB")
    if maximum and initial["memory_current_bytes"] / maximum >= 0.90:
        raise RuntimeError("stage 启动前 cgroup memory 已达到 90%")
    if initial["disk_free_bytes"] < 60 * 1024**3:
        raise RuntimeError("stage 启动前数据盘空闲低于 60 GiB")
    if require_gpu_free:
        if initial["gpu_memory_total_mb"] < 24 * 1024:
            raise RuntimeError("GPU 总显存低于 24 GiB")
        if initial["gpu_memory_used_mb"] > 2048:
            raise RuntimeError("GPU 已占用超过 2 GiB")

    stdout_path = run_dir / f"logs/{name}.stdout.log"
    stderr_path = run_dir / f"logs/{name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    samples = [initial]
    started = now()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        high_count = 0
        stopped = None
        while process.poll() is None:
            time.sleep(5)
            sample = read_resource()
            samples.append(sample)
            stopped, high_count = stop_reason(
                sample, initial["memory_events"], high_count
            )
            if stopped:
                terminate(process)
                break
        return_code = process.wait()
    text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(
        errors="replace"
    )
    cuda_oom = "CUDA out of memory" in text or "torch.OutOfMemoryError" in text
    expected_matched = (
        expected_failure is not None
        and return_code != 0
        and expected_failure in text
        and stopped is None
    )
    allowed_oom = allow_cuda_oom and cuda_oom and return_code != 0 and stopped is None
    done = expected_matched or allowed_oom or (
        expected_failure is None and return_code == 0 and stopped is None
    )
    result = {
        "stage": name,
        "status": "done" if done else "blocked",
        "started_at": started,
        "finished_at": now(),
        "command": command,
        "cwd": str(cwd),
        "return_code": return_code,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stop_reason": stopped,
        "expected_failure": expected_failure,
        "expected_failure_matched": expected_matched,
        "cuda_oom_observed": cuda_oom,
        "resource": {
            "samples": samples,
            "peak_memory_current_bytes": max(
                row["memory_current_bytes"] for row in samples
            ),
            "peak_gpu_memory_used_mb": max(
                row["gpu_memory_used_mb"] for row in samples
            ),
        },
    }
    write_json(marker, result)
    with (run_dir / "resource.jsonl").open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps({"stage": name, **sample}, ensure_ascii=False) + "\n"
            )
    if not done:
        raise RuntimeError(f"stage {name} 失败，rc={return_code}, stop={stopped}")
    return result


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    nvidia_root = ENV / "lib/python3.10/site-packages/nvidia"
    include_paths = ":".join(
        str(path)
        for path in sorted(nvidia_root.glob("*/include"))
        if path.is_dir()
    )
    library_paths = ":".join(
        str(path)
        for path in sorted(nvidia_root.glob("*/lib"))
        if path.is_dir()
    )
    inherited_cpath = env.get("CPATH")
    inherited_ld_path = env.get("LD_LIBRARY_PATH")
    env.update(
        {
            "PROJECT_ROOT": str(PROJECT),
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
            "PIP_CACHE_DIR": "/root/autodl-tmp/cache/pip",
            "TMPDIR": "/root/autodl-tmp/tmp",
            "CUDA_HOME": str(ENV),
            "PATH": f"{ENV / 'bin'}:" + env.get("PATH", ""),
            "CPATH": include_paths
            + (f":{inherited_cpath}" if inherited_cpath else ""),
            "LD_LIBRARY_PATH": f"{ENV / 'lib'}:{library_paths}"
            + (f":{inherited_ld_path}" if inherited_ld_path else ""),
            "OMP_NUM_THREADS": "8",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "PYTHONPATH": str(DGGT),
            "MPLCONFIGDIR": "/root/autodl-tmp/cache/matplotlib-dggt-v2",
        }
    )
    return env


def parse_metrics(stage: dict[str, Any]) -> dict[str, float]:
    text = Path(stage["stdout"]).read_text(errors="replace")
    values = {}
    for name, pattern in METRICS.items():
        matches = pattern.findall(text)
        if len(matches) != 1:
            raise RuntimeError(f"{stage['stage']} 的 {name} 匹配数为 {len(matches)}")
        value = float(matches[0])
        if not math.isfinite(value):
            raise RuntimeError(f"{stage['stage']} 的 {name} 非 finite")
        values[name] = value
    return values


def inference_command(
    source_root: Path, scene: str, views: int, output: Path
) -> list[str]:
    return [
        PYTHON,
        "inference.py",
        "--image_dir",
        str(INPUT_ROOT),
        "--scene_names",
        scene,
        "--input_views",
        str(views),
        "--sequence_length",
        "4",
        "--start_idx",
        "0",
        "--mode",
        "2",
        "--ckpt_path",
        str(CHECKPOINT),
        "--output_path",
        str(output),
        "-images",
        "-depth",
        "-metrics",
    ]


def validate_output(output: Path, views: int) -> dict[str, Any]:
    scene_dir = output / "001"
    expected = 4 * views
    images = sorted(scene_dir.glob("view_*.png"))
    depths = sorted(scene_dir.glob("view_*.npy"))
    videos = [scene_dir / "rendered_video.mp4", scene_dir / "comparison.mp4"]
    if len(images) != (4 if views == 1 else 4):
        # 3-view 把每个时间步拼成一张同步三相机图。
        raise RuntimeError(f"{output} image 数错误: {len(images)}")
    if len(depths) != expected:
        raise RuntimeError(f"{output} depth 数错误: {len(depths)} != {expected}")
    for path in [*images, *depths, *videos]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"DGGT 输出缺失或为空: {path}")
    return {
        "output": str(output),
        "image_count": len(images),
        "depth_count": len(depths),
        "video_count": len(videos),
        "total_bytes": sum(
            path.stat().st_size for path in [*images, *depths, *videos]
        ),
    }


def prepare(run_dir: Path) -> dict[str, Any]:
    if INPUT_ROOT.exists():
        raise RuntimeError(f"输入 staging 已存在，禁止覆盖: {INPUT_ROOT}")
    stage = run_stage(
        run_dir,
        "prepare_fixed_inputs",
        [PYTHON, str(ADAPTER), "--output-root", str(INPUT_ROOT)],
        cwd=PROJECT,
        env=base_env(),
    )
    manifest = load_json(INPUT_ROOT / "manifest.json")
    if manifest["pseudo_scene_count"] != 18 or manifest["frame_camera_count"] != 216:
        raise RuntimeError("DGGT 固定窗口 staging coverage 不完整")
    return {"stage": stage, "manifest": manifest}


def make_patched_worktree(run_dir: Path) -> Path:
    worktree = run_dir / "source_snapshot/dggt_patched_worktree"
    run_stage(
        run_dir,
        "create_patched_worktree",
        ["git", "-C", str(DGGT), "worktree", "add", "--detach", str(worktree), DGGT_COMMIT],
        cwd=PROJECT,
        env=base_env(),
    )
    run_stage(
        run_dir,
        "apply_inference_compatibility_patch",
        ["git", "-C", str(worktree), "apply", str(PATCH)],
        cwd=PROJECT,
        env=base_env(),
    )
    status = subprocess.check_output(
        ["git", "-C", str(worktree), "status", "--porcelain"], text=True
    ).strip()
    if status != "M inference.py":
        raise RuntimeError(f"patch 后 worktree 状态异常: {status}")
    return worktree


def row_for(
    scene: str,
    views: int,
    stage: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    return {
        "pseudo_scene": scene,
        "views": views,
        "metrics": parse_metrics(stage),
        "stage": stage["stage"],
        "resource": stage["resource"],
        "artifacts": validate_output(output, views),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(METRICS)
    return {
        "coverage": len(rows) / len(PSEUDO_SCENES),
        "count": len(rows),
        "means": {
            name: sum(row["metrics"][name] for row in rows) / len(rows)
            for name in names
        },
        "worst": {
            "PSNR": min(rows, key=lambda row: row["metrics"]["PSNR"]),
            "SSIM": min(rows, key=lambda row: row["metrics"]["SSIM"]),
            "LPIPS(ALEX)": max(
                rows, key=lambda row: row["metrics"]["LPIPS(ALEX)"]
            ),
        },
    }


def execute(run_dir: Path) -> None:
    terminal = load_json(run_dir / "terminal.json")
    if terminal.get("status") != "running":
        raise RuntimeError("M1 env run 不是 running")
    if not (run_dir / "stages/pointops2_cuda_smoke.json").is_file():
        raise RuntimeError("pointops2 CUDA smoke 尚未通过")
    if not ENV.is_dir() or not CHECKPOINT.is_file():
        raise RuntimeError("dggt-v2 环境或 checkpoint hardlink 缺失")
    if subprocess.check_output(
        ["git", "-C", str(DGGT), "rev-parse", "HEAD"], text=True
    ).strip() != DGGT_COMMIT:
        raise RuntimeError("DGGT commit 漂移")

    write_json(
        run_dir / "resolved.yaml",
        {
            "task_id": TASK_ID,
            "scenes": [f"scene-{scene}" for scene in ("0230", "0242", "0255", "0295", "0518", "0749")],
            "raw_windows": [[10, 11, 12, 13], [34, 35, 36, 37], [66, 67, 68, 69]],
            "pseudo_scenes": PSEUDO_SCENES,
            "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
            "sequence_length": 4,
            "mode": 2,
            "diffusion": False,
            "model_input_hw": [294, 518],
            "input_root": str(INPUT_ROOT),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
        },
    )
    prepared = prepare(run_dir)
    run_stage(
        run_dir,
        "untouched_help",
        [PYTHON, "inference.py", "--help"],
        cwd=DGGT,
        env=base_env(),
    )
    untouched_output = run_dir / "outputs/untouched_1view_000"
    untouched = run_stage(
        run_dir,
        "untouched_1view_000",
        inference_command(DGGT, "000", 1, untouched_output),
        cwd=DGGT,
        env=base_env(),
        require_gpu_free=True,
        expected_failure=EXPECTED_UNTOUCHED_FAILURE,
    )
    patched = make_patched_worktree(run_dir)

    rows_1view = []
    for ordinal, scene in enumerate(PSEUDO_SCENES):
        output = run_dir / f"outputs/1view_{scene}"
        name = "patched_smoke_1view_000" if ordinal == 0 else f"native_1view_{scene}"
        stage = run_stage(
            run_dir,
            name,
            inference_command(patched, scene, 1, output),
            cwd=patched,
            env=base_env(),
            require_gpu_free=True,
        )
        rows_1view.append(row_for(scene, 1, stage, output))

    rows_3view = []
    three_view_status: dict[str, Any]
    for ordinal, scene in enumerate(PSEUDO_SCENES):
        output = run_dir / f"outputs/3view_{scene}"
        stage = run_stage(
            run_dir,
            f"native_3view_{scene}",
            inference_command(patched, scene, 3, output),
            cwd=patched,
            env=base_env(),
            require_gpu_free=True,
            allow_cuda_oom=True,
        )
        if stage["cuda_oom_observed"]:
            three_view_status = {
                "status": "blocked",
                "reason": "24 GiB GPU OOM on preregistered 3-view protocol",
                "failed_pseudo_scene": scene,
                "coverage": len(rows_3view) / len(PSEUDO_SCENES),
            }
            break
        rows_3view.append(row_for(scene, 3, stage, output))
    else:
        three_view_status = {"status": "done", "reason": None, "coverage": 1.0}

    metrics = {
        "native_1view_rows": rows_1view,
        "native_1view_summary": summarize(rows_1view),
        "native_3view_rows": rows_3view,
        "native_3view_summary": summarize(rows_3view) if rows_3view else None,
        "native_3view_status": three_view_status,
    }
    write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows_1view + rows_3view:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "status": "native_done",
        "input_manifest": prepared["manifest"],
        "untouched_failure": {
            "stage": untouched["stage"],
            "return_code": untouched["return_code"],
            "expected_failure_matched": untouched["expected_failure_matched"],
        },
        "compatibility_patch": {
            "path": str(PATCH),
            "sha256": sha256_file(PATCH),
            "scope": "args.difix -> args.diffusion only",
        },
        "native_1view": metrics["native_1view_summary"],
        "native_3view": three_view_status,
        "next": "run common-observation diagnostic before final terminal",
    }
    write_json(run_dir / "native_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.run_dir.resolve())
    except Exception as exc:
        if args.run_dir.exists():
            write_json(
                args.run_dir / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": f"{type(exc).__name__}: {exc}",
                },
            )
        raise


if __name__ == "__main__":
    main()
