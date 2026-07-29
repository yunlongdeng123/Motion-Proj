#!/usr/bin/env python3
"""在 M4 通过后安装、复现并运行 DGGT inference-only 对照。"""

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
import sys
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
DGGT = Path("/root/autodl-tmp/third_party/dggt")
DGGT_ENV = Path("/root/autodl-tmp/envs/dggt")
DGGT_PYTHON = str(DGGT_ENV / "bin/python")
CONDA = "/root/miniconda3/bin/conda"
TASK_ID = "DR-M5-DGGT-NUSC-01"
EXPECTED_PROJECT_COMMIT = "d90226cbba3854fe67cf32e6cb6be323a106e778"
DGGT_COMMIT = "a3276d2bbe4cbb03bcc117830b1836110a27adeb"
MODEL_REVISION = "735ac9a6486057b1eb886c33a8c6dc79e0b43214"
CHECKPOINT = Path(
    "/root/autodl-tmp/checkpoints/dggt/model_latest_nuscenes.pt"
)
CHECKPOINT_BYTES = 5411266466
CHECKPOINT_URL = (
    "https://hf-mirror.com/xiaomi-research/dggt/resolve/"
    + MODEL_REVISION
    + "/model_latest_nuscenes.pt"
)
INPUT_ROOT = Path(
    "/root/autodl-tmp/data/dynamic_recon/processed/dggt_nuscenes_v1"
)
ADAPTER = PROJECT / "scripts/prepare_dr_m5_dggt_inputs.py"
COMPATIBILITY_PATCH = PROJECT / "compatibility/DGGT-2026-07-28.patch"
EXPECTED_ADAPTER_SHA256 = (
    "e8a629583eeb26ea6d60149c8340a38119dbfcff73270dcd6b2da32de295dfcf"
)
EXPECTED_PATCH_SHA256 = (
    "a433785a84fffe44e5a84354b2aacf3bb3c21b308186fb88e52848b3476cb3a1"
)
PSEUDO_SCENES = ["{:03d}".format(index) for index in range(18)]
METRIC_PATTERN = {
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


def now():
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return sha256_bytes(encoded)


def atomic_text(path, payload):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(payload)
    os.replace(str(tmp), str(path))


def atomic_json(path, payload):
    atomic_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def append_event(run_dir, event, **payload):
    row = {"timestamp": now(), "event": event, **payload}
    with (run_dir / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def load_json(path):
    if not path.is_file():
        raise RuntimeError("缺少必需文件: {}".format(path))
    return json.loads(path.read_text())


def read_resource():
    events = {}
    for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines():
        key, value = line.split()
        events[key] = int(value)
    memory_max_raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    memory_max = int(memory_max_raw) if memory_max_raw != "max" else None
    gpu_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        universal_newlines=True,
    ).strip()
    gpu_used, gpu_total, gpu_util = [
        int(value.strip()) for value in gpu_raw.split(",")
    ]
    disk = shutil.disk_usage("/root/autodl-tmp")
    return {
        "timestamp": now(),
        "memory_current_bytes": int(
            Path("/sys/fs/cgroup/memory.current").read_text().strip()
        ),
        "memory_max_bytes": memory_max,
        "memory_events": events,
        "disk_free_bytes": disk.free,
        "gpu_memory_used_mb": gpu_used,
        "gpu_memory_total_mb": gpu_total,
        "gpu_utilization_percent": gpu_util,
    }


def resource_stop_reason(sample, initial_events, high_count):
    memory_max = sample["memory_max_bytes"]
    if memory_max and sample["memory_current_bytes"] / memory_max >= 0.90:
        high_count += 1
    else:
        high_count = 0
    if high_count >= 2:
        return "cgroup memory 连续两个采样达到 90%", high_count
    events = sample["memory_events"]
    if (
        events.get("oom", 0) > initial_events.get("oom", 0)
        or events.get("oom_kill", 0) > initial_events.get("oom_kill", 0)
    ):
        return "memory.events 的 oom/oom_kill 增加", high_count
    if sample["disk_free_bytes"] < 20 * 1024 ** 3:
        return "数据盘空闲低于 20 GiB", high_count
    return None, high_count


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def run_stage(
    run_dir,
    name,
    command,
    cwd,
    env,
    require_gpu_free=False,
    expected_failure_pattern=None,
    allow_gpu_oom=False,
):
    marker = run_dir / "stages/{}.json".format(name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        previous = load_json(marker)
        if previous.get("status") == "done":
            print("[skip] {} 已完成".format(name), flush=True)
            return previous
        raise RuntimeError("{} 已有非 done stage".format(name))

    initial = read_resource()
    if (
        initial["memory_max_bytes"] is not None
        and initial["memory_max_bytes"] < 32 * 1024 ** 3
    ):
        raise RuntimeError("cgroup memory.max 低于 32 GiB")
    if (
        initial["memory_max_bytes"] is not None
        and initial["memory_current_bytes"]
        / initial["memory_max_bytes"] >= 0.90
    ):
        raise RuntimeError("stage 启动前 cgroup memory 已达到 90%")
    if initial["disk_free_bytes"] < 60 * 1024 ** 3:
        raise RuntimeError("stage 启动前数据盘空闲低于 60 GiB")
    if require_gpu_free:
        if initial["gpu_memory_total_mb"] < 24 * 1024:
            raise RuntimeError("GPU 总显存低于 24 GiB")
        if initial["gpu_memory_used_mb"] > 2048:
            raise RuntimeError("GPU 已被其他任务占用超过 2 GiB")

    stdout_path = run_dir / "logs/{}.stdout.log".format(name)
    stderr_path = run_dir / "logs/{}.stderr.log".format(name)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    samples = [initial]
    started_at = now()
    print("[run] {}: {}".format(name, " ".join(command)), flush=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
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
            stopped, high_count = resource_stop_reason(
                sample, initial["memory_events"], high_count
            )
            if stopped:
                terminate_group(process)
                break
        return_code = process.wait()

    stdout_text = stdout_path.read_text(errors="replace")
    stderr_text = stderr_path.read_text(errors="replace")
    combined = stdout_text + "\n" + stderr_text
    expected_failure_matched = False
    gpu_oom_observed = (
        "CUDA out of memory" in combined
        or "torch.OutOfMemoryError" in combined
    )
    if expected_failure_pattern is not None:
        expected_failure_matched = (
            return_code != 0 and expected_failure_pattern in combined
        )
        stage_done = expected_failure_matched and stopped is None
    elif allow_gpu_oom and gpu_oom_observed and return_code != 0:
        stage_done = stopped is None
    else:
        stage_done = return_code == 0 and stopped is None

    result = {
        "stage": name,
        "status": "done" if stage_done else "blocked",
        "started_at": started_at,
        "finished_at": now(),
        "command": command,
        "cwd": str(cwd),
        "return_code": return_code,
        "stop_reason": stopped,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "expected_failure_pattern": expected_failure_pattern,
        "expected_failure_matched": expected_failure_matched,
        "gpu_oom_observed": gpu_oom_observed,
        "resource": {
            "n_samples": len(samples),
            "peak_memory_current_bytes": max(
                sample["memory_current_bytes"] for sample in samples
            ),
            "peak_gpu_memory_used_mb": max(
                sample["gpu_memory_used_mb"] for sample in samples
            ),
            "minimum_disk_free_bytes": min(
                sample["disk_free_bytes"] for sample in samples
            ),
            "oom_delta": (
                samples[-1]["memory_events"].get("oom", 0)
                - initial["memory_events"].get("oom", 0)
            ),
            "oom_kill_delta": (
                samples[-1]["memory_events"].get("oom_kill", 0)
                - initial["memory_events"].get("oom_kill", 0)
            ),
        },
    }
    atomic_json(marker, result)
    with (run_dir / "resource_samples.jsonl").open("a") as handle:
        for sample in samples:
            handle.write(json.dumps(
                {"stage": name, **sample}, ensure_ascii=False
            ) + "\n")
    if not stage_done:
        raise RuntimeError(
            "stage {} blocked: rc={} stop={} gpu_oom={}".format(
                name, return_code, stopped, gpu_oom_observed
            )
        )
    return result


def base_env():
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": "/root/autodl-tmp/hf_cache",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "TORCH_HOME": "/root/autodl-tmp/checkpoints/dggt/torch_home",
        "PYTHONPATH": str(DGGT),
    })
    env["PATH"] = (
        str(DGGT_ENV / "bin")
        + ":/usr/local/cuda/bin:"
        + env.get("PATH", "")
    )
    return env


def verify_prerequisite(m4_aggregate_run):
    terminal = load_json(m4_aggregate_run / "terminal.json")
    if terminal.get("status") != "done":
        raise RuntimeError("M4 aggregate 尚未 done: {}".format(terminal))
    summary = load_json(m4_aggregate_run / "summary.json")
    if not summary.get("all_gates_passed"):
        raise RuntimeError("M4 aggregate 没有通过全部数值门禁")
    if (m4_aggregate_run / "launcher.rc").is_file():
        launcher_rc = int(
            (m4_aggregate_run / "launcher.rc").read_text().strip()
        )
        if launcher_rc != 0:
            raise RuntimeError("M4 aggregate launcher.rc 非 0")
    return {
        "run_dir": str(m4_aggregate_run),
        "terminal_sha256": sha256_file(m4_aggregate_run / "terminal.json"),
        "summary_sha256": sha256_file(m4_aggregate_run / "summary.json"),
    }


def verify_frozen_sources():
    project_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    if project_commit != EXPECTED_PROJECT_COMMIT:
        raise RuntimeError(
            "project commit 已变化: {} != {}".format(
                project_commit, EXPECTED_PROJECT_COMMIT
            )
        )
    dggt_commit = subprocess.check_output(
        ["git", "-C", str(DGGT), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    if dggt_commit != DGGT_COMMIT:
        raise RuntimeError(
            "DGGT commit 已变化: {} != {}".format(
                dggt_commit, DGGT_COMMIT
            )
        )
    dggt_status = subprocess.check_output(
        ["git", "-C", str(DGGT), "status", "--porcelain"],
        universal_newlines=True,
    )
    if dggt_status:
        raise RuntimeError("DGGT worktree 启动时非 clean: " + dggt_status)
    if sha256_file(ADAPTER) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("M5 input adapter SHA-256 已变化")
    if sha256_file(COMPATIBILITY_PATCH) != EXPECTED_PATCH_SHA256:
        raise RuntimeError("DGGT compatibility patch SHA-256 已变化")
    subprocess.check_call(
        [
            "git", "-C", str(DGGT), "apply", "--check",
            str(COMPATIBILITY_PATCH),
        ]
    )
    return {
        "project_commit": project_commit,
        "dggt_commit": dggt_commit,
        "dggt_clean": True,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "compatibility_patch_sha256": EXPECTED_PATCH_SHA256,
    }


def initialize(run_dir, m4_evidence, sources):
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("M5 run 目录非空，禁止覆盖: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "source_snapshot").mkdir()
    snapshot_inputs = [
        ("runner.py", Path(__file__).resolve()),
        ("input_adapter.py", ADAPTER),
        ("compatibility.patch", COMPATIBILITY_PATCH),
        ("dggt_inference_original.py", DGGT / "inference.py"),
        ("dggt_dataset_original.py", DGGT / "datasets/dataset.py"),
        ("dggt_license.txt", DGGT / "LICENSE"),
        ("dggt_notice.txt", DGGT / "NOTICE"),
    ]
    snapshots = []
    for name, source in snapshot_inputs:
        target = run_dir / "source_snapshot" / name
        shutil.copy2(str(source), str(target))
        snapshots.append({
            "name": name,
            "source": str(source),
            "snapshot": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "m4_prerequisite": m4_evidence,
        "protocol": {
            "native_1view": True,
            "native_3view_if_24gb_supported": True,
            "sequence_length": 4,
            "windows": [[10, 11, 12, 13], [34, 35, 36, 37],
                        [66, 67, 68, 69]],
            "official_scenes": [
                "scene-0230", "scene-0242", "scene-0255",
                "scene-0295", "scene-0518", "scene-0749",
            ],
            "model_input_resize_hw": [294, 518],
            "diffusion": False,
            "poses_as_input": False,
            "intrinsics_as_input": False,
            "per_scene_optimization": False,
            "metric_lpips_backbone": "Alex",
        },
        "dggt_commit": DGGT_COMMIT,
        "model_revision": MODEL_REVISION,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_expected_bytes": CHECKPOINT_BYTES,
        "input_root": str(INPUT_ROOT),
        "sources": sources,
    }
    config_fingerprint = canonical_sha256(resolved)
    resolved["config_fingerprint"] = config_fingerprint
    atomic_json(run_dir / "resolved.yaml", resolved)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "started_at": now(),
        "project_commit": sources["project_commit"],
        "upstream_commit": DGGT_COMMIT,
        "model_revision": MODEL_REVISION,
        "config_fingerprint": config_fingerprint,
        "seed": 0,
        "code_license": "Apache-2.0 with VGGT NOTICE boundary",
        "model_license": "CC BY-NC 4.0",
        "source_snapshot": snapshots,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    append_event(run_dir, "m5_started")


def ensure_environment(run_dir):
    env = base_env()
    if DGGT_ENV.exists():
        raise RuntimeError(
            "DGGT env 在正式 run 前已存在，无法证明本实例创建: {}".format(
                DGGT_ENV
            )
        )
    run_stage(
        run_dir,
        "env_create",
        [CONDA, "create", "-p", str(DGGT_ENV), "python=3.10", "pip", "-y"],
        PROJECT,
        env,
    )
    run_stage(
        run_dir,
        "env_torch",
        [
            str(DGGT_ENV / "bin/pip"), "install",
            "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
            "--index-url", "https://download.pytorch.org/whl/cu118",
        ],
        PROJECT,
        env,
    )
    run_stage(
        run_dir,
        "env_requirements",
        [
            str(DGGT_ENV / "bin/pip"), "install",
            "-r", str(DGGT / "requirements.txt"), "ninja",
        ],
        DGGT,
        env,
    )
    run_stage(
        run_dir,
        "env_pointops2",
        [str(DGGT_ENV / "bin/pip"), "install", "."],
        DGGT / "third_party/pointops2",
        env,
    )
    audit_commands = {
        "python_version.txt": [DGGT_PYTHON, "-V"],
        "pip_freeze.txt": [str(DGGT_ENV / "bin/pip"), "freeze"],
        "conda_explicit.txt": [CONDA, "list", "-p", str(DGGT_ENV), "--explicit"],
        "torch_cuda_versions.txt": [
            DGGT_PYTHON, "-c",
            (
                "import torch, torchvision, torchaudio; "
                "print('torch', torch.__version__); "
                "print('torchvision', torchvision.__version__); "
                "print('torchaudio', torchaudio.__version__); "
                "print('torch_cuda', torch.version.cuda); "
                "print('cuda_available', torch.cuda.is_available())"
            ),
        ],
        "nvcc_version.txt": ["/usr/local/cuda/bin/nvcc", "--version"],
        "gcc_version.txt": ["gcc", "--version"],
        "gpu_driver.txt": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    }
    env_audit_dir = run_dir / "env_audit"
    env_audit_dir.mkdir()
    for name, command in audit_commands.items():
        output = subprocess.check_output(
            command, cwd=str(DGGT), env=env, stderr=subprocess.STDOUT
        )
        (env_audit_dir / name).write_bytes(output)
    run_stage(
        run_dir,
        "upstream_help_original",
        [DGGT_PYTHON, "inference.py", "--help"],
        DGGT,
        env,
    )


def prepare_inputs(run_dir):
    if INPUT_ROOT.exists():
        raise RuntimeError(
            "DGGT input staging 在正式 run 前已存在: {}".format(INPUT_ROOT)
        )
    run_stage(
        run_dir,
        "prepare_fixed_windows",
        [
            "/root/autodl-tmp/envs/motionproj/bin/python",
            str(ADAPTER),
            "--output-root",
            str(INPUT_ROOT),
        ],
        PROJECT,
        base_env(),
    )
    manifest = load_json(INPUT_ROOT / "manifest.json")
    if manifest.get("frame_camera_count") != 216:
        raise RuntimeError("DGGT input staging 不是 216 个 frame-camera")
    return manifest


def download_checkpoint(run_dir):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        raise RuntimeError(
            "DGGT checkpoint 在正式 run 前已存在，来源不可归属本实例"
        )
    partial = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".partial")
    run_stage(
        run_dir,
        "download_nuscenes_checkpoint",
        [
            "curl", "-L", "--fail", "--retry", "8",
            "--retry-all-errors", "--continue-at", "-",
            CHECKPOINT_URL, "--output", str(partial),
        ],
        PROJECT,
        base_env(),
    )
    if partial.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError(
            "DGGT checkpoint bytes 不匹配: {} != {}".format(
                partial.stat().st_size, CHECKPOINT_BYTES
            )
        )
    os.replace(str(partial), str(CHECKPOINT))
    checkpoint_record = {
        "path": str(CHECKPOINT),
        "bytes": CHECKPOINT.stat().st_size,
        "sha256": sha256_file(CHECKPOINT),
        "model_revision": MODEL_REVISION,
        "model_license": "CC BY-NC 4.0",
    }
    atomic_json(run_dir / "checkpoint.json", checkpoint_record)
    return checkpoint_record


def inference_command(pseudo_scene, views, output_path):
    return [
        DGGT_PYTHON,
        "inference.py",
        "--image_dir",
        str(INPUT_ROOT),
        "--scene_names",
        pseudo_scene,
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
        str(output_path),
        "-images",
        "-depth",
        "-metrics",
    ]


def parse_metrics(stdout_path):
    text = stdout_path.read_text(errors="replace")
    result = {}
    for name, pattern in METRIC_PATTERN.items():
        matches = pattern.findall(text)
        if len(matches) != 1:
            raise RuntimeError(
                "{} 中 {} 匹配数不是 1: {}".format(
                    stdout_path, name, len(matches)
                )
            )
        value = float(matches[0])
        if not math.isfinite(value):
            raise RuntimeError("{} 非 finite".format(name))
        result[name] = value
    return result


def apply_compatibility_patch(run_dir):
    run_stage(
        run_dir,
        "apply_compatibility_patch",
        [
            "git", "-C", str(DGGT), "apply",
            str(COMPATIBILITY_PATCH),
        ],
        PROJECT,
        base_env(),
    )
    status = subprocess.check_output(
        ["git", "-C", str(DGGT), "status", "--porcelain"],
        universal_newlines=True,
    )
    if status.strip() != "M inference.py":
        raise RuntimeError("DGGT patch 后 worktree 非预期: " + status)
    patched = run_dir / "source_snapshot/dggt_inference_patched.py"
    shutil.copy2(str(DGGT / "inference.py"), str(patched))


def run_inference_suite(run_dir, input_manifest):
    env = base_env()
    original_output = run_dir / "outputs/original_upstream_1view_000"
    original = run_stage(
        run_dir,
        "upstream_original_1view_000",
        inference_command("000", 1, original_output),
        DGGT,
        env,
        require_gpu_free=True,
        expected_failure_pattern=(
            "AttributeError: 'Namespace' object has no attribute 'difix'"
        ),
    )
    apply_compatibility_patch(run_dir)
    patched_smoke_output = run_dir / "outputs/patched_smoke_1view_000"
    patched_smoke = run_stage(
        run_dir,
        "patched_smoke_1view_000",
        inference_command("000", 1, patched_smoke_output),
        DGGT,
        env,
        require_gpu_free=True,
    )
    rows_1view = [{
        "pseudo_scene": "000",
        "views": 1,
        "metrics": parse_metrics(
            Path(patched_smoke["stdout"])
        ),
        "stage": patched_smoke["stage"],
        "resource": patched_smoke["resource"],
    }]
    for pseudo_scene in PSEUDO_SCENES[1:]:
        output = run_dir / "outputs/1view_{}".format(pseudo_scene)
        stage = run_stage(
            run_dir,
            "native_1view_{}".format(pseudo_scene),
            inference_command(pseudo_scene, 1, output),
            DGGT,
            env,
            require_gpu_free=True,
        )
        rows_1view.append({
            "pseudo_scene": pseudo_scene,
            "views": 1,
            "metrics": parse_metrics(Path(stage["stdout"])),
            "stage": stage["stage"],
            "resource": stage["resource"],
        })

    rows_3view = []
    output = run_dir / "outputs/3view_000"
    three_view_smoke = run_stage(
        run_dir,
        "native_3view_000",
        inference_command("000", 3, output),
        DGGT,
        env,
        require_gpu_free=True,
        allow_gpu_oom=True,
    )
    if three_view_smoke["gpu_oom_observed"]:
        three_view_status = {
            "status": "blocked",
            "reason": "24 GiB GPU OOM on preregistered 3-view smoke",
            "coverage": 0.0,
        }
    else:
        rows_3view.append({
            "pseudo_scene": "000",
            "views": 3,
            "metrics": parse_metrics(Path(three_view_smoke["stdout"])),
            "stage": three_view_smoke["stage"],
            "resource": three_view_smoke["resource"],
        })
        three_view_status = {
            "status": "running",
            "reason": None,
            "coverage": 1 / len(PSEUDO_SCENES),
        }
        for pseudo_scene in PSEUDO_SCENES[1:]:
            output = run_dir / "outputs/3view_{}".format(pseudo_scene)
            stage = run_stage(
                run_dir,
                "native_3view_{}".format(pseudo_scene),
                inference_command(pseudo_scene, 3, output),
                DGGT,
                env,
                require_gpu_free=True,
                allow_gpu_oom=True,
            )
            if stage["gpu_oom_observed"]:
                three_view_status = {
                    "status": "blocked",
                    "reason": (
                        "24 GiB GPU OOM at pseudo scene {}".format(
                            pseudo_scene
                        )
                    ),
                    "coverage": len(rows_3view) / len(PSEUDO_SCENES),
                }
                break
            rows_3view.append({
                "pseudo_scene": pseudo_scene,
                "views": 3,
                "metrics": parse_metrics(Path(stage["stdout"])),
                "stage": stage["stage"],
                "resource": stage["resource"],
            })
        else:
            three_view_status = {
                "status": "done",
                "reason": None,
                "coverage": 1.0,
            }
    return {
        "original_upstream_failure": {
            "return_code": original["return_code"],
            "expected_failure_matched": original[
                "expected_failure_matched"
            ],
            "stage": original["stage"],
        },
        "input_manifest": input_manifest,
        "native_1view": rows_1view,
        "native_3view": rows_3view,
        "native_3view_status": three_view_status,
    }


def summarize_rows(rows):
    metric_names = [
        "PSNR", "SSIM", "LPIPS(ALEX)", "inference_time_seconds"
    ]
    means = {
        metric: sum(row["metrics"][metric] for row in rows) / len(rows)
        for metric in metric_names
    }
    worst = {
        "PSNR": min(rows, key=lambda row: row["metrics"]["PSNR"]),
        "SSIM": min(rows, key=lambda row: row["metrics"]["SSIM"]),
        "LPIPS(ALEX)": max(
            rows, key=lambda row: row["metrics"]["LPIPS(ALEX)"]
        ),
        "inference_time_seconds": max(
            rows,
            key=lambda row: row["metrics"]["inference_time_seconds"],
        ),
    }
    return {
        "count": len(rows),
        "expected_count": len(PSEUDO_SCENES),
        "coverage": len(rows) / len(PSEUDO_SCENES),
        "mean": means,
        "worst": {
            metric: {
                "pseudo_scene": row["pseudo_scene"],
                "value": row["metrics"][metric],
            }
            for metric, row in worst.items()
        },
        "peak_vram_mb": max(
            row["resource"]["peak_gpu_memory_used_mb"] for row in rows
        ),
        "peak_cgroup_memory_bytes": max(
            row["resource"]["peak_memory_current_bytes"] for row in rows
        ),
    }


def write_artifacts(run_dir):
    paths = []
    for name in [
        "manifest.json", "resolved.yaml", "checkpoint.json",
        "metrics.json", "metrics.jsonl", "summary.json", "summary.md",
        "terminal.json", "events.jsonl", "resource_samples.jsonl",
    ]:
        paths.append(run_dir / name)
    paths.extend(sorted((run_dir / "source_snapshot").glob("*")))
    paths.extend(sorted((run_dir / "stages").glob("*.json")))
    paths.extend(sorted((run_dir / "env_audit").glob("*")))
    artifacts = []
    for path in paths:
        if path.is_file():
            artifacts.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_json(run_dir / "artifacts.json", {"artifacts": artifacts})


def write_done(run_dir, results, checkpoint):
    one_summary = summarize_rows(results["native_1view"])
    three_rows = results["native_3view"]
    three_summary = summarize_rows(three_rows) if three_rows else None
    status = "done"
    metrics = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": status,
        "protocol_boundary": (
            "inference-only characterization; not matched to AD-GS "
            "60-frame per-scene optimization"
        ),
        "checkpoint": checkpoint,
        "original_upstream_failure": results[
            "original_upstream_failure"
        ],
        "native_1view_rows": results["native_1view"],
        "native_1view_summary": one_summary,
        "native_3view_rows": three_rows,
        "native_3view_summary": three_summary,
        "native_3view_status": results["native_3view_status"],
    }
    atomic_json(run_dir / "metrics.json", metrics)
    with (run_dir / "metrics.jsonl").open("w") as handle:
        for row in results["native_1view"]:
            handle.write(json.dumps(
                {"protocol": "native_1view", **row}, ensure_ascii=False
            ) + "\n")
        for row in three_rows:
            handle.write(json.dumps(
                {"protocol": "native_3view", **row}, ensure_ascii=False
            ) + "\n")
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "status": status,
        "completed_at": now(),
        "upstream_smoke_passed_after_minimal_patch": True,
        "original_upstream_failure_preserved": True,
        "native_1view": one_summary,
        "native_3view": three_summary,
        "native_3view_status": results["native_3view_status"],
        "next_action": "进入 M6 baseline 压力测试；不得将 M5 写成 matched leaderboard",
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_text(
        run_dir / "summary.md",
        """# DR-M5 DGGT nuScenes inference-only 对照

- 状态：`done`
- official upstream 原始 `difix` AttributeError：已保存
- 最小 compatibility patch 后 1-view coverage：`{count}/18`
- 1-view mean PSNR / SSIM / LPIPS(Alex)：`{psnr:.6f} / {ssim:.6f} / {lpips:.6f}`
- 1-view mean inference time：`{time:.6f} s / 4-frame window`
- 3-view 状态：`{three_status}`；coverage `{three_coverage:.1%}`
- 下一步：M6 压力测试

DGGT 只使用 4 张（或 3-view 时 12 张）resize 后 RGB，不接收 pose，也不逐场景优化；AD-GS 使用完整
60 帧、三相机与 60k 逐场景优化。本结果只作速度/泛化/失败 characterization，不是 matched leaderboard。
""".format(
            count=one_summary["count"],
            psnr=one_summary["mean"]["PSNR"],
            ssim=one_summary["mean"]["SSIM"],
            lpips=one_summary["mean"]["LPIPS(ALEX)"],
            time=one_summary["mean"]["inference_time_seconds"],
            three_status=results["native_3view_status"]["status"],
            three_coverage=results["native_3view_status"]["coverage"],
        ),
    )
    atomic_json(
        run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    append_event(run_dir, "m5_done")
    write_artifacts(run_dir)


def write_blocked(run_dir, exc):
    failure = {"type": type(exc).__name__, "message": str(exc)}
    atomic_json(
        run_dir / "metrics.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "blocked",
            "failure": failure,
        },
    )
    atomic_text(
        run_dir / "metrics.jsonl",
        json.dumps({"type": "failure", **failure}, ensure_ascii=False) + "\n",
    )
    atomic_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "instance_id": run_dir.name,
            "status": "blocked",
            "completed_at": now(),
            "failure": failure,
            "next_action": "保留原始 upstream/资源证据；禁止静默缩窗口、降 view 或改模型",
        },
    )
    atomic_text(
        run_dir / "summary.md",
        "# DR-M5 DGGT nuScenes inference-only 对照\n\n"
        "- 状态：`blocked`\n"
        "- 失败：`{}: {}`\n".format(type(exc).__name__, exc),
    )
    atomic_json(
        run_dir / "terminal.json",
        {"status": "blocked", "updated_at": now(), "failure": failure},
    )
    append_event(run_dir, "m5_blocked", failure=failure)
    write_artifacts(run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m4-aggregate-run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    m4_aggregate_run = Path(args.m4_aggregate_run)
    initialized = False
    try:
        m4_evidence = verify_prerequisite(m4_aggregate_run)
        sources = verify_frozen_sources()
        initialize(run_dir, m4_evidence, sources)
        initialized = True
        ensure_environment(run_dir)
        input_manifest = prepare_inputs(run_dir)
        checkpoint = download_checkpoint(run_dir)
        results = run_inference_suite(run_dir, input_manifest)
        write_done(run_dir, results, checkpoint)
        return 0
    except Exception as exc:
        if initialized:
            write_blocked(run_dir, exc)
        else:
            print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
