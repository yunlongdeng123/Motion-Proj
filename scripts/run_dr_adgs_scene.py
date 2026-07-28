#!/usr/bin/env python3
"""按冻结顺序执行单个 AD-GS 官方场景的预处理与分级训练门禁。"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
ADGS = Path("/root/autodl-tmp/third_party/AD-GS")
DPT = Path("/root/autodl-tmp/third_party/Depth-Anything-V2")
SAM = Path("/root/autodl-tmp/third_party/Grounded-SAM-2")
RAW = Path(
    "/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1"
)
PROCESSED_ROOT = Path(
    "/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1"
)
VALID_SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
SCENE_NAME = "scene-0230"
SCENE = PROCESSED_ROOT / SCENE_NAME
TASK_ID = "DR-M3-ADGS-0230-01"
ADGS_PYTHON = "/root/autodl-tmp/envs/adgs/bin/python"
DPT_PYTHON = "/root/autodl-tmp/envs/adgs-dpt/bin/python"
SAM_PYTHON = "/root/autodl-tmp/envs/adgs-sam/bin/python"
M2_RUN = Path(
    "/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/"
    "20260727T180733__e49a4e-4080s-r3"
)
DATA_MANIFEST = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/"
    "adgs_nuscenes_v1_manifest.json"
)
SOURCE_INPUTS = [
    ("runner.py", PROJECT / "scripts/run_dr_adgs_scene.py"),
    ("audit_processed.py", PROJECT / "scripts/audit_adgs_processed_scene.py"),
    ("compatibility.patch", PROJECT / "compatibility/AD-GS-2026-07-27.patch"),
    ("nuscene.py", ADGS / "scripts/nuscene/nuscene.py"),
    ("run_dpt.py", ADGS / "scripts/run-dpt.py"),
    ("semantic.py", ADGS / "scripts/semantic.py"),
    ("segment_pcd.py", ADGS / "scripts/segment_pcd.py"),
    ("flow.py", ADGS / "scripts/flow.py"),
    ("colmap.py", ADGS / "scripts/colmap.py"),
    ("train.py", ADGS / "train.py"),
    ("render.py", ADGS / "render.py"),
    ("arguments_init.py", ADGS / "arguments/__init__.py"),
    ("arguments_nuscenes.py", ADGS / "arguments/nuscenes.py"),
]


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(path))


def now():
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


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


def stop_reason(sample, initial_events, high_count):
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


def run_stage(run_dir, name, command, cwd, env):
    marker = run_dir / "stages/{}.json".format(name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get("status") == "done":
            print("[skip] {} 已完成".format(name), flush=True)
            return previous
        raise RuntimeError(
            "{} 已有非 done 终态；按 run contract 必须新建实例".format(name)
        )

    stdout_path = run_dir / "logs/{}.stdout.log".format(name)
    stderr_path = run_dir / "logs/{}.stderr.log".format(name)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    initial = read_resource()
    if (
        initial["memory_max_bytes"] is not None
        and initial["memory_max_bytes"] < 32 * 1024 ** 3
    ):
        raise RuntimeError("cgroup memory.max 低于 32 GiB")
    if (
        initial["memory_max_bytes"] is not None
        and initial["memory_current_bytes"] / initial["memory_max_bytes"] >= 0.90
    ):
        raise RuntimeError("stage 启动前 cgroup memory 已达到 90%")
    if initial["disk_free_bytes"] < 60 * 1024 ** 3:
        raise RuntimeError("stage 启动前数据盘空闲低于 60 GiB")
    if initial["gpu_memory_total_mb"] < 24 * 1024:
        raise RuntimeError("GPU 总显存低于 24 GiB")
    if initial["gpu_memory_used_mb"] > 2048:
        raise RuntimeError("stage 启动前 GPU 已被其他任务占用超过 2 GiB")
    samples = [initial]
    started = now()
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
            stopped, high_count = stop_reason(
                sample, initial["memory_events"], high_count
            )
            if stopped:
                terminate_group(process)
                break
        return_code = process.wait()

    status = "done" if return_code == 0 and not stopped else "blocked"
    result = {
        "stage": name,
        "status": status,
        "started_at": started,
        "finished_at": now(),
        "command": command,
        "cwd": str(cwd),
        "return_code": return_code,
        "stop_reason": stopped,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
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
    if status != "done":
        raise RuntimeError(
            "stage {} blocked: rc={} reason={}".format(
                name, return_code, stopped
            )
        )
    return result


def base_env():
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "16",
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": "/root/autodl-tmp/hf_cache",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "COTRACKER_REPO": "/root/autodl-tmp/third_party/co-tracker",
        "COTRACKER_CHECKPOINT": (
            "/root/autodl-tmp/checkpoints/cotracker3/scaled_offline.pth"
        ),
    })
    env["PATH"] = (
        "/root/autodl-tmp/envs/adgs/bin:"
        "/usr/local/cuda/bin:" + env.get("PATH", "")
    )
    return env


def commands(run_dir):
    env = base_env()
    dpt_env = dict(env)
    dpt_env["PYTHONPATH"] = str(DPT)
    sam_env = dict(env)
    sam_env["PYTHONPATH"] = str(SAM)
    preprocess_audit = run_dir / "preprocess_audit.json"
    common_train = [
        ADGS_PYTHON,
        "train.py",
        "-c",
        str(ADGS / "arguments/nuscenes.py"),
        "-s",
        str(SCENE),
        "--data_device",
        "cuda:0",
    ]
    common_render = [
        ADGS_PYTHON,
        "render.py",
        "-c",
        str(ADGS / "arguments/nuscenes.py"),
        "--data_device",
        "cuda:0",
        "-v",
        "--cam_order",
        "1",
        "0",
        "2",
    ]
    return [
        (
            "prepare_raw",
            [
                ADGS_PYTHON,
                str(ADGS / "scripts/nuscene/nuscene.py"),
                str(RAW),
                str(PROCESSED_ROOT),
                SCENE_NAME,
                "--first_frame",
                "10",
                "--last_frame",
                "69",
                "--use_color",
            ],
            ADGS,
            env,
            "preprocess",
        ),
        (
            "depth",
            [
                DPT_PYTHON,
                str(ADGS / "scripts/run-dpt.py"),
                "--img-path",
                str(SCENE / "image"),
                "--outdir",
                str(SCENE / "depth"),
                "--encoder",
                "vitl",
                "--checkpoint",
                (
                    "/root/autodl-tmp/checkpoints/depth_anything_v2/"
                    "depth_anything_v2_vitl.pth"
                ),
            ],
            DPT,
            dpt_env,
            "preprocess",
        ),
        (
            "sky_mask",
            [
                SAM_PYTHON,
                str(ADGS / "scripts/semantic.py"),
                str(SCENE),
                "--text",
                "sky.",
                "--name",
                "sky",
                "--work_dir",
                str(run_dir / "work/sky"),
            ],
            SAM,
            sam_env,
            "preprocess",
        ),
        (
            "object_mask",
            [
                SAM_PYTHON,
                str(ADGS / "scripts/semantic.py"),
                str(SCENE),
                "--text",
                "car.bus.truck.van.human.bike.",
                "--name",
                "semantic",
                "--work_dir",
                str(run_dir / "work/semantic"),
            ],
            SAM,
            sam_env,
            "preprocess",
        ),
        (
            "segment_points",
            [
                ADGS_PYTHON,
                str(ADGS / "scripts/segment_pcd.py"),
                str(SCENE),
            ],
            ADGS,
            env,
            "preprocess",
        ),
        (
            "flow",
            [
                ADGS_PYTHON,
                str(ADGS / "scripts/flow.py"),
                str(SCENE),
                "--device",
                "cuda:0",
                "--step",
                "4",
                "--seed",
                "0",
            ],
            ADGS,
            env,
            "preprocess",
        ),
        (
            "colmap",
            [
                ADGS_PYTHON,
                str(ADGS / "scripts/colmap.py"),
                str(SCENE),
                "--cam",
                "3",
                "--cmd",
                "/root/autodl-tmp/envs/adgs/bin/colmap",
                "--num_threads",
                "16",
            ],
            ADGS,
            env,
            "preprocess",
        ),
        (
            "audit_preprocess",
            [
                ADGS_PYTHON,
                str(PROJECT / "scripts/audit_adgs_processed_scene.py"),
                "--scene",
                str(SCENE),
                "--out-json",
                str(preprocess_audit),
            ],
            PROJECT,
            env,
            "preprocess",
        ),
        (
            "train_100",
            common_train + [
                "-m",
                str(run_dir / "model_100"),
                "--iterations",
                "100",
                "--save_iterations",
                "100",
            ],
            ADGS,
            env,
            "train100",
        ),
        (
            "render_100",
            common_render + [
                "-m",
                str(run_dir / "model_100"),
                "--iteration",
                "100",
            ],
            ADGS,
            env,
            "train100",
        ),
        (
            "train_1000",
            common_train + [
                "-m",
                str(run_dir / "model_1000"),
                "--iterations",
                "1000",
                "--save_iterations",
                "1000",
            ],
            ADGS,
            env,
            "train1000",
        ),
        (
            "render_1000",
            common_render + [
                "-m",
                str(run_dir / "model_1000"),
                "--iteration",
                "1000",
            ],
            ADGS,
            env,
            "train1000",
        ),
        (
            "train_60000",
            common_train + [
                "-m",
                str(run_dir / "model_60000"),
                "--iterations",
                "60000",
                "--save_iterations",
                "60000",
            ],
            ADGS,
            env,
            "train60000",
        ),
        (
            "render_60000",
            common_render + [
                "-m",
                str(run_dir / "model_60000"),
                "--iteration",
                "60000",
            ],
            ADGS,
            env,
            "train60000",
        ),
    ]


def initialize_run(run_dir):
    terminal = M2_RUN / "terminal.json"
    if not terminal.is_file():
        raise RuntimeError("M2 尚无 terminal.json，禁止启动 AD-GS scene")
    m2_terminal = json.loads(terminal.read_text())
    if m2_terminal.get("status") != "done":
        raise RuntimeError("M2 未通过，禁止启动 AD-GS scene")
    run_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = [
        {
            "name": name,
            "source": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in SOURCE_INPUTS
    ]
    upstream_diff = subprocess.check_output(
        ["git", "-C", str(ADGS), "diff", "--binary"],
    )
    resolved = {
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "scene": SCENE_NAME,
        "frames": [10, 69],
        "sensors_in_upstream_order": [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ],
        "resolution": [900, 1600],
        "seed": 0,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(ADGS), "rev-parse", "HEAD"],
            universal_newlines=True,
        ).strip(),
        "compatibility_patch_sha256": sha256_file(
            PROJECT / "compatibility/AD-GS-2026-07-27.patch"
        ),
        "upstream_diff_sha256": hashlib.sha256(upstream_diff).hexdigest(),
        "source_sha256": source_hashes,
        "data_manifest_sha256": sha256_file(DATA_MANIFEST),
        "m2_terminal": str(terminal),
        "processed_scene": str(SCENE),
        "profiles": [100, 1000, 60000],
    }
    fingerprint = hashlib.sha256(json.dumps(
        resolved, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    resolved["config_fingerprint"] = fingerprint
    existing_manifest = run_dir / "manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text())
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError("已有 AD-GS 实例 fingerprint 与当前配置不一致")
        return
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot_rows = []
    for row, (_, source) in zip(source_hashes, SOURCE_INPUTS):
        destination = snapshot / row["name"]
        shutil.copy2(str(source), str(destination))
        snapshot_row = dict(row)
        snapshot_row["snapshot"] = str(destination)
        snapshot_rows.append(snapshot_row)
    (run_dir / "resolved.yaml").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "project_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
            universal_newlines=True,
        ).strip(),
        "config_fingerprint": fingerprint,
        "data_manifest_sha256": resolved["data_manifest_sha256"],
        "seed": 0,
        "upstream_diff_sha256": resolved["upstream_diff_sha256"],
        "source_snapshot": snapshot_rows,
    }
    atomic_json(run_dir / "manifest.json", manifest)


def reuse_completed_preprocess(run_dir, source_run):
    """把同一 processed scene 的已完成前缀移交给新的修复实例。"""
    source_run = Path(source_run)
    reusable_stages = [
        "prepare_raw",
        "depth",
        "sky_mask",
        "object_mask",
        "segment_points",
        "flow",
    ]
    reusable_sources = [
        "nuscene.py",
        "run_dpt.py",
        "semantic.py",
        "segment_pcd.py",
        "flow.py",
    ]
    source_terminal = json.loads((source_run / "terminal.json").read_text())
    if (
        source_terminal.get("status") != "blocked"
        or "stage colmap blocked" not in (source_terminal.get("failure") or "")
    ):
        raise RuntimeError("只允许复用因 colmap stage 阻塞的前一实例")

    source_manifest_path = source_run / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    current_manifest = json.loads((run_dir / "manifest.json").read_text())
    source_resolved = json.loads((source_run / "resolved.yaml").read_text())
    if source_resolved.get("scene") != SCENE_NAME:
        raise RuntimeError(
            "source scene {} != current scene {}".format(
                source_resolved.get("scene"), SCENE_NAME
            )
        )
    source_hashes = {
        row["name"]: row["sha256"]
        for row in source_manifest["source_snapshot"]
    }
    current_hashes = {
        row["name"]: row["sha256"]
        for row in current_manifest["source_snapshot"]
    }
    mismatched_sources = [
        name for name in reusable_sources
        if source_hashes.get(name) != current_hashes.get(name)
    ]
    if mismatched_sources:
        raise RuntimeError(
            "pre-colmap source 已变化，禁止复用: {}".format(mismatched_sources)
        )

    stage_rows = []
    for name in reusable_stages:
        marker = source_run / "stages/{}.json".format(name)
        payload = json.loads(marker.read_text())
        if payload.get("status") != "done":
            raise RuntimeError("source stage 未完成: {}".format(name))
        stage_rows.append({
            "stage": name,
            "marker": str(marker),
            "marker_sha256": sha256_file(marker),
            "payload": payload,
        })

    expected_groups = [
        ("image", "image", "*.png", 180),
        ("depth", "depth", "*.npy", 180),
        ("sky", "sky", "mask_*.npy", 180),
        ("semantic", "semantic", "mask_*.npy", 180),
        ("flow", "flow", "*.npz", 138),
    ]
    output_paths = [SCENE / "meta.npz", SCENE / "points3d.ply"]
    counts = {}
    for label, folder, pattern, expected in expected_groups:
        paths = sorted((SCENE / folder).glob(pattern))
        counts[label] = len(paths)
        if len(paths) != expected:
            raise RuntimeError(
                "{} 输出计数 {} != {}".format(label, len(paths), expected)
            )
        output_paths.extend(paths)
    output_rows = []
    for path in sorted(output_paths):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("复用输出缺失或为空: {}".format(path))
        output_rows.append({
            "path": str(path),
            "relative_path": str(path.relative_to(SCENE)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    output_fingerprint = hashlib.sha256(json.dumps(
        output_rows, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()

    reuse_record = {
        "schema_version": 1,
        "adopted_at": now(),
        "source_run": str(source_run),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "verified_source_files": reusable_sources,
        "stages": stage_rows,
        "output_counts": counts,
        "output_files": len(output_rows),
        "output_bytes": sum(row["bytes"] for row in output_rows),
        "output_fingerprint": output_fingerprint,
    }
    atomic_json(run_dir / "reuse_preprocess.json", reuse_record)
    marker_dir = run_dir / "stages"
    marker_dir.mkdir(parents=True, exist_ok=True)
    for row in stage_rows:
        payload = dict(row["payload"])
        payload["reused_from_run"] = str(source_run)
        payload["reuse_output_fingerprint"] = output_fingerprint
        payload["adopted_at"] = reuse_record["adopted_at"]
        atomic_json(
            marker_dir / "{}.json".format(row["stage"]),
            payload,
        )
    print(
        "[reuse] adopted stages {} from {} fingerprint={}".format(
            ",".join(reusable_stages), source_run, output_fingerprint
        ),
        flush=True,
    )


def update_run_summary(run_dir, requested_through, failure=None):
    stages = []
    for marker in sorted((run_dir / "stages").glob("*.json")):
        stages.append(json.loads(marker.read_text()))
    stage_status = {row["stage"]: row["status"] for row in stages}
    if failure:
        status = "blocked"
    elif stage_status.get("render_60000") == "done":
        status = "done"
    else:
        status = "running"
    metrics = {
        "status": status,
        "requested_through": requested_through,
        "stages": stages,
        "failure": failure,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    summary = """# {task_id} Summary

- 状态：`{status}`
- 实例：`{instance}`
- scene：`{scene_name}`
- 本次推进到：`{through}`
- 已完成 stages：{done}
- 失败：{failure}
- processed scene：`{scene}`

100/1,000 iterations 只用于工程画像；只有 60,000 iterations + official render/metrics 完整后，当前 scene 才能标记 `done`。
""".format(
        task_id=TASK_ID,
        status=status,
        instance=run_dir.name,
        scene_name=SCENE_NAME,
        through=requested_through,
        done=", ".join(
            row["stage"] for row in stages if row["status"] == "done"
        ) or "无",
        failure=failure or "无",
        scene=SCENE,
    )
    (run_dir / "summary.md").write_text(summary)
    terminal = {
        "status": status,
        "updated_at": now(),
        "failure": failure,
    }
    atomic_json(run_dir / "terminal.json", terminal)
    artifact_paths = [
        run_dir / "manifest.json",
        run_dir / "resolved.yaml",
        run_dir / "metrics.json",
        run_dir / "summary.md",
        run_dir / "terminal.json",
        run_dir / "preprocess_audit.json",
        run_dir / "reuse_preprocess.json",
    ]
    artifact_paths.extend(sorted((run_dir / "source_snapshot").glob("*")))
    artifact_paths.extend(sorted((run_dir / "stages").glob("*.json")))
    artifact_paths.extend(sorted((run_dir / "logs").glob("*.log")))
    for model_dir in sorted(run_dir.glob("model_*")):
        artifact_paths.extend(sorted(model_dir.glob("results*.json")))
        artifact_paths.extend(sorted(model_dir.glob("cfg_args")))
        artifact_paths.extend(sorted(
            model_dir.glob("point_cloud/iteration_*/point_cloud.ply")
        ))
    artifacts = []
    for path in artifact_paths:
        if path.is_file():
            artifacts.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_json(run_dir / "artifacts.json", {"artifacts": artifacts})


def main():
    global SCENE_NAME, SCENE, TASK_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--scene",
        choices=VALID_SCENES,
        default="scene-0230",
    )
    parser.add_argument(
        "--task-id",
        default="DR-M3-ADGS-0230-01",
    )
    parser.add_argument(
        "--through",
        choices=["preprocess", "train100", "train1000", "train60000"],
        required=True,
    )
    parser.add_argument(
        "--reuse-completed-from",
        help="复用仅在 colmap 阶段阻塞的前一实例之已完成预处理前缀",
    )
    args = parser.parse_args()
    SCENE_NAME = args.scene
    SCENE = PROCESSED_ROOT / SCENE_NAME
    TASK_ID = args.task_id
    run_dir = Path(args.run_dir)
    initialize_run(run_dir)
    order = {
        "preprocess": 0,
        "train100": 1,
        "train1000": 2,
        "train60000": 3,
    }
    failure = None
    try:
        if args.reuse_completed_from:
            if not (run_dir / "reuse_preprocess.json").is_file():
                reuse_completed_preprocess(
                    run_dir, args.reuse_completed_from
                )
        for name, command, cwd, env, group in commands(run_dir):
            if order[group] > order[args.through]:
                break
            run_stage(run_dir, name, command, cwd, env)
    except Exception as exc:
        failure = "{}: {}".format(type(exc).__name__, exc)
        update_run_summary(run_dir, args.through, failure=failure)
        raise
    update_run_summary(run_dir, args.through)


if __name__ == "__main__":
    main()
