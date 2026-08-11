#!/usr/bin/env python3
"""从冻结的本地环境恢复 WorldSim V4 AD-GS runtime，并生成不可变证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
SNAPSHOT_RELPATHS = (
    "configs/worldsim_v4/adgs_environment_v1.yaml",
    "scripts/restore_worldsim_v4_adgs_env.py",
    "scripts/smoke_worldsim_v4_adgs_env.py",
    "tests/test_restore_worldsim_v4_adgs_env.py",
)


class AdgsEnvironmentError(RuntimeError):
    """AD-GS 环境恢复合同不满足。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdgsEnvironmentError("config 必须为 mapping")
    if value.get("task_id") != TASK_ID or value.get("status") != "running":
        raise AdgsEnvironmentError("task/status 漂移")
    policy = value.get("policy", {})
    if policy != {
        "restore_mode": "clone_frozen_local_environment_then_build_vendored_extensions",
        "network_access": False,
        "test_quality_read": False,
    }:
        raise AdgsEnvironmentError("离线恢复策略漂移")
    runtime = value.get("runtime_contract", {})
    if runtime.get("torch_cuda_arch_list") != "8.6" or runtime.get("torch_cuda") != "11.8":
        raise AdgsEnvironmentError("RTX3090/CUDA 合同漂移")
    extensions = value.get("extensions", [])
    if [row.get("import") for row in extensions] != ["simple_knn._C", "diff_gaussian_rasterization"]:
        raise AdgsEnvironmentError("AD-GS CUDA extension 集合漂移")
    return value


def git(root: Path, *args: str) -> str:
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise AdgsEnvironmentError(process.stderr.strip())
    # porcelain status 的首列允许空格；只能裁掉末尾换行，不能破坏首行 XY 字段。
    return process.stdout.rstrip()


def inspect_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    source = config["source_contract"]
    adgs_root = Path(paths["adgs_root"])
    base = Path(paths["base_environment"])
    patch = Path(paths["compatibility_patch"])
    wheel = Path(paths["plyfile_wheel"])
    if not (base / "bin/python").is_file():
        raise AdgsEnvironmentError(f"base environment 缺失：{base}")
    if git(adgs_root, "rev-parse", "HEAD") != source["adgs_commit"]:
        raise AdgsEnvironmentError("AD-GS commit 漂移")
    changed = sorted(line[3:] for line in git(adgs_root, "status", "--short").splitlines() if line)
    if changed != sorted(source["expected_modified_files"]):
        raise AdgsEnvironmentError(f"AD-GS compatibility 文件集合漂移：{changed}")
    if sha256_file(patch) != source["compatibility_patch_sha256"]:
        raise AdgsEnvironmentError("compatibility patch 哈希漂移")
    reverse_check = subprocess.run(
        [
            "git",
            "-C",
            str(adgs_root),
            "apply",
            "--check",
            "--reverse",
            "--unidiff-zero",
            str(patch),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if reverse_check.returncode != 0:
        raise AdgsEnvironmentError(
            "AD-GS working tree 不是冻结 compatibility patch 的精确已应用状态："
            + reverse_check.stderr.strip()
        )
    wheel_expected = config["wheel_contract"]
    wheel_actual = {"bytes": wheel.stat().st_size, "sha256": sha256_file(wheel)} if wheel.is_file() else None
    if wheel_actual != {"bytes": wheel_expected["bytes"], "sha256": wheel_expected["sha256"]}:
        raise AdgsEnvironmentError(f"plyfile wheel 漂移：{wheel_actual}")
    base_probe = subprocess.run(
        [str(base / "bin/python"), "-c", "import json,sys,torch;print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda':torch.version.cuda}))"],
        capture_output=True,
        text=True,
        check=True,
    )
    versions = json.loads(base_probe.stdout)
    runtime = config["runtime_contract"]
    if not versions["python"].startswith(runtime["python"]) or versions["torch"] != runtime["torch"] or versions["torch_cuda"] != runtime["torch_cuda"]:
        raise AdgsEnvironmentError(f"base environment 版本漂移：{versions}")
    return {
        "adgs_root": str(adgs_root),
        "adgs_commit": source["adgs_commit"],
        "modified_files": changed,
        "compatibility_patch": {"path": str(patch), "bytes": patch.stat().st_size, "sha256": sha256_file(patch)},
        "plyfile_wheel": {"path": str(wheel), **wheel_actual},
        "base_environment": {"path": str(base), **versions},
    }


def materialize_extension_sources(config: Mapping[str, Any], build_root: Path) -> Path:
    """把 vendored CUDA extension 复制到 run 内，避免 pip 污染 official checkout。"""
    if build_root.exists():
        raise AdgsEnvironmentError(f"extension build root 已存在：{build_root}")
    adgs = Path(config["paths"]["adgs_root"])
    for extension in config["extensions"]:
        relative = Path(extension["source"])
        source = adgs / relative
        destination = build_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    return build_root


def build_commands(
    config: Mapping[str, Any], extension_root: Path | None = None
) -> list[tuple[str, list[str], Path]]:
    paths = config["paths"]
    target = Path(paths["target_environment"])
    adgs = Path(paths["adgs_root"])
    python = str(target / "bin/python")
    commands: list[tuple[str, list[str], Path]] = [
        ("install_plyfile", [python, "-m", "pip", "install", "--no-index", "--no-deps", paths["plyfile_wheel"]], adgs),
    ]
    for extension in config["extensions"]:
        source = (extension_root or adgs) / extension["source"]
        commands.append((f"build_{extension['name']}", [python, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation", "."], source))
    commands.append(("cuda_smoke", [python, str(Path(paths["project_root"]) / "scripts/smoke_worldsim_v4_adgs_env.py")], adgs))
    return commands


def resource_sample() -> dict[str, Any]:
    memory_events = {}
    for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines():
        key, value = line.split()
        memory_events[key] = int(value)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().split(",")
    return {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
        "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "memory_events": memory_events,
        "disk_free_bytes": shutil.disk_usage("/root/autodl-tmp").free,
        "gpu": {"name": gpu[0].strip(), "memory_used_mib": int(gpu[1]), "memory_total_mib": int(gpu[2]), "utilization_percent": int(gpu[3])},
    }


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(payload))


def run_stage(run_dir: Path, name: str, command: list[str], cwd: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    before = resource_sample()
    stdout_path = run_dir / "logs" / f"{name}.stdout.log"
    stderr_path = run_dir / "logs" / f"{name}.stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.run(command, cwd=str(cwd), env=dict(environment), stdout=stdout, stderr=stderr, check=False)
    after = resource_sample()
    result = {
        "stage": name,
        "status": "done" if process.returncode == 0 else "blocked",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(cwd),
        "return_code": process.returncode,
        "resource_before": before,
        "resource_after": after,
    }
    write_json(run_dir / "stages" / f"{name}.json", result)
    append_jsonl(run_dir / "resource.jsonl", {"stage": name, "event": "before", **before})
    append_jsonl(run_dir / "resource.jsonl", {"stage": name, "event": "after", **after})
    if process.returncode != 0:
        raise AdgsEnvironmentError(f"stage {name} failed: rc={process.returncode}; stderr={stderr_path}")
    return result


def audit_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["target_environment"])
    python = target / "bin/python"
    imports = ["plyfile"] + [row["import"] for row in config["extensions"]]
    probe_code = "import importlib,json,sys,torch; names=" + repr(imports) + "; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda':torch.version.cuda,'imports':{n:importlib.import_module(n).__file__ for n in names}}))"
    probe = subprocess.run([str(python), "-c", probe_code], capture_output=True, text=True, check=True)
    extensions = []
    for path in sorted((target / "lib").rglob("*.so")):
        if "simple_knn" in str(path) or "diff_gaussian_rasterization" in str(path):
            extensions.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if len(extensions) < 2:
        raise AdgsEnvironmentError("AD-GS CUDA extension 二进制不完整")
    return {"target_environment": str(target), "probe": json.loads(probe.stdout), "extension_binaries": extensions}


def final_manifest(run_dir: Path, status: str) -> None:
    artifacts = {
        str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(run_dir / "manifest.json", {"schema_version": "worldsim_v4_adgs_environment_manifest_v1", "task_id": TASK_ID, "status": status, "artifacts": artifacts, "test_quality_read": False})


def run(config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path, run_dir, project_root = config_path.resolve(), run_dir.resolve(), project_root.resolve()
    if run_dir.exists():
        raise AdgsEnvironmentError(f"run 目录已存在，禁止复用：{run_dir}")
    config = load_config(config_path)
    target = Path(config["paths"]["target_environment"])
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    run_dir.mkdir(parents=True)
    for folder in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (run_dir / folder).mkdir(exist_ok=True)
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    inputs = inspect_inputs(config)
    write_json(run_dir / "artifacts" / "input_audit.json", inputs)
    preflight_resource = resource_sample()
    runtime = config["runtime_contract"]
    if preflight_resource["disk_free_bytes"] < int(runtime["minimum_disk_free_gib"]) * 1024**3:
        raise AdgsEnvironmentError("环境恢复前数据盘空闲低于门槛")
    memory_max = preflight_resource["memory_max"]
    if memory_max != "max" and int(memory_max) < int(runtime["minimum_cgroup_memory_gib"]) * 1024**3:
        raise AdgsEnvironmentError("环境恢复前 cgroup memory.max 低于门槛")
    write_json(run_dir / "artifacts" / "preflight_resource.json", preflight_resource)
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        destination = run_dir / "source_snapshot" / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshots[relpath] = {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
    project_git = {"head": git(project_root, "rev-parse", "HEAD"), "branch": git(project_root, "branch", "--show-current"), "dirty": bool(git(project_root, "status", "--porcelain"))}
    environment = os.environ.copy()
    environment.update({"CUDA_HOME": runtime["cuda_home"], "TORCH_CUDA_ARCH_LIST": runtime["torch_cuda_arch_list"], "MAX_JOBS": str(runtime["max_jobs"]), "PIP_NO_INDEX": "1", "PYTHONDONTWRITEBYTECODE": "1", "ADGS_ROOT": config["paths"]["adgs_root"]})
    present_before = target.is_dir()
    if not present_before:
        if partial.exists():
            raise AdgsEnvironmentError(f"partial environment 已存在，需人工审计：{partial}")
        clone_result = run_stage(run_dir, "clone_base_environment", ["cp", "-a", "--reflink=auto", config["paths"]["base_environment"], str(partial)], Path("/root/autodl-tmp/envs"), environment)
        os.replace(partial, target)
        clone_result["published_target"] = str(target)
        write_json(run_dir / "stages" / "clone_base_environment.json", clone_result)
        extension_root = materialize_extension_sources(
            config, run_dir / "build_source")
        for name, command, cwd in build_commands(config, extension_root):
            if name == "cuda_smoke":
                sample = resource_sample()
                if sample["gpu"]["memory_used_mib"] > int(runtime["maximum_gpu_used_at_smoke_start_mib"]):
                    raise AdgsEnvironmentError(f"CUDA smoke 启动前 GPU 占用过高：{sample['gpu']['memory_used_mib']} MiB")
            run_stage(run_dir, name, command, cwd, environment)
        restore_mode = "created_from_frozen_local_environment"
    else:
        restore_mode = "existing_environment_audited"
        run_stage(run_dir, "cuda_smoke", build_commands(config)[-1][1], build_commands(config)[-1][2], environment)
    audit = audit_environment(config)
    write_json(run_dir / "environment" / "environment_audit.json", audit)
    with (run_dir / "environment" / "pip_freeze.txt").open("wb") as handle:
        subprocess.run([str(target / "bin/python"), "-m", "pip", "freeze"], stdout=handle, check=True)
    fingerprint = {"config_sha256": sha256_file(config_path), "input_audit_sha256": sha256_file(run_dir / "artifacts/input_audit.json"), "environment_audit_sha256": sha256_file(run_dir / "environment/environment_audit.json"), "source_snapshots": snapshots, "project_git": project_git}
    write_json(run_dir / "fingerprint.json", fingerprint)
    finished = datetime.now(timezone.utc).isoformat()
    append_jsonl(run_dir / "events.jsonl", {"at_utc": finished, "event": "adgs_environment_restore_complete", "status": "done", "restore_mode": restore_mode})
    summary = {"schema_version": "worldsim_v4_adgs_environment_summary_v1", "task_id": TASK_ID, "status": "done", "finished_at_utc": finished, "restore_mode": restore_mode, "target_environment": str(target), "adgs_commit": inputs["adgs_commit"], "compatibility_patch_sha256": inputs["compatibility_patch"]["sha256"], "environment_audit_sha256": fingerprint["environment_audit_sha256"], "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "project_git": project_git, "training_started": False, "model_inference_started": False, "test_quality_read": False}
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "finished_at_utc": finished, "summary_sha256": sha256_file(run_dir / "summary.json")})
    final_manifest(run_dir, "done")
    return summary


def record_blocked(config_path: Path, run_dir: Path, error: BaseException) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.is_file() and not (run_dir / "resolved.yaml").exists():
        shutil.copy2(config_path, run_dir / "resolved.yaml")
    finished = datetime.now(timezone.utc).isoformat()
    event = {"at_utc": finished, "event": "adgs_environment_restore_blocked", "error_type": type(error).__name__, "message": str(error)}
    append_jsonl(run_dir / "events.jsonl", event)
    write_json(run_dir / "fingerprint.json", {"config_sha256": sha256_file(config_path) if config_path.is_file() else None, "error": event})
    summary = {"schema_version": "worldsim_v4_adgs_environment_summary_v1", "task_id": TASK_ID, "status": "blocked", "finished_at_utc": finished, "reason": "adgs_environment_restore_failed", "error": event, "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"), "training_started": False, "model_inference_started": False, "test_quality_read": False}
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "blocked", "finished_at_utc": finished, "summary_sha256": sha256_file(run_dir / "summary.json")})
    final_manifest(run_dir, "blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复 WorldSim V4 AD-GS 环境")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    existed_before = args.run_dir.resolve().exists()
    try:
        summary = run(args.config, args.run_dir, args.project_root)
    except BaseException as error:
        if not existed_before:
            record_blocked(args.config.resolve(), args.run_dir.resolve(), error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
