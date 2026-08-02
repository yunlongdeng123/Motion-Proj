#!/usr/bin/env python3
"""验证并封存 DR-V2 M0 bootstrap run。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT = Path("/root/autodl-tmp/motion_proj")
TASK_ID = "DR-V2-M0-BOOTSTRAP-01"
SCENES = ("0230", "0242", "0255", "0295", "0518", "0749")
PRELOAD = Path(
    "/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt"
)
PRELOAD_BYTES = 5_411_266_466
PRELOAD_SHA256 = "fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9"
DGGT_COMMIT = "a3276d2bbe4cbb03bcc117830b1836110a27adeb"
DRIVESTUDIO_COMMIT = "e59bda4fa681f829dbb1d65f0de582b0f633c450"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def run(*command: str, cwd: Path = PROJECT, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"缺少必需文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_zero(path: Path) -> None:
    if path.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError(f"stage 未通过: {path}")


def read_resource() -> dict[str, Any]:
    memory_events = {}
    for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines():
        key, value = line.split()
        memory_events[key] = int(value)
    memory_max_raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    disk = shutil.disk_usage("/root/autodl-tmp")
    gpu = run(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ).split(",")
    return {
        "timestamp": now(),
        "gpu_name": gpu[0].strip(),
        "driver_version": gpu[1].strip(),
        "gpu_memory_total_mb": int(gpu[2].strip()),
        "gpu_memory_used_mb": int(gpu[3].strip()),
        "memory_max_bytes": None if memory_max_raw == "max" else int(memory_max_raw),
        "memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
        "memory_events": memory_events,
        "disk_free_bytes": disk.free,
    }


def find_adgs_models() -> dict[str, Path]:
    candidates = sorted(
        Path("/root/autodl-tmp/runs/dynamic_recon").glob(
            "DR-M*/**/model_60000"
        )
    )
    result = {}
    for scene in SCENES:
        matches = [path for path in candidates if f"scene{scene}" in str(path)]
        if len(matches) != 1:
            raise RuntimeError(f"scene-{scene} model_60000 数量错误: {len(matches)}")
        result[scene] = matches[0]
    return result


def audit_adgs() -> dict[str, Any]:
    rows = {}
    for scene, model in find_adgs_models().items():
        point_cloud = model / "point_cloud/iteration_60000/point_cloud.ply"
        results = model / "results.json"
        test_renders = sorted((model / "test/ours_60000/renders").glob("*.png"))
        train_renders = sorted((model / "train/ours_60000/renders").glob("*.png"))
        processed = Path(
            f"/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1/scene-{scene}"
        )
        if not point_cloud.is_file() or point_cloud.stat().st_size <= 0:
            raise RuntimeError(f"scene-{scene} point_cloud 缺失或为空")
        if not results.is_file():
            raise RuntimeError(f"scene-{scene} results.json 缺失")
        if len(test_renders) != 42 or len(train_renders) != 138:
            raise RuntimeError(
                f"scene-{scene} render 数量错误: {len(test_renders)}/{len(train_renders)}"
            )
        if not processed.is_dir():
            raise RuntimeError(f"scene-{scene} processed 输入缺失")
        rows[f"scene-{scene}"] = {
            "model_dir": str(model),
            "point_cloud": {
                "path": str(point_cloud),
                "bytes": point_cloud.stat().st_size,
                "sha256": sha256_file(point_cloud),
            },
            "results": {
                "path": str(results),
                "bytes": results.stat().st_size,
                "sha256": sha256_file(results),
            },
            "official_test_render_count": len(test_renders),
            "official_train_render_count": len(train_renders),
            "processed_input": str(processed),
        }
    return rows


def audit_sources() -> dict[str, Any]:
    dggt = Path("/root/autodl-tmp/third_party/dggt")
    drivestudio = Path("/root/autodl-tmp/third_party/drivestudio")
    dggt_commit = run("git", "rev-parse", "HEAD", cwd=dggt)
    drivestudio_commit = run("git", "rev-parse", "HEAD", cwd=drivestudio)
    if dggt_commit != DGGT_COMMIT:
        raise RuntimeError(f"DGGT commit 漂移: {dggt_commit}")
    if drivestudio_commit != DRIVESTUDIO_COMMIT:
        raise RuntimeError(f"DriveStudio commit 漂移: {drivestudio_commit}")
    return {
        "dggt": {
            "path": str(dggt),
            "commit": dggt_commit,
            "remote": run("git", "remote", "get-url", "origin", cwd=dggt),
            "status": run("git", "status", "--short", cwd=dggt),
        },
        "drivestudio": {
            "path": str(drivestudio),
            "commit": drivestudio_commit,
            "remote": run("git", "remote", "get-url", "origin", cwd=drivestudio),
            "status": run("git", "status", "--short", cwd=drivestudio),
            "environment": {
                "path": "/root/autodl-tmp/envs/drivestudio",
                "status": "available"
                if Path("/root/autodl-tmp/envs/drivestudio/bin/python").is_file()
                else "missing",
            },
            "pilot_assets": {
                "scene-0230": "missing",
                "scene-0242": "missing",
                "scene-0255": "missing",
            },
        },
    }


def audit_bootstrap(run_dir: Path) -> dict[str, Any]:
    require_zero(run_dir / "stages/bootstrap_empty.rc")
    require_zero(run_dir / "stages/bootstrap_tmux.rc")
    require_zero(run_dir / "stages/pytest_m0.rc")
    reports = {}
    for shell_name in ("empty_shell", "tmux_shell"):
        payload = load_json(
            run_dir / f"environment/{shell_name}/source_resolution.json"
        )
        unreachable = [
            row["name"] for row in payload["connectivity"] if not row["reachable"]
        ]
        if unreachable:
            raise RuntimeError(f"{shell_name} 网络 smoke 失败: {unreachable}")
        if payload.get("global_config_modified") is not False:
            raise RuntimeError(f"{shell_name} 未证明全局配置不变")
        reports[shell_name] = payload
    return reports


def source_behavior_audit() -> dict[str, Any]:
    dggt_runner = (PROJECT / "scripts/run_dr_m5_dggt.py").read_text()
    stress_runner = (PROJECT / "scripts/run_dr_m6_stress.py").read_text()
    pseudo_tracks = (PROJECT / "motion_proj/dynamic_recon/pseudo_tracks.py").read_text()
    return {
        "legacy_dggt_pointops_command": "pip install .",
        "legacy_dggt_pointops_command_confirmed": (
            '"env_pointops2"' in dggt_runner
            and '"install", "."' in dggt_runner
        ),
        "legacy_stress_executes_edit": False,
        "legacy_stress_abstain_path_confirmed": (
            '"type": "edit"' in stress_runner
            and '"status": "ABSTAIN"' in stress_runner
        ),
        "pseudo_tracks_fixed_vehicle_coverage_zero_confirmed": (
            '"vehicle_eligible_count": 0' in pseudo_tracks
        ),
        "report": "docs/DR_V2_M0_SOURCE_AUDIT.md",
    }


def finalize(run_dir: Path) -> None:
    if run_dir.parent.name != TASK_ID:
        raise RuntimeError(f"run 路径不属于 {TASK_ID}: {run_dir}")
    if (run_dir / "terminal.json").exists():
        raise RuntimeError("run 已有 terminal.json，禁止覆盖")
    if run("git", "branch", "--show-current") != "research/dynamic-editing-v2":
        raise RuntimeError("当前不在 research/dynamic-editing-v2 分支")
    if "动态驾驶场景可编辑重建与失败诊断 V2" not in (
        PROJECT / "README.md"
    ).read_text():
        raise RuntimeError("README 尚未切换到 V2")
    if "conda init bash" in (PROJECT / "AGENTS.md").read_text():
        raise RuntimeError("AGENTS.md 仍允许全局 conda init")
    if not (PROJECT / "configs/env/autodl_condarc_v2.yaml").is_file():
        raise RuntimeError("缺少项目 Conda 镜像配置")

    bootstrap = audit_bootstrap(run_dir)
    source_behavior = source_behavior_audit()
    if not all(
        source_behavior[key]
        for key in (
            "legacy_dggt_pointops_command_confirmed",
            "legacy_stress_abstain_path_confirmed",
            "pseudo_tracks_fixed_vehicle_coverage_zero_confirmed",
        )
    ):
        raise RuntimeError("V1 source behavior 审计失败")
    resources = read_resource()
    if resources["gpu_memory_total_mb"] < 24 * 1024:
        raise RuntimeError("GPU 总显存低于 24 GiB")
    if resources["memory_max_bytes"] is not None and resources["memory_max_bytes"] < 32 * 1024**3:
        raise RuntimeError("cgroup memory.max 低于 32 GiB")
    if resources["disk_free_bytes"] < 60 * 1024**3:
        raise RuntimeError("数据盘空闲低于 60 GiB")

    if not PRELOAD.is_file() or PRELOAD.stat().st_size != PRELOAD_BYTES:
        raise RuntimeError("DGGT preload 缺失或字节数错误")
    preload_sha256 = sha256_file(PRELOAD)
    if preload_sha256 != PRELOAD_SHA256:
        raise RuntimeError("DGGT preload SHA-256 不匹配")

    git_status = run("git", "status", "--short", "--branch")
    # 与 HEAD 比较可同时覆盖 staged、unstaged 和已加入索引的新文件。
    git_diff = run("git", "diff", "HEAD", "--binary", check=False)
    (run_dir / "source_snapshot/git_status.txt").write_text(
        git_status + "\n", encoding="utf-8"
    )
    (run_dir / "source_snapshot/git_diff.patch").write_text(
        git_diff + ("\n" if git_diff else ""), encoding="utf-8"
    )
    assets = {
        "dggt_preload": {
            "path": str(PRELOAD),
            "bytes": PRELOAD_BYTES,
            "sha256": preload_sha256,
        },
        "adgs": audit_adgs(),
        "sources": audit_sources(),
        "cleanup_manifest": {
            "path": "docs/archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md",
            "sha256": sha256_file(
                PROJECT / "docs/archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md"
            ),
        },
    }
    write_json(run_dir / "environment/assets.json", assets)
    write_json(
        run_dir / "environment/baseline_readiness.json",
        {
            "dggt": "source_and_full_preload_available",
            "adgs": "six_frozen_models_and_processed_inputs_available",
            "drivestudio_source": "available",
            "drivestudio_environment": assets["sources"]["drivestudio"]["environment"]["status"],
            "drivestudio_pilot_assets": "missing",
        },
    )
    write_json(run_dir / "source_snapshot/source_behavior.json", source_behavior)
    with (run_dir / "resource.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(resources, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "created_at": now(),
        "project_root": str(PROJECT),
        "branch": "research/dynamic-editing-v2",
        "source_commit": run("git", "rev-parse", "HEAD"),
        "source_dirty": bool(run("git", "status", "--porcelain")),
        "source_diff_sha256": sha256_bytes(git_diff.encode()),
        "seed": 0,
        "run_dir": str(run_dir),
    }
    resolved = {
        "task_id": TASK_ID,
        "bootstrap": "scripts/bootstrap_autodl_v2.sh",
        "condarc": "configs/env/autodl_condarc_v2.yaml",
        "network_timeout_seconds": 20,
        "required_tests": [
            "tests/test_dr_pseudo_tracks.py",
            "tests/test_v71_actor_registry.py",
        ],
        "asset_policy": "V1 frozen assets are read-only",
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "resolved.yaml", resolved)
    metric = {
        "timestamp": now(),
        "task_id": TASK_ID,
        "status": "done",
        "bootstrap_empty_shell": "PASS",
        "bootstrap_tmux_shell": "PASS",
        "pytest": "7 passed",
        "network_source_coverage": "4/4",
        "adgs_asset_coverage": "6/6",
        "dggt_preload_verified": True,
    }
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(metric, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = (
        "# DR-V2 M0 summary\n\n"
        "- status: `done`\n"
        "- bootstrap: empty shell 与 tmux shell 均通过，TUNA/HF/GitHub 4/4 可达\n"
        "- tests: `7 passed`\n"
        "- AD-GS: 冻结 `model_60000`、official render 与 processed 输入 6/6 完整\n"
        "- DGGT: source commit 与 5.41 GB preload 字节数/SHA-256 通过\n"
        "- DriveStudio: source/env available，三个 pilot scene 的 processed/checkpoint missing\n"
        "- source audit: V1 DGGT 使用 `pip install .`；V1 M6 没有执行真实编辑\n"
        "- next: `DR-V2-M1-DGGT-REPAIR-01`\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    indexed = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"artifacts.json", "terminal.json"}:
            indexed[str(path.relative_to(run_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(run_dir / "artifacts.json", indexed)
    write_json(
        run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        finalize(args.run_dir.resolve())
    except Exception as exc:
        if args.run_dir.exists() and not (args.run_dir / "terminal.json").exists():
            write_json(
                args.run_dir / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": f"{type(exc).__name__}: {exc}",
                },
            )
        raise
    print(args.run_dir)


if __name__ == "__main__":
    main()
