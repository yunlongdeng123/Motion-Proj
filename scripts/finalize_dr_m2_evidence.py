#!/usr/bin/env python3
"""闭合 DR-M2-ENV-ASSET-01 的正式 run contract 与证据摘要。"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
DATA_MANIFEST = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/"
    "adgs_nuscenes_v1_manifest.json"
)
FRAME_TABLES = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/"
    "adgs_nuscenes_v1_frame_tables.json"
)
MEMBER_SHARDS = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/"
    "adgs_nuscenes_v1_member_shards.tsv"
)
REQUIRED_MEMBERS = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/"
    "adgs_nuscenes_v1_required_members.txt"
)
SOURCE_FILES = [
    PROJECT / "scripts/build_adgs_nuscenes_assets.py",
    PROJECT / "scripts/audit_adgs_nuscenes_assets.py",
    PROJECT / "scripts/env_report_adgs.sh",
    PROJECT / "scripts/env_report_all.sh",
    PROJECT / "scripts/smoke_adgs_env.py",
    PROJECT / "scripts/smoke_dpt_env.py",
    PROJECT / "scripts/smoke_sam_env.py",
    PROJECT / "scripts/smoke_cotracker_env.py",
    PROJECT / "scripts/smoke_grounding_dino_hf.py",
    PROJECT / "scripts/finalize_dr_m2_evidence.py",
    PROJECT / "compatibility/AD-GS-2026-07-27.patch",
    PROJECT / "docs/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md",
    Path("/root/autodl-tmp/third_party/AD-GS/scripts/flow.py"),
    Path("/root/autodl-tmp/third_party/AD-GS/scripts/run-dpt.py"),
    Path("/root/autodl-tmp/third_party/AD-GS/scripts/semantic.py"),
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


def command(*args):
    return subprocess.check_output(
        list(args),
        cwd=str(PROJECT),
        universal_newlines=True,
    ).strip()


def read_rc(path):
    return int(path.read_text().strip())


def parse_peak_gpu_mb(log):
    text = log.read_text(errors="replace")
    patterns = [
        r"max_allocated MB\s+([0-9.]+)",
        r"peak GPU mem\s+([0-9.]+)\s*MB",
    ]
    values = []
    for pattern in patterns:
        values.extend(float(value) for value in re.findall(pattern, text))
    return max(values) if values else None


def worktree_snapshot():
    head = command("git", "rev-parse", "HEAD")
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=str(PROJECT),
    )
    untracked_raw = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(PROJECT),
    )
    untracked = []
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        path = PROJECT / rel
        if path.is_file():
            untracked.append({
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    payload = {
        "head": head,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked": untracked,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def resource_metrics(run_dir):
    samples_path = run_dir / "resource_samples.tsv"
    rows = []
    if samples_path.is_file():
        for line in samples_path.read_text().splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) != 6:
                continue
            rows.append({
                "timestamp": fields[0],
                "memory_current_bytes": int(fields[1]),
                "memory_max_bytes": int(fields[2]),
                "disk_avail_bytes": int(fields[3]),
                "oom": int(fields[4]),
                "oom_kill": int(fields[5]),
            })
    return {
        "n_samples": len(rows),
        "peak_memory_current_bytes": max(
            (row["memory_current_bytes"] for row in rows), default=None
        ),
        "memory_max_bytes": rows[-1]["memory_max_bytes"] if rows else None,
        "minimum_disk_avail_bytes": min(
            (row["disk_avail_bytes"] for row in rows), default=None
        ),
        "oom_delta": (
            rows[-1]["oom"] - rows[0]["oom"] if len(rows) >= 2 else 0
        ),
        "oom_kill_delta": (
            rows[-1]["oom_kill"] - rows[0]["oom_kill"] if len(rows) >= 2 else 0
        ),
    }


def write_license_audit(path):
    path.write_text(
        """# DR-M2 许可证审计

- AD-GS `9a208512...`：仓库顶层未发现 LICENSE/COPYING；仅限当前内部研究复现，禁止随本项目再分发其源码或产物包，发布前需作者许可或补充书面条款。
- PyTorch3D `3145dd4...`：仓库顶层 LICENSE 声明 BSD License。
- Depth Anything V2 `a561b84...`：仓库顶层 LICENSE 声明 Apache License 2.0。
- Grounded-SAM-2 `b7a9c29...`：聚合仓库顶层为 Apache License 2.0，同时保留 GroundingDINO、SAM2 等组件独立 LICENSE。
- CoTracker3 `82e02e8...`：仓库 LICENSE.md 声明 CC BY-NC 4.0；固定权重同样按非商业研究边界处理。
- nuScenes：按项目计划登记为 CC BY-NC-SA 4.0 与官方附加条款；原始数据不进入 Git 或对外复现包。

本文件是工程 provenance 记录，不构成法律意见。AD-GS 许可证缺失不会阻止本机内部 smoke，但会阻止源码/模型再分发。
"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--asset-audit", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    asset_audit_path = Path(args.asset_audit)

    required_rc = {
        "asset_build": run_dir / "asset_build.rc",
        "adgs": run_dir / "environment/adgs/smoke.rc",
        "adgs_dpt": run_dir / "environment/adgs-dpt/smoke.rc",
        "adgs_sam": run_dir / "environment/adgs-sam/smoke.rc",
        "grounding_dino_hf": (
            run_dir / "environment/adgs-sam/grounding_dino_hf_smoke.rc"
        ),
        "cotracker": run_dir / "environment/cotracker/smoke.rc",
    }
    failures = []
    exit_codes = {}
    for name, path in required_rc.items():
        if not path.is_file():
            failures.append("缺少退出码: {}".format(path))
            continue
        exit_codes[name] = read_rc(path)
        if exit_codes[name] != 0:
            failures.append("{} 退出码不是 0".format(name))

    if not asset_audit_path.is_file():
        failures.append("缺少资产审计 JSON")
        asset_audit = {}
    else:
        asset_audit = json.loads(asset_audit_path.read_text())
        if asset_audit.get("status") != "done":
            failures.append("资产结构审计未通过")

    data_manifest = json.loads(DATA_MANIFEST.read_text())
    if not data_manifest.get("complete"):
        failures.append("数据 manifest 未闭合")

    env_logs = {
        "adgs": run_dir / "environment/adgs/smoke.log",
        "adgs_dpt": run_dir / "environment/adgs-dpt/smoke.log",
        "adgs_sam": run_dir / "environment/adgs-sam/smoke.log",
        "grounding_dino_hf": (
            run_dir / "environment/adgs-sam/grounding_dino_hf_smoke.log"
        ),
        "cotracker": run_dir / "environment/cotracker/smoke.log",
    }
    peak_gpu = {
        name: parse_peak_gpu_mb(path)
        for name, path in env_logs.items()
        if path.is_file()
    }

    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir(parents=True, exist_ok=True)
    source_hashes = []
    for source in SOURCE_FILES:
        if not source.is_file():
            failures.append("缺少 source snapshot 输入: {}".format(source))
            continue
        destination = source_snapshot / source.name
        shutil.copy2(str(source), str(destination))
        source_hashes.append({
            "source": str(source),
            "snapshot": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
    executed_builder = (
        run_dir / "executed_source/build_adgs_nuscenes_assets.py"
    )
    if executed_builder.is_file():
        source_hashes.append({
            "source": str(executed_builder),
            "snapshot": str(executed_builder),
            "bytes": executed_builder.stat().st_size,
            "sha256": sha256_file(executed_builder),
            "role": "executed_source",
        })
    else:
        failures.append("缺少实际执行时的 asset builder source snapshot")

    license_audit = run_dir / "environment/license_audit.md"
    write_license_audit(license_audit)
    worktree = worktree_snapshot()
    gpu = command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    )
    resource = resource_metrics(run_dir)

    upstream_commits = {
        "AD-GS": command(
            "git", "-C", "/root/autodl-tmp/third_party/AD-GS", "rev-parse", "HEAD"
        ),
        "pytorch3d-v0.7.2": command(
            "git", "-C", "/root/autodl-tmp/third_party/pytorch3d-v0.7.2",
            "rev-parse", "HEAD"
        ),
        "Depth-Anything-V2": command(
            "git", "-C", "/root/autodl-tmp/third_party/Depth-Anything-V2",
            "rev-parse", "HEAD"
        ),
        "Grounded-SAM-2": command(
            "git", "-C", "/root/autodl-tmp/third_party/Grounded-SAM-2",
            "rev-parse", "HEAD"
        ),
        "co-tracker": command(
            "git", "-C", "/root/autodl-tmp/third_party/co-tracker",
            "rev-parse", "HEAD"
        ),
    }
    checkpoint_hashes = {
        "depth_anything_v2_vitl": sha256_file(
            Path("/root/autodl-tmp/checkpoints/depth_anything_v2/"
                 "depth_anything_v2_vitl.pth")
        ),
        "sam2_1_hiera_large": sha256_file(
            Path("/root/autodl-tmp/third_party/Grounded-SAM-2/"
                 "checkpoints/sam2.1_hiera_large.pt")
        ),
        "groundingdino_swint_ogc": sha256_file(
            Path("/root/autodl-tmp/third_party/Grounded-SAM-2/"
                 "gdino_checkpoints/groundingdino_swint_ogc.pth")
        ),
        "cotracker3_scaled_offline": sha256_file(
            Path("/root/autodl-tmp/checkpoints/cotracker3/scaled_offline.pth")
        ),
    }
    hf_smoke_log = env_logs["grounding_dino_hf"]
    if hf_smoke_log.is_file():
        match = re.search(
            r"snapshot fingerprint\s+([0-9a-f]{64})",
            hf_smoke_log.read_text(errors="replace"),
        )
        if match:
            checkpoint_hashes["grounding_dino_hf_snapshot"] = match.group(1)
        else:
            failures.append("Grounding DINO HF smoke 缺少 snapshot fingerprint")
    compatibility_patch_sha256 = sha256_file(
        PROJECT / "compatibility/AD-GS-2026-07-27.patch"
    )

    resolved = {
        "task_id": "DR-M2-ENV-ASSET-01",
        "protocol": {
            "scenes": data_manifest["protocol"]["scenes"],
            "sensors_in_upstream_order": data_manifest["protocol"]["sensors"],
            "first_frame": 10,
            "last_frame": 69,
            "resolution": [900, 1600],
            "tar_scan_workers": 5,
        },
        "paths": {
            "project": str(PROJECT),
            "data_manifest": str(DATA_MANIFEST),
            "raw_subset": data_manifest["raw_subset"],
            "run_dir": str(run_dir),
        },
        "resources": {
            "minimum_gpu_vram_gib": 24,
            "minimum_memory_gib": 32,
            "minimum_start_disk_free_gib": 60,
            "stop_disk_free_gib": 20,
        },
        "upstream_commits": upstream_commits,
        "compatibility_patch_sha256": compatibility_patch_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "seed": None,
    }
    resolved_bytes = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    config_fingerprint = hashlib.sha256(resolved_bytes).hexdigest()
    (run_dir / "resolved.yaml").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n"
    )

    inputs = {
        "data_manifest": str(DATA_MANIFEST),
        "data_manifest_sha256": sha256_file(DATA_MANIFEST),
        "frame_tables_sha256": sha256_file(FRAME_TABLES),
        "required_members_sha256": sha256_file(REQUIRED_MEMBERS),
        "member_shards_sha256": sha256_file(MEMBER_SHARDS),
        "checkpoints": checkpoint_hashes,
    }
    atomic_json(run_dir / "inputs.json", inputs)

    metrics = {
        "environment_smoke": {
            name: {"exit_code": exit_codes.get(name), "peak_gpu_memory_mb": peak_gpu.get(name)}
            for name in [
                "adgs",
                "adgs_dpt",
                "adgs_sam",
                "grounding_dino_hf",
                "cotracker",
            ]
        },
        "assets": {
            "n_required": data_manifest.get("n_required"),
            "n_present": data_manifest.get("n_present"),
            "complete": data_manifest.get("complete"),
            "payload": asset_audit.get("payload"),
            "auxiliary": asset_audit.get("auxiliary"),
            "per_scene": asset_audit.get("per_scene"),
        },
        "resources": resource,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"stage": "m2_final", **metrics}, ensure_ascii=False) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "task_id": "DR-M2-ENV-ASSET-01",
        "instance_id": run_dir.name,
        "project_commit": worktree["head"],
        "worktree_fingerprint": worktree["fingerprint"],
        "worktree": worktree,
        "config_fingerprint": config_fingerprint,
        "data_manifest_sha256": inputs["data_manifest_sha256"],
        "seed": None,
        "gpu": gpu,
        "upstream_commits": upstream_commits,
        "compatibility_patch_sha256": compatibility_patch_sha256,
        "source_snapshot": source_hashes,
    }
    atomic_json(run_dir / "manifest.json", manifest)

    artifact_paths = [
        run_dir / "manifest.json",
        run_dir / "resolved.yaml",
        run_dir / "inputs.json",
        run_dir / "metrics.json",
        run_dir / "metrics.jsonl",
        asset_audit_path,
        run_dir / "asset_build.log",
        run_dir / "resource_samples.tsv",
        license_audit,
        DATA_MANIFEST,
        FRAME_TABLES,
        MEMBER_SHARDS,
        REQUIRED_MEMBERS,
    ]
    artifacts = []
    for path in artifact_paths:
        if path.is_file():
            artifacts.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_json(run_dir / "artifacts.json", {"artifacts": artifacts})

    old_summary = run_dir / "summary.md"
    if old_summary.is_file() and not (run_dir / "environment_summary.md").exists():
        shutil.copy2(str(old_summary), str(run_dir / "environment_summary.md"))

    status = "done" if not failures else "blocked"
    summary = """# DR-M2-ENV-ASSET-01 Summary

- 终态：`{status}`
- 实例：`{instance}`
- 项目 commit：`{commit}`
- worktree fingerprint：`{worktree_fingerprint}`
- config fingerprint：`{config_fingerprint}`
- GPU：`{gpu}`
- 环境 smoke：AD-GS / DPT / Grounded-SAM-2 / pinned Grounding DINO HF / CoTracker3 均以独立退出码记录
- 精确资产：`{present}/{required}`，六场景各 180 RGB + 60 nearest LiDAR
- nuScenes auxiliary：4 个静态 map masks 已逐文件哈希审计
- 数据 manifest SHA-256：`{data_hash}`
- 资源峰值（cgroup 采样）：`{peak_memory}` bytes；OOM/OOM-kill delta=`{oom_delta}/{oom_kill_delta}`
- 许可证边界：AD-GS 顶层许可证缺失，当前只允许内部研究复现，不进入再分发包
- 失败：{failures}

下一门禁是 `DR-M3-ADGS-0230-01`；M3 必须从 100-iteration smoke 开始，不能直接把工程 smoke 当论文复现结果。
""".format(
        status=status,
        instance=run_dir.name,
        commit=worktree["head"],
        worktree_fingerprint=worktree["fingerprint"],
        config_fingerprint=config_fingerprint,
        gpu=gpu,
        present=data_manifest.get("n_present"),
        required=data_manifest.get("n_required"),
        data_hash=inputs["data_manifest_sha256"],
        peak_memory=resource.get("peak_memory_current_bytes"),
        oom_delta=resource.get("oom_delta"),
        oom_kill_delta=resource.get("oom_kill_delta"),
        failures="无" if not failures else "；".join(failures),
    )
    old_summary.write_text(summary)
    (run_dir / "stdout.log").write_text(
        "环境日志位于 environment/*/smoke.log；资产日志位于 asset_build.log。\n"
    )
    (run_dir / "stderr.log").write_text(
        "" if not failures else "\n".join(failures) + "\n"
    )

    finished = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    terminal = {
        "status": status,
        "finished_at": finished,
        "exit_code": 0 if not failures else 2,
        "failures": failures,
    }
    atomic_json(run_dir / "terminal.json", terminal)

    final_artifact_paths = artifact_paths + [
        run_dir / "summary.md",
        run_dir / "environment_summary.md",
        run_dir / "stdout.log",
        run_dir / "stderr.log",
        run_dir / "terminal.json",
    ] + [Path(item["snapshot"]) for item in source_hashes]
    final_artifacts = []
    for path in final_artifact_paths:
        if path.is_file():
            final_artifacts.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_json(run_dir / "artifacts.json", {"artifacts": final_artifacts})

    print(summary)
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
