#!/usr/bin/env python3
"""依据 M6 failure matrix 完成 M7 唯一假设与 novelty fail-closed 裁决。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT = Path("/root/autodl-tmp/motion_proj")
TASK_ID = "DR-M7-HYPOTHESIS-01"
AUDIT_DATE = "2026-07-29"
CANDIDATE = {
    "decision_table_branch": "A",
    "failure": "原轨迹内的身份/生命周期不可用",
    "hypothesis": "可编辑运动表示与轨迹不确定性",
    "provisional_mechanism": (
        "从 SAM pseudo masks 恢复跨时间实例身份，将 Gaussian 绑定到持久 actor，"
        "并以置信度/ABSTAIN 约束轨迹编辑"
    ),
}
NOVELTY_MATRIX = [
    {
        "work": "InstDrive: Instance-Aware 3D Gaussian Splatting for Driving Scenes",
        "year": 2025,
        "primary_source": "https://arxiv.org/abs/2508.12015",
        "official_claim": (
            "以 SAM masks 作 pseudo ground truth，学习 2D/3D instance identity，"
            "用 voxel consistency 与 codebook 得到动态驾驶场景离散身份并支持交互编辑"
        ),
        "overlap": "direct",
        "overlapped_candidate_components": [
            "SAM pseudo identity",
            "3D Gaussian instance binding",
            "driving-scene interactive editing",
        ],
    },
    {
        "work": "Director: Instance-aware Gaussian Splatting for Dynamic Scene Modeling and Understanding",
        "year": 2026,
        "primary_source": "https://arxiv.org/abs/2604.01678",
        "official_claim": (
            "用时间对齐实例 masks 监督 4D Gaussian instance-consistent semantics，"
            "并结合 optical flow 稳定运动和身份"
        ),
        "overlap": "direct",
        "overlapped_candidate_components": [
            "temporal identity consistency",
            "motion-aware Gaussian assignment",
            "dynamic-scene instance representation",
        ],
    },
    {
        "work": "OmniRe: Omni Urban Scene Reconstruction",
        "year": 2025,
        "primary_source": "https://openreview.net/forum?id=9cwxZxJixB",
        "official_claim": (
            "以 3DGS scene graph 和 canonical actor representations 分解车辆、"
            "行人等动态对象，并明确支持动态 actor simulation"
        ),
        "overlap": "direct",
        "overlapped_candidate_components": [
            "persistent actor nodes",
            "object-centric Gaussians",
            "driving simulation",
        ],
    },
    {
        "work": "HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles",
        "year": 2026,
        "primary_source": "https://arxiv.org/abs/2602.21333",
        "official_claim": (
            "将场景重建为可编辑 Gaussian Splats/Meshes，支持任意车辆轨迹和对象操作，"
            "并提供 agent-level trajectory/object editing benchmark"
        ),
        "overlap": "direct",
        "overlapped_candidate_components": [
            "trajectory editing",
            "vehicle manipulation",
            "Gaussian driving edit benchmark",
        ],
    },
    {
        "work": "G2Editor: Realistic and Controllable 3D Gaussian-Guided Object Editing",
        "year": 2025,
        "primary_source": "https://arxiv.org/abs/2508.20471",
        "official_claim": (
            "统一支持驾驶视频对象 reposition/insert/delete，并以 3D layout 重建"
            "非目标遮挡区域"
        ),
        "overlap": "direct",
        "overlapped_candidate_components": [
            "object reposition/delete",
            "occlusion-aware completion",
            "non-target preservation",
        ],
    },
]


def now() -> str:
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(str(temporary), str(path))


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
        )


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def initialize(run_dir: Path, m6_run: Path) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run 目录非空，禁止覆盖: {run_dir}")
    m6_summary = load_json(m6_run / "summary.json")
    m6_terminal = load_json(m6_run / "terminal.json")
    if m6_terminal.get("status") != "done" or m6_summary.get("status") != "done":
        raise RuntimeError("M6 不是 done")
    stable = m6_summary.get("stable_failure", {})
    if stable.get("type") != "persistent_object_identity_unavailable":
        raise RuntimeError(f"M6 failure type 不匹配: {stable}")
    if int(stable.get("scene_count", 0)) < 3:
        raise RuntimeError(f"M6 failure 未跨 3 scenes: {stable}")

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir()
    runner = Path(__file__).resolve()
    snapshot = snapshot_dir / runner.name
    shutil.copy2(runner, snapshot)
    project_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(PROJECT), "status", "--short"], text=True
    )
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "m6_run": str(m6_run),
        "m6_stable_failure": stable,
        "candidate": CANDIDATE,
        "novelty_audit_date": AUDIT_DATE,
        "novelty_matrix": NOVELTY_MATRIX,
        "decision_rule": (
            "若候选主机制已被 instance-aware driving GS、actor scene graph 或"
            "trajectory/object editing 工作直接覆盖，则 M7=rejected，不写方法、"
            "不注册事后 primary endpoint、不启动 M8/M9"
        ),
    }
    resolved["config_fingerprint"] = canonical_sha256(resolved)
    atomic_json(run_dir / "resolved.yaml", resolved)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "started_at": now(),
        "project_commit": project_commit,
        "project_git_status": git_status.splitlines(),
        "project_git_status_sha256": hashlib.sha256(
            git_status.encode("utf-8")
        ).hexdigest(),
        "config_fingerprint": resolved["config_fingerprint"],
        "m6_summary_sha256": sha256_file(m6_run / "summary.json"),
        "source_snapshot": {
            "path": str(snapshot),
            "bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    return resolved


def finalize(run_dir: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    for row in NOVELTY_MATRIX:
        append_jsonl(
            run_dir / "metrics.jsonl",
            {"type": "novelty_overlap", "audit_date": AUDIT_DATE, **row},
        )
    direct_overlap_count = sum(
        row["overlap"] == "direct" for row in NOVELTY_MATRIX
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "execution_status": "done",
        "research_status": "rejected",
        "completed_at": now(),
        "m6_failure_gate_passed": True,
        "candidate_branch": CANDIDATE["decision_table_branch"],
        "candidate_hypothesis": CANDIDATE["hypothesis"],
        "novelty_direct_overlap_count": direct_overlap_count,
        "novelty_gate_passed": False,
        "rejection_reason": (
            "候选 A 的持久实例身份、actor-centric Gaussian binding、时序一致性"
            "与轨迹/对象编辑核心机制已被 InstDrive、Director、OmniRe、"
            "HorizonForge 与 G2Editor 直接覆盖；confidence/ABSTAIN 是评测与"
            "安全护栏，不构成独立技术 delta，当前差异只剩 AD-GS 适配工程。"
        ),
        "primary_endpoint_preregistered": False,
        "primary_endpoint_reason": "novelty gate 先失败，禁止事后注册 endpoint",
        "m8": {
            "status": "rejected",
            "authorized": False,
            "method_code_written": False,
            "seeds_run": 0,
            "reason": "M7 novelty gate failed",
        },
        "m9": {
            "status": "rejected",
            "authorized": False,
            "blind_samples": 0,
            "human_verdict": None,
            "reason": "M8 not authorized; no method result exists to review",
        },
        "retained_contribution": (
            "AD-GS 官方六场景复现、DGGT upstream smoke 证据，以及"
            "AD-GS pseudo identity/checkpoint identity collapse 的负结果"
        ),
        "next_action": "停止该方法路线；仅保留复现与负结果，不重命名为创新模块",
    }
    atomic_json(run_dir / "novelty_matrix.json", NOVELTY_MATRIX)
    atomic_json(run_dir / "downstream_stop.json", {
        "m8": summary["m8"],
        "m9": summary["m9"],
    })
    atomic_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m6-run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    try:
        resolved = initialize(run_dir, Path(args.m6_run))
        summary = finalize(run_dir, resolved)
    except Exception as exc:
        if run_dir.exists():
            atomic_json(
                run_dir / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": f"{type(exc).__name__}: {exc}",
                },
            )
        raise
    atomic_json(
        run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
