"""从已完成 renderer、但汇总失败的 R3 run 恢复正式分析。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from motion_proj.worldsim_v6.r3_support import analyze_support_deviation


TASK_ID = "WS-V6-R3-SUPPORT-DEVIATION-01"


class R3RecoveryError(RuntimeError):
    """恢复来源不满足不可变证据合同。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_render_source(source_run: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    terminal_path = source_run / "TERMINAL.json"
    terminal = _read_json(terminal_path)
    if terminal.get("status") != "failed" or "numpy.ndarray" not in terminal.get("error", ""):
        raise R3RecoveryError("来源 run 不是冻结的 structured-array 汇总失败")
    expected_per_worker = len(config["cohort"]["scenes"][0]["source_frame_indices"]) * (
        len(config["camera_deviation_profile"]["lateral_offsets_m"])
        + 1
        + len(config["actor_edit_profile"]["operations"])
    )
    worker_rows = []
    content_hashes = []
    auxiliary_inputs = []
    for scene_row in config["cohort"]["scenes"]:
        scene = scene_row["scene"]
        adapter = source_run / "development_adapters" / scene
        partition = _read_json(adapter / "partition.json")
        if "heldout" in partition["included_partitions"] or any(
            row["partition"] == "heldout" for row in partition["rows"]
        ):
            raise R3RecoveryError(f"{scene} adapter 含 heldout")
        for required in (
            "adapter_manifest.json",
            "R3_INFERENCE_ONLY_DEPTH_PLACEHOLDERS.json",
            "R3_ADGS_POINT_CLOUD_BINDING.json",
        ):
            if not (adapter / required).is_file():
                raise R3RecoveryError(f"{scene} 缺少 adapter audit：{required}")
        street_root = source_run / "renders" / scene / "streetgs"
        support_paths = [street_root / "TRAINING_CAMERA_SUPPORT.json"] + [
            street_root / f"support_frame_{int(frame):03d}.npz"
            for frame in scene_row["source_frame_indices"]
        ]
        for path in support_paths:
            if not path.is_file():
                raise R3RecoveryError(f"{scene} 缺少 support 输入：{path.name}")
            auxiliary_inputs.append(
                {
                    "path": str(path.relative_to(source_run)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        for frontend in ("streetgs", "ad_gs"):
            root = source_run / "renders" / scene / frontend
            audit_path = root / "AUDIT.json"
            render_map_path = root / "RENDER_MAP.jsonl"
            audit = _read_json(audit_path)
            if audit.get("training_started") is not False or audit.get("confirmation_content_read") is not False:
                raise R3RecoveryError(f"{scene}/{frontend} 训练或确认集 audit 失败")
            if frontend == "streetgs":
                if audit["checkpoint_sha256_before"] != audit["checkpoint_sha256_after"]:
                    raise R3RecoveryError(f"{scene}/{frontend} checkpoint 漂移")
            elif audit["checkpoint_sha256_before"] != audit["checkpoint_sha256_after"]:
                raise R3RecoveryError(f"{scene}/{frontend} checkpoint bundle 漂移")
            rows = [
                json.loads(line)
                for line in render_map_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if len(rows) != expected_per_worker or audit["render_count"] != expected_per_worker:
                raise R3RecoveryError(f"{scene}/{frontend} render 数量不完整")
            for row in rows:
                path = (root / row["path"]).resolve()
                if path.parent != root.resolve() or _sha256(path) != row["sha256"]:
                    raise R3RecoveryError(f"{scene}/{frontend} render hash 漂移：{row['path']}")
                content_hashes.append(row["sha256"])
            worker_rows.append(
                {
                    "scene": scene,
                    "frontend": frontend,
                    "render_count": len(rows),
                    "audit_sha256": _sha256(audit_path),
                    "render_map_sha256": _sha256(render_map_path),
                    "peak_torch_allocated_bytes": audit["peak_torch_allocated_bytes"],
                    "peak_torch_reserved_bytes": audit["peak_torch_reserved_bytes"],
                }
            )
    aggregate = hashlib.sha256("\n".join(sorted(content_hashes)).encode("ascii")).hexdigest()
    return {
        "schema_version": "worldsim_v6.r3_reused_render_evidence.v1",
        "source_run": str(source_run),
        "source_terminal_sha256": _sha256(terminal_path),
        "source_terminal_status": "failed",
        "source_failure_stage": "post_render_structured_array_analysis",
        "render_count": len(content_hashes),
        "render_content_hash_aggregate_sha256": aggregate,
        "all_render_hashes_reverified": True,
        "auxiliary_support_inputs": auxiliary_inputs,
        "checkpoint_before_after_invariant_passed": True,
        "training_started": False,
        "confirmation_content_read": False,
        "workers": worker_rows,
    }


def recover(
    repo_root: Path,
    config_path: Path,
    source_run: Path,
    source_project_commit: str,
    run_root: Path,
) -> Path:
    repo_root = repo_root.resolve()
    source_run = source_run.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R3RecoveryError("正式恢复禁止 dirty source")
    current_commit = _git(repo_root, "rev-parse", "HEAD")
    _git(repo_root, "cat-file", "-e", f"{source_project_commit}^{{commit}}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R3RecoveryError("R3 task_id 漂移")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__support-deviation-analysis-recovery-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        evidence = _verify_render_source(source_run, config)
        evidence["source_project_commit"] = source_project_commit
        evidence["analysis_project_commit"] = current_commit
        _write_json(run_dir / "REUSED_RENDER_EVIDENCE.json", evidence)
        case_rows, ranking, actor_rows, forward_rows = analyze_support_deviation(source_run, config)
        _write_jsonl(run_dir / "PER_CASE_METRICS.jsonl", case_rows)
        _write_json(run_dir / "SUPPORT_RANKING.json", ranking)
        _write_jsonl(run_dir / "ACTOR_EDIT_EFFECTS.jsonl", actor_rows)
        _write_jsonl(run_dir / "FORWARD_EXTENSION_METRICS.jsonl", forward_rows)
        peak_reserved = max(row["peak_torch_reserved_bytes"] for row in evidence["workers"])
        resources = {
            "schema_version": "worldsim_v6.r3_recovery_resource_audit.v1",
            "source_worker_peak_torch_reserved_bytes": peak_reserved,
            "source_worker_peak_torch_reserved_mib": peak_reserved / (1024**2),
            "frozen_limit_mib": config["resources"]["maximum_peak_gpu_memory_mib"],
            "passed": peak_reserved / (1024**2) <= config["resources"]["maximum_peak_gpu_memory_mib"],
            "analysis_gpu_used": False,
        }
        _write_json(run_dir / "RESOURCE_AUDIT.json", resources)
        summary = {
            "schema_version": "worldsim_v6.r3_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "hypothesis_outcome": "accepted" if ranking["gate"]["passed"] else "rejected",
            "method_decision": ranking["decision"],
            "source_commit": current_commit,
            "render_source_commit": source_project_commit,
            "render_source_run": str(source_run),
            "recovery_kind": "analysis_only_after_all_render_hashes_reverified",
            "scene_count": ranking["scene_count"],
            "frontend_count": ranking["frontend_count"],
            "training_started": False,
            "confirmation_content_read": False,
            "development_content_read": True,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = (
            "REUSED_RENDER_EVIDENCE.json",
            "PER_CASE_METRICS.jsonl",
            "SUPPORT_RANKING.json",
            "ACTOR_EDIT_EFFECTS.jsonl",
            "FORWARD_EXTENSION_METRICS.jsonl",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        )
        manifest = {
            "schema_version": "worldsim_v6.r3_run_manifest.v1",
            "files": {
                relative: {
                    "bytes": (run_dir / relative).stat().st_size,
                    "sha256": _sha256(run_dir / relative),
                }
                for relative in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "done",
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/worldsim_v6/r3_support_deviation_v1.yaml")
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--source-project-commit", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    recover(args.repo_root, args.config, args.source_run, args.source_project_commit, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
