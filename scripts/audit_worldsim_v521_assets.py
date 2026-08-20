#!/usr/bin/env python3
"""执行 V5.2.1 P1 exact 资产 census 与 quality-blind partition freeze。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v521.asset_audit import (
    AssetAuditError,
    audit_bundle,
    audit_file,
    ensure_matched_asset,
    enumerate_quality_blind_targets,
    freeze_summary,
)
from motion_proj.worldsim_v521.protocol import (
    atomic_json,
    atomic_jsonl,
    atomic_text,
    inventory_files,
    sha256_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssetAuditError(f"配置根节点非法：{path}")
    return value


def _git(path: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise AssetAuditError(process.stderr.strip() or f"git {' '.join(arguments)} failed")
    return process.stdout.strip()


def _source_record(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(spec["root"]))
    expected_commit = str(spec["commit"])
    patch = PROJECT_ROOT / str(spec["patch"])
    row = {
        "name": name,
        "root": str(root),
        "expected_commit": expected_commit,
        "observed_commit": _git(root, "rev-parse", "HEAD"),
        "worktree_status": _git(root, "status", "--short"),
        "patch_path": str(patch),
        "patch_bytes": patch.stat().st_size if patch.is_file() else None,
        "patch_sha256": sha256_file(patch) if patch.is_file() else None,
        "expected_patch_sha256": spec["patch_sha256"],
    }
    row["state"] = (
        "RECONSTRUCTABLE_EXACT_ISOLATED"
        if row["observed_commit"] == expected_commit and row["patch_sha256"] == row["expected_patch_sha256"]
        else "PRESENT_HASH_MISMATCH"
    )
    return row


def _historical_adgs(metrics_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    blockers = []
    for scene in metrics["scenes"]:
        files = scene["checkpoint"]
        point = audit_file(files["point_cloud.ply"])
        other_files = {}
        for name in ("deform.pth", "env.pth"):
            spec = files[name]
            path = Path(spec["path"])
            other_files[name] = {
                "path": str(path),
                "expected_bytes": int(spec["bytes"]),
                "present": path.is_file(),
                "state": "PRESENT_EXACT" if path.is_file() and path.stat().st_size == int(spec["bytes"]) else "MISSING_BUT_MANIFESTED",
            }
        states = {point["state"], *(value["state"] for value in other_files.values())}
        state = "PRESENT_EXACT" if states == {"PRESENT_EXACT"} else "MISSING_BUT_MANIFESTED"
        row = {
            "base": "adgs",
            "protocol": "native_historical",
            "scene": scene["scene"],
            "state": state,
            "source_run": scene["source_run"],
            "checkpoint": {"point_cloud.ply": point, **other_files},
            "render_counts_expected": scene["render_counts"],
            "historical_metrics_present": True,
            "historical_metrics_only": True,
        }
        rows.append(row)
        if state != "PRESENT_EXACT":
            blockers.append(
                {
                    "base": "adgs",
                    "protocol": "native_historical",
                    "scene": scene["scene"],
                    "state": state,
                    "reason": "historical heavy checkpoint/render assets missing; metrics manifest remains exact",
                }
            )
    return rows, blockers


def _markdown(registry: Mapping[str, Any], freeze: Mapping[str, Any]) -> str:
    lines = [
        "# WorldSim V5.2.1 P1 Base Asset Registry",
        "",
        f"- coverage terminal candidate: `{registry['coverage_terminal_candidate']}`",
        f"- quality bytes decoded: `{freeze['summary']['quality_bytes_decoded']}`",
        f"- frozen samples/views: `{freeze['summary']['candidate_samples']} / {freeze['summary']['candidate_views']}`",
        "",
        "| base/protocol | exact | missing | mismatch | protocol mismatch |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, counts in registry["state_counts"].items():
        lines.append(
            f"| {key} | {counts.get('PRESENT_EXACT', 0)} | {counts.get('MISSING_BUT_MANIFESTED', 0)} | "
            f"{counts.get('PRESENT_HASH_MISMATCH', 0)} | {counts.get('PROTOCOL_MISMATCH', 0)} |"
        )
    lines.extend(
        [
            "",
            "Matched Tier D 使用 V4 strict mod-5 六场 exact checkpoint；AD-GS V1 native historical 六场重载荷缺失仅记 blocker，",
            "不把资产缺失写成算法失败，也不以近似 checkpoint 补洞。当前 third-party checkout 不原地改写；正式 renderer 可从",
            "冻结 commit + compatibility patch 在隔离目录重建 exact source。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldsim_v521/base_asset_registry_v1.yaml")
    parser.add_argument("--run-root", default="/root/autodl-tmp/runs/worldsim_v521")
    parser.add_argument("--run-id")
    arguments = parser.parse_args()

    config_path = PROJECT_ROOT / arguments.config
    config = _load_yaml(config_path)
    if _git(PROJECT_ROOT, "status", "--porcelain=v1"):
        raise AssetAuditError("P1 正式 run 要求 clean worktree")
    matrix_path = PROJECT_ROOT / config["inputs"]["baseline_matrix"]
    matrix = _load_yaml(matrix_path)
    street_training = _load_yaml(PROJECT_ROOT / config["inputs"]["streetgs_training"])
    adgs_training = _load_yaml(PROJECT_ROOT / config["inputs"]["adgs_training"])

    street = matrix["baselines"]["streetgs"]
    adgs = matrix["baselines"]["ad_gs"]
    sources = {
        "streetgs": _source_record(
            "streetgs",
            {
                "root": street["implementation_root"],
                "commit": street["implementation_commit"],
                "patch": street_training["implementation"]["compatibility_patch"],
                "patch_sha256": street_training["implementation"]["compatibility_patch_sha256"],
            },
        ),
        "adgs": _source_record(
            "adgs",
            {
                "root": adgs["implementation_root"],
                "commit": adgs["implementation_commit"],
                "patch": adgs_training["implementation"]["compatibility_patch"],
                "patch_sha256": adgs_training["implementation"]["compatibility_patch_sha256"],
            },
        ),
    }

    matched_rows = []
    for scene, spec in sorted(street["checkpoints"].items()):
        audited = audit_file(spec)
        row = {"base": "streetgs", "protocol": "strict_mod5", "scene": scene, **audited}
        ensure_matched_asset(row)
        matched_rows.append(row)
    for scene, spec in sorted(adgs["executable_checkpoints"].items()):
        bundle = audit_bundle(spec["files"])
        row = {"base": "adgs", "protocol": "strict_mod5", "scene": scene, **bundle}
        ensure_matched_asset(row)
        matched_rows.append(row)

    mismatch_rows = [
        {
            "base": "streetgs",
            "protocol": "stride10",
            "scene": scene,
            "state": "PROTOCOL_MISMATCH",
            "path": spec["path"],
            "reason": street["protocol_mismatch_runs"]["common_reason"],
        }
        for scene, spec in sorted(street["protocol_mismatch_runs"].items())
        if scene != "common_reason"
    ]
    historical_rows, blockers = _historical_adgs(Path(config["inputs"]["adgs_historical_metrics"]))

    sample_rows = []
    view_rows = []
    cohort = config["cohort"]
    for scene, spec in sorted(cohort["scenes"].items()):
        root = Path(cohort["processed_root"]) / f"{int(spec['scene_index']):03d}"
        scene_samples, scene_views = enumerate_quality_blind_targets(
            dataset=cohort["dataset"],
            scene=scene,
            scene_index=int(spec["scene_index"]),
            scene_root=root,
            expected_frames=int(cohort["expected_frames"]),
            cameras=[int(value) for value in cohort["cameras"]],
            eligible_bases=["adgs", "streetgs"],
            development_remainder=int(cohort["partition"]["development_remainder"]),
            modulus=int(cohort["partition"]["modulus"]),
        )
        sample_rows.extend(scene_samples)
        view_rows.extend(scene_views)

    state_groups: dict[str, Counter[str]] = {}
    for label, rows in {
        "matched_mod5": matched_rows,
        "adgs_native_historical": historical_rows,
        "streetgs_stride10_provenance": mismatch_rows,
    }.items():
        state_groups[label] = Counter(str(row["state"]) for row in rows)
    exact_bases = {
        base
        for base in ("streetgs", "adgs")
        if all(row["state"] == "PRESENT_EXACT" for row in matched_rows if row["base"] == base)
    }
    coverage_terminal = "complete_full" if exact_bases == {"streetgs", "adgs"} else "complete_partial_base_blocked"
    registry = {
        "schema": "worldsim_v521_base_asset_registry_v1",
        "task_id": config["task_id"],
        "source_head": _git(PROJECT_ROOT, "rev-parse", "HEAD"),
        "sources": sources,
        "assets": matched_rows + historical_rows + mismatch_rows,
        "state_counts": {key: dict(sorted(value.items())) for key, value in state_groups.items()},
        "exact_matched_bases": sorted(exact_bases),
        "coverage_terminal_candidate": coverage_terminal,
        "quality_bytes_decoded": 0,
    }
    split_summary = freeze_summary(sample_rows, view_rows)
    freeze = {
        "schema": "worldsim_v521_discovery_confirmation_freeze_v1",
        "frozen_before_quality_read": True,
        "partition_unit": "dataset|scene|sample_token_or_canonical_sample_index",
        "hash_excludes": ["base", "camera", "actor"],
        "algorithm": "sha256(UTF-8(unit_key)) % 5; bucket 0 confirmation",
        "membership": sample_rows,
        "summary": split_summary,
        "blocked_samples": [],
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = arguments.run_id or f"{stamp}__p1-base-asset-census-s0-r001"
    run_root = Path(arguments.run_root)
    target = run_root / run_id
    partial = run_root / f"{run_id}.partial"
    if target.exists() or partial.exists():
        raise AssetAuditError(f"run ID 已存在：{run_id}")
    partial.mkdir(parents=True)
    shutil.copy2(config_path, partial / "resolved_config.yaml")
    atomic_json(partial / "status.json", {"task_id": config["task_id"], "status": "running"})
    atomic_jsonl(partial / "events.jsonl", [{"event": "run_started", "at_utc": _utc_now()}])
    atomic_json(partial / config["outputs"]["base_asset_registry"], registry)
    atomic_text(partial / config["outputs"]["base_asset_report"], _markdown(registry, freeze))
    atomic_jsonl(partial / config["outputs"]["matched_frames"], view_rows)
    atomic_jsonl(partial / config["outputs"]["blockers"], blockers)
    atomic_json(partial / config["outputs"]["split_freeze"], freeze)

    input_paths = [PROJECT_ROOT / value for value in config["inputs"].values() if not str(value).startswith("/")]
    input_paths.append(Path(config["inputs"]["adgs_historical_metrics"]))
    fingerprint = {
        "source_head": registry["source_head"],
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "checkpoint_files_hashed": sum(
            1
            for row in matched_rows
            for value in ([row] if row["base"] == "streetgs" else row["files"].values())
            if value.get("observed_sha256")
        ),
        "target_files_hashed": len(view_rows),
        "quality_bytes_decoded": 0,
    }
    atomic_json(partial / "input_fingerprint.json", fingerprint)
    summary = {
        "task_id": config["task_id"],
        "status": "done",
        "outcome": "p1_gate_pass",
        "coverage_terminal_candidate": coverage_terminal,
        "exact_matched_bases": sorted(exact_bases),
        "matched_scene_count": len(cohort["scenes"]),
        "asset_blockers": len(blockers),
        "split": split_summary,
        "quality_bytes_decoded": 0,
        "next_task": "WS-V521-P2-BASE-CENSUS-EVAL-01",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
    }
    atomic_json(partial / "summary.json", summary)
    atomic_jsonl(
        partial / "events.jsonl",
        [
            {"event": "run_started", "at_utc": _utc_now()},
            {"event": "membership_frozen", "at_utc": _utc_now(), "samples": len(sample_rows)},
            {"event": "run_completed", "at_utc": _utc_now(), "outcome": "p1_gate_pass"},
        ],
    )
    atomic_json(partial / "status.json", {"task_id": config["task_id"], "status": "done", "outcome": "p1_gate_pass"})
    atomic_json(
        partial / "run_manifest.json",
        {
            "schema": "worldsim_v521_run_manifest_v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "source_head": registry["source_head"],
            "failure_ledger_refs": config["failure_ledger_refs"],
            "failure_ledger_delta": "none",
            "inventory_before_manifest": inventory_files(partial),
        },
    )
    partial.rename(target)
    print(json.dumps({"run_dir": str(target), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
