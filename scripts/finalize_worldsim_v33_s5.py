#!/usr/bin/env python3
"""冻结 S5 开发选择、留出确认、生产 fallback 与终端证据。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.semantic_gate import development_selection  # noqa: E402
from motion_proj.worldsim_v33.spatial_delta import atomic_json, sha256_file  # noqa: E402


def verify(path: str | Path, expected: str, role: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{role} 不存在: {target}")
    actual = sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"{role} SHA 漂移: expected={expected} actual={actual}")
    return {"path": str(target), "sha256": actual, "bytes": target.stat().st_size}


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def copy_verified(source: str | Path, expected: str, target: Path) -> dict[str, Any]:
    verify(source, expected, f"production source {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return verify(target, expected, f"production target {target}")


def region_aggregate(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
    return {
        "raw_l1_uint8": float(
            np.mean([row["metrics"][name]["raw_l1_uint8"] for row in rows])
        ),
        "gated_l1_uint8": float(
            np.mean([row["metrics"][name]["gated_l1_uint8"] for row in rows])
        ),
        "l1_delta": float(
            np.mean([row["metrics"][name]["l1_delta"] for row in rows])
        ),
    }


def source_snapshot(run_dir: Path, config_path: Path) -> list[dict[str, Any]]:
    sources = [
        config_path,
        PROJECT / "motion_proj/worldsim_v32/harmonizer_adapter.py",
        PROJECT / "motion_proj/worldsim_v33/semantic_gate.py",
        PROJECT / "scripts/prepare_worldsim_v33_s5_inputs.py",
        PROJECT / "scripts/run_worldsim_v33_s5_harmonizer.py",
        PROJECT / "scripts/run_worldsim_v33_s5_sam2_detector.py",
        PROJECT / "scripts/finalize_worldsim_v33_s5.py",
        PROJECT / "scripts/run_worldsim_v33_s5.sh",
    ]
    destination = run_dir / "artifacts/source_snapshot"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    records = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(f"S5 source snapshot 缺文件: {source}")
        target = destination / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source.resolve()),
                "snapshot": str(target),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    atomic_json(destination / "manifest.json", {"files": records})
    return records


def git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *arguments], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha", required=True)
    parser.add_argument("--harmonizer-manifest", type=Path, required=True)
    parser.add_argument("--harmonizer-manifest-sha", required=True)
    parser.add_argument("--sam2-manifest", type=Path, required=True)
    parser.add_argument("--sam2-manifest-sha", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifests = {
        "input": (
            args.input_manifest,
            args.input_manifest_sha,
            "worldsim_v33_s5_input_manifest_v1",
        ),
        "harmonizer": (
            args.harmonizer_manifest,
            args.harmonizer_manifest_sha,
            "worldsim_v33_s5_harmonizer_manifest_v1",
        ),
        "sam2": (
            args.sam2_manifest,
            args.sam2_manifest_sha,
            "worldsim_v33_s5_sam2_detector_manifest_v1",
        ),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, (path, expected, schema) in manifests.items():
        verify(path, expected, f"S5 {name} manifest")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        if loaded[name].get("schema_version") != schema:
            raise RuntimeError(f"S5 {name} manifest schema 漂移")
        if loaded[name].get("config_sha256") != sha256_file(args.config):
            raise RuntimeError(f"S5 {name} manifest config SHA 漂移")
    if loaded["harmonizer"]["input_manifest_sha256"] != args.input_manifest_sha:
        raise RuntimeError("S5 Harmonizer provenance 链断裂")
    if loaded["sam2"]["harmonizer_manifest_sha256"] != args.harmonizer_manifest_sha:
        raise RuntimeError("S5 SAM2 provenance 链断裂")

    harmonizer_rows = {
        (int(row["frame"]), int(row["camera_id"])): row
        for row in loaded["harmonizer"]["rows"]
    }
    sam2_rows = {
        (int(row["frame"]), int(row["camera_id"])): row
        for row in loaded["sam2"]["rows"]
    }
    if harmonizer_rows.keys() != sam2_rows.keys() or len(harmonizer_rows) != 5:
        raise RuntimeError("S5 Harmonizer/SAM2 view 集不一致")

    # 先构造且只读取 development rows；heldout 对象在选择完成后才解引用。
    development_rows = []
    for key, harmonizer_row in harmonizer_rows.items():
        if harmonizer_row["phase"] != "development":
            continue
        semantic_row = sam2_rows[key]
        development_rows.append(
            {
                **harmonizer_row,
                "semantic_reintroduction": semantic_row[
                    "semantic_reintroduction"
                ],
            }
        )
    if len(development_rows) != 3:
        raise RuntimeError("S5 development view 数必须为 3")
    dev_decision = development_selection(
        development_rows, config["selection_gates"]
    )
    selected_before_heldout = dev_decision["selected_arm"]
    heldout_audit: dict[str, Any]
    if selected_before_heldout == "G1_semantic_gate":
        heldout_rows = []
        for key, harmonizer_row in harmonizer_rows.items():
            if harmonizer_row["phase"] != "heldout_confirmation":
                continue
            heldout_rows.append(
                {
                    **harmonizer_row,
                    "semantic_reintroduction": sam2_rows[key][
                        "semantic_reintroduction"
                    ],
                }
            )
        if len(heldout_rows) != 2:
            raise RuntimeError("S5 heldout confirmation view 数必须为 2")
        limits = config["selection_gates"]
        checks = {
            "boundary_non_degradation": all(
                row["metrics"]["boundary_ring"]["l1_delta"]
                <= float(limits["heldout_maximum_boundary_l1_degradation"])
                for row in heldout_rows
            ),
            "contact_non_degradation": all(
                row["metrics"]["ground_contact"]["l1_delta"]
                <= float(limits["heldout_maximum_contact_l1_degradation"])
                for row in heldout_rows
            ),
            "actor_interior_preserved": all(
                row["metrics"]["actor_interior"]["l1_delta"]
                <= float(
                    limits["heldout_maximum_actor_interior_l1_degradation"]
                )
                for row in heldout_rows
            ),
            "far_non_target_exact": all(
                row["blend_audit"]["changed_far_non_target_pixels"] == 0
                for row in heldout_rows
            ),
            "delete_production_exact": all(
                row["delete_raw_production_exact"] for row in heldout_rows
            ),
            "delete_semantic_safe": all(
                row["semantic_reintroduction"]["production_safe"]
                for row in heldout_rows
            ),
        }
        heldout_passed = all(checks.values())
        heldout_audit = {
            "entered": True,
            "confirmation_only": True,
            "checks": checks,
            "passed": heldout_passed,
            "views": [[row["frame"], row["camera_id"]] for row in heldout_rows],
        }
    else:
        heldout_passed = False
        heldout_audit = {
            "entered": False,
            "confirmation_only": True,
            "reason": "development_did_not_select_G1",
            "passed": None,
        }
    selected_arm = (
        "G1_semantic_gate"
        if selected_before_heldout == "G1_semantic_gate" and heldout_passed
        else "G0_raw_3d"
    )
    if selected_before_heldout == "G0_raw_3d":
        selection_reason = (
            "semantic_gate_no_gain"
            if all(dev_decision["safeguards"].values())
            else "development_safeguard_failure"
        )
    elif not heldout_passed:
        selection_reason = "rejected_on_heldout_confirmation"
    else:
        selection_reason = "development_gain_confirmed_on_heldout"

    production_root = args.run_dir / "artifacts/production"
    if production_root.exists():
        raise FileExistsError(production_root)
    production_rows = []
    for key, row in sorted(harmonizer_rows.items()):
        view_root = production_root / f"f{key[0]:03d}_c{key[1]}"
        insertion_source = row["outputs"][
            "full_gated" if selected_arm == "G1_semantic_gate" else "full_raw"
        ]
        delete_source = row["outputs"]["delete_raw"]
        insertion = copy_verified(
            insertion_source["path"],
            insertion_source["sha256"],
            view_root / "insertion.png",
        )
        deletion = copy_verified(
            delete_source["path"],
            delete_source["sha256"],
            view_root / "delete.png",
        )
        production_rows.append(
            {
                "frame": key[0],
                "camera_id": key[1],
                "phase": row["phase"],
                "selected_arm": selected_arm,
                "insertion": insertion,
                "delete": deletion,
                "delete_raw_sha256": row["outputs"]["delete_raw"]["sha256"],
                "delete_exact": deletion["sha256"]
                == row["outputs"]["delete_raw"]["sha256"],
            }
        )
    atomic_json(production_root / "production_manifest.json", {"rows": production_rows})

    snapshots = source_snapshot(args.run_dir, args.config)
    all_rows = []
    for key, harmonizer_row in sorted(harmonizer_rows.items()):
        all_rows.append(
            {
                **harmonizer_row,
                "semantic_mass": sam2_rows[key]["semantic_mass"],
                "semantic_reintroduction": sam2_rows[key][
                    "semantic_reintroduction"
                ],
            }
        )
    aggregate = {
        name: region_aggregate(all_rows, name)
        for name in (
            "boundary_ring",
            "ground_contact",
            "shadow_support",
            "actor_interior",
            "far_non_target",
        )
    }
    resource_cfg = config["resources"]
    stage_resources = {
        "harmonizer": loaded["harmonizer"]["resource"],
        "sam2": loaded["sam2"]["resource"],
    }
    resource_checks = {
        "stage_wall_seconds": all(
            stage["wall_seconds"]
            <= float(resource_cfg["maximum_stage_wall_seconds"])
            for stage in stage_resources.values()
        ),
        "peak_nvidia_memory": all(
            stage["peak_nvidia_memory_mib"]
            <= int(resource_cfg["maximum_peak_nvidia_memory_mib"])
            for stage in stage_resources.values()
        ),
        "oom_events": sum(
            int(stage["oom_events_delta"]) for stage in stage_resources.values()
        )
        <= int(resource_cfg["maximum_oom_events_delta"]),
        "oom_kill_events": sum(
            int(stage["oom_kill_events_delta"]) for stage in stage_resources.values()
        )
        <= int(resource_cfg["maximum_oom_kill_events_delta"]),
    }
    safety_checks = {
        "five_view_contract": len(all_rows) == 5,
        "far_non_target_exact": all(
            row["blend_audit"]["changed_far_non_target_pixels"] == 0
            for row in all_rows
        ),
        "residual_cap": all(
            row["blend_audit"]["maximum_applied_abs_residual_uint8"]
            <= float(config["semantic_gate"]["residual_cap_uint8"])
            for row in all_rows
        ),
        "delete_pixel_exact": all(row["delete_raw_production_exact"] for row in all_rows)
        and all(row["delete_exact"] for row in production_rows),
        "delete_semantic_safe": all(
            row["semantic_reintroduction"]["production_safe"] for row in all_rows
        ),
        "immutable_3d_assets": loaded["harmonizer"]["immutable_before"]
        == loaded["harmonizer"]["immutable_after"]
        == loaded["sam2"]["immutable_before"]
        == loaded["sam2"]["immutable_after"],
        "r3d2_not_fabricated": loaded["harmonizer"]["r3d2"]["disposition"]
        == "blocked_pretrained_model_unavailable"
        and not loaded["harmonizer"]["r3d2"]["model_loaded"]
        and not loaded["harmonizer"]["r3d2"]["training_performed"],
    }
    accepted = all(resource_checks.values()) and all(safety_checks.values())
    if not accepted:
        raise RuntimeError(
            f"S5 工程/安全 gate 失败: resources={resource_checks} safety={safety_checks}"
        )

    decision = {
        "schema_version": "worldsim_v33_s5_decision_v1",
        "task_id": config["task_id"],
        "accepted": True,
        "selected_arm": selected_arm,
        "selection_reason": selection_reason,
        "development_selection": dev_decision,
        "heldout_confirmation": heldout_audit,
        "heldout_read_for_development_selection": False,
        "enhancement_accepted": selected_arm == "G1_semantic_gate",
        "delete_policy": "raw_3d_render_only",
        "r3d2_disposition": "blocked_pretrained_model_unavailable",
        "temporal_consistency": {
            "status": "not_evaluated_non_temporal_frozen_five_view_protocol",
            "claimed": False,
            "reason": "冻结视图不是相邻视频帧，禁止伪造 temporal 指标",
        },
        "resource_checks": resource_checks,
        "safety_checks": safety_checks,
    }
    decision_path = args.run_dir / "artifacts/decision.json"
    atomic_json(decision_path, decision)
    summary = {
        "schema_version": "worldsim_v33_s5_summary_v1",
        "task_id": config["task_id"],
        "state": "completed",
        "accepted": True,
        "selected_arm": selected_arm,
        "selection_reason": selection_reason,
        "config": verify(args.config, sha256_file(args.config), "S5 config"),
        "manifests": {
            name: verify(path, expected, f"S5 {name} manifest")
            for name, (path, expected, _) in manifests.items()
        },
        "aggregate_metrics": aggregate,
        "candidate_semantic_reintroduction_flagged_views": sum(
            int(row["semantic_reintroduction"]["unconstrained_candidate_flagged"])
            for row in all_rows
        ),
        "production_semantic_safe_views": sum(
            int(row["semantic_reintroduction"]["production_safe"])
            for row in all_rows
        ),
        "rows": all_rows,
        "production_manifest": verify(
            production_root / "production_manifest.json",
            sha256_file(production_root / "production_manifest.json"),
            "S5 production manifest",
        ),
        "source_snapshot": snapshots,
        "repository": {
            "branch": git_text("branch", "--show-current"),
            "head": git_text("rev-parse", "HEAD"),
            "dirty_at_runtime": bool(git_text("status", "--porcelain")),
        },
        "resources": {
            "stages": stage_resources,
            "checks": resource_checks,
            "run_bytes": 0,
            "maximum_run_bytes": int(resource_cfg["maximum_run_bytes"]),
        },
        "decision": decision,
    }
    summary_path = args.run_dir / "summary.json"
    status_path = args.run_dir / "status.json"
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = {
        "schema_version": "worldsim_v33_s5_status_v1",
        "task_id": config["task_id"],
        "state": "completed",
        "accepted": True,
        "selected_arm": selected_arm,
        "selection_reason": selection_reason,
        "completed_at_utc": completed_at,
        "summary": str(summary_path),
        "decision": str(decision_path),
        "run_bytes": 0,
    }
    # 迭代到 JSON 内记录的 run_bytes 与落盘目录大小一致。
    for _ in range(8):
        atomic_json(summary_path, summary)
        atomic_json(status_path, status)
        current_bytes = directory_bytes(args.run_dir)
        if (
            summary["resources"]["run_bytes"] == current_bytes
            and status["run_bytes"] == current_bytes
        ):
            break
        summary["resources"]["run_bytes"] = current_bytes
        status["run_bytes"] = current_bytes
    atomic_json(summary_path, summary)
    atomic_json(status_path, status)
    final_bytes = directory_bytes(args.run_dir)
    if final_bytes != summary["resources"]["run_bytes"]:
        # 字节位数变化最多只会触发一次；再写一轮即可稳定。
        summary["resources"]["run_bytes"] = final_bytes
        status["run_bytes"] = final_bytes
        atomic_json(summary_path, summary)
        atomic_json(status_path, status)
        final_bytes = directory_bytes(args.run_dir)
    if final_bytes > int(resource_cfg["maximum_run_bytes"]):
        raise RuntimeError(
            f"S5 run bytes 超限: {final_bytes} > {resource_cfg['maximum_run_bytes']}"
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "accepted": True,
                "selected_arm": selected_arm,
                "selection_reason": selection_reason,
                "summary_sha256": sha256_file(summary_path),
                "status_sha256": sha256_file(status_path),
                "decision_sha256": sha256_file(decision_path),
                "run_bytes": final_bytes,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
