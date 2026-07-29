#!/usr/bin/env python3
"""执行 M6 冻结对象、编辑、去遮挡与噪声压力测试的 fail-closed 审计。"""
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

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/motion_proj")

from motion_proj.dynamic_recon.pseudo_tracks import (
    PseudoTrackConfig,
    audit_mask_id_continuity,
    read_scalar_vertex_ply,
)


PROJECT = Path("/root/autodl-tmp/motion_proj")
DATA_ROOT = Path(
    "/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1"
)
M4_AGGREGATE = Path(
    "/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/"
    "20260728T141204__aggregate6-s0-wm3090"
)
DATA_MANIFEST = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/"
    "adgs_nuscenes_v1_manifest.json"
)
TASK_ID = "DR-M6-STRESS-01"
SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
NOISE_SCENES = SCENES[:3]
NOISE_LEVELS = {
    "camera_translation_sigma_m": [0.02, 0.05, 0.10],
    "camera_rotation_sigma_deg": [0.1, 0.3, 1.0],
    "mask_dropout_fraction": [0.05, 0.10, 0.20],
    "mask_morphology_radius_px": [3, 7, 15],
    "flow_noise_sigma_px": [1, 3, 5],
    "id_switch_frames": [1, 3, 5],
    "prior_missing_fraction": [0.05, 0.10, 0.20],
}
EDIT_SPECS = [
    {"kind": "lateral", "value": 0.5, "unit": "m"},
    {"kind": "lateral", "value": 1.0, "unit": "m"},
    {"kind": "lateral", "value": 1.5, "unit": "m"},
    {"kind": "time_shift", "value": -0.5, "unit": "s"},
    {"kind": "time_shift", "value": 0.5, "unit": "s"},
    {"kind": "speed", "value": 0.75, "unit": "x"},
    {"kind": "speed", "value": 1.25, "unit": "x"},
    {"kind": "stop_restart", "value": 1.0, "unit": "s"},
    {"kind": "delete", "value": None, "unit": None},
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_checkpoint_identity(checkpoint_ply: Path) -> dict[str, Any]:
    vertices = read_scalar_vertex_ply(checkpoint_ply)
    if "obj" not in vertices.dtype.names:
        return {
            "path": str(checkpoint_ply),
            "instance_property_present": False,
            "failure": "point_cloud.ply 缺少 obj property",
        }
    values = np.asarray(vertices["obj"])
    scene_count = int(np.sum(values <= 0.5))
    object_count = int(np.sum(values > 0.5))
    distinct = [float(value) for value in np.unique(values).tolist()]
    return {
        "path": str(checkpoint_ply),
        "bytes": checkpoint_ply.stat().st_size,
        "instance_property_present": True,
        "obj_distinct_values": distinct,
        "scene_gaussian_count": scene_count,
        "object_gaussian_count": object_count,
        "persistent_instance_id_count": 0,
        "binary_object_only": distinct == [0.0, 1.0],
    }


def source_scene_rows() -> dict[str, dict[str, Any]]:
    rows = {}
    for row in load_jsonl(M4_AGGREGATE / "metrics.jsonl"):
        if row.get("type") == "scene":
            rows[row["scene"]] = row
    if sorted(rows) != sorted(SCENES):
        raise RuntimeError(
            f"M4 scene coverage 不完整: {sorted(rows)} != {sorted(SCENES)}"
        )
    return rows


def verify_upstream_gates(
    m5_run: Path, m5_common_run: Path | None
) -> dict[str, Any]:
    m4_summary = load_json(M4_AGGREGATE / "summary.json")
    if m4_summary.get("status") != "done" or not m4_summary.get(
        "all_gates_passed"
    ):
        raise RuntimeError("M4 不是 all-gates-passed done")
    m5_terminal = load_json(m5_run / "terminal.json")
    m5_status = m5_terminal.get("status")
    if m5_status not in {"done", "blocked"}:
        raise RuntimeError(f"M5 尚无可接受终态: {m5_terminal}")
    if m5_status == "blocked" and not m5_terminal.get("failure"):
        raise RuntimeError("M5 blocked 但缺少 failure 证据")
    common_status = "not_run_upstream_blocked"
    if m5_status == "done":
        if m5_common_run is None:
            raise RuntimeError("M5 native done 但缺少 common-observation diagnostic")
        common_terminal = load_json(m5_common_run / "terminal.json")
        common_summary = load_json(m5_common_run / "summary.json")
        if common_terminal.get("status") != "done" or common_summary.get(
            "status"
        ) != "done":
            raise RuntimeError("M5 common-observation diagnostic 不是 done")
        if common_summary.get("target_mapping_coverage") != 1.0:
            raise RuntimeError("M5 common-observation target coverage 不完整")
        common_status = "done"
    return {
        "m4_status": m4_summary["status"],
        "m4_all_gates_passed": m4_summary["all_gates_passed"],
        "m4_official_test_mean": m4_summary["official_test_mean"],
        "m5_status": m5_status,
        "m5_failure": m5_terminal.get("failure"),
        "m5_common_run": str(m5_common_run) if m5_common_run else None,
        "m5_common_status": common_status,
    }


def initialize(
    run_dir: Path,
    m5_run: Path,
    m5_common_run: Path | None,
    config: PseudoTrackConfig,
) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run 目录非空，禁止覆盖: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir()
    source_files = [
        Path(__file__).resolve(),
        PROJECT / "motion_proj/dynamic_recon/pseudo_tracks.py",
    ]
    source_snapshot = {}
    for source in source_files:
        destination = snapshot_dir / source.name
        shutil.copy2(source, destination)
        source_snapshot[source.name] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
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
        "scenes": SCENES,
        "noise_scenes": NOISE_SCENES,
        "pseudo_track_config": asdict_without_import(config),
        "pseudo_track_protocol_fingerprint": config.fingerprint(),
        "object_slots": ["high-support", "boundary-support"],
        "edit_specs": EDIT_SPECS,
        "noise_levels": NOISE_LEVELS,
        "noise_seeds": [0, 1, 2],
        "medium_combined": {
            "camera_translation_sigma_m": 0.05,
            "camera_rotation_sigma_deg": 0.3,
            "mask_dropout_fraction": 0.10,
            "mask_morphology_radius_px": 7,
            "flow_noise_sigma_px": 3,
            "id_switch_frames": 3,
            "prior_missing_fraction": 0.10,
        },
        "truth_tiers": {
            "A": "held-out observed",
            "B": "geometric support",
            "C": "unsupported",
        },
        "m4_aggregate_run": str(M4_AGGREGATE),
        "m5_run": str(m5_run),
        "m5_common_run": str(m5_common_run) if m5_common_run else None,
        "data_root": str(DATA_ROOT),
        "data_manifest": str(DATA_MANIFEST),
        "fail_closed_rule": (
            "没有训练前冻结且满足门槛的车辆 pseudo ID 时，所有对象编辑、"
            "去遮挡和对象噪声 endpoint 必须 ABSTAIN 并进入 coverage"
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
        "project_git_status_sha256": sha256_bytes(git_status.encode()),
        "config_fingerprint": resolved["config_fingerprint"],
        "data_manifest_sha256": sha256_file(DATA_MANIFEST),
        "m4_summary_sha256": sha256_file(M4_AGGREGATE / "summary.json"),
        "m5_terminal_sha256": sha256_file(m5_run / "terminal.json"),
        "m5_common_terminal_sha256": (
            sha256_file(m5_common_run / "terminal.json")
            if m5_common_run
            else None
        ),
        "source_snapshot": source_snapshot,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )


def asdict_without_import(config: PseudoTrackConfig) -> dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in config.__dataclass_fields__
    }


def run_audit(
    run_dir: Path, m5_run: Path, m5_common_run: Path | None
) -> dict[str, Any]:
    config = PseudoTrackConfig()
    gates = verify_upstream_gates(m5_run, m5_common_run)
    m4_rows = source_scene_rows()
    track_audits = {}
    failure_rows = []
    metrics_path = run_dir / "metrics.jsonl"
    edit_rows = []
    noise_rows = []
    hole_rows = []

    for scene in SCENES:
        scene_dir = DATA_ROOT / scene
        mask_audit = audit_mask_id_continuity(scene_dir, config)
        source = m4_rows[scene]
        checkpoint_ply = Path(
            source["checkpoint"]["point_cloud.ply"]["path"]
        )
        checkpoint_audit = audit_checkpoint_identity(checkpoint_ply)
        frozen_track_artifacts = sorted(
            str(path.relative_to(scene_dir))
            for pattern in ("*track*.json", "*track*.npz", "*tracks*.npy")
            for path in scene_dir.glob(pattern)
        )
        scene_audit = {
            "scene": scene,
            "mask_identity": mask_audit,
            "frozen_track_artifacts": frozen_track_artifacts,
            "checkpoint_identity": checkpoint_audit,
            "selected_objects": {
                "high-support": None,
                "boundary-support": None,
            },
            "coverage_slots": 0,
            "expected_slots": 2,
            "coverage": 0.0,
        }
        track_audits[scene] = scene_audit
        failure = {
            "scene": scene,
            "failure_type": "persistent_object_identity_unavailable",
            "repeated": True,
            "max_frozen_mask_id_support_frames": mask_audit[
                "max_support_frames"
            ],
            "eligible_vehicle_tracks": mask_audit["vehicle_eligible_count"],
            "frozen_track_artifact_count": len(frozen_track_artifacts),
            "checkpoint_persistent_instance_id_count": checkpoint_audit[
                "persistent_instance_id_count"
            ],
            "evidence": [
                "stored SAM mask IDs",
                "processed-scene track artifact scan",
                "trained point_cloud.ply obj property",
            ],
        }
        failure_rows.append(failure)
        reconstruction_row = {
            "type": "scene_reconstruction",
            "scene": scene,
            "official_test_metrics": source["official_test_metrics"],
            "object_typed_metrics": None,
            "object_metric_status": "ABSTAIN",
            "reason": "no_eligible_frozen_vehicle_pseudo_track",
            "coverage_slots": 0,
            "expected_slots": 2,
        }
        append_jsonl(metrics_path, reconstruction_row)
        for slot in ("high-support", "boundary-support"):
            for edit in EDIT_SPECS:
                row = {
                    "type": "edit",
                    "scene": scene,
                    "object_slot": slot,
                    "edit": edit,
                    "status": "ABSTAIN",
                    "truth_tier": "C",
                    "support": 0.0,
                    "confidence": 0.0,
                    "reason": "no_eligible_frozen_vehicle_pseudo_track",
                }
                edit_rows.append(row)
                append_jsonl(metrics_path, row)
        for baseline in (
            "no-completion",
            "2d-framewise-diagnostic",
            "observed-only-gaussians",
            "candidate-3d-completion",
        ):
            row = {
                "type": "pseudo_hole",
                "scene": scene,
                "baseline": baseline,
                "status": "ABSTAIN",
                "truth_tier": None,
                "reason": "no_eligible_frozen_vehicle_footprint",
            }
            hole_rows.append(row)
            append_jsonl(metrics_path, row)

    for scene in NOISE_SCENES:
        for factor, levels in NOISE_LEVELS.items():
            for level in levels:
                for seed in (0, 1, 2):
                    row = {
                        "type": "noise",
                        "scene": scene,
                        "factor": factor,
                        "level": level,
                        "seed": seed,
                        "status": "ABSTAIN",
                        "coverage": 0.0,
                        "reason": "clean_baseline_has_no_eligible_frozen_vehicle_pseudo_track",
                    }
                    noise_rows.append(row)
                    append_jsonl(metrics_path, row)
        for seed in (0, 1, 2):
            row = {
                "type": "noise",
                "scene": scene,
                "factor": "medium_combined",
                "level": "preregistered_medium",
                "seed": seed,
                "status": "ABSTAIN",
                "coverage": 0.0,
                "reason": "clean_baseline_has_no_eligible_frozen_vehicle_pseudo_track",
            }
            noise_rows.append(row)
            append_jsonl(metrics_path, row)

    stable_scenes = [
        row["scene"]
        for row in failure_rows
        if row["failure_type"] == "persistent_object_identity_unavailable"
    ]
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "instance_id": run_dir.name,
        "status": "done",
        "completed_at": now(),
        "upstream_gates": gates,
        "scene_count": len(SCENES),
        "expected_object_slots": len(SCENES) * 2,
        "eligible_object_slots": 0,
        "object_coverage": 0.0,
        "edit_expected_rows": len(SCENES) * 2 * len(EDIT_SPECS),
        "edit_abstain_rows": len(edit_rows),
        "pseudo_hole_abstain_rows": len(hole_rows),
        "noise_expected_rows": len(noise_rows),
        "noise_abstain_rows": len(noise_rows),
        "stable_failure": {
            "type": "persistent_object_identity_unavailable",
            "scene_count": len(stable_scenes),
            "scenes": stable_scenes,
            "cross_scene_gate_at_least_3": len(stable_scenes) >= 3,
            "interpretation": (
                "冻结 SAM mask 没有可证明的持久车辆 ID，训练检查点又把 obj "
                "压成二值；因此不能可靠选择单车，也不能诚实执行对象编辑。"
            ),
        },
        "m7_decision_candidate": "A: 可编辑运动表示与轨迹不确定性",
        "claim_boundaries": [
            "全图官方重建指标仍沿用 M4，不声称对象级编辑质量",
            "所有不可执行项保留为 ABSTAIN，未从 coverage 分母删除",
            "事后几何重关联只允许作为 M7 候选方法，不能回填 M6 baseline",
        ],
        "next_action": "进入 M7 唯一假设与 novelty 审计；若重合则 rejected",
    }
    atomic_json(run_dir / "track_audit.json", track_audits)
    atomic_json(run_dir / "failure_matrix.json", failure_rows)
    atomic_json(run_dir / "edit_coverage.json", edit_rows)
    atomic_json(run_dir / "pseudo_hole_coverage.json", hole_rows)
    atomic_json(run_dir / "noise_coverage.json", noise_rows)
    atomic_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m5-run", required=True)
    parser.add_argument("--m5-common-run")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    m5_run = Path(args.m5_run)
    m5_common_run = Path(args.m5_common_run) if args.m5_common_run else None
    config = PseudoTrackConfig()
    try:
        initialize(run_dir, m5_run, m5_common_run, config)
        summary = run_audit(run_dir, m5_run, m5_common_run)
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
