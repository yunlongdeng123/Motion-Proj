#!/usr/bin/env python3
"""Independently replay the frozen r018 D0 H metric and gate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.run_worldsim_v5_m1_graph_diagnostic import _metric_row


METRICS = (
    "boundary_f1",
    "iou_at_frozen_threshold",
    "false_negative_semantic_mass",
    "false_positive_semantic_mass",
    "brier",
    "ece",
    "nll",
)


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as table:
        return {name: np.asarray(table[name]) for name in table.files}


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-15):
        raise AuditError(f"metric drift: {label}: {actual} != {expected}")


def _verify_inventory(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {row["path"]: row for row in manifest["inventory"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    }
    if set(observed) != set(expected):
        raise AuditError("run manifest inventory coverage drift")
    total_bytes = 0
    for relative, path in observed.items():
        record = expected[relative]
        size = path.stat().st_size
        if size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise AuditError(f"run manifest identity drift: {relative}")
        total_bytes += size
    return {"entry_count": len(observed), "bytes": total_bytes}


def _verified_graph_inventory(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    graph_run = Path(scene["graph_run"]["path"])
    manifest_path = graph_run / "manifest.json"
    if _sha256(manifest_path) != scene["graph_run"]["manifest_sha256"]:
        raise AuditError(f"graph manifest drift: {scene['scene']}")
    manifest = _json(manifest_path)
    if manifest.get("status") != "done":
        raise AuditError(f"graph terminal drift: {scene['scene']}")
    return {row["path"]: row for row in manifest["inventory"]}


def _verified_baseline(
    scene: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    directory: str,
    frame: int,
    camera_id: int,
) -> dict[str, np.ndarray]:
    relative = f"artifacts/evaluation/{directory}/f{frame:03d}_c{camera_id}.npz"
    path = Path(scene["graph_run"]["path"]) / relative
    record = inventory.get(relative)
    if (
        record is None
        or not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256(path) != record["sha256"]
    ):
        raise AuditError(f"baseline identity drift: {scene['scene']}/{relative}")
    return _npz(path)


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise AuditError("empty metric denominator")
    return {
        name: float(np.mean([row[name] for row in rows], dtype=np.float64))
        for name in METRICS
    }


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status = _json(run_dir / "status.json")
    summary = _json(run_dir / "summary.json")
    report = _json(run_dir / "artifacts/h_evaluation_report.json")
    manifest = _json(run_dir / "manifest.json")
    if status.get("status") != "rejected" or manifest.get("status") != "rejected":
        raise AuditError("r018 terminal must remain rejected")
    if summary.get("status") != "rejected" or report.get("method_status") != "rejected":
        raise AuditError("r018 summary/report terminal drift")
    conclusion = "d0_progressive_rejected_skip_d1_advance_super_primitive_or_anchor"
    if summary.get("conclusion") != conclusion or report.get("conclusion") != conclusion:
        raise AuditError("r018 conclusion drift")
    if (run_dir / "resolved_config.yaml").read_text(encoding="utf-8") != config_path.read_text(
        encoding="utf-8"
    ):
        raise AuditError("resolved config is not byte-exact")
    inventory_report = _verify_inventory(run_dir, manifest)
    if summary["source_commit"] != status["source_commit"]:
        raise AuditError("source commit drift")
    if summary["scene_reports"] != report["scene_reports"]:
        raise AuditError("summary/report scene payload drift")

    audited_scenes = []
    for scene_config, scene_report in zip(config["scenes"], summary["scene_reports"]):
        scene = scene_config["scene"]
        if scene_report["scene"] != scene:
            raise AuditError("scene order drift")
        expected_views = int(scene_config["expected_evaluation_view_count"])
        if scene_report["evaluation_view_count"] != expected_views:
            raise AuditError(f"view denominator drift: {scene}")
        inventory = _verified_graph_inventory(scene_config)
        recomputed_by_arm: dict[str, list[dict[str, float]]] = {
            "U2_B3_G0": [],
            "U2_B3_G_V5": [],
            "D0": [],
        }
        reference_rows = scene_report["evaluation_rows"]["U2_B3_G0"]
        for index, reference_row in enumerate(reference_rows):
            frame = int(reference_row["frame"])
            camera_id = int(reference_row["camera_id"])
            g0 = _verified_baseline(scene_config, inventory, "B3_G0", frame, camera_id)
            gv5 = _verified_baseline(scene_config, inventory, "B3_G3", frame, camera_id)
            d0_path = (
                run_dir
                / f"artifacts/evaluation/{scene}/D0/f{frame:03d}_c{camera_id}.npz"
            )
            if not d0_path.is_file():
                raise AuditError(f"D0 persisted render missing: {scene}/{frame}/{camera_id}")
            d0 = _npz(d0_path)
            tables = {"U2_B3_G0": g0, "U2_B3_G_V5": gv5, "D0": d0}
            target = np.asarray(g0["target"])
            for arm, table in tables.items():
                if table["probability"].dtype != np.float16:
                    raise AuditError(f"persisted probability precision drift: {scene}/{arm}")
                if not np.array_equal(table["target"], target):
                    raise AuditError(f"matched target drift: {scene}/{arm}")
                observed = _metric_row(
                    table["probability"].astype(np.float32),
                    target.astype(bool),
                    threshold=float(config["evaluation"]["probability_threshold"]),
                    boundary_tolerance=int(config["evaluation"]["boundary_tolerance_px"]),
                    ece_bins=int(config["evaluation"]["ece_bins"]),
                )
                recorded = scene_report["evaluation_rows"][arm][index]
                if int(recorded["frame"]) != frame or int(recorded["camera_id"]) != camera_id:
                    raise AuditError(f"evaluation row order drift: {scene}/{arm}")
                for name in METRICS:
                    _assert_close(observed[name], recorded[name], f"{scene}/{arm}/{index}/{name}")
                recomputed_by_arm[arm].append({name: float(observed[name]) for name in METRICS})
        recomputed_aggregate = {
            arm: _aggregate(rows) for arm, rows in recomputed_by_arm.items()
        }
        for arm, metrics in recomputed_aggregate.items():
            for name in METRICS:
                _assert_close(
                    metrics[name],
                    scene_report["evaluation_aggregate"][arm][name],
                    f"{scene}/{arm}/aggregate/{name}",
                )
        if scene_report["checkpoint_sha256_before"] != scene_report["checkpoint_sha256_after"]:
            raise AuditError(f"checkpoint mutation: {scene}")
        audited_scenes.append(
            {
                "scene": scene,
                "view_count": expected_views,
                "checkpoint_exact": True,
                "metric_replay_exact": True,
            }
        )

    gate = config["h_gate"]
    scene_deltas = []
    for scene_report in summary["scene_reports"]:
        base = scene_report["evaluation_aggregate"]["U2_B3_G0"]
        candidate = scene_report["evaluation_aggregate"]["D0"]
        scene_deltas.append({name: candidate[name] - base[name] for name in METRICS})
    balanced = {
        name: float(np.mean([row[name] for row in scene_deltas], dtype=np.float64))
        for name in METRICS
    }
    positive_count = sum(row["boundary_f1"] > 0.0 for row in scene_deltas)
    checks = {
        "positive_boundary_f1_scene_count": positive_count
        >= int(gate["minimum_positive_boundary_f1_scenes"]),
        "scene_balanced_boundary_f1_positive": balanced["boundary_f1"]
        > float(gate["minimum_scene_balanced_boundary_f1_delta_exclusive"]),
        "scene_balanced_iou_nonnegative": balanced["iou_at_frozen_threshold"]
        >= float(gate["minimum_scene_balanced_iou_delta"]),
        "scene_balanced_false_negative_safeguard": balanced[
            "false_negative_semantic_mass"
        ]
        <= float(gate["maximum_scene_balanced_false_negative_semantic_mass_delta"]),
    }
    recorded_gate = summary["h_gate"]
    if any(checks[name] is not bool(recorded_gate["checks"][name]) for name in checks):
        raise AuditError("H gate check replay drift")
    if positive_count != recorded_gate["positive_boundary_f1_scene_count"]:
        raise AuditError("H positive-scene count replay drift")
    for name in METRICS:
        _assert_close(
            balanced[name],
            recorded_gate["scene_balanced_delta_vs_u2_b3_g0"][name],
            f"H gate/{name}",
        )
    if bool(recorded_gate["pass"]) or all(checks.values()):
        raise AuditError("r018 rejection gate no longer fails")

    resources = summary["resources"]
    resource_samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in resource_samples if "monitor_error" not in row]
    if len(valid) != int(resources["sample_count"]) or resources["monitor_error_count"] != 0:
        raise AuditError("resource sample denominator drift")
    if max(int(row["gpu_used_mib"]) for row in valid) != int(resources["nvidia_peak_mib"]):
        raise AuditError("NVIDIA peak replay drift")
    if max(int(row["cgroup_memory_current_bytes"]) for row in valid) != int(
        resources["cgroup_memory_peak_bytes"]
    ):
        raise AuditError("cgroup peak replay drift")
    if not all(summary["resource_checks"].values()):
        raise AuditError("resource gate drift")
    for lock in (
        "parameter_search",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
    ):
        if summary.get(lock) is not False:
            raise AuditError(f"forbidden quality/search lock drift: {lock}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("M2/M3 status drift")

    return {
        "status": "passed",
        "conclusion": "r018_rejected_result_and_matched_metric_gate_replay_exact",
        "run": str(run_dir),
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "inventory": inventory_report,
        "scene_audits": audited_scenes,
        "total_view_count": sum(row["view_count"] for row in audited_scenes),
        "h_gate": recorded_gate,
        "resources": resources,
        "hashes": {
            name: _sha256(run_dir / name)
            for name in (
                "summary.json",
                "status.json",
                "manifest.json",
                "events.jsonl",
                "artifacts/h_evaluation_report.json",
                "artifacts/resources.json",
                "artifacts/resource_samples.jsonl",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_d_progressive_h_evaluation_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.config.resolve(), args.run_dir.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
