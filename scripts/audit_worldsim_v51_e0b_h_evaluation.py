#!/usr/bin/env python3
"""Independently replay and audit the frozen Stage E r022 E0b H result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.progressive_evaluation import METRICS
from scripts.run_worldsim_v5_m1_graph_diagnostic import _metric_row


SCENES = ("scene-0471", "scene-1087", "scene-0379")
ARMS = ("U2_B3_G0", "D0", "E0B")


class AuditError(RuntimeError):
    """The frozen r022 evidence no longer satisfies its recorded contract."""


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


def _assert_payload(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AuditError(f"payload field drift: {label}")
        for key, value in expected.items():
            _assert_payload(actual[key], value, f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditError(f"payload list drift: {label}")
        for index, value in enumerate(expected):
            _assert_payload(actual[index], value, f"{label}/{index}")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise AuditError(f"payload value drift: {label}")
        return
    if isinstance(expected, (int, float)):
        _assert_close(float(actual), float(expected), label)
        return
    if actual != expected:
        raise AuditError(f"payload value drift: {label}")


def _inventory(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["path"]: row for row in manifest["inventory"]}


def _verify_inventory(run_dir: Path, manifest: dict[str, Any]) -> dict[str, int]:
    expected = _inventory(manifest)
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


def _verified_manifest(path: Path, expected_sha256: str, status: str) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise AuditError(f"manifest identity drift: {path}")
    manifest = _json(path)
    if manifest.get("status") != status:
        raise AuditError(f"manifest terminal drift: {path}")
    return manifest


def _verified_npz(
    root: Path,
    inventory: dict[str, dict[str, Any]],
    relative: str,
) -> dict[str, np.ndarray]:
    path = root / relative
    record = inventory.get(relative)
    if (
        record is None
        or not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256(path) != record["sha256"]
    ):
        raise AuditError(f"persisted artifact identity drift: {relative}")
    return _npz(path)


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise AuditError("empty metric denominator")
    return {
        name: float(np.mean([row[name] for row in rows], dtype=np.float64))
        for name in METRICS
    }


def _delta(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    return {name: float(candidate[name] - baseline[name]) for name in METRICS}


def _balanced(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in rows], dtype=np.float64))
        for name in METRICS
    }


def _replay_gate(
    scene_reports: list[dict[str, Any]],
    primary: Mapping[str, Any],
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    if len(scene_reports) != int(primary["scene_count"]):
        raise AuditError("primary H scene denominator drift")
    if len(scene_reports) != int(mechanism["scene_count"]):
        raise AuditError("mechanism H scene denominator drift")
    primary_rows = []
    mechanism_rows = []
    scenes = []
    for scene in scene_reports:
        aggregate = scene["evaluation_aggregate"]
        primary_delta = _delta(aggregate["E0B"], aggregate["U2_B3_G0"])
        mechanism_delta = _delta(aggregate["E0B"], aggregate["D0"])
        primary_rows.append(primary_delta)
        mechanism_rows.append(mechanism_delta)
        scenes.append(
            {
                "scene": scene["scene"],
                "delta_vs_u2_b3_g0": primary_delta,
                "delta_vs_d0": mechanism_delta,
            }
        )

    primary_balanced = _balanced(primary_rows)
    mechanism_balanced = _balanced(mechanism_rows)
    positive_primary = sum(row["boundary_f1"] > 0.0 for row in primary_rows)
    nonnegative_mechanism = sum(row["boundary_f1"] >= 0.0 for row in mechanism_rows)
    primary_checks = {
        "positive_boundary_f1_scene_count": positive_primary
        >= int(primary["minimum_positive_boundary_f1_scenes"]),
        "scene_balanced_boundary_f1_positive": primary_balanced["boundary_f1"]
        > float(primary["minimum_scene_balanced_boundary_f1_delta_exclusive"]),
        "scene_balanced_iou_nonnegative": primary_balanced["iou_at_frozen_threshold"]
        >= float(primary["minimum_scene_balanced_iou_delta"]),
        "scene_balanced_false_negative_safeguard": primary_balanced[
            "false_negative_semantic_mass"
        ]
        <= float(primary["maximum_scene_balanced_false_negative_semantic_mass_delta"]),
    }
    mechanism_checks = {
        "nonnegative_boundary_f1_scene_count": nonnegative_mechanism
        >= int(mechanism["minimum_nonnegative_boundary_f1_scenes"]),
        "scene_balanced_boundary_f1_positive": mechanism_balanced["boundary_f1"]
        > float(mechanism["minimum_scene_balanced_boundary_f1_delta_exclusive"]),
        "scene_balanced_iou_nonnegative": mechanism_balanced["iou_at_frozen_threshold"]
        >= float(mechanism["minimum_scene_balanced_iou_delta"]),
        "scene_balanced_false_negative_nonincreasing": mechanism_balanced[
            "false_negative_semantic_mass"
        ]
        <= float(mechanism["maximum_scene_balanced_false_negative_semantic_mass_delta"]),
    }
    primary_pass = bool(all(primary_checks.values()))
    mechanism_pass = bool(all(mechanism_checks.values()))
    return {
        "pass": primary_pass and mechanism_pass,
        "primary_gate": {
            "pass": primary_pass,
            "checks": primary_checks,
            "positive_boundary_f1_scene_count": positive_primary,
            "scene_balanced_delta_vs_u2_b3_g0": primary_balanced,
        },
        "mechanism_gate": {
            "pass": mechanism_pass,
            "checks": mechanism_checks,
            "nonnegative_boundary_f1_scene_count": nonnegative_mechanism,
            "scene_balanced_delta_vs_d0": mechanism_balanced,
        },
        "scenes": scenes,
    }


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status = _json(run_dir / "status.json")
    summary = _json(run_dir / "summary.json")
    report = _json(run_dir / "artifacts/h_evaluation_report.json")
    manifest = _json(run_dir / "manifest.json")
    conclusion = "e0b_rejected_stop_e1_e2_advance_gaussian_grouping"
    if status.get("status") != "rejected" or manifest.get("status") != "rejected":
        raise AuditError("r022 status/manifest terminal must remain rejected")
    if summary.get("status") != "rejected" or report.get("method_status") != "rejected":
        raise AuditError("r022 summary/report terminal must remain rejected")
    if summary.get("conclusion") != conclusion or report.get("conclusion") != conclusion:
        raise AuditError("r022 conclusion drift")
    if status.get("conclusion") != conclusion:
        raise AuditError("r022 status conclusion drift")
    if (run_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"
    ) != config_path.read_text(encoding="utf-8"):
        raise AuditError("r022 resolved config is not byte-exact")
    inventory_report = _verify_inventory(run_dir, manifest)
    run_inventory = _inventory(manifest)
    if summary["source_commit"] != status["source_commit"]:
        raise AuditError("r022 source commit drift")
    source_tree = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", f"{summary['source_commit']}^{{tree}}"],
        text=True,
    ).strip()
    if source_tree != summary["source_tree"]:
        raise AuditError("r022 source tree drift")
    if summary["scene_reports"] != report["scene_reports"]:
        raise AuditError("r022 summary/report scene payload drift")

    d0_root = Path(config["d0_h_run"]["path"])
    d0_manifest = _verified_manifest(
        d0_root / "manifest.json",
        config["d0_h_run"]["manifest_sha256"],
        config["d0_h_run"]["required_status"],
    )
    d0_inventory = _inventory(d0_manifest)

    audited_scenes = []
    recomputed_scene_reports = []
    for scene_config, scene_report in zip(config["scenes"], summary["scene_reports"]):
        scene = scene_config["scene"]
        if scene not in SCENES or scene_report["scene"] != scene:
            raise AuditError("r022 scene order drift")
        expected_views = int(scene_config["expected_evaluation_view_count"])
        if int(scene_report["evaluation_view_count"]) != expected_views:
            raise AuditError(f"r022 view denominator drift: {scene}")
        graph_root = Path(scene_config["graph_run"]["path"])
        graph_manifest = _verified_manifest(
            graph_root / "manifest.json",
            scene_config["graph_run"]["manifest_sha256"],
            "done",
        )
        graph_inventory = _inventory(graph_manifest)
        persisted_scene = _json(run_dir / f"artifacts/reports/{scene}.json")
        _assert_payload(persisted_scene, scene_report, f"{scene}/persisted_report")

        rows = {arm: [] for arm in ARMS}
        reference_rows = scene_report["evaluation_rows"]["U2_B3_G0"]
        if len(reference_rows) != expected_views:
            raise AuditError(f"r022 reference row denominator drift: {scene}")
        for index, reference in enumerate(reference_rows):
            frame = int(reference["frame"])
            camera_id = int(reference["camera_id"])
            suffix = f"f{frame:03d}_c{camera_id}.npz"
            tables = {
                "U2_B3_G0": _verified_npz(
                    graph_root, graph_inventory, f"artifacts/evaluation/B3_G0/{suffix}"
                ),
                "D0": _verified_npz(
                    d0_root,
                    d0_inventory,
                    f"artifacts/evaluation/{scene}/D0/{suffix}",
                ),
                "E0B": _verified_npz(
                    run_dir,
                    run_inventory,
                    f"artifacts/evaluation/{scene}/E0B/{suffix}",
                ),
            }
            target = np.asarray(tables["U2_B3_G0"]["target"])
            for arm, table in tables.items():
                if set(table) != {"probability", "target"}:
                    raise AuditError(f"r022 persisted schema drift: {scene}/{arm}/{suffix}")
                if table["probability"].dtype != np.float16:
                    raise AuditError(f"r022 precision drift: {scene}/{arm}/{suffix}")
                if not np.array_equal(table["target"], target):
                    raise AuditError(f"r022 matched target drift: {scene}/{arm}/{suffix}")
                observed = _metric_row(
                    table["probability"].astype(np.float32),
                    target.astype(bool),
                    threshold=float(config["evaluation"]["probability_threshold"]),
                    boundary_tolerance=int(config["evaluation"]["boundary_tolerance_px"]),
                    ece_bins=int(config["evaluation"]["ece_bins"]),
                )
                recorded = scene_report["evaluation_rows"][arm][index]
                if int(recorded["frame"]) != frame or int(recorded["camera_id"]) != camera_id:
                    raise AuditError(f"r022 row order drift: {scene}/{arm}/{index}")
                for name in METRICS:
                    _assert_close(observed[name], recorded[name], f"{scene}/{arm}/{index}/{name}")
                rows[arm].append({name: float(observed[name]) for name in METRICS})

        aggregate = {arm: _aggregate(arm_rows) for arm, arm_rows in rows.items()}
        for arm in ARMS:
            for name in METRICS:
                _assert_close(
                    aggregate[arm][name],
                    scene_report["evaluation_aggregate"][arm][name],
                    f"{scene}/{arm}/aggregate/{name}",
                )
        primary_deltas = {
            arm: _delta(aggregate[arm], aggregate["U2_B3_G0"])
            for arm in ("D0", "E0B")
        }
        mechanism_delta = _delta(aggregate["E0B"], aggregate["D0"])
        _assert_payload(
            primary_deltas, scene_report["delta_vs_u2_b3_g0"], f"{scene}/primary_delta"
        )
        _assert_payload(
            mechanism_delta, scene_report["delta_e0b_vs_d0"], f"{scene}/mechanism_delta"
        )
        if scene_report["checkpoint_sha256_before"] != scene_report["checkpoint_sha256_after"]:
            raise AuditError(f"r022 checkpoint mutation: {scene}")
        recomputed_scene_reports.append({"scene": scene, "evaluation_aggregate": aggregate})
        audited_scenes.append(
            {
                "scene": scene,
                "view_count": expected_views,
                "checkpoint_exact": True,
                "persisted_identity_exact": True,
                "metric_replay_exact": True,
            }
        )

    replayed_gate = _replay_gate(
        recomputed_scene_reports,
        config["primary_h_gate_vs_u2_b3_g0"],
        config["mechanism_h_gate_vs_d0"],
    )
    _assert_payload(replayed_gate, summary["h_gate"], "h_gate")
    if replayed_gate["pass"]:
        raise AuditError("r022 rejection gate unexpectedly passed")

    resources = summary["resources"]
    samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in samples if "monitor_error" not in row]
    if len(samples) != int(resources["sample_count"]) or len(valid) != len(samples):
        raise AuditError("r022 resource sample denominator/error drift")
    if max(int(row["gpu_used_mib"]) for row in valid) != int(resources["nvidia_peak_mib"]):
        raise AuditError("r022 NVIDIA peak replay drift")
    if max(int(row["cgroup_memory_current_bytes"]) for row in valid) != int(
        resources["cgroup_memory_peak_bytes"]
    ):
        raise AuditError("r022 cgroup peak replay drift")
    if not all(summary["resource_checks"].values()):
        raise AuditError("r022 resource gate drift")
    if summary.get("h_quality_read") is not True:
        raise AuditError("r022 H read lock drift")
    for lock in (
        "parameter_search",
        "e1_panogs_execution",
        "e2_ag2aussian_execution",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
    ):
        if summary.get(lock) is not False:
            raise AuditError(f"r022 forbidden execution/read lock drift: {lock}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("r022 M2/M3 status drift")

    return {
        "status": "passed",
        "conclusion": "r022_rejected_result_and_dual_matched_h_gate_replay_exact",
        "run": str(run_dir),
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "inventory": inventory_report,
        "scene_audits": audited_scenes,
        "total_view_count": sum(row["view_count"] for row in audited_scenes),
        "h_gate": replayed_gate,
        "resources": resources,
        "quality_locks": {
            "h_quality_read": True,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
        },
        "hashes": {
            name: _sha256(run_dir / name)
            for name in (
                "resolved_config.yaml",
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
        default=PROJECT / "configs/worldsim_v51/stage_e_e0b_h_evaluation_v1.yaml",
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
