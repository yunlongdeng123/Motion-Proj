#!/usr/bin/env python
"""只读集成 WorldSim V3 A0–A4/F0 canonical 证据并生成 R0 复现包。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/r0_integration_protocol_v1.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"R0 integration invalid: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        return {
            "command": list(command),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "wall_seconds": time.monotonic() - started,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": str(error),
            "wall_seconds": time.monotonic() - started,
        }


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-R0-INTEGRATION-01", "task_id")
    require(protocol["profile_id"] == "R0-INTEGRATION-v1", "profile_id")
    require(
        protocol["protocol_status"] == "frozen_before_canonical_integration",
        "protocol_status",
    )
    require(protocol["seed"] == 0, "seed")
    require(len(protocol["prerequisite_closeout_commit"]) == 40, "closeout commit")

    authorization = protocol["authorization"]
    for key in (
        "read_existing_evidence_authorized",
        "hash_existing_assets_authorized",
        "derive_json_csv_markdown_reports_authorized",
        "snapshot_documentation_authorized",
        "verify_chunk_package_payloads_authorized",
    ):
        require(authorization[key], f"authorization {key}")
    for key in (
        "training_authorized",
        "model_inference_authorized",
        "gpu_launch_authorized",
        "checkpoint_or_registry_mutation_authorized",
        "dependency_install_authorized",
        "download_authorized",
        "f1_pilot_authorized",
        "p4_lod_authorized",
        "a3_additional_refinement_authorized",
        "d3_d4_authorized",
    ):
        require(not authorization[key], f"authorization {key}")

    formal = protocol["formal_integration"]
    require(len(protocol["protocol_inputs"]) == formal["expected_protocol_input_count"], "protocol input count")
    canonical_count = sum(len(group["files"]) for group in protocol["canonical_evidence"].values())
    require(canonical_count == formal["expected_canonical_evidence_file_count"], "canonical evidence count")
    require(formal["expected_selected_asset_file_count"] == 3, "selected asset count")
    require(
        len(protocol["visualization_package"]["files"])
        == formal["expected_visualization_file_count"],
        "visualization file count",
    )
    require(len(protocol["deliverables"]) == formal["expected_deliverable_count"] == 12, "deliverable count")
    require(
        [row["id"] for row in protocol["deliverables"]]
        == [f"D{index:02d}" for index in range(1, 13)],
        "deliverable IDs",
    )
    require(len({row["output"] for row in protocol["deliverables"]}) == 12, "deliverable outputs")

    expected = protocol["expected_decisions"]
    require(expected["a1"]["decision_status"] == "done_off", "A1 decision")
    require(expected["f0"]["prerequisite_passed"] == 4, "F0 prerequisite pass count")
    require(expected["f0"]["f1_decision"] == "conditional_not_unlocked", "F1 decision")
    require(expected["a2"]["fixed_quality_verdict"] == "tradeoff_non_dominated", "A2 verdict")
    require(expected["a3"]["selected_arm"] == "R0-off", "A3 selected arm")
    require(expected["a4"]["p1_method_state"] == "rejected_quality_or_integrity_gate", "P1 result")
    require(expected["a4"]["p3_selected_arm"] == "p3-chunk-package", "P3 selection")

    selected = protocol["selected_production_asset"]
    require(selected["source_checkpoint"]["sha256"] == "7be87e8b0bdaf86c4a8066fa99a8c8a0a691ba9b5ea1c91ebf7b7fbf60287448", "selected checkpoint")
    require(selected["actor_registry"]["sha256"] == "69c4f38ae2a06d5fd3cd92d61dcc3595e3f05e70a4229f8827acd53d87448a27", "selected registry")
    require(selected["chunk_package"]["expected_files"] == 159, "package file count")
    require(selected["chunk_package"]["static_assets"] == 133, "static assets")
    require(selected["chunk_package"]["actor_assets"] == 24, "actor assets")

    allowed_conclusions = {
        "calibration_enhanced",
        "calibration_native_or_off_preferred",
        "actor_aware_supported",
        "actor_aware_rejected",
        "local_refine_supported",
        "local_refine_limited_to_observed_support",
        "deployment_pareto_supported",
        "engineering_blocked",
    }
    require(set(protocol["final_conclusions"]) <= allowed_conclusions, "final conclusion vocabulary")
    require("calibration_enhanced" not in protocol["final_conclusions"], "calibration enhancement inflation")
    require("local_refine_supported" not in protocol["final_conclusions"], "local refine inflation")
    require(all(protocol["claim_boundary"].values()), "claim boundary")


def audit_file(path: Path, expected: Mapping[str, Any], role: str) -> dict[str, Any]:
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_sha = sha256_file(path) if exists else None
    return {
        "role": role,
        "path": str(path),
        "exists": exists,
        "expected_bytes": int(expected["bytes"]),
        "actual_bytes": actual_bytes,
        "expected_sha256": expected["sha256"],
        "actual_sha256": actual_sha,
        "exact": exists and actual_bytes == int(expected["bytes"]) and actual_sha == expected["sha256"],
    }


def audit_all_inputs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for name, entry in protocol["protocol_inputs"].items():
        rows.append(audit_file(Path(entry["path"]), entry, f"protocol:{name}"))

    terminal_checks = {}
    canonical_rows = []
    for group_name, group in protocol["canonical_evidence"].items():
        root = Path(group["root"])
        for relative, expected in group["files"].items():
            row = audit_file(root / relative, expected, f"canonical:{group_name}:{relative}")
            rows.append(row)
            canonical_rows.append(row)
        terminal = load_json(root / "terminal.json")
        terminal_checks[group_name] = {
            "path": str(root / "terminal.json"),
            "expected": group["expected_terminal_status"],
            "actual": terminal.get("status"),
            "exact": terminal.get("status") == group["expected_terminal_status"],
        }

    selected_rows = []
    selected = protocol["selected_production_asset"]
    for name in ("source_checkpoint", "actor_registry", "source_config"):
        row = audit_file(Path(selected[name]["path"]), selected[name], f"selected:{name}")
        rows.append(row)
        selected_rows.append(row)

    visualization_rows = []
    for entry in protocol["visualization_package"]["files"]:
        row = audit_file(Path(entry["path"]), entry, f"visualization:{entry['role']}")
        rows.append(row)
        visualization_rows.append(row)

    require(all(row["exact"] for row in rows), "one or more frozen evidence files drifted")
    require(all(row["exact"] for row in terminal_checks.values()), "terminal status drift")
    return {
        "rows": rows,
        "protocol_input_count": len(protocol["protocol_inputs"]),
        "canonical_evidence_file_count": len(canonical_rows),
        "selected_asset_file_count": len(selected_rows),
        "visualization_file_count": len(visualization_rows),
        "terminal_checks": terminal_checks,
        "all_exact": True,
    }


def verify_chunk_package(protocol: Mapping[str, Any]) -> dict[str, Any]:
    package = protocol["selected_production_asset"]["chunk_package"]
    root = Path(package["root"])
    manifest_path = root / "manifest.json"
    require(sha256_file(manifest_path) == package["manifest_sha256"], "package manifest hash")
    manifest = load_json(manifest_path)
    payloads = [manifest["skeleton"], *manifest["static_assets"], *manifest["actor_assets"]]
    rows = []
    for entry in payloads:
        path = root / entry["path"]
        rows.append(audit_file(path, entry, f"chunk_payload:{entry['path']}"))
    require(all(row["exact"] for row in rows), "chunk package payload drift")
    require(len(manifest["static_assets"]) == package["static_assets"], "static package count")
    require(len(manifest["actor_assets"]) == package["actor_assets"], "actor package count")
    require(len(rows) + 1 == package["expected_files"], "package total file count")
    require(sum(row["actual_bytes"] for row in rows) + manifest_path.stat().st_size == package["total_bytes"], "package bytes")
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "payload_file_count": len(rows),
        "total_file_count": len(rows) + 1,
        "static_asset_count": len(manifest["static_assets"]),
        "actor_asset_count": len(manifest["actor_assets"]),
        "total_bytes": sum(row["actual_bytes"] for row in rows) + manifest_path.stat().st_size,
        "all_payloads_exact": all(row["exact"] for row in rows),
        "rows": rows,
    }


def evidence_json(protocol: Mapping[str, Any], group: str, relative: str = "summary.json") -> Any:
    return load_json(Path(protocol["canonical_evidence"][group]["root"]) / relative)


def validate_frozen_decisions(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = protocol["expected_decisions"]
    a0 = evidence_json(protocol, "a0")
    a1 = evidence_json(protocol, "a1")
    f0 = evidence_json(protocol, "f0")
    a2 = evidence_json(protocol, "a2")
    a3 = evidence_json(protocol, "a3_synthetic")
    p1 = evidence_json(protocol, "a4_p1")
    p2 = evidence_json(protocol, "a4_p2")
    p3 = evidence_json(protocol, "a4_p3")

    checks = {
        "a0_status": a0["status"] == expected["a0"]["status"],
        "a0_scene_count": a0["scene_count"] == expected["a0"]["scene_count"],
        "a1_status": a1["status"] == expected["a1"]["status"],
        "a1_decision_status": a1["decision_status"] == expected["a1"]["decision_status"],
        "a1_selected": a1["selected_variant"] == expected["a1"]["selected_variant"],
        "a1_logical_count": a1["decision"]["logical_matrix_entries"] == expected["a1"]["logical_matrix_entries"],
        "a1_training_count": a1["decision"]["unique_training_runs"] == expected["a1"]["unique_training_runs"],
        "f0_outcome": f0["audit_outcome"] == expected["f0"]["audit_outcome"],
        "f0_prerequisites": f0["smoke_prerequisites"]["passed_count"] == expected["f0"]["prerequisite_passed"] and f0["smoke_prerequisites"]["total_count"] == expected["f0"]["prerequisite_total"],
        "f0_inference": f0["inference_smoke"] == expected["f0"]["inference_smoke"],
        "f0_f1": f0["f1_decision"] == expected["f0"]["f1_decision"],
        "a2_status": a2["status"] == expected["a2"]["status"],
        "a2_complete": a2["d2_formal_complete"] == expected["a2"]["d2_formal_complete"],
        "a2_d3_locked": a2["d3_unlocked"] == expected["a2"]["d3_unlocked"],
        "a2_quality_tradeoff": a2["fixed_step"]["comparison"]["quality"]["verdict"] == expected["a2"]["fixed_quality_verdict"],
        "a2_cost_tradeoff": a2["fixed_step"]["comparison"]["quality_cost_pareto"]["verdict"] == expected["a2"]["fixed_quality_cost_verdict"],
        "a2_checkpoint": a2["matched_gaussian_budget"]["selected"]["checkpoint_sha256"] == expected["a2"]["selected_checkpoint_sha256"],
        "a3_synthetic_status": a3["status"] == "done",
        "a3_no_formal": a3["formal_training_authorized"] == expected["a3"]["formal_training_authorized"],
        "p1_rejected": p1["method_state"] == expected["a4"]["p1_method_state"],
        "p2_selected": p2["selection"]["selected_arm"] == expected["a4"]["p2_selected_arm"],
        "p3_status": p3["status"] == expected["a4"]["status"],
        "p3_selected": p3["selection"]["selected_arm"] == expected["a4"]["p3_selected_arm"],
    }
    require(all(checks.values()), "frozen decision drift")
    return {"checks": checks, "passed_count": sum(checks.values()), "total_count": len(checks), "all_passed": True}


def snapshot_documentation(protocol: Mapping[str, Any], artifacts: Path) -> dict[str, Any]:
    project = Path(protocol["documentation"]["project_root"])
    target = artifacts / "documentation"
    target.mkdir(parents=True)
    rows = []
    for relative in protocol["documentation"]["files"]:
        source = project / relative
        require(source.is_file(), f"documentation missing: {relative}")
        destination = target / Path(relative).name
        shutil.copyfile(source, destination)
        rows.append({
            "source_path": str(source),
            "snapshot_path": str(destination.relative_to(artifacts.parent)),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "source_snapshot_exact": sha256_file(source) == sha256_file(destination),
        })
    return {"source_commit": current_commit(), "files": rows, "all_exact": all(row["source_snapshot_exact"] for row in rows)}


def compact_a0(protocol: Mapping[str, Any]) -> dict[str, Any]:
    summary = evidence_json(protocol, "a0")
    rows = []
    for row in summary["rows"]:
        rows.append({key: row.get(key) for key in (
            "scene", "global_psnr", "global_ssim", "global_lpips",
            "background_gaussians", "rigid_gaussians", "total_gaussians",
            "train_seconds", "train_peak_gpu_mib", "high_actor_psnr",
            "boundary_actor_psnr", "boundary_status", "checkpoint_sha256",
        )})
    return {"status": "done", "scene_count": 3, "rows": rows, "source_summary": protocol["canonical_evidence"]["a0"]["root"] + "/summary.json"}


def compact_a1(protocol: Mapping[str, Any]) -> dict[str, Any]:
    summary = evidence_json(protocol, "a1")
    matrix = evidence_json(protocol, "a1", "artifacts/a1_matrix.json")
    return {
        "status": "done_off",
        "selected_variant": summary["selected_variant"],
        "decision": summary["decision"],
        "confirmation_matrix_rows": matrix["matrix_rows"],
        "source_summary": protocol["canonical_evidence"]["a1"]["root"] + "/summary.json",
    }


def compact_f0(protocol: Mapping[str, Any]) -> dict[str, Any]:
    summary = evidence_json(protocol, "f0")
    return {
        "status": "done",
        "audit_outcome": summary["audit_outcome"],
        "standalone_cli_export": summary["standalone_cli_export"],
        "standalone_cli_reads_lidar": summary["standalone_cli_reads_lidar"],
        "smoke_prerequisites": summary["smoke_prerequisites"],
        "inference_smoke": summary["inference_smoke"],
        "f1_decision": summary["f1_decision"],
        "f1_launched": summary["f1_launched"],
        "capability_matrix": evidence_json(protocol, "f0", "artifacts/capability_matrix.json"),
    }


def build_actor_quality(protocol: Mapping[str, Any]) -> dict[str, Any]:
    summary = evidence_json(protocol, "a2")
    actor = evidence_json(protocol, "a2", "artifacts/evaluations/fixed-d2-boundary-residual/actor_metrics/summary.json")
    return {
        "schema_version": 1,
        "derivation": "exact projection of frozen A2 D2 formal and actor-metrics summaries",
        "scene": actor["scene_name"],
        "selected_research_asset": "D2-boundary-residual",
        "selected_checkpoint": summary["matched_gaussian_budget"]["selected"],
        "fixed_step_quality_verdict": summary["fixed_step"]["comparison"]["quality"]["verdict"],
        "fixed_step_quality_cost_verdict": summary["fixed_step"]["comparison"]["quality_cost_pareto"]["verdict"],
        "roles": actor["roles"],
        "non_target": actor["non_target"],
        "mask_contract": actor["mask_contract"],
        "registry": {"path": actor["registry"], "sha256": actor["registry_sha256"]},
        "claim_boundary": summary["claim_boundary"],
        "not_a_dominance_claim": True,
    }


def build_a3_support(protocol: Mapping[str, Any]) -> dict[str, Any]:
    synthetic = evidence_json(protocol, "a3_synthetic")
    negative_terminal = evidence_json(protocol, "a3_negative", "terminal.json")
    resource = evidence_json(protocol, "a3_negative", "artifacts/resource_audit.json")
    return {
        "task_status": "done",
        "selected_arm": "R0-off",
        "selected_asset": "A2-D2 immutable exact alias",
        "support_types": {
            "S-A_observed": "local optimization is conceptually allowed, but no independent S-A formal arm was launched",
            "S-B_geometric": "depth/opacity/scale only; RGB loss forbidden; engineering chain replayed",
            "S-C_unsupported": "abstain; no pseudo 3D truth",
        },
        "synthetic_contract": synthetic,
        "negative_terminal": negative_terminal,
        "negative_resource_audit": resource,
        "r1_method_state": "rejected_resource_gate_and_diagnostic_tradeoff",
        "formal_training_authorized": False,
        "r2_r4_launched": False,
    }


def build_a4(protocol: Mapping[str, Any], package_audit: Mapping[str, Any]) -> dict[str, Any]:
    p0 = evidence_json(protocol, "a4_p0")
    p5 = evidence_json(protocol, "a4_p5")
    p1 = evidence_json(protocol, "a4_p1")
    p2 = evidence_json(protocol, "a4_p2")
    p3 = evidence_json(protocol, "a4_p3")
    return {
        "status": "done",
        "p0": {"status": p0["status"], "performance": p0["performance"], "resources": p0["resources"]},
        "p5": {"status": p5["status"], "deployment_registry": p5["deployment_registry"], "reload": p5["reload"], "recovery": p5["recovery"]},
        "p1": {"status": p1["status"], "selection": p1["selection"], "resources": p1["resources"]},
        "p2": {"status": p2["status"], "selection": p2["selection"], "selected_asset": p2["selected_asset"], "runtime": p2["runtime"], "resources": p2["resources"]},
        "p3": {"status": p3["status"], "selection": p3["selection"], "selected_asset": p3["selected_asset"], "runtime": p3["runtime"], "resources": p3["resources"]},
        "selected_production_asset": protocol["selected_production_asset"],
        "package_audit": {key: package_audit[key] for key in (
            "manifest", "payload_file_count", "total_file_count", "static_asset_count",
            "actor_asset_count", "total_bytes", "all_payloads_exact",
        )},
    }


def main_table(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    roots = protocol["canonical_evidence"]
    return [
        {"stage": "A0", "status": "done", "selected": "native StreetGS per scene", "result": "3/3 scene baseline", "evidence": roots["a0"]["root"]},
        {"stage": "A1", "status": "done_off", "selected": "C0-off", "result": "10 logical / 8 unique; confirmation directions scene-dependent", "evidence": roots["a1"]["root"]},
        {"stage": "F0", "status": "done", "selected": "audit only; F1 not unlocked", "result": "4/11 prerequisites; local inference not run", "evidence": roots["f0"]["root"]},
        {"stage": "A2", "status": "done", "selected": "D2 boundary-priority; D1 fallback", "result": "fixed/matched tradeoff_non_dominated", "evidence": roots["a2"]["root"]},
        {"stage": "A3", "status": "done", "selected": "R0-off / D2 exact alias", "result": "R1 rejected by resource gate and diagnostic tradeoff", "evidence": roots["a3_negative"]["root"]},
        {"stage": "A4", "status": "done", "selected": "P2 mixed + P3 exact chunk + P5 registry/resume", "result": "P1 rejected; no size/load/render speedup claim for P3", "evidence": roots["a4_p3"]["root"]},
    ]


def build_pareto(protocol: Mapping[str, Any]) -> dict[str, Any]:
    a0 = compact_a0(protocol)
    a1 = compact_a1(protocol)
    a2 = evidence_json(protocol, "a2")
    p1 = evidence_json(protocol, "a4_p1")
    p2 = evidence_json(protocol, "a4_p2")
    p3 = evidence_json(protocol, "a4_p3")
    return {
        "scope": "frozen report-only axes; values are not statistically generalized",
        "a0_scene_baselines": a0["rows"],
        "a1_confirmation_rows": a1["confirmation_matrix_rows"],
        "a2_fixed_quality": a2["fixed_step"]["comparison"]["quality"],
        "a2_fixed_quality_cost": a2["fixed_step"]["comparison"]["quality_cost_pareto"],
        "a2_matched_quality": a2["matched_gaussian_budget"]["comparison"]["quality"],
        "a2_matched_quality_cost": a2["matched_gaussian_budget"]["comparison"]["quality_cost_pareto"],
        "a3": {"qualified_pareto": False, "reason": "R1 failed frozen GPU resource ceiling; recalculation is diagnostic only"},
        "a4_p1": {"selection": p1["selection"], "arms": p1["arms"], "runtime": p1["runtime"], "resources": p1["resources"]},
        "a4_p2": {"selection": p2["selection"], "arms": p2["arms"], "runtime": p2["runtime"], "resources": p2["resources"]},
        "a4_p3": {"selection": p3["selection"], "runtime": p3["runtime"], "resources": p3["resources"]},
    }


def build_negative_results(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "results": [
            {"component": "A1", "outcome": "C1/C2/C3 did not pass the complete frozen contract; C0-off selected", "status": "done_off"},
            {"component": "F0", "outcome": "standalone CLI is static PLY only and local inference prerequisites failed", "status": "done"},
            {"component": "A2", "outcome": "D1 versus D2 is a boundary/global/cost tradeoff, not dominance", "status": "done"},
            {"component": "A3", "outcome": "R1 failed the frozen GPU ceiling and diagnostic Pareto remained a tradeoff", "status": "rejected"},
            {"component": "A4-P1", "outcome": "all prune candidates failed one or more quality safeguards", "status": "rejected"},
            {"component": "A4-P2", "outcome": "storage reduction selected; render speed did not improve", "status": "done"},
            {"component": "A4-P3", "outcome": "exact asset separation selected; package grew and load/render did not improve", "status": "done"},
        ],
        "claim_boundary": protocol["claim_boundary"],
        "final_conclusions": protocol["final_conclusions"],
    }


def build_reproducibility(
    protocol: Mapping[str, Any], input_audit: Mapping[str, Any], package_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "source_commit": current_commit(),
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "commands": [
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/finalize_worldsim_v3_a0.py",
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/finalize_worldsim_v3_a1.py",
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/run_worldsim_v3_a2_d2_formal.py",
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/eval_worldsim_v3_a3_r1_heldout.py",
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/run_worldsim_v3_a4_p3_chunk.py",
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/run_worldsim_v3_f0_instant_nurec_audit.py",
            "/root/autodl-tmp/envs/motionproj/bin/python scripts/run_worldsim_v3_r0_integration.py",
        ],
        "protocol_inputs": protocol["protocol_inputs"],
        "selected_production_asset": protocol["selected_production_asset"],
        "evidence_audit": input_audit,
        "chunk_package_audit": package_audit,
        "seed": protocol["seed"],
        "no_new_training_or_inference": True,
    }


def visualization_index(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": protocol["visualization_package"]["kind"],
        "scope": "existing scene-0230 D2 held-out offline videos; no new rendering or UI",
        "files": protocol["visualization_package"]["files"],
        "selected_asset_manifest": str(Path(protocol["selected_production_asset"]["chunk_package"]["root"]) / "manifest.json"),
        "raw_media_copied_into_r0": False,
    }


def csv_main_table(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "status", "selected", "result", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_report(protocol: Mapping[str, Any], table: list[dict[str, Any]]) -> str:
    lines = [
        "# WorldSim V3.1 R0 Final Model Report",
        "",
        "## Integrated model chain",
        "",
        f"`{protocol['selected_production_asset']['method_chain']}`",
        "",
        "| Stage | Status | Selected | Result |",
        "|---|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['stage']} | {row['status']} | {row['selected']} | {row['result']} |")
    lines.extend([
        "",
        "## Final conclusions",
        "",
        *[f"- `{value}`" for value in protocol["final_conclusions"]],
        "",
        "`actor_aware_supported` means the D2 boundary-priority research asset is auditable and selected for the final chain; it does not mean D2 dominates D1.",
        "`engineering_blocked` is scoped to local Instant NuRec inference on this host, not to the complete project or permanent upstream feasibility.",
        "",
        "## Evidence boundary",
        "",
        *[f"- `{key}`" for key, value in protocol["claim_boundary"].items() if value],
        "",
        "All detailed metrics, hashes, commands and asset paths are recorded in the sibling JSON/CSV artifacts.",
        "",
    ])
    return "\n".join(lines)


def artifact_fingerprint(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_int_file(path: Path) -> int | str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else value


def oom_counts() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {"oom": 0, "oom_kill": 0}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    return {"oom": values.get("oom", 0), "oom_kill": values.get("oom_kill", 0)}


def current_commit() -> str:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=PROJECT)
    require(result["exit_code"] == 0, "cannot resolve source commit")
    return result["stdout"].strip()


def make_run_dir(protocol: Mapping[str, Any], explicit: str | None) -> Path:
    if explicit:
        run_dir = Path(explicit)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(protocol["formal_integration"]["run_root"]) / f"{timestamp}__{protocol['formal_integration']['run_slug']}"
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def execute(protocol_path: Path, run_dir_override: str | None = None) -> Path:
    started = time.monotonic()
    oom_before = oom_counts()
    protocol = OmegaConf.to_container(OmegaConf.load(protocol_path), resolve=True)
    validate_schema(protocol)
    require(current_commit() != protocol["prerequisite_closeout_commit"], "R0 protocol must be committed before formal integration")
    run_dir = make_run_dir(protocol, run_dir_override)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    shutil.copyfile(protocol_path, run_dir / "protocol.yaml")

    input_audit = audit_all_inputs(protocol)
    decision_audit = validate_frozen_decisions(protocol)
    package_audit = verify_chunk_package(protocol)
    documentation = snapshot_documentation(protocol, artifacts)
    a0 = compact_a0(protocol)
    a1 = compact_a1(protocol)
    f0 = compact_f0(protocol)
    actor_quality = build_actor_quality(protocol)
    a3 = build_a3_support(protocol)
    a4 = build_a4(protocol, package_audit)
    table = main_table(protocol)
    pareto = build_pareto(protocol)
    negatives = build_negative_results(protocol)
    reproducibility = build_reproducibility(protocol, input_audit, package_audit)
    visualization = visualization_index(protocol)

    deliverables = {
        "documentation_snapshot.json": documentation,
        "a0_baseline.json": a0,
        "a1_calibration.json": a1,
        "f0_audit.json": f0,
        "actor_quality.json": actor_quality,
        "a3_local_refine_support.json": a3,
        "a4_deployment.json": a4,
        "a0_a4_main_table.json": {"rows": table},
        "quality_scale_time_vram_pareto.json": pareto,
        "negative_results_and_boundaries.json": negatives,
        "reproducibility_manifest.json": reproducibility,
        "visualization_index.json": visualization,
    }
    for name, value in deliverables.items():
        atomic_json(artifacts / name, value)
    atomic_json(artifacts / "input_audit.json", input_audit)
    atomic_json(artifacts / "decision_audit.json", decision_audit)
    atomic_json(artifacts / "chunk_package_audit.json", package_audit)
    csv_main_table(artifacts / "a0_a4_main_table.csv", table)
    atomic_text(artifacts / "FINAL_MODEL_REPORT.md", build_report(protocol, table))

    required_outputs = [run_dir / row["output"] for row in protocol["deliverables"]]
    require(all(path.is_file() for path in required_outputs), "required deliverable missing")
    deliverable_audit = {
        "rows": [artifact_fingerprint(path, run_dir) for path in required_outputs],
        "passed_count": len(required_outputs),
        "total_count": len(required_outputs),
        "all_passed": True,
    }
    atomic_json(artifacts / "deliverable_audit.json", deliverable_audit)

    oom_after = oom_counts()
    disk = shutil.disk_usage(Path(protocol["formal_integration"]["run_root"]).parent)
    gpu_processes = run_command([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
    ])
    resource = {
        "wall_seconds": time.monotonic() - started,
        "cgroup_memory_current_bytes": read_int_file(Path("/sys/fs/cgroup/memory.current")),
        "cgroup_memory_max_bytes": read_int_file(Path("/sys/fs/cgroup/memory.max")),
        "disk_free_bytes": disk.free,
        "oom_events_delta": oom_after["oom"] - oom_before["oom"],
        "oom_kill_events_delta": oom_after["oom_kill"] - oom_before["oom_kill"],
        "torch_imported": "torch" in sys.modules,
        "gpu_launched": False,
        "training_launched": False,
        "model_inference_launched": False,
        "dependency_install_or_download_launched": False,
        "gpu_compute_processes": gpu_processes,
    }
    ceilings = protocol["formal_integration"]["resource_ceilings"]
    resource["checks"] = {
        "wall_time_seconds": resource["wall_seconds"] <= ceilings["wall_time_seconds"],
        "cgroup_memory": resource["cgroup_memory_current_bytes"] <= ceilings["peak_cgroup_memory_bytes"],
        "disk_free": resource["disk_free_bytes"] >= ceilings["disk_free_floor_bytes"],
        "oom": resource["oom_events_delta"] == ceilings["oom_events_delta"],
        "oom_kill": resource["oom_kill_events_delta"] == ceilings["oom_kill_events_delta"],
        "no_torch_gpu_training_inference_install_download": not any((
            resource["torch_imported"], resource["gpu_launched"], resource["training_launched"],
            resource["model_inference_launched"], resource["dependency_install_or_download_launched"],
        )),
    }
    require(all(resource["checks"].values()), "resource or no-launch audit failed")
    atomic_json(artifacts / "resource_audit.json", resource)

    summary = {
        "schema_version": 1,
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "status": "done",
        "source_commit": current_commit(),
        "protocol_sha256": sha256_file(protocol_path),
        "evidence_files_exact": len(input_audit["rows"]),
        "terminal_checks_exact": len(input_audit["terminal_checks"]),
        "decision_checks": decision_audit,
        "chunk_package": {key: package_audit[key] for key in (
            "payload_file_count", "total_file_count", "static_asset_count", "actor_asset_count",
            "total_bytes", "all_payloads_exact",
        )},
        "deliverables": deliverable_audit,
        "selected_production_asset": protocol["selected_production_asset"],
        "final_conclusions": protocol["final_conclusions"],
        "claim_boundary": protocol["claim_boundary"],
        "resource_audit": resource,
        "next_action": "none_plan_complete",
    }
    atomic_json(run_dir / "summary.json", summary)

    run_files = [run_dir / "protocol.yaml", *sorted(path for path in artifacts.rglob("*") if path.is_file()), run_dir / "summary.json"]
    manifest = {
        "schema_version": 1,
        "task_id": protocol["task_id"],
        "source_commit": summary["source_commit"],
        "protocol_sha256": summary["protocol_sha256"],
        "files": [artifact_fingerprint(path, run_dir) for path in run_files],
        "source_assets_mutated": False,
        "new_training_or_inference": False,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    run_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    require(run_bytes <= ceilings["run_bytes"], "run bytes ceiling")
    atomic_json(run_dir / "terminal.json", {"status": "done", "exit_code": 0})
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = execute(args.protocol, args.run_dir)
    except Exception as error:
        print(f"R0 integration failed: {error}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
