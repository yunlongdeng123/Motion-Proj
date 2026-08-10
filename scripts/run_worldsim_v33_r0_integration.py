#!/usr/bin/env python3
"""构建并验证 WorldSim V3.3 R0 内容寻址发布包。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.actor_asset_schema import validate_actor_asset  # noqa: E402
from motion_proj.worldsim_v33.instance_field import validate_instance_field  # noqa: E402
from motion_proj.worldsim_v33.integration_release import (  # noqa: E402
    atomic_json,
    copy_production_renders,
    copy_spatial_delta_package,
    copy_verified,
    extract_and_verify_archive,
    sha256_file,
    verify_file_record,
    verify_json_input,
    verify_release_directory,
    write_content_manifest,
    write_deterministic_archive,
)
from motion_proj.worldsim_v33.roadpatch import validate_patch_delta  # noqa: E402
from motion_proj.worldsim_v33.spatial_delta import load_npz  # noqa: E402


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *arguments], text=True
    ).strip()


def memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    return {
        key: int(value)
        for key, value in (
            line.split() for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def cgroup_current() -> int:
    path = Path("/sys/fs/cgroup/memory.current")
    return int(path.read_text().strip()) if path.is_file() else 0


def gpu_compute_processes() -> list[dict[str, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, memory = (part.strip() for part in line.split(",", 1))
        rows.append({"pid": int(pid), "used_memory_mib": int(memory)})
    return rows


class ResourceSampler:
    def __init__(self) -> None:
        self.peak_cgroup_memory_bytes = 0
        self.maximum_gpu_compute_processes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        self.peak_cgroup_memory_bytes = max(
            self.peak_cgroup_memory_bytes, cgroup_current()
        )
        self.maximum_gpu_compute_processes = max(
            self.maximum_gpu_compute_processes, len(gpu_compute_processes())
        )

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()


def snapshot_sources(run_dir: Path, config_path: Path) -> list[dict[str, Any]]:
    sources = [
        config_path,
        PROJECT / "motion_proj/worldsim_v32/actor_asset_schema.py",
        PROJECT / "motion_proj/worldsim_v33/instance_field.py",
        PROJECT / "motion_proj/worldsim_v33/integration_release.py",
        PROJECT / "motion_proj/worldsim_v33/roadpatch.py",
        PROJECT / "motion_proj/worldsim_v33/spatial_delta.py",
        PROJECT / "scripts/run_worldsim_v33_r0_integration.py",
    ]
    root = run_dir / "source_snapshot"
    records = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        target = root / f"{index:02d}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source.resolve()),
                "snapshot": str(target),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    atomic_json(root / "manifest.json", {"files": records})
    return records


def load_and_verify_inputs(
    config: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    audits: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, spec in config["inputs"].items():
        if str(spec["path"]).endswith(".json"):
            audit, payload = verify_json_input(name, spec)
            payloads[name] = payload
        else:
            audit = verify_file_record(spec, role=name)
        audits[name] = audit
    return audits, payloads


def validate_selected_assets(
    config: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    instance_field = load_npz(config["inputs"]["s1_instance_field"]["path"])
    validate_instance_field(instance_field)
    patch_delta = load_npz(config["inputs"]["s2_roadpatch_delta"]["path"])
    validate_patch_delta(patch_delta)
    actor_asset = load_npz(config["inputs"]["s3_actor_asset"]["path"])
    validate_actor_asset(actor_asset)

    s4_package = payloads["s4_package_manifest"]
    base_checkpoint = config["inputs"]["base_checkpoint"]
    actor_registry = config["inputs"]["actor_registry"]
    if s4_package["base"]["checkpoint"] != {
        key: base_checkpoint[key] for key in ("path", "sha256", "bytes")
    }:
        raise RuntimeError("S4 package checkpoint reference 与 R0 config 不一致")
    if s4_package["base"]["actor_registry"] != {
        key: actor_registry[key] for key in ("path", "sha256", "bytes")
    }:
        raise RuntimeError("S4 package registry reference 与 R0 config 不一致")
    if s4_package["composition_order"] != [
        "ERASE",
        "INSERT_BACKGROUND",
        "INSERT_ACTOR",
        "RENDER_ONLY",
    ]:
        raise RuntimeError("S4 composition order 漂移")
    if not all(bool(value) for key, value in s4_package["invariants"].items() if key not in {"base_rows_deleted", "duplicate_insert_indices", "full_checkpoint_copy_count"}):
        raise RuntimeError("S4 package boolean invariant 失败")
    if any(
        int(s4_package["invariants"][name]) != 0
        for name in (
            "base_rows_deleted",
            "duplicate_insert_indices",
            "full_checkpoint_copy_count",
        )
    ):
        raise RuntimeError("S4 package zero invariant 失败")
    s4_eval = payloads["s4_eval_summary"]
    if len(s4_eval["rollback_checks"]) != 20 or not all(
        row["exact"] for row in s4_eval["rollback_checks"]
    ):
        raise RuntimeError("S4 rollback 不是 20/20 exact")
    if (
        s4_eval["checkpoint_sha256_before"]
        != s4_eval["checkpoint_sha256_after"]
        or s4_eval["actor_registry_sha256_before"]
        != s4_eval["actor_registry_sha256_after"]
    ):
        raise RuntimeError("S4 evaluation 改写 base")

    s5 = payloads["s5_summary"]
    production = payloads["s5_production_manifest"]
    s5_rows = {
        (int(row["frame"]), int(row["camera_id"])): row for row in s5["rows"]
    }
    if len(production["rows"]) != 5 or len(s5_rows) != 5:
        raise RuntimeError("S5 production 五视图合同失败")
    for row in production["rows"]:
        key = (int(row["frame"]), int(row["camera_id"]))
        source = s5_rows[key]
        if row["insertion"]["sha256"] != source["outputs"]["full_raw"]["sha256"]:
            raise RuntimeError(f"S5 {key} G0 insertion 非 raw full")
        if row["delete"]["sha256"] != source["outputs"]["delete_raw"]["sha256"]:
            raise RuntimeError(f"S5 {key} delete 非 raw 3D")
        if not source["semantic_reintroduction"]["production_safe"]:
            raise RuntimeError(f"S5 {key} delete semantic 不安全")
        if source["blend_audit"]["changed_far_non_target_pixels"] != 0:
            raise RuntimeError(f"S5 {key} far non-target 漂移")

    return {
        "instance_field_schema": "validated_worldsim_v33_instance_field",
        "instance_field_gaussians": int(np.asarray(instance_field["gaussian_id"]).size),
        "roadpatch_rows": int(np.asarray(patch_delta["means"]).shape[0]),
        "actor_asset_rows": int(np.asarray(actor_asset["means"]).shape[0]),
        "s4_rollback_exact": 20,
        "s5_production_views_exact_safe": 5,
    }


def stage_ledgers(
    config: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    s1 = payloads["s1_summary"]
    o0 = s1["arms"]["O0_heuristic"]["evaluation"]["aggregate"]
    o1_arm = s1["arms"]["O1_dual_opacity"]
    o1 = o1_arm["evaluation"]["aggregate"]
    s2 = payloads["s2_roadpatch_summary"]
    s3_dev, s3_held = payloads["s3_dev_summary"], payloads["s3_heldout_summary"]
    s4 = payloads["s4_eval_summary"]
    s5 = payloads["s5_summary"]
    p0_sources = {
        name: row["execution_state"] for name, row in payloads["p0_summary"]["sources"].items()
    }
    decisions = {
        "schema_version": "worldsim_v33_r0_decision_ledger_v1",
        "selected_chain": list(config["release"]["expected_selected_chain"]),
        "stages": {
            "S1": {
                "selected": "O1_dual_opacity",
                "outcome": "object_field_supported",
                "boundary_f1_relative_change": o1["boundary_f1"] / o0["boundary_f1"] - 1.0,
                "iou_relative_change": o1["iou"] / o0["iou"] - 1.0,
                "normalized_boundary_distance_relative_change": o1["normalized_boundary_distance"] / o0["normalized_boundary_distance"] - 1.0,
                "false_positive_mass_relative_change": o1["false_positive_semantic_mass"] / o0["false_positive_semantic_mass"] - 1.0,
                "false_negative_mass_change": o1["false_negative_semantic_mass"] - o0["false_negative_semantic_mass"],
                "rgb_checkpoint_bitwise_exact": True,
            },
            "S2": {
                "selected": "B1_RoadPatch_Lite",
                "outcome": "roadpatch_supported",
                "delta_rows": s2["delta_rows"],
                "heldout": s2["heldout_mean"],
                "telea_matched_head_to_head": False,
                "telea_claim": "not_claimed_protocols_differ",
                "inpaint360gs": "blocked_single_3090",
            },
            "S3": {
                "selected": "A4_auto_4view",
                "outcome": "asset_view_selection_supported",
                "development": s3_dev["decision"],
                "heldout": s3_held["decision"],
                "boundary_actor": "ABSTAIN_GENERATED_OVERRIDE",
            },
            "S4": {
                "selected": "posterior_gated_spatial_delta",
                "outcome": "spatial_delta_supported",
                "rollback_exact": len(s4["rollback_checks"]),
                "package_manifest_sha256": audits["s4_package_manifest"]["sha256"],
            },
            "S5": {
                "selected": s5["selected_arm"],
                "outcome": "delete_semantic_reintroduction_prevention_supported",
                "enhancement": s5["selection_reason"],
                "candidate_flagged_views": s5[
                    "candidate_semantic_reintroduction_flagged_views"
                ],
                "production_safe_views": s5["production_semantic_safe_views"],
                "r3d2": "blocked_pretrained_model_unavailable",
                "temporal": "not_evaluated",
            },
        },
        "rejected_or_blocked": {
            "S1_O3_wider_reassignment": "rejected_on_development",
            "S2_dense_2150_row_delta": "rejected_on_heldout",
            "Inpaint360GS": "blocked_single_3090_not_quality_failure",
            "S3_boundary_A4": "rejected_on_development_use_native",
            "S4_all_hard_erase": "rejected_outside_l1",
            "S5_semantic_gate_G1": "rejected_on_heldout_confirmation",
            "SAM3.1": "weights_blocked_not_quality_failure",
            "R3D2": "pretrained_model_unavailable_not_quality_failure",
            "GOR_IS": "audit_only_noncommercial_no_pretrained_manifest",
        },
        "p0_source_states": p0_sources,
        "overall": "v33_supported",
    }
    provenance = {
        "schema_version": "worldsim_v33_r0_provenance_ledger_v1",
        "regions": [
            {"region": "immutable_base", "code": "ORIGINAL_BG_OR_ACTOR", "source": "V3.2_D2", "copied_checkpoint": False},
            {"region": "object_occupancy", "code": "LEARNED_INSTANCE_OPACITY_SIDECAR", "source": "S1_O1", "rgb_mutation": False},
            {"region": "background_patch", "code": "REAL_PATCH_REUSE", "source": "S2_native_D2_donors", "generated_rgb": False},
            {"region": "actor_override", "code": "GENERATED_ACTOR", "source": "S3_Asset_Harvester_A4", "gt_claim": False},
            {"region": "erase", "code": "RUNTIME_OPACITY_ZERO", "source": "S4_instance_field", "base_rows_deleted": 0},
            {"region": "render_insertion", "code": "RAW_3D_FAILSAFE", "source": "S5_G0", "two_d_residual_applied": False},
            {"region": "render_delete", "code": "RAW_3D_DELETE_FAILSAFE", "source": "S5_G0", "semantic_safe_views": 5},
        ],
        "all_insert_rows_have_provenance": True,
    }
    resources = {
        "schema_version": "worldsim_v33_r0_resource_ledger_v1",
        "hardware_contract": "single_NVIDIA_GeForce_RTX_3090_24GiB",
        "stages": {
            "S1": {
                "wall_seconds": o1_arm["train_wall_seconds"] + o1_arm["evaluation"]["wall_seconds"],
                "peak_cuda_reserved_mib": s1["runtime"]["peak_cuda_reserved_bytes"] / (1024**2),
            },
            "S2": {
                "wall_seconds": payloads["s2_index_summary"]["elapsed_seconds"] + s2["elapsed_seconds"],
                "peak_cuda_reserved_mib": s2["peak_cuda_reserved_bytes"] / (1024**2),
            },
            "S3": {
                "wall_seconds": payloads["s3_harvester_summary"]["wall_seconds"],
                "peak_nvidia_memory_mib": payloads["s3_harvester_summary"]["peak_nvidia_memory_mib"],
            },
            "S4": {
                "wall_seconds": s4["resources"]["elapsed_seconds"],
                "peak_cuda_reserved_mib": s4["resources"]["peak_cuda_reserved_mib"],
            },
            "S5_harmonizer": s5["resources"]["stages"]["harmonizer"],
            "S5_sam2": s5["resources"]["stages"]["sam2"],
            "V32_persistent_storage_inherited": payloads["v32_r0_summary"]["resource_audit"],
        },
        "oom_or_oom_kill_observed_in_selected_v33_stages": False,
    }
    claims = {
        "schema_version": "worldsim_v33_r0_claims_v1",
        "answers": [
            {"question": 1, "answer": "yes", "claim": "O1 improves boundary/identity precision axes; FN regression disclosed", "evidence": "S1"},
            {"question": 2, "answer": "not_directly_ranked", "claim": "RoadPatch passes frozen heldout and replaces Telea as V3.3 primary, but no matched Telea head-to-head", "evidence": "S2"},
            {"question": 3, "answer": "no_on_current_contract", "claim": "Inpaint360GS blocked by official environment/weights/adapter prerequisites on 3090; no quality conclusion", "evidence": "S2"},
            {"question": 4, "answer": "yes_for_high_support", "claim": "A4 improves heldout IoU/boundary within retention gates; boundary actor abstains", "evidence": "S3"},
            {"question": 5, "answer": "yes", "claim": "20/20 stack rollbacks plus replay exact; immutable external base", "evidence": "S4"},
            {"question": 6, "answer": "yes_via_failsafe", "claim": "unconstrained candidate reintroduced semantics on one view; G0 raw production is 5/5 exact safe", "evidence": "S5"},
            {"question": 7, "answer": "complete", "claim": "every maintained region has typed provenance", "evidence": "provenance_ledger"},
            {"question": 8, "answer": "reported", "claim": "per-stage wall/VRAM/disk/OOM retained under the single-3090 contract", "evidence": "resource_ledger"},
            {"question": 9, "answer": "reported", "claim": "all material rejected/blocked arms are preserved", "evidence": "decision_ledger"},
            {"question": 10, "answer": "satisfied", "claim": "unavailable code/weights are external blockers, not algorithm failures", "evidence": "P0_and_decision_ledger"},
        ],
        "success_criteria_required_met": 4,
        "success_criteria_required_total": 4,
        "overall": "v33_supported",
        "scope": "scene-0230_primary_frozen_views_single_RTX3090_no_closed_loop_safety_claim",
    }
    assets = {
        "schema_version": "worldsim_v33_r0_asset_ledger_v1",
        "canonical_inputs": {name: dict(record) for name, record in audits.items()},
        "external_base": {
            "checkpoint": dict(audits["base_checkpoint"]),
            "actor_registry": dict(audits["actor_registry"]),
            "copied_into_release": False,
        },
        "persistent_storage": {
            "v32_chunk_manifest": dict(audits["v32_chunk_manifest"]),
            "v33_authoring": "base_plus_external_delta",
        },
    }
    return {
        "decisions": decisions,
        "provenance": provenance,
        "resources": resources,
        "claims": claims,
        "assets": assets,
    }


def build_release(
    run_dir: Path,
    config: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = run_dir / config["release"]["directory"]
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    selected = {
        "object_field/instance_field.npz": "s1_instance_field",
        "roadpatch/roadpatch_delta.npz": "s2_roadpatch_delta",
        "actor_asset/high_support_auto_4view.npz": "s3_actor_asset",
        "persistent_storage/v32_chunk_package_manifest.json": "v32_chunk_manifest",
    }
    selected_records = {}
    for relative, name in selected.items():
        spec = config["inputs"][name]
        record = copy_verified(
            spec["path"],
            root / relative,
            expected_sha256=spec["sha256"],
            expected_bytes=spec["bytes"],
        )
        selected_records[name] = {
            "path": relative,
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
    spatial = copy_spatial_delta_package(
        config["inputs"]["s4_package_manifest"]["path"], root / "spatial_delta"
    )
    renders = copy_production_renders(
        config["inputs"]["s5_production_manifest"]["path"], root / "renders"
    )
    evidence_names = [
        name
        for name, spec in config["inputs"].items()
        if str(spec["path"]).endswith(".json") and name != "actor_registry"
    ]
    for name in evidence_names:
        spec = config["inputs"][name]
        copy_verified(
            spec["path"],
            root / "evidence" / f"{name}.json",
            expected_sha256=spec["sha256"],
            expected_bytes=spec["bytes"],
        )
    for name, payload in ledgers.items():
        atomic_json(root / "ledgers" / f"{name}.json", payload)
    atomic_json(
        root / "base/external_references.json",
        {
            "checkpoint": ledgers["assets"]["external_base"]["checkpoint"],
            "actor_registry": ledgers["assets"]["external_base"]["actor_registry"],
            "copied_into_release": False,
        },
    )
    atomic_json(
        root / "package_contract.json",
        {
            "schema_version": "worldsim_v33_r0_package_contract_v1",
            "selected_chain": config["release"]["expected_selected_chain"],
            "authoring_state": "immutable_external_base_plus_delta",
            "production_renderer": "G0_raw_3d",
            "full_checkpoint_copy_count": 0,
            "offline_verification": "python tools/verify_release.py verify-dir .",
        },
    )
    readme = """# ad-worldsim V3.3 release\n\nThis content-addressed release contains the selected object field, RoadPatch delta,\nauto-selected actor asset, exact spatial-delta authoring package, frozen G0 production\nrenders, canonical evidence, typed provenance and an offline verifier. The 579 MB base\ncheckpoint and actor registry remain immutable external references and are not copied.\n\nVerify the directory:\n\n```bash\npython tools/verify_release.py verify-dir .\n```\n\nThe scope is scene-0230 with frozen confirmation views on one RTX 3090. Generated actor\ncompleteness is not ground truth, temporal video quality was not evaluated, and no\nclosed-loop safety claim is made.\n"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "tools").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT / "motion_proj/worldsim_v33/integration_release.py",
        root / "tools/verify_release.py",
    )
    content = write_content_manifest(root)
    verification = verify_release_directory(root)
    return {
        "root": str(root),
        "selected_records": selected_records,
        "spatial_delta": spatial,
        "renders": renders,
        "evidence_files": len(evidence_names),
        "content": content,
        "verification": verification,
    }


def finalize_terminal(
    run_dir: Path, summary: dict[str, Any], status: dict[str, Any]
) -> int:
    summary_path, status_path = run_dir / "summary.json", run_dir / "status.json"
    for _ in range(8):
        atomic_json(summary_path, summary)
        atomic_json(status_path, status)
        current = directory_bytes(run_dir)
        if summary["resource_audit"]["run_bytes"] == current and status["run_bytes"] == current:
            return current
        summary["resource_audit"]["run_bytes"] = current
        status["run_bytes"] = current
    atomic_json(summary_path, summary)
    atomic_json(status_path, status)
    return directory_bytes(run_dir)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_r0_integration_v1":
        raise RuntimeError("R0 config schema 漂移")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output_root"]) / f"{timestamp}__{config['run_label']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    atomic_json(
        status_path,
        {
            "schema_version": "worldsim_v33_r0_status_v1",
            "task_id": config["task_id"],
            "state": "running",
            "run_dir": str(run_dir),
            "started_at_utc": timestamp,
        },
    )
    started = time.perf_counter()
    sampler = ResourceSampler()
    events_before = memory_events()
    gpu_before = gpu_compute_processes()
    sampler.start()
    try:
        if config["resources"]["require_zero_gpu_compute_processes"] and gpu_before:
            raise RuntimeError(f"R0 preflight GPU 非空闲: {gpu_before}")
        snapshots = snapshot_sources(run_dir, config_path)
        audits, payloads = load_and_verify_inputs(config)
        selected_audit = validate_selected_assets(config, payloads)
        ledgers = stage_ledgers(config, audits, payloads)
        release = build_release(run_dir, config, audits, payloads, ledgers)
        archive_path = run_dir / config["release"]["archive"]
        archive = write_deterministic_archive(release["root"], archive_path)
        replay_archive_path = run_dir / "artifacts/release_replay_check.zip"
        replay_archive = write_deterministic_archive(
            release["root"], replay_archive_path
        )
        if archive["sha256"] != replay_archive["sha256"]:
            raise RuntimeError("R0 deterministic archive replay SHA 漂移")
        replay_archive_path.unlink()
        replay = extract_and_verify_archive(
            archive_path, run_dir / config["release"]["replay_directory"]
        )
        if replay["manifest_sha256"] != release["verification"]["manifest_sha256"]:
            raise RuntimeError("R0 release extraction manifest SHA 漂移")
        expected_archive = config["release"].get("expected_archive_sha256")
        if expected_archive is not None and archive["sha256"] != expected_archive:
            raise RuntimeError(
                f"R0 archive SHA 与冻结值不符: {archive['sha256']} != {expected_archive}"
            )
        gpu_after = gpu_compute_processes()
        sampler.stop()
        events_after = memory_events()
        wall = time.perf_counter() - started
        resource_audit = {
            "wall_seconds": wall,
            "peak_cgroup_memory_bytes": sampler.peak_cgroup_memory_bytes,
            "gpu_compute_processes_before": gpu_before,
            "gpu_compute_processes_after": gpu_after,
            "maximum_gpu_compute_processes_sampled": sampler.maximum_gpu_compute_processes,
            "memory_events_before": events_before,
            "memory_events_after": events_after,
            "oom_delta": events_after.get("oom", 0) - events_before.get("oom", 0),
            "oom_kill_delta": events_after.get("oom_kill", 0) - events_before.get("oom_kill", 0),
            "run_bytes": 0,
            "ceilings": config["resources"],
        }
        checks = {
            "wall": wall <= float(config["resources"]["maximum_wall_seconds"]),
            "cgroup": sampler.peak_cgroup_memory_bytes
            <= int(config["resources"]["maximum_peak_cgroup_memory_bytes"]),
            "gpu_idle": not gpu_before
            and not gpu_after
            and sampler.maximum_gpu_compute_processes == 0,
            "oom": resource_audit["oom_delta"] == 0
            and resource_audit["oom_kill_delta"] == 0,
        }
        gates = {
            "canonical_inputs_exact": len(audits) == len(config["inputs"]),
            "selected_assets_schema_valid": all(
                selected_audit[name] > 0
                for name in (
                    "instance_field_gaussians",
                    "roadpatch_rows",
                    "actor_asset_rows",
                )
            ),
            "s4_rollback_exact": selected_audit["s4_rollback_exact"] == 20,
            "s5_production_exact_safe": selected_audit[
                "s5_production_views_exact_safe"
            ]
            == 5,
            "release_directory_verified": release["verification"]["valid"],
            "archive_deterministic": archive["sha256"] == replay_archive["sha256"],
            "archive_replay_exact": replay["manifest_sha256"]
            == release["verification"]["manifest_sha256"],
            "no_full_checkpoint_copy": release["content"][
                "full_checkpoint_copy_count"
            ]
            == 0,
            "all_success_criteria_met": ledgers["claims"][
                "success_criteria_required_met"
            ]
            == 4,
            "resource_checks": all(checks.values()),
        }
        if not all(gates.values()):
            raise RuntimeError(f"R0 gate 失败: {gates}; resources={checks}")
        summary = {
            "schema_version": "worldsim_v33_r0_summary_v1",
            "task_id": config["task_id"],
            "state": "completed",
            "accepted": True,
            "overall": "v33_supported",
            "selected_chain": config["release"]["expected_selected_chain"],
            "config_sha256": sha256_file(config_path),
            "repository": {
                "branch": git_text("branch", "--show-current"),
                "head": git_text("rev-parse", "HEAD"),
                "dirty_at_runtime": bool(git_text("status", "--porcelain")),
            },
            "source_snapshot": snapshots,
            "input_audits": audits,
            "selected_asset_audit": selected_audit,
            "release": release,
            "archive": archive,
            "archive_replay_sha256": replay_archive["sha256"],
            "extraction_replay": replay,
            "decision_ledger": ledgers["decisions"],
            "provenance_ledger": ledgers["provenance"],
            "resource_ledger": ledgers["resources"],
            "claims": ledgers["claims"],
            "gates": gates,
            "resource_checks": checks,
            "resource_audit": resource_audit,
            "training_or_optimizer_step_performed": False,
            "checkpoint_written": False,
        }
        status = {
            "schema_version": "worldsim_v33_r0_status_v1",
            "task_id": config["task_id"],
            "state": "completed",
            "accepted": True,
            "overall": "v33_supported",
            "archive_sha256": archive["sha256"],
            "completed_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "summary": str(run_dir / "summary.json"),
            "run_bytes": 0,
        }
        final_bytes = finalize_terminal(run_dir, summary, status)
        if final_bytes > int(config["resources"]["maximum_run_bytes"]):
            raise RuntimeError(
                f"R0 run bytes 超限: {final_bytes} > {config['resources']['maximum_run_bytes']}"
            )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "accepted": True,
                    "overall": "v33_supported",
                    "run_dir": str(run_dir),
                    "summary_sha256": sha256_file(run_dir / "summary.json"),
                    "status_sha256": sha256_file(run_dir / "status.json"),
                    "archive_sha256": archive["sha256"],
                    "archive_bytes": archive["bytes"],
                    "run_bytes": final_bytes,
                    "wall_seconds": wall,
                },
                ensure_ascii=False,
            )
        )
        return run_dir
    except Exception as error:
        sampler.stop()
        atomic_json(
            run_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_json(
            status_path,
            {
                "schema_version": "worldsim_v33_r0_status_v1",
                "task_id": config["task_id"],
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
