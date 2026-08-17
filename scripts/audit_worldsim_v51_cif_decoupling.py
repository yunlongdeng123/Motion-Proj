#!/usr/bin/env python3
"""Pre-quality identifiability audit for V5.1 A4 CIF-style decoupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.ownership_renderer import compose_ownership_opacity
from motion_proj.worldsim_v51.evidence.cif_decoupling import (
    compose_decoupled_actor_opacity,
)
from motion_proj.worldsim_v51.protocol import V51_BRANCH, load_yaml, sha256_file
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V51-M1-A-UNARY-OBSERVABILITY-01"
SCHEMA_VERSION = "worldsim_v51_m1_cif_decoupling_audit_v1"
RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v51")


class CifAuditError(RuntimeError):
    """A4 source binding or identifiability audit failure."""


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _prepare_run(run_dir: Path) -> str:
    resolved = run_dir.resolve()
    task_root = (RUN_ROOT / TASK_ID).resolve()
    if resolved.exists():
        raise CifAuditError(f"run directory already exists: {resolved}")
    if task_root not in resolved.parents:
        raise CifAuditError(f"run must be under {task_root}")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise CifAuditError(f"A4 audit must execute on {V51_BRANCH}")
    if _git("status", "--short"):
        raise CifAuditError("A4 formal audit requires a clean worktree")
    source_commit = _git("rev-parse", "HEAD")
    resolved.mkdir(parents=True)
    return source_commit


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CifAuditError("A4 config schema drift")
    if config.get("task_id") != TASK_ID or config.get("phase") != "a4_cif_decoupling_mechanism_audit":
        raise CifAuditError("A4 task/phase drift")
    if config.get("status") != "frozen_before_quality_read":
        raise CifAuditError("A4 audit was not frozen before quality read")
    a2_config_binding = config["inputs"]["a2_config"]
    verify_file(
        PROJECT / a2_config_binding["path"], a2_config_binding["sha256"]
    )
    a2_run_binding = config["inputs"]["a2_run"]
    a2_run = Path(a2_run_binding["path"])
    verify_file(a2_run / "summary.json", a2_run_binding["summary_sha256"])
    verify_file(a2_run / "manifest.json", a2_run_binding["manifest_sha256"])
    source_text = {}
    for name, binding in config["inputs"]["source_contracts"].items():
        path = PROJECT / binding["path"]
        verify_file(path, binding["sha256"])
        source_text[name] = path.read_text(encoding="utf-8")

    posterior_paths = sorted((a2_run / "artifacts/posteriors").glob("*.npz"))
    expected_names = [f"{scene}.npz" for scene in sorted(config["scenes"])]
    if [path.name for path in posterior_paths] != expected_names:
        raise CifAuditError("A4 A2 posterior scene denominator drift")
    scene_rows = []
    all_fields = set()
    for path in posterior_paths:
        payload = _load_npz(path)
        fields = sorted(payload)
        all_fields.update(fields)
        occupancy_fields = [name for name in fields if "occupancy" in name]
        conditional = np.asarray(
            payload["conditional_actor_probability"], dtype=np.float32
        )
        sample_size = min(4096, int(conditional.size))
        sample = conditional[:sample_size]
        appearance = np.linspace(0.05, 0.95, sample_size, dtype=np.float32)
        existing = compose_ownership_opacity(appearance, sample)
        constant_occupancy = compose_decoupled_actor_opacity(
            appearance_opacity=appearance,
            occupancy_probability=np.ones(sample_size, dtype=np.float32),
            conditional_actor_probability=sample,
        )
        opacity_reuse = compose_decoupled_actor_opacity(
            appearance_opacity=appearance,
            occupancy_probability=appearance,
            conditional_actor_probability=sample,
        )
        scene_rows.append(
            {
                "scene": path.stem,
                "posterior_path": str(path),
                "posterior_sha256": sha256_file(path),
                "gaussian_count": int(conditional.size),
                "fields": fields,
                "occupancy_fields": occupancy_fields,
                "constant_occupancy_existing_renderer_bit_exact": bool(
                    np.array_equal(existing, constant_occupancy)
                ),
                "appearance_opacity_reuse_bit_exact": bool(
                    np.array_equal(existing, opacity_reuse)
                ),
                "appearance_opacity_reuse_max_abs_delta": float(
                    np.max(np.abs(existing - opacity_reuse), initial=0.0)
                ),
            }
        )

    renderer = source_text["ownership_renderer"]
    checks = {
        "renderer_keeps_base_opacity_and_ownership_as_distinct_inputs": (
            "base_opacities" in renderer and "probability" in renderer
        ),
        "renderer_multiplies_appearance_opacity_by_ownership": (
            "semantic_opacity = opacity * ownership" in renderer
        ),
        "a2_has_no_independent_occupancy_field": not any(
            "occupancy" in name for name in all_fields
        ),
        "constant_occupancy_is_exact_noop_all_scenes": all(
            row["constant_occupancy_existing_renderer_bit_exact"]
            for row in scene_rows
        ),
        "appearance_opacity_reuse_is_not_existing_renderer": all(
            not row["appearance_opacity_reuse_bit_exact"] for row in scene_rows
        ),
        "visibility_already_has_separate_semantic_eligibility": (
            "semantic_visibility_mask" in source_text["visibility"]
        ),
        "unknown_already_has_separate_probability": (
            "unknown_probability" in source_text["abstention"]
        ),
        "independent_occupancy_observable_available": False,
    }
    reject = (
        all(value for name, value in checks.items() if name != "independent_occupancy_observable_available")
        and not checks["independent_occupancy_observable_available"]
    )
    diagnostics_path = run_dir / "artifacts/diagnostics.json"
    atomic_json(
        diagnostics_path,
        {
            "schema_version": "worldsim_v51_m1_a4_cif_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "checks": checks,
            "scene_rows": scene_rows,
            "reference": config["reference"],
            "occupancy_candidates": config["occupancy_candidates"],
        },
    )
    return {
        "schema_version": "worldsim_v51_m1_a4_cif_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": config["phase"],
        "conclusion": (
            "a4_cif_decoupling_rejected_no_independent_occupancy_observable"
            if reject
            else "a4_cif_decoupling_mechanism_audit_inconclusive"
        ),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "seed": int(config["seed"]),
        "duration_seconds": 0.0,
        "checks": checks,
        "scene_summaries": scene_rows,
        "reference": config["reference"],
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "method_quality_inference_started": False,
        "gpu_renderer_started": False,
        "evaluation_artifact_read": False,
        "evaluation_quality_read": False,
        "training_started": False,
        "parameter_search_performed": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": "V51-F07",
    }


def _write_terminal(
    run_dir: Path,
    *,
    status: str,
    source_commit: str | None,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None = None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v51_m1_a4_cif_status_v1",
            "task_id": TASK_ID,
            "status": status,
            "source_commit": source_commit,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/m1_cif_decoupling_audit_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_commit = _prepare_run(run_dir)
    config = load_yaml(args.config.resolve())
    resolved_config = write_resolved_config(run_dir, config)
    events = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_commit}
    ]
    write_events(run_dir, events)
    started = time.perf_counter()
    try:
        summary = run(args.config.resolve(), run_dir)
        summary["duration_seconds"] = time.perf_counter() - started
        source_snapshot = copy_source_snapshot(
            run_dir,
            [
                args.config.resolve(),
                PROJECT / "configs/worldsim_v51/m1_unary_unknown_v1.yaml",
                PROJECT / "motion_proj/worldsim_v5/ownership_renderer.py",
                PROJECT / "motion_proj/worldsim_v51/evidence/cif_decoupling.py",
                PROJECT / "motion_proj/worldsim_v51/evidence/visibility.py",
                PROJECT / "motion_proj/worldsim_v51/evidence/abstention.py",
                PROJECT / "scripts/audit_worldsim_v51_cif_decoupling.py",
                PROJECT / "tests/test_worldsim_v51_cif_decoupling.py",
            ],
            PROJECT,
        )
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v51_m1_a4_cif_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_commit,
                "source_branch": V51_BRANCH,
                "worktree_clean": True,
                "resolved_config": resolved_config,
                "source_snapshot": source_snapshot,
                "python": sys.version,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        manifest_path = run_dir / "manifest.json"
        atomic_json(
            manifest_path,
            {
                "schema_version": "worldsim_v51_m1_a4_cif_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": inventory_files(
                    run_dir, {"manifest.json", "status.json"}
                ),
            },
        )
        _write_terminal(
            run_dir,
            status="done",
            source_commit=source_commit,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        write_events(run_dir, events)
        _write_terminal(
            run_dir,
            status="blocked",
            source_commit=source_commit,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
