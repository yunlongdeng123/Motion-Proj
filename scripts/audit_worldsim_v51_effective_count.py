#!/usr/bin/env python3
"""Pre-quality mechanism audit for V5.1 A3 Kish effective count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.bayesian_unary import observation_reliability
from motion_proj.worldsim_v51.evidence.effective_count import (
    audit_fractional_concentration_cap,
)
from motion_proj.worldsim_v51.evidence.visibility import semantic_visibility_mask
from motion_proj.worldsim_v51.protocol import (
    V51_BRANCH,
    load_yaml,
    sha256_file,
    verify_canonical_run,
)
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
SCHEMA_VERSION = "worldsim_v51_m1_effective_count_audit_v1"
RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v51")


class EffectiveCountAuditError(RuntimeError):
    """A3 mechanism-audit input or invariant failure."""


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
        raise EffectiveCountAuditError(f"run directory already exists: {resolved}")
    if task_root not in resolved.parents:
        raise EffectiveCountAuditError(f"run must be under {task_root}")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise EffectiveCountAuditError(f"A3 audit must execute on {V51_BRANCH}")
    if _git("status", "--short"):
        raise EffectiveCountAuditError("A3 formal audit requires a clean worktree")
    source_commit = _git("rev-parse", "HEAD")
    resolved.mkdir(parents=True)
    return source_commit


def _scene_audit(
    scene: str,
    spec: Mapping[str, Any],
    source_config_path: Path,
    config: Mapping[str, Any],
    a1_config: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = verify_canonical_run(scene, spec)
    source_config = load_yaml(source_config_path)
    unary = source_config["unary"]
    source_run = Path(spec["path"])
    a2_binding = config["inputs"]["a2_posteriors"][scene]
    a2_path = Path(a2_binding["path"])
    verify_file(a2_path, a2_binding["sha256"])
    a2_posterior = _load_npz(a2_path)
    frozen_count = np.asarray(
        a2_posterior["effective_observation_count"], dtype=np.float32
    )
    gaussian_count = int(frozen_count.size)
    weight_sum = np.zeros(gaussian_count, dtype=np.float64)
    squared_weight_sum = np.zeros(gaussian_count, dtype=np.float64)
    eligible_observation_count = np.zeros(gaussian_count, dtype=np.int64)
    observed_weight_min = np.inf
    observed_weight_max = -np.inf
    observation_paths = sorted((source_run / "artifacts/observations").glob("*.npz"))
    if len(observation_paths) != int(spec["evidence_view_count"]):
        raise EffectiveCountAuditError(f"A3 evidence denominator drift: {scene}")
    for path in observation_paths:
        observations = _load_npz(path)
        eligibility = semantic_visibility_mask(
            observations,
            minimum_visibility=float(a1_config["visibility"]["minimum_visibility"]),
        )
        reliability = observation_reliability(
            observations,
            sam_confidence_floor=float(unary["sam_confidence_floor"]),
            boundary_distance_scale_px=float(unary["boundary_distance_scale_px"]),
            depth_residual_scale_m=float(unary["depth_residual_scale_m"]),
        ).astype(np.float64)
        reliability *= eligibility
        if np.any((reliability < 0.0) | (reliability > 1.0)):
            raise EffectiveCountAuditError("A3 reliability left closed unit interval")
        positive = reliability[reliability > 0.0]
        if positive.size:
            observed_weight_min = min(observed_weight_min, float(positive.min()))
            observed_weight_max = max(observed_weight_max, float(positive.max()))
        gaussian_id = np.asarray(observations["gaussian_id"], dtype=np.int64)
        weight_sum += np.bincount(
            gaussian_id, weights=reliability, minlength=gaussian_count
        )
        squared_weight_sum += np.bincount(
            gaussian_id, weights=np.square(reliability), minlength=gaussian_count
        )
        eligible_observation_count += np.bincount(
            gaussian_id,
            weights=eligibility.astype(np.int64),
            minlength=gaussian_count,
        ).astype(np.int64)
    if not np.array_equal(weight_sum.astype(np.float32), frozen_count):
        raise EffectiveCountAuditError(f"A3 parent A2 effective count drift: {scene}")

    epsilon = float(config["effective_count"]["epsilon"])
    audit = audit_fractional_concentration_cap(
        weight_sum, squared_weight_sum, epsilon=epsilon
    )
    observed = weight_sum > 0.0
    raw = audit["fractional_concentration"][observed]
    kish_zero = audit["kish_effective_count_without_epsilon"][observed]
    kish = audit["kish_effective_count"][observed]
    cap_reduction = audit["cap_reduction"][observed]
    amplification = audit["replacement_amplification"][observed]
    relative_cap_reduction = np.divide(
        cap_reduction,
        raw,
        out=np.zeros_like(cap_reduction),
        where=raw > 0.0,
    )
    if np.any(kish_zero < raw):
        raise EffectiveCountAuditError(
            f"A3 mathematical unit-weight invariant failed: {scene}"
        )
    threshold = float(
        config["effective_count"]["prequality_mechanism_checks"][
            "maximum_meaningful_relative_cap_change"
        ]
    )
    return {
        **canonical,
        "gaussian_count": gaussian_count,
        "observed_gaussian_count": int(observed.sum()),
        "eligible_observation_count": int(eligible_observation_count.sum()),
        "observation_file_count": len(observation_paths),
        "weight_domain": {
            "minimum_positive": float(observed_weight_min),
            "maximum": float(observed_weight_max),
        },
        "parent_effective_count_float32_exact": True,
        "no_epsilon_kish_below_fractional_count": int((kish_zero < raw).sum()),
        "cap_changed_gaussian_count": int((cap_reduction > 0.0).sum()),
        "meaningful_cap_changed_gaussian_count": int(
            (relative_cap_reduction > threshold).sum()
        ),
        "maximum_absolute_cap_reduction": float(cap_reduction.max(initial=0.0)),
        "maximum_relative_cap_reduction": float(
            relative_cap_reduction.max(initial=0.0)
        ),
        "replacement_amplified_gaussian_count": int((amplification > 0.0).sum()),
        "replacement_equal_gaussian_count": int((amplification == 0.0).sum()),
        "replacement_amplification_quantiles": {
            str(q): float(np.quantile(amplification, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "fractional_count_quantiles": {
            str(q): float(np.quantile(raw, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "kish_count_quantiles": {
            str(q): float(np.quantile(kish, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "a2_posterior_path": str(a2_path),
        "a2_posterior_sha256": a2_binding["sha256"],
        "evaluation_artifacts_read": False,
        "evaluation_quality_read": False,
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise EffectiveCountAuditError("A3 config schema drift")
    if config.get("task_id") != TASK_ID or config.get("phase") != "a3_kish_effective_count_mechanism_audit":
        raise EffectiveCountAuditError("A3 task/phase drift")
    if config.get("status") != "frozen_before_quality_read":
        raise EffectiveCountAuditError("A3 audit was not frozen before quality read")
    a1_binding = config["inputs"]["a1_config"]
    a2_binding = config["inputs"]["a2_config"]
    a1_path = PROJECT / a1_binding["path"]
    a2_path = PROJECT / a2_binding["path"]
    verify_file(a1_path, a1_binding["sha256"])
    verify_file(a2_path, a2_binding["sha256"])
    a1_config = load_yaml(a1_path)
    a2_run_binding = config["inputs"]["a2_run"]
    a2_run = Path(a2_run_binding["path"])
    verify_file(a2_run / "summary.json", a2_run_binding["summary_sha256"])
    verify_file(a2_run / "manifest.json", a2_run_binding["manifest_sha256"])
    baseline_binding = a1_config["inputs"]["a0_baseline_config"]
    baseline_path = PROJECT / baseline_binding["path"]
    verify_file(baseline_path, baseline_binding["sha256"])
    baseline = load_yaml(baseline_path)
    if list(config["scenes"]) != list(baseline["canonical_runs"]):
        raise EffectiveCountAuditError("A3 H scene set or ordering drift")
    started = time.perf_counter()
    scenes = [
        _scene_audit(
            scene,
            baseline["canonical_runs"][scene],
            PROJECT / baseline["source_configs"][scene],
            config,
            a1_config,
        )
        for scene in config["scenes"]
    ]
    meaningful = sum(row["meaningful_cap_changed_gaussian_count"] for row in scenes)
    no_epsilon_violations = sum(
        row["no_epsilon_kish_below_fractional_count"] for row in scenes
    )
    amplified = sum(row["replacement_amplified_gaussian_count"] for row in scenes)
    observed = sum(row["observed_gaussian_count"] for row in scenes)
    checks = {
        "parent_effective_count_exact_all_scenes": all(
            row["parent_effective_count_float32_exact"] for row in scenes
        ),
        "no_epsilon_kish_never_below_fractional_mass": no_epsilon_violations == 0,
        "cap_has_no_meaningful_reduction": meaningful == 0,
        "formula_has_no_correlation_observable": config["effective_count"][
            "correlation_observable_present"
        ]
        is False,
    }
    diagnostics_path = run_dir / "artifacts/diagnostics.json"
    atomic_json(
        diagnostics_path,
        {
            "schema_version": "worldsim_v51_m1_a3_effective_count_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "checks": checks,
            "scenes": scenes,
        },
    )
    rejected = all(checks.values())
    return {
        "schema_version": "worldsim_v51_m1_a3_effective_count_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": config["phase"],
        "conclusion": (
            "a3_kish_cap_rejected_structural_noop_not_correlation_aware"
            if rejected
            else "a3_kish_cap_mechanism_audit_inconclusive"
        ),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "seed": int(config["seed"]),
        "duration_seconds": time.perf_counter() - started,
        "checks": checks,
        "scene_summaries": scenes,
        "aggregate": {
            "observed_gaussian_count": observed,
            "meaningful_cap_changed_gaussian_count": meaningful,
            "no_epsilon_kish_below_fractional_count": no_epsilon_violations,
            "replacement_amplified_gaussian_count": amplified,
            "replacement_amplified_ratio": float(amplified / observed),
        },
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "method_quality_inference_started": False,
        "gpu_renderer_started": False,
        "evaluation_artifact_read": False,
        "evaluation_quality_read": False,
        "parameter_search_performed": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": "V51-F05",
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
            "schema_version": "worldsim_v51_m1_a3_effective_count_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/m1_effective_count_audit_v1.yaml",
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
    try:
        summary = run(args.config.resolve(), run_dir)
        source_snapshot = copy_source_snapshot(
            run_dir,
            [
                args.config.resolve(),
                PROJECT / "configs/worldsim_v51/m1_unary_visibility_v1.yaml",
                PROJECT / "configs/worldsim_v51/m1_unary_unknown_v1.yaml",
                PROJECT / "motion_proj/worldsim_v51/evidence/effective_count.py",
                PROJECT / "motion_proj/worldsim_v51/evidence/visibility.py",
                PROJECT / "scripts/audit_worldsim_v51_effective_count.py",
                PROJECT / "tests/test_worldsim_v51_effective_count.py",
            ],
            PROJECT,
        )
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v51_m1_a3_effective_count_fingerprint_v1",
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
                "schema_version": "worldsim_v51_m1_a3_effective_count_manifest_v1",
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
