#!/usr/bin/env python3
"""用 development 冻结参数执行六场景 M1 validation confirmation。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.evidence_calibration import calibrator_from_dict
from motion_proj.worldsim_v4.evidence_state import (
    atomic_save_evidence_state,
    evidence_state_summary,
)
from motion_proj.worldsim_v4.evidence_temporal import TemporalEvidenceMemory
from motion_proj.worldsim_v4.beta_fusion import BetaEvidence
from scripts.run_worldsim_v4_m1 import (
    M1RunError,
    SNAPSHOT_FILES,
    TASK_ID,
    RUN_ROOT,
    _rgb_exact_pair,
    _state_for_actor,
    _verified,
    accepted_evaluation_rows,
    aggregate_methods,
    atomic_json,
    candidate_probability_vectors,
    gate_preview,
    gpu_compute_processes,
    load_json,
    load_yaml,
    sha256_file,
    _render_probabilities,
)


SNAPSHOT_VALIDATION_FILES = (
    *SNAPSHOT_FILES,
    "scripts/materialize_worldsim_v4_m1_validation_scene_config.py",
    "scripts/run_worldsim_v4_m1_validation.py",
)


def verify_development_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "development_frozen":
        raise M1RunError("validation requires status=development_frozen")
    registration = config.get("development_result")
    selection = config.get("frozen_selection")
    if not isinstance(registration, Mapping) or not isinstance(selection, Mapping):
        raise M1RunError("development registration/frozen selection is missing")
    if registration.get("status") != "done" or registration.get("gate_status") != "pass":
        raise M1RunError("registered development result is not done/pass")
    run_dir = Path(str(registration["run"]))
    files = {}
    for name, binding in registration["files"].items():
        files[name] = {
            "path": str(
                _verified(
                    {"path": str(run_dir / name), "sha256": binding["sha256"]},
                    label=f"development freeze/{name}",
                )
            ),
            "sha256": binding["sha256"],
        }
    status = load_json(Path(files["status.json"]["path"]))
    summary = load_json(Path(files["summary.json"]["path"]))
    calibration = load_json(Path(files["calibration.json"]["path"]))
    if (
        status.get("status") != "done"
        or status.get("phase") != "six_scene_development"
        or status.get("heldout_content_read") is not False
        or status.get("test_quality_read") is not False
    ):
        raise M1RunError("registered development status provenance drift")
    expected = {
        "evidence_arm": summary.get("selected_evidence_arm"),
        "calibration": summary.get("selected_calibration"),
        "mask_threshold": summary.get("selected_mask_threshold"),
        "temporal_retention": float(config["evidence"]["temporal"]["retention"]),
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise M1RunError("frozen M1 selection drift")
    if calibration.get("selected") != selection["calibration"]:
        raise M1RunError("frozen calibration artifact selection drift")
    parameters = calibration.get("candidates", {}).get(selection["calibration"])
    if not isinstance(parameters, Mapping):
        raise M1RunError("frozen calibration parameters are missing")
    calibrator = calibrator_from_dict(parameters)

    audit = config.get("development_freeze_audit")
    if not isinstance(audit, Mapping) or audit.get("status") != "done":
        raise M1RunError("development freeze audit registration is missing")
    audit_run = Path(str(audit["run"]))
    audit_files = {}
    for name, binding in audit["files"].items():
        audit_files[name] = {
            "path": str(
                _verified(
                    {"path": str(audit_run / name), "sha256": binding["sha256"]},
                    label=f"development freeze audit/{name}",
                )
            ),
            "sha256": binding["sha256"],
        }
    audit_status = load_json(Path(audit_files["status.json"]["path"]))
    if audit_status.get("status") != "done":
        raise M1RunError("development freeze audit is not terminal")
    return {
        "development_run": str(run_dir),
        "selection": dict(selection),
        "calibrator": calibrator,
        "calibrator_parameters": dict(parameters),
        "files": files,
        "freeze_audit_run": str(audit_run),
        "freeze_audit_files": audit_files,
    }


def validation_confirmation_gate(
    *,
    per_scene: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scene_mean: Mapping[str, Mapping[str, float]],
    gates: Mapping[str, Any],
    required_scene_count: int,
    base_exact: bool,
) -> dict[str, Any]:
    mean_gate = gate_preview(
        reference=scene_mean["v33_o1"],
        candidate=scene_mean["frozen_m1"],
        gates=gates,
        base_rgb_exact=base_exact,
        scope="six_scene_validation_read_only_confirmation",
    )
    support = []
    directions = {}
    for scene, methods in sorted(per_scene.items()):
        reference = methods["v33_o1"]["aggregate"]
        candidate = methods["frozen_m1"]["aggregate"]
        delta = {
            name: float(candidate[name] - reference[name])
            for name in (
                "boundary_f1",
                "false_negative_semantic_mass",
                "ece",
                "brier",
            )
        }
        directional = (
            delta["boundary_f1"] > 0.0
            and delta["false_negative_semantic_mass"]
            <= float(gates["false_negative_semantic_mass_delta_max"])
            and (delta["ece"] < 0.0 or delta["brier"] < 0.0)
        )
        directions[scene] = {"delta": delta, "directional_support": directional}
        if directional:
            support.append(scene)
    required_support = required_scene_count // 2 + 1
    checks = {
        "development_mean_gate_reapplied_without_selection": mean_gate["status"] == "pass",
        "strict_majority_of_all_required_scenes": len(support) >= required_support,
        "base_rgb_and_checkpoint_exact": base_exact,
    }
    return {
        "status": "pass" if all(checks.values()) else "reject",
        "checks": checks,
        "required_scene_count": required_scene_count,
        "evaluable_scene_count": len(per_scene),
        "required_directional_support_scene_count": required_support,
        "directional_support_scene_count": len(support),
        "directional_support_scenes": support,
        "per_scene_directions": directions,
        "mean_gate": mean_gate,
        "arm_search_performed": False,
        "calibration_fit_performed": False,
        "threshold_search_performed": False,
    }


def verify_runtime_python(
    configs: Sequence[Mapping[str, Any]], *, executable: str | Path
) -> str:
    ready = [config for config in configs if config.get("status") == "ready"]
    expected = {
        Path(str(config["runtime"]["drivestudio_python"])).resolve()
        for config in ready
    }
    if len(expected) != 1:
        raise M1RunError(f"validation runtime Python drift: {sorted(map(str, expected))}")
    expected_python = next(iter(expected))
    actual_python = Path(executable).resolve()
    if actual_python != expected_python:
        raise M1RunError(
            "M1 validation must run with the frozen DriveStudio Python: "
            f"expected={expected_python} actual={actual_python}"
        )
    return str(expected_python)


def run(
    *,
    project_root: Path,
    run_dir: Path,
    scene_config_paths: Sequence[Path],
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run directory exists: {run_dir}")
    if RUN_ROOT.resolve() not in run_dir.resolve().parents:
        raise M1RunError(f"M1 validation run must be under {RUN_ROOT}")
    git_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    ).strip():
        raise M1RunError("formal M1 validation requires a clean project")
    if gpu_compute_processes():
        raise M1RunError("GPU preflight is not empty")
    configs = [load_yaml(path) for path in scene_config_paths]
    if any(config.get("status") not in {"ready", "abstain"} for config in configs):
        raise M1RunError("validation scene config is not terminal")
    if any(config.get("partition") != "validation" for config in configs):
        raise M1RunError("validation scene config partition drift")
    source_path = _verified(configs[0]["source_config"], label="M1 source config")
    source_config = load_yaml(source_path)
    expected = list(source_config["protocol"]["validation_scenes"])
    scenes = [str(config["scene"]) for config in configs]
    if scenes != expected or len(set(scenes)) != len(expected):
        raise M1RunError(f"validation scene order/set drift: {scenes} != {expected}")
    if any(
        Path(config["source_config"]["path"]).resolve() != source_path.resolve()
        or config["source_config"]["sha256"] != configs[0]["source_config"]["sha256"]
        for config in configs
    ):
        raise M1RunError("validation configs do not share one M1 source config")
    if any(
        config.get(name) is not False
        for config in configs
        for name in (
            "development_content_read",
            "validation_content_read",
            "heldout_content_read",
            "test_quality_read",
        )
    ):
        raise M1RunError("validation scene provenance is not sealed")
    freeze = verify_development_freeze(source_config)
    selected_arm = str(freeze["selection"]["evidence_arm"])
    selected_calibration = str(freeze["selection"]["calibration"])
    selected_threshold = float(freeze["selection"]["mask_threshold"])
    ready_configs = [config for config in configs if config["status"] == "ready"]
    if not ready_configs:
        raise M1RunError("validation has no evaluable scene")
    runtime_python = verify_runtime_python(configs, executable=sys.executable)
    for config in ready_configs:
        arms = config["evaluation"].get("candidate_arms", {})
        if selected_arm.removeprefix("raw__") not in arms:
            raise M1RunError("validation config lacks the frozen evidence arm")

    started = time.monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts" / "states").mkdir(parents=True)
    (run_dir / "artifacts" / "predictions").mkdir(parents=True)
    (run_dir / "source_snapshot").mkdir()
    for path in scene_config_paths:
        shutil.copy2(path, run_dir / "source_snapshot" / f"{path.stem}.yaml")
    for relative in SNAPSHOT_VALIDATION_FILES:
        source = project_root / relative
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    status = {
        "schema_version": "worldsim_v4_m1_validation_status_v1",
        "task_id": TASK_ID,
        "status": "running",
        "phase": "six_scene_validation_confirmation",
        "scenes": scenes,
        "project_git_head": git_head,
        "development_frozen_selection_read": True,
        "runtime_python": runtime_python,
        "validation_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "status.json", status)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[dict[str, Any]] = []
    scene_records = [
        {"scene": config["scene"], "status": "abstain", "reason": config["reason"]}
        for config in configs
        if config["status"] == "abstain"
    ]
    base_rgb_checks = []
    checkpoint_checks = []
    temporal_records = []

    for scene_config in ready_configs:
        from motion_proj.worldsim_v33.instance_field import load_instance_field
        from scripts.lift_worldsim_v32_semantics import build_runtime

        scene = str(scene_config["scene"])
        checkpoint = _verified(scene_config["inputs"]["checkpoint"], label="checkpoint")
        runtime_source = _verified(
            scene_config["inputs"]["drivestudio_source_config"],
            label="DriveStudio source config",
        )
        field_path = _verified(
            scene_config["inputs"]["v33_o1_instance_field"], label="V3.3 O1 field"
        )
        checkpoint_before = sha256_file(checkpoint)
        field = load_instance_field(field_path)
        rows = accepted_evaluation_rows(scene_config)
        states = {}
        for role in scene_config["actors"]:
            state, _ = _state_for_actor(scene_config, field, role)
            states[role] = state
            atomic_save_evidence_state(
                run_dir / "artifacts" / "states" / scene / f"{role}.npz", state
            )
            memory = TemporalEvidenceMemory(
                state["gaussian_id"],
                retention=float(freeze["selection"]["temporal_retention"]),
            )
            observed = BetaEvidence(state["alpha"], state["beta"])
            actor_rows = [row for row in rows if row["role"] == role]
            final = None
            for _ in actor_rows:
                final = memory.update(state["gaussian_id"], observed)
            if final is None:
                raise M1RunError(f"validation actor has no accepted row: {scene}/{role}")
            temporal_records.append(
                {
                    "scene": scene,
                    "role": role,
                    "updates": len(actor_rows),
                    "retention": memory.retention,
                    "stationary_observation_max_abs": float(
                        max(
                            np.max(np.abs(final.alpha - observed.alpha)),
                            np.max(np.abs(final.beta - observed.beta)),
                        )
                    ),
                }
            )
        runtime_config = {
            "inputs": {
                "checkpoint": str(checkpoint),
                "source_config": str(runtime_source),
            },
            "runtimes": scene_config["runtime"],
        }
        dataset, trainer = build_runtime(runtime_config, device)
        first = rows[0]
        first_actor = scene_config["actors"][first["role"]]
        rgb_before = _rgb_exact_pair(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            row=first,
            model_index=int(first_actor["rigid_model_index"]),
            device=device,
        )
        for row in rows:
            role = str(row["role"])
            candidate_ids, vectors = candidate_probability_vectors(
                field=field,
                state=states[role],
                actor_instance_id=int(
                    scene_config["actors"][role]["dataset_instance_id"]
                ),
                candidate_arms=scene_config["evaluation"]["candidate_arms"],
            )
            rendered = _render_probabilities(
                trainer=trainer,
                dataset=dataset,
                frame=int(row["frame"]),
                camera=int(row["camera_id"]),
                global_ids=torch.from_numpy(candidate_ids.astype(np.int64)).to(device),
                probabilities=[vectors["v33_o1"], vectors[selected_arm]],
                device=device,
            )
            with np.load(row["mask"], allow_pickle=False) as arrays:
                target = arrays["binary"].astype(bool)
            frozen = freeze["calibrator"].transform(rendered[1].reshape(-1)).reshape(
                rendered[1].shape
            )
            samples.append(
                {
                    "scene": scene,
                    "role": role,
                    "frame": int(row["frame"]),
                    "camera_id": int(row["camera_id"]),
                    "target": target,
                    "ece_bins": int(scene_config["calibration"]["ece_bins"]),
                    "probabilities": {
                        "v33_o1": rendered[0].astype(np.float32),
                        "frozen_m1": frozen.astype(np.float32),
                    },
                }
            )
        rgb_after = _rgb_exact_pair(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            row=first,
            model_index=int(first_actor["rigid_model_index"]),
            device=device,
        )
        checkpoint_after = sha256_file(checkpoint)
        base_rgb_checks.append(
            {"scene": scene, "before": rgb_before, "after": rgb_after, "exact": rgb_before == rgb_after}
        )
        checkpoint_checks.append(
            {"scene": scene, "before": checkpoint_before, "after": checkpoint_after, "exact": checkpoint_before == checkpoint_after}
        )
        scene_records.append(
            {
                "scene": scene,
                "status": "done",
                "evaluation_view_count": len(rows),
                "states": {
                    role: evidence_state_summary(state) for role, state in states.items()
                },
            }
        )
        del trainer, dataset, field, states
        torch.cuda.empty_cache()

    methods = ["v33_o1", "frozen_m1"]
    per_scene, scene_mean = aggregate_methods(
        samples,
        methods=methods,
        evaluation=ready_configs[0]["evaluation"],
        thresholds={
            "v33_o1": float(ready_configs[0]["evaluation"]["mask_threshold"]),
            "frozen_m1": selected_threshold,
        },
    )
    base_exact = all(row["exact"] for row in base_rgb_checks + checkpoint_checks)
    confirmation = validation_confirmation_gate(
        per_scene=per_scene,
        scene_mean=scene_mean,
        gates=ready_configs[0]["gates"],
        required_scene_count=len(configs),
        base_exact=base_exact,
    )
    for sample in samples:
        stem = f"{sample['role']}__f{sample['frame']:03d}__c{sample['camera_id']}"
        for method in methods:
            output = run_dir / "artifacts" / "predictions" / sample["scene"] / method / f"{stem}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(
                output,
                np.round(np.clip(sample["probabilities"][method], 0.0, 1.0) * 255.0).astype(np.uint8),
            )
    atomic_json(
        run_dir / "calibration.json",
        {
            "schema_version": "worldsim_v4_m1_validation_calibration_v1",
            "fit_partition": "development",
            "application_partition": "validation",
            "selected": selected_calibration,
            "parameters": freeze["calibrator_parameters"],
            "development_calibration_artifact": freeze["files"]["calibration.json"],
            "fit_performed": False,
        },
    )
    accounting = {
        "required_scene_count": len(configs),
        "evaluable_scene_count": len(ready_configs),
        "abstain_scene_count": len(configs) - len(ready_configs),
        "evaluable_fraction": len(ready_configs) / len(configs),
        "quality_metric_denominator": "evaluable_scenes_only",
        "coverage_denominator": "all_required_scenes",
    }
    atomic_json(
        run_dir / "metrics.json",
        {
            "schema_version": "worldsim_v4_m1_validation_metrics_v1",
            "partition": "validation",
            "per_scene": per_scene,
            "scene_mean": scene_mean,
            "frozen_selection": freeze["selection"],
            "confirmation_gate": confirmation,
            "cohort_accounting": accounting,
            "arm_search_performed": False,
            "calibration_fit_performed": False,
            "threshold_search_performed": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v4_m1_validation_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": "six_scene_validation_confirmation",
        "scenes": scenes,
        "evaluable_scenes": [config["scene"] for config in ready_configs],
        "abstain_scenes": [config["scene"] for config in configs if config["status"] == "abstain"],
        "project_git_head": git_head,
        "scene_records": scene_records,
        "frozen_selection": freeze["selection"],
        "confirmation_gate": confirmation,
        "cohort_accounting": accounting,
        "development_freeze": {
            "run": freeze["development_run"],
            "audit_run": freeze["freeze_audit_run"],
        },
        "base_rgb_render_checks": base_rgb_checks,
        "checkpoint_checks": checkpoint_checks,
        "temporal_memory_checks": temporal_records,
        "development_frozen_selection_read": True,
        "validation_content_read": True,
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "duration_seconds": time.monotonic() - started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "summary.json", summary)
    manifest_rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "status.json"}:
            manifest_rows.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v4_m1_validation_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "files": manifest_rows,
        },
    )
    status.update(
        status="done",
        validation_content_read=True,
        validation_optimization_read=False,
        confirmation_status=confirmation["status"],
        summary_sha256=sha256_file(run_dir / "summary.json"),
        manifest_sha256=sha256_file(run_dir / "manifest.json"),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(run_dir / "status.json", status)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        summary = run(
            project_root=args.project_root.resolve(),
            run_dir=args.run_dir.resolve(),
            scene_config_paths=[path.resolve() for path in args.scene_config],
        )
    except Exception as error:
        if args.run_dir.exists():
            atomic_json(
                args.run_dir / "status.json",
                {
                    "schema_version": "worldsim_v4_m1_validation_status_v1",
                    "task_id": TASK_ID,
                    "status": "failed",
                    "phase": "six_scene_validation_confirmation",
                    "reason": type(error).__name__,
                    "error": str(error),
                    "validation_optimization_read": False,
                    "heldout_content_read": False,
                    "test_quality_read": False,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
