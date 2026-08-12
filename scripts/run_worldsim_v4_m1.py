#!/usr/bin/env python3
"""Run the two-scene WorldSim V4 M1 evidence-field smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.beta_fusion import BetaEvidence
from motion_proj.worldsim_v4.evidence_calibration import (
    RawCalibrator,
    fit_beta_calibration,
    fit_temperature,
)
from motion_proj.worldsim_v4.evidence_metrics import (
    brier_score,
    expected_calibration_error,
)
from motion_proj.worldsim_v4.evidence_renderer import rasterize_evidence_mask
from motion_proj.worldsim_v4.evidence_state import (
    atomic_save_evidence_state,
    build_evidence_state,
    evidence_state_summary,
)
from motion_proj.worldsim_v4.evidence_temporal import TemporalEvidenceMemory


TASK_ID = "WS-V4-M1-EVIDENCE-FIELD-01"
RUN_ROOT = Path(f"/root/autodl-tmp/runs/worldsim_v4/{TASK_ID}")
SNAPSHOT_FILES = (
    "configs/worldsim_v4/m1_evidence_v1.yaml",
    "motion_proj/worldsim_v4/beta_fusion.py",
    "motion_proj/worldsim_v4/evidence_state.py",
    "motion_proj/worldsim_v4/evidence_calibration.py",
    "motion_proj/worldsim_v4/evidence_temporal.py",
    "motion_proj/worldsim_v4/evidence_renderer.py",
    "motion_proj/worldsim_v4/evidence_metrics.py",
    "scripts/materialize_worldsim_v4_m1_scene_config.py",
    "scripts/run_worldsim_v4_m1.py",
)


class M1RunError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M1RunError(f"YAML root is not a mapping: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M1RunError(f"JSON root is not a mapping: {path}")
    return payload


def gpu_compute_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _verified(binding: Mapping[str, Any], *, label: str) -> Path:
    path = Path(binding["path"])
    if not path.is_file():
        raise M1RunError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise M1RunError(
            f"{label} SHA drift: expected={binding['sha256']} actual={actual}"
        )
    return path


def accepted_evaluation_rows(scene_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_path = _verified(
        scene_config["inputs"]["development_evaluation_masks"],
        label="development evaluation masks",
    )
    manifest = load_json(manifest_path)
    if not manifest.get("optimization_forbidden"):
        raise M1RunError("evaluation masks are not optimization-forbidden")
    if manifest.get("evaluation_partition") != "development":
        raise M1RunError("evaluation mask partition is not development")
    actors = scene_config["actors"]
    rows = []
    seen: set[tuple[str, int, int]] = set()
    for source in manifest["masks"]:
        key = (str(source["role"]), int(source["frame"]), int(source["camera_id"]))
        if key in seen:
            raise M1RunError(f"duplicate evaluation row: {key}")
        seen.add(key)
        if key[0] not in actors:
            raise M1RunError(f"evaluation row references unknown actor: {key[0]}")
        mask = Path(source["mask"])
        if sha256_file(mask) != source["mask_sha256"]:
            raise M1RunError(f"evaluation mask SHA drift: {mask}")
        if bool(source["accepted"]) and int(source["positive_pixels"]) > 0:
            rows.append(dict(source))
    if not rows:
        raise M1RunError("scene has no accepted development evaluation mask")
    return rows


def _render_probabilities(
    *, trainer, dataset, frame: int, camera: int, global_ids: torch.Tensor,
    probabilities: Sequence[np.ndarray], device: torch.device,
) -> list[np.ndarray]:
    from scripts.eval_worldsim_v3_a3_r1_heldout import release_trainer_render_info
    from scripts.run_worldsim_v33_s1_instance_field import collect_frozen_geometry

    gaussians, camera_model = collect_frozen_geometry(
        trainer, dataset, frame, camera, device
    )
    if int(gaussians.means.shape[0]) <= int(global_ids.max()):
        raise M1RunError("evidence Gaussian identity exceeds frozen geometry")
    means = gaussians.means.detach()[global_ids]
    quats = gaussians.quats.detach()[global_ids]
    scales = gaussians.scales.detach()[global_ids]
    outputs = []
    try:
        with torch.inference_mode():
            for probability in probabilities:
                rendered, _ = rasterize_evidence_mask(
                    means=means,
                    quats=quats,
                    scales=scales,
                    probability=torch.from_numpy(probability).to(device),
                    viewmats=torch.linalg.inv(camera_model.camtoworlds)[None, ...],
                    intrinsics=camera_model.Ks[None, ...],
                    width=int(camera_model.W),
                    height=int(camera_model.H),
                    near_plane=float(trainer.render_cfg.near_plane),
                    far_plane=float(trainer.render_cfg.far_plane),
                    packed=bool(trainer.render_cfg.packed),
                    radius_clip=float(trainer.render_cfg.get("radius_clip", 0.0)),
                    antialiased=bool(trainer.render_cfg.antialiased),
                )
                outputs.append(rendered.detach().float().cpu().numpy())
    finally:
        release_trainer_render_info(trainer)
    return outputs


def _state_for_actor(
    scene_config: Mapping[str, Any], field: Mapping[str, np.ndarray], role: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    actor = scene_config["actors"][role]
    sidecar_path = _verified(actor["semantic_sidecar"], label=f"semantic sidecar/{role}")
    with np.load(sidecar_path, allow_pickle=False) as arrays:
        semantic = {name: arrays[name] for name in arrays.files}
    evidence = scene_config["evidence"]
    update = evidence["update"]
    state = build_evidence_state(
        instance_field=field,
        semantic_sidecar=semantic,
        actor_instance_id=int(actor["dataset_instance_id"]),
        actor_token=str(actor["instance_token"]),
        prior_strength=float(evidence["prior"]["strength"]),
        unassigned_probability=float(evidence["prior"]["unassigned_probability"]),
        visibility_saturation_mass=float(update["visibility_saturation_mass"]),
        mask_confidence_floor=float(update["mask_confidence_floor"]),
        depth_confidence_floor=float(update["depth_confidence_floor"]),
        lidar_confidence_floor=float(update["lidar_confidence_floor"]),
        observed_authenticity=float(evidence["authenticity"]["observed_cross_view"]),
    )
    selected = np.flatnonzero(
        np.asarray(field["hard_instance_id"], dtype=np.int32)
        == int(actor["dataset_instance_id"])
    )
    if selected.size == 0:
        raise M1RunError(f"V3.3 O1 assigns no Gaussian to {role}")
    return state, selected


def _rgb_exact_pair(
    *, trainer, dataset, checkpoint: Path, row: Mapping[str, Any],
    model_index: int, device: torch.device,
) -> str:
    from scripts.materialize_worldsim_v3_a3_s_b_sidecar import render_variant

    rendered = render_variant(
        trainer=trainer,
        dataset=dataset,
        checkpoint=checkpoint,
        frame=int(row["frame"]),
        camera=int(row["camera_id"]),
        model_index=model_index,
        variant="original",
        device=device,
    )["rgb"]
    return array_sha256(rendered)


def mean_numeric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    names = sorted(
        set.intersection(
            *(
                {
                    name
                    for name, value in row.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                    and name not in {"frame", "camera_id"}
                }
                for row in rows
            )
        )
    )
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in names}


def candidate_probability_vectors(
    *,
    field: Mapping[str, np.ndarray],
    state: Mapping[str, np.ndarray],
    actor_instance_id: int,
    candidate_arms: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build O1-owned posterior-minus-uncertainty development arms."""

    if not candidate_arms or float(candidate_arms.get("risk_000", -1.0)) != 0.0:
        raise ValueError("candidate arms must include zero-penalty risk_000")
    hard = np.asarray(field["hard_instance_id"], dtype=np.int32)
    posterior = np.asarray(state["posterior"], dtype=np.float32)
    uncertainty = np.asarray(state["uncertainty"], dtype=np.float32)
    o1 = np.asarray(field["instance_opacity"], dtype=np.float32)
    if not (hard.shape == posterior.shape == uncertainty.shape == o1.shape):
        raise ValueError("field/state Gaussian shapes differ")
    owned = hard == int(actor_instance_id)
    ids = np.flatnonzero(owned)
    if ids.size == 0:
        raise ValueError("candidate arm union is empty")
    vectors = {"v33_o1": o1[ids].astype(np.float32)}
    standard_deviation = np.sqrt(np.maximum(uncertainty[ids], 0.0))
    for name, penalty in candidate_arms.items():
        value = float(penalty)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"candidate arm {name} penalty is invalid")
        vectors[f"raw__{name}"] = np.clip(
            posterior[ids] - value * standard_deviation, 0.0, 1.0
        ).astype(np.float32)
    return ids, vectors


def score_prediction(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    boundary_tolerance_pixels: float,
    ece_bins: int,
) -> dict[str, float]:
    from motion_proj.worldsim_v33.instance_renderer import binary_mask_metrics

    metrics = binary_mask_metrics(
        probability >= threshold,
        target,
        boundary_tolerance_pixels=boundary_tolerance_pixels,
    )
    metrics.update(
        brier=brier_score(probability, target),
        ece=expected_calibration_error(probability, target, bins=ece_bins),
    )
    return metrics


def aggregate_methods(
    samples: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    evaluation: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_scene: dict[str, Any] = {}
    for scene in sorted({str(sample["scene"]) for sample in samples}):
        selected = [sample for sample in samples if sample["scene"] == scene]
        per_scene[scene] = {}
        for method in methods:
            rows = []
            for sample in selected:
                metrics = score_prediction(
                    sample["probabilities"][method],
                    sample["target"],
                    threshold=float(
                        thresholds.get(method, evaluation["mask_threshold"])
                        if thresholds is not None
                        else evaluation["mask_threshold"]
                    ),
                    boundary_tolerance_pixels=float(
                        evaluation["boundary_tolerance_pixels"]
                    ),
                    ece_bins=int(sample["ece_bins"]),
                )
                rows.append(
                    {
                        "role": sample["role"],
                        "frame": sample["frame"],
                        "camera_id": sample["camera_id"],
                        **metrics,
                    }
                )
            per_scene[scene][method] = {
                "aggregate": mean_numeric(rows),
                "rows": rows,
            }
    scene_mean = {}
    for method in methods:
        scene_mean[method] = mean_numeric(
            [per_scene[scene][method]["aggregate"] for scene in sorted(per_scene)]
        )
    return per_scene, scene_mean


def choose_calibration(
    scene_mean: Mapping[str, Mapping[str, float]],
    candidates: Sequence[str],
    *,
    boundary_f1_max_degradation: float = 0.01,
    false_negative_mass_max_degradation: float = 0.01,
    calibration_metric_max_degradation: float = 0.01,
) -> str:
    if not candidates or any(name not in scene_mean for name in candidates):
        raise ValueError("calibration candidates are incomplete")
    raw = scene_mean["raw"]
    eligible = ["raw"]
    for name in candidates:
        if name == "raw":
            continue
        candidate = scene_mean[name]
        probability_improves = (
            candidate["ece"] < raw["ece"]
            and candidate["brier"]
            <= raw["brier"] + calibration_metric_max_degradation
        ) or (
            candidate["brier"] < raw["brier"]
            and candidate["ece"] <= raw["ece"] + calibration_metric_max_degradation
        )
        mask_preserved = (
            candidate["boundary_f1"]
            >= raw["boundary_f1"] - boundary_f1_max_degradation
            and candidate["false_negative_semantic_mass"]
            <= raw["false_negative_semantic_mass"]
            + false_negative_mass_max_degradation
        )
        if probability_improves and mask_preserved:
            eligible.append(name)
    return min(
        eligible,
        key=lambda name: (
            scene_mean[name]["brier"],
            scene_mean[name]["ece"],
            -scene_mean[name]["boundary_f1"],
            name,
        ),
    )


def choose_evidence_arm(
    scene_mean: Mapping[str, Mapping[str, float]],
    candidates: Sequence[str],
    *,
    false_negative_mass_max_degradation: float,
) -> str:
    if not candidates or any(name not in scene_mean for name in candidates):
        raise ValueError("evidence-arm metrics are incomplete")
    reference = scene_mean["v33_o1"]
    eligible = [
        name
        for name in candidates
        if scene_mean[name]["false_negative_semantic_mass"]
        <= reference["false_negative_semantic_mass"]
        + false_negative_mass_max_degradation
    ]
    pool = eligible or list(candidates)
    return max(
        pool,
        key=lambda name: (
            scene_mean[name]["boundary_f1"],
            scene_mean[name]["iou"],
            -scene_mean[name]["brier"],
            -scene_mean[name]["ece"],
            name,
        ),
    )


def choose_mask_threshold(
    search: Mapping[float, Mapping[str, float]],
    *,
    reference: Mapping[str, float],
    false_negative_mass_max_degradation: float,
) -> float:
    if not search:
        raise ValueError("mask-threshold search is empty")
    eligible = [
        threshold
        for threshold, metrics in search.items()
        if metrics["false_negative_semantic_mass"]
        <= reference["false_negative_semantic_mass"]
        + false_negative_mass_max_degradation
    ]
    pool = eligible or list(search)
    return float(
        max(
            pool,
            key=lambda threshold: (
                search[threshold]["boundary_f1"],
                search[threshold]["iou"],
                -search[threshold]["false_positive_semantic_mass"],
                threshold,
            ),
        )
    )


def gate_preview(
    *, reference: Mapping[str, float], candidate: Mapping[str, float],
    gates: Mapping[str, Any], base_rgb_exact: bool,
) -> dict[str, Any]:
    boundary_delta = candidate["boundary_f1"] - reference["boundary_f1"]
    fn_delta = (
        candidate["false_negative_semantic_mass"]
        - reference["false_negative_semantic_mass"]
    )
    ece_delta = candidate["ece"] - reference["ece"]
    brier_delta = candidate["brier"] - reference["brier"]
    tolerance = float(gates["other_calibration_metric_max_degradation"])
    checks = {
        "boundary_f1": boundary_delta
        >= float(gates["boundary_f1_scene_mean_delta_min"]),
        "false_negative_semantic_mass": fn_delta
        <= float(gates["false_negative_semantic_mass_delta_max"]),
        "calibration": (ece_delta < 0.0 and brier_delta <= tolerance)
        or (brier_delta < 0.0 and ece_delta <= tolerance),
        "base_rgb_exact": bool(base_rgb_exact),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "deltas": {
            "boundary_f1": float(boundary_delta),
            "false_negative_semantic_mass": float(fn_delta),
            "ece": float(ece_delta),
            "brier": float(brier_delta),
        },
        "scope": "two_scene_development_smoke_preview_not_frozen_m1_decision",
    }


def run(
    *, project_root: Path, run_dir: Path, scene_config_paths: Sequence[Path]
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run directory exists: {run_dir}")
    if RUN_ROOT.resolve() not in run_dir.resolve().parents:
        raise M1RunError(f"M1 run must be under {RUN_ROOT}")
    if len(scene_config_paths) != 2:
        raise M1RunError("M1 smoke requires exactly two scene configs")
    git_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    ).strip():
        raise M1RunError("formal M1 smoke requires a clean project")
    if gpu_compute_processes():
        raise M1RunError("GPU preflight is not empty")
    configs = [load_yaml(path) for path in scene_config_paths]
    if any(config.get("status") != "ready" for config in configs):
        raise M1RunError("M1 smoke scene config is not ready")
    scenes = [config["scene"] for config in configs]
    source_m1_config = _verified(configs[0]["source_config"], label="M1 source config")
    expected = list(load_yaml(source_m1_config)["protocol"]["smoke_scenes"])
    if scenes != expected or len(set(scenes)) != 2:
        raise M1RunError(f"M1 smoke scene order/set drift: expected={expected} actual={scenes}")
    if any(
        config.get(name) is not False
        for config in configs
        for name in ("development_content_read", "heldout_content_read", "test_quality_read")
    ):
        raise M1RunError("M1 scene config provenance is not sealed")
    if any(
        config["evaluation"].get("candidate_policy")
        != "development_uncertainty_penalty_search"
        or not isinstance(config["evaluation"].get("candidate_arms"), Mapping)
        for config in configs
    ):
        raise M1RunError("M1 render-candidate policy drift")

    started = time.monotonic()
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts" / "states").mkdir(parents=True)
    (run_dir / "artifacts" / "predictions").mkdir(parents=True)
    (run_dir / "source_snapshot").mkdir()
    for path in scene_config_paths:
        shutil.copy2(path, run_dir / "source_snapshot" / f"{path.stem}.yaml")
    for relative in SNAPSHOT_FILES:
        source = project_root / relative
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    status = {
        "schema_version": "worldsim_v4_m1_status_v1",
        "task_id": TASK_ID,
        "status": "running",
        "phase": "two_scene_smoke",
        "scenes": scenes,
        "project_git_head": git_head,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "status.json", status)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[dict[str, Any]] = []
    scene_records = []
    base_rgb_checks = []
    checkpoint_checks = []
    temporal_records = []

    for scene_config in configs:
        from motion_proj.worldsim_v33.instance_field import load_instance_field
        from scripts.lift_worldsim_v32_semantics import build_runtime

        scene = str(scene_config["scene"])
        checkpoint = _verified(scene_config["inputs"]["checkpoint"], label="checkpoint")
        source_config = _verified(
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
        selected_ids = {}
        for role in scene_config["actors"]:
            state, selected = _state_for_actor(scene_config, field, role)
            states[role] = state
            selected_ids[role] = selected
            state_path = run_dir / "artifacts" / "states" / scene / f"{role}.npz"
            atomic_save_evidence_state(state_path, state)
            memory = TemporalEvidenceMemory(
                state["gaussian_id"],
                retention=float(scene_config["evidence"]["temporal"]["retention"]),
            )
            observed = BetaEvidence(state["alpha"], state["beta"])
            actor_rows = [row for row in rows if row["role"] == role]
            final = None
            for _ in actor_rows:
                final = memory.update(state["gaussian_id"], observed)
            assert final is not None
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
            "inputs": {"checkpoint": str(checkpoint), "source_config": str(source_config)},
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
            state = states[role]
            actor_id = int(
                scene_config["actors"][role]["dataset_instance_id"]
            )
            candidate, vectors = candidate_probability_vectors(
                field=field,
                state=state,
                actor_instance_id=actor_id,
                candidate_arms=scene_config["evaluation"]["candidate_arms"],
            )
            global_ids = torch.from_numpy(candidate.astype(np.int64)).to(device)
            method_names = list(vectors)
            rendered = _render_probabilities(
                trainer=trainer,
                dataset=dataset,
                frame=int(row["frame"]),
                camera=int(row["camera_id"]),
                global_ids=global_ids,
                probabilities=[vectors[name] for name in method_names],
                device=device,
            )
            with np.load(row["mask"], allow_pickle=False) as arrays:
                target = arrays["binary"].astype(bool)
            samples.append(
                {
                    "scene": scene,
                    "role": role,
                    "frame": int(row["frame"]),
                    "camera_id": int(row["camera_id"]),
                    "target": target,
                    "ece_bins": int(scene_config["calibration"]["ece_bins"]),
                    "probabilities": {
                        name: value.astype(np.float32)
                        for name, value in zip(method_names, rendered)
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
            {
                "scene": scene,
                "before": checkpoint_before,
                "after": checkpoint_after,
                "exact": checkpoint_before == checkpoint_after,
            }
        )
        scene_records.append(
            {
                "scene": scene,
                "evaluation_view_count": len(rows),
                "states": {
                    role: {
                        **evidence_state_summary(state),
                        "render_candidate_count": int(
                            candidate_probability_vectors(
                                field=field,
                                state=state,
                                actor_instance_id=int(
                                    scene_config["actors"][role][
                                        "dataset_instance_id"
                                    ]
                                ),
                                candidate_arms=scene_config["evaluation"][
                                    "candidate_arms"
                                ],
                            )[0].size
                        ),
                    }
                    for role, state in states.items()
                },
            }
        )
        del trainer, dataset, field, states
        torch.cuda.empty_cache()

    arm_methods = [
        f"raw__{name}" for name in configs[0]["evaluation"]["candidate_arms"]
    ]
    arm_per_scene, arm_scene_mean = aggregate_methods(
        samples,
        methods=["v33_o1", *arm_methods],
        evaluation=configs[0]["evaluation"],
    )
    selected_arm = choose_evidence_arm(
        arm_scene_mean,
        arm_methods,
        false_negative_mass_max_degradation=float(
            configs[0]["evaluation"]["candidate_arm_fn_mass_max_degradation"]
        ),
    )
    for sample in samples:
        sample["probabilities"]["raw"] = sample["probabilities"][selected_arm]
    pooled_probability = np.concatenate(
        [sample["probabilities"]["raw"].reshape(-1) for sample in samples]
    )
    pooled_target = np.concatenate([sample["target"].reshape(-1) for sample in samples])
    calibration_config = configs[0]["calibration"]
    calibrators = {
        "raw": RawCalibrator(),
        "temperature": fit_temperature(pooled_probability, pooled_target),
        "beta": fit_beta_calibration(
            pooled_probability,
            pooled_target,
            l2_regularization=float(calibration_config["beta_l2_regularization"]),
        ),
    }
    for sample in samples:
        raw = sample["probabilities"]["raw"]
        sample["probabilities"]["temperature"] = calibrators["temperature"].transform(
            raw.reshape(-1)
        ).reshape(raw.shape).astype(np.float32)
        sample["probabilities"]["beta"] = calibrators["beta"].transform(
            raw.reshape(-1)
        ).reshape(raw.shape).astype(np.float32)
    methods = ["v33_o1", "raw", "temperature", "beta"]
    _, reference_scene_mean = aggregate_methods(
        samples,
        methods=["v33_o1"],
        evaluation=configs[0]["evaluation"],
        thresholds={
            "v33_o1": float(configs[0]["evaluation"]["mask_threshold"])
        },
    )
    reference_metrics = reference_scene_mean["v33_o1"]
    threshold_candidates = [
        float(value)
        for value in configs[0]["evaluation"]["mask_threshold_candidates"]
    ]
    method_thresholds = {
        "v33_o1": float(configs[0]["evaluation"]["mask_threshold"])
    }
    threshold_search = {}
    for method in ("raw", "temperature", "beta"):
        rows = {}
        for threshold in threshold_candidates:
            _, candidate_scene_mean = aggregate_methods(
                samples,
                methods=[method],
                evaluation=configs[0]["evaluation"],
                thresholds={method: threshold},
            )
            rows[threshold] = candidate_scene_mean[method]
        selected_threshold = choose_mask_threshold(
            rows,
            reference=reference_metrics,
            false_negative_mass_max_degradation=float(
                configs[0]["evaluation"]["threshold_fn_mass_max_degradation"]
            ),
        )
        method_thresholds[method] = selected_threshold
        threshold_search[method] = {
            "selected": selected_threshold,
            "candidates": {str(key): value for key, value in rows.items()},
        }
    per_scene, scene_mean = aggregate_methods(
        samples,
        methods=methods,
        evaluation=configs[0]["evaluation"],
        thresholds=method_thresholds,
    )
    selected_calibration = choose_calibration(
        scene_mean,
        [str(name) for name in calibration_config["candidates"]],
        boundary_f1_max_degradation=float(
            calibration_config["boundary_f1_max_degradation"]
        ),
        false_negative_mass_max_degradation=float(
            calibration_config["false_negative_mass_max_degradation"]
        ),
        calibration_metric_max_degradation=float(
            calibration_config["non_selected_metric_max_degradation"]
        ),
    )
    rgb_exact = all(row["exact"] for row in base_rgb_checks)
    checkpoint_exact = all(row["exact"] for row in checkpoint_checks)
    preview = gate_preview(
        reference=scene_mean["v33_o1"],
        candidate=scene_mean[selected_calibration],
        gates=configs[0]["gates"],
        base_rgb_exact=rgb_exact and checkpoint_exact,
    )
    for sample in samples:
        stem = f"{sample['role']}__f{sample['frame']:03d}__c{sample['camera_id']}"
        for method in methods:
            output = (
                run_dir
                / "artifacts"
                / "predictions"
                / sample["scene"]
                / method
                / f"{stem}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(
                output,
                np.round(np.clip(sample["probabilities"][method], 0.0, 1.0) * 255.0).astype(np.uint8),
            )
    calibration_payload = {
        "schema_version": "worldsim_v4_m1_calibration_v1",
        "fit_partition": "development",
        "fit_scene_count": len(configs),
        "fit_pixel_count": int(pooled_target.size),
        "development_fit_reused_for_smoke_scoring": True,
        "candidates": {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
        "selected": selected_calibration,
    }
    atomic_json(run_dir / "calibration.json", calibration_payload)
    metrics_payload = {
        "schema_version": "worldsim_v4_m1_metrics_v1",
        "partition": "development",
        "per_scene": per_scene,
        "scene_mean": scene_mean,
        "selected_calibration": selected_calibration,
        "selected_evidence_arm": selected_arm,
        "selected_mask_thresholds": method_thresholds,
        "mask_threshold_search": threshold_search,
        "evidence_arm_search": {
            "per_scene": arm_per_scene,
            "scene_mean": arm_scene_mean,
        },
        "gate_preview": preview,
    }
    atomic_json(run_dir / "metrics.json", metrics_payload)
    summary = {
        "schema_version": "worldsim_v4_m1_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": "two_scene_smoke",
        "scenes": scenes,
        "project_git_head": git_head,
        "scene_records": scene_records,
        "selected_calibration": selected_calibration,
        "selected_evidence_arm": selected_arm,
        "selected_mask_threshold": method_thresholds[selected_calibration],
        "selected_mask_thresholds": method_thresholds,
        "gate_preview": preview,
        "base_rgb_render_checks": base_rgb_checks,
        "checkpoint_checks": checkpoint_checks,
        "temporal_memory_checks": temporal_records,
        "development_content_read": True,
        "development_optimization_read": True,
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
            "schema_version": "worldsim_v4_m1_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "files": manifest_rows,
        },
    )
    status.update(
        status="done",
        development_content_read=True,
        development_optimization_read=True,
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
                    "schema_version": "worldsim_v4_m1_status_v1",
                    "task_id": TASK_ID,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
