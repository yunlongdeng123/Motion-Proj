"""Train anchor authority from producer-side build evidence with frozen M8 geometry."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m6_gt_supervised_gaussian_relocation as m6_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m13_local_signed_field as field_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
    weighted_gaussian_energy,
)


FEATURE_NAMES = (
    "local_x_over_half_length",
    "local_y_over_half_width",
    "local_z_over_half_height",
    "abs_local_x_over_half_length",
    "abs_local_y_over_half_width",
    "abs_local_z_over_half_height",
    "provenance_keep",
    "provenance_project",
    "source_ray_x",
    "source_ray_y",
    "source_ray_z",
    "source_range_over_actor_diagonal",
    "projection_dx_over_length",
    "projection_dy_over_width",
    "projection_dz_over_height",
    "projection_norm_over_actor_diagonal",
    "canonical_log_hit_fraction",
    "canonical_temporal_support_fraction",
    "canonical_view_support_fraction",
    "build_free_mass",
    "build_occupied_mass",
    "build_unknown_mass",
    "build_log_opportunity_fraction",
    "canonical_view_x",
    "canonical_view_y",
    "canonical_view_z",
    "canonical_range_over_actor_diagonal",
    "source_canonical_view_cosine",
    "log_actor_length",
    "log_actor_width",
    "log_actor_height",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else value)


def _load_sidecar(
    actor: Mapping[str, Any], sidecar_root: Path
) -> dict[str, np.ndarray]:
    scene_name = _scalar_text(actor["scene_name"])
    track_id = _scalar_text(actor["track_id"])
    path = sidecar_root / "train" / scene_name / f"{track_id}.npz"
    with np.load(path, allow_pickle=False) as payload:
        sidecar = {key: payload[key] for key in payload.files}
    if len(sidecar["anchors"]) != len(actor["anchors"]):
        raise RuntimeError(f"sidecar anchor count mismatch: {scene_name}/{track_id}")
    return sidecar


def _anchor_features(
    actor: Mapping[str, Any],
    sidecar: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    anchors = torch.as_tensor(actor["anchors"], dtype=torch.float32, device=device)
    size = torch.as_tensor(actor["size_lwh_m"], dtype=torch.float32, device=device).clamp_min(0.10)
    half = 0.5 * size
    diagonal = torch.linalg.vector_norm(size).clamp_min(1.0e-6)
    normalized = anchors / half.reshape(1, 3)
    provenance = torch.as_tensor(
        sidecar["input_provenance"], dtype=torch.long, device=device
    )
    provenance_one_hot = F.one_hot(provenance, num_classes=2).to(torch.float32)
    source_directions = torch.as_tensor(
        sidecar["input_source_ray_directions"], dtype=torch.float32, device=device
    )
    source_ranges = torch.as_tensor(
        sidecar["input_source_ranges_m"], dtype=torch.float32, device=device
    ).reshape(-1, 1)
    projection = torch.as_tensor(
        sidecar["input_projection_displacement_xyz_m"],
        dtype=torch.float32,
        device=device,
    )
    hit_count = torch.as_tensor(
        sidecar["input_canonical_hit_count"], dtype=torch.float32, device=device
    ).reshape(-1, 1)
    temporal_support = torch.as_tensor(
        sidecar["input_canonical_temporal_support"], dtype=torch.float32, device=device
    ).reshape(-1, 1)
    view_support = torch.as_tensor(
        sidecar["input_canonical_view_support"], dtype=torch.float32, device=device
    ).reshape(-1, 1)
    build_frame_count = max(int(np.asarray(sidecar["input_build_frame_count"]).item()), 1)
    build_masses = torch.as_tensor(
        sidecar["input_build_evidence_masses"], dtype=torch.float32, device=device
    )
    opportunities = torch.as_tensor(
        sidecar["input_build_evidence_opportunities"],
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    canonical_origins = torch.as_tensor(
        sidecar["input_canonical_sensor_origins"], dtype=torch.float32, device=device
    )
    canonical_vectors = anchors - canonical_origins
    canonical_ranges = torch.linalg.vector_norm(canonical_vectors, dim=1, keepdim=True)
    canonical_directions = canonical_vectors / canonical_ranges.clamp_min(1.0e-6)
    view_cosine = torch.sum(
        source_directions * canonical_directions, dim=1, keepdim=True
    )
    log_size = torch.log1p(size).reshape(1, 3).expand(len(anchors), -1)
    maximum_opportunities = float(config["maximum_build_evidence_points"])
    features = torch.cat(
        [
            normalized,
            normalized.abs(),
            provenance_one_hot,
            source_directions,
            source_ranges / diagonal,
            projection / size.reshape(1, 3),
            torch.linalg.vector_norm(projection, dim=1, keepdim=True) / diagonal,
            torch.log1p(hit_count) / np.log1p(maximum_opportunities),
            temporal_support / float(build_frame_count),
            view_support / 8.0,
            build_masses,
            torch.log1p(opportunities) / np.log1p(maximum_opportunities),
            canonical_directions,
            canonical_ranges / diagonal,
            view_cosine,
            log_size,
        ],
        dim=1,
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"M34 feature dimension {features.shape[1]} != {len(FEATURE_NAMES)}")
    return features


def _attach_frozen_authority_state(
    actor: dict[str, Any],
    sidecar_root: Path,
    anchor_scale_m: float,
    feature_config: Mapping[str, Any],
    device: torch.device,
) -> None:
    sidecar = _load_sidecar(actor, sidecar_root)
    anchors = actor["anchors_t"].clone()
    children = actor["m8_children_t"].clone()
    actor["m8_children_t"] = children
    actor["m8_scales_t"] = actor["m8_scales_t"].clone()
    actor["authority_centers_t"] = torch.cat([anchors, children], dim=0)
    actor["authority_scales_t"] = torch.cat(
        [
            torch.full(
                (len(anchors),),
                float(anchor_scale_m),
                dtype=anchors.dtype,
                device=device,
            ),
            actor["m8_scales_t"].reshape(-1),
        ]
    ).clamp_min(1.0e-4)
    actor["authority_anchor_features_t"] = _anchor_features(
        actor, sidecar, feature_config, device
    )
    actor["authority_target_masses_t"] = torch.as_tensor(
        sidecar["supervision_evidence_masses"], dtype=torch.float32, device=device
    )
    actor["authority_build_masses_t"] = torch.as_tensor(
        sidecar["input_build_evidence_masses"], dtype=torch.float32, device=device
    )
    actor["authority_provenance_t"] = torch.as_tensor(
        sidecar["input_provenance"], dtype=torch.long, device=device
    )


def _occupied_with_unit_children(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_logits = authority(actor["authority_anchor_features_t"])
    anchor_occupied = occupied_masses(anchor_logits)
    child_occupied = torch.ones(
        len(actor["m8_children_t"]),
        dtype=anchor_occupied.dtype,
        device=anchor_occupied.device,
    )
    return torch.cat([anchor_occupied, child_occupied]), anchor_logits


def _ray_values(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, origins = m6_runner._limit_target(
        actor, int(config["maximum_training_rays"]), actor["features"].device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(config["training_cuboid_padding_m"])
    entry, exit_depth, valid_box = field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    valid = valid_box & (target_depth >= entry) & (target_depth <= exit_depth)
    fractions = torch.linspace(
        0.0,
        1.0,
        int(config["categorical_train_bins"]),
        dtype=targets.dtype,
        device=targets.device,
    )
    depths = entry[valid, None] + (
        exit_depth[valid] - entry[valid]
    )[:, None] * fractions[None, :]
    queries = origins[valid, None, :] + depths[:, :, None] * directions[valid, None, :]
    occupied, anchor_logits = _occupied_with_unit_children(authority, actor)
    energy = weighted_gaussian_energy(
        queries.reshape(-1, 3),
        actor["authority_centers_t"],
        actor["authority_scales_t"],
        occupied,
    ).reshape(len(depths), -1)
    return depths, target_depth[valid], energy, anchor_logits


def _losses(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, energy, anchor_logits = _ray_values(authority, actor, config)
    target_bins = torch.abs(depths - target_depth[:, None]).argmin(dim=1)
    categorical = F.cross_entropy(energy, target_bins)
    probabilities = torch.softmax(energy, dim=1)
    expected_depth = torch.sum(probabilities * depths, dim=1)
    depth_l1 = torch.abs(expected_depth - target_depth).mean()
    target_masses = actor["authority_target_masses_t"]
    evidential = -torch.sum(
        target_masses * torch.log_softmax(anchor_logits, dim=1), dim=1
    ).mean()
    loss = (
        categorical
        + float(config["categorical_depth_weight"]) * depth_l1
        + float(config["evidential_mass_weight"]) * evidential
    )
    return {
        "loss": loss,
        "categorical_nll": categorical,
        "depth_l1": depth_l1,
        "evidential_cross_entropy": evidential,
        "target_probability": probabilities.gather(1, target_bins[:, None]).mean(),
    }


def _train(
    authority: EvidentialGaussianAuthority,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "loss",
        "categorical_nll",
        "depth_l1",
        "evidential_cross_entropy",
        "target_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [_losses(authority, actors[index], config) for index in indices]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: total / len(actors) for name, total in totals.items()},
        }
        history.append(row)
        print(json.dumps({"stage": "m34_authority_train", **row}), flush=True)
    return history


def _weighted_partition(
    authority: EvidentialGaussianAuthority,
    actor: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    device = actor["features"].device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1).clamp_min(1.0e-6)
    directions = (targets - origins) / target_depth[:, None]
    bounds = actor["size_t"] * 0.5 + float(evaluation["cuboid_padding_m"])
    entry, exit_depth, valid_box = field_runner._ray_box_intervals(
        origins, directions, bounds
    )
    fractions = torch.linspace(
        0.0,
        1.0,
        int(evaluation["field_sample_count"]),
        dtype=targets.dtype,
        device=device,
    )
    occupied, _ = _occupied_with_unit_children(authority, actor)
    first_depths = []
    threshold = float(evaluation["categorical_median_threshold"])
    chunk = int(evaluation["ray_chunk_size"])
    for start in range(0, len(targets), chunk):
        local_entry = entry[start : start + chunk]
        local_exit = exit_depth[start : start + chunk]
        depths = local_entry[:, None] + (
            local_exit - local_entry
        )[:, None] * fractions[None, :]
        queries = (
            origins[start : start + chunk, None, :]
            + depths[:, :, None] * directions[start : start + chunk, None, :]
        )
        energy = weighted_gaussian_energy(
            queries.reshape(-1, 3),
            actor["authority_centers_t"],
            actor["authority_scales_t"],
            occupied,
        ).reshape(len(local_entry), -1)
        cdf = torch.softmax(energy, dim=1).cumsum(dim=1)
        indices = (cdf >= threshold).to(torch.int64).argmax(dim=1)
        previous = (indices - 1).clamp_min(0)
        right_cdf = cdf.gather(1, indices[:, None]).squeeze(1)
        gathered_left_cdf = cdf.gather(1, previous[:, None]).squeeze(1)
        left_cdf = torch.where(
            indices > 0, gathered_left_cdf, torch.zeros_like(gathered_left_cdf)
        )
        right_depth = depths.gather(1, indices[:, None]).squeeze(1)
        gathered_left_depth = depths.gather(1, previous[:, None]).squeeze(1)
        left_depth = torch.where(indices > 0, gathered_left_depth, local_entry)
        ratio = (threshold - left_cdf) / (right_cdf - left_cdf).clamp_min(1.0e-6)
        first_depths.append(
            left_depth + ratio.clamp(0.0, 1.0) * (right_depth - left_depth)
        )
    first_depth = torch.cat(first_depths)
    tolerance = float(evaluation["literal_depth_tolerance_m"])
    return {
        "observable": valid_box.cpu().numpy().astype(bool),
        "early": (
            valid_box & (first_depth < target_depth - tolerance)
        ).cpu().numpy(),
        "hit": (
            valid_box & (torch.abs(first_depth - target_depth) <= tolerance)
        ).cpu().numpy(),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        baseline_early = sum(int(row["baseline_early_count"]) for row in selected)
        learned_early = sum(int(row["learned_early_count"]) for row in selected)
        baseline_hit = sum(int(row["baseline_hit_count"]) for row in selected)
        learned_hit = sum(int(row["learned_hit_count"]) for row in selected)
        return {
            "actor_count": len(selected),
            "ray_count": rays,
            "baseline_early_rate": baseline_early / rays,
            "learned_early_rate": learned_early / rays,
            "early_rate_delta": (learned_early - baseline_early) / rays,
            "baseline_hit_rate": baseline_hit / rays,
            "learned_hit_rate": learned_hit / rays,
            "hit_rate_delta": (learned_hit - baseline_hit) / rays,
        }

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M34 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = m22_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["model"]["maximum_training_actors"])
        )
        actors = [
            actor for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = centers
                children, _, scales = m7_runner._predict(surface, actor, surface_config)
                actor["m8_children_t"] = children
                actor["m8_scales_t"] = scales
        for actor in actors:
            _attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["model"]["anchor_scale_m"]),
                config["features"],
                device,
            )
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        authority = EvidentialGaussianAuthority(
            hidden_dim=int(config["model"]["hidden_dim"]),
            input_dim=len(FEATURE_NAMES),
        ).to(device)
        optimizer = torch.optim.AdamW(
            authority.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
        authority.train()
        history = _train(authority, train_actors, config["model"], optimizer)
        authority.eval()
        torch.save(
            {
                "state_dict": authority.state_dict(),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "input_dim": len(FEATURE_NAMES),
                "feature_names": FEATURE_NAMES,
                "m8_run": str(config["m8_run"]),
                "sidecar_root": str(config["sidecar_root"]),
                "seed": int(config["model"]["seed"]),
                "states": ["FREE", "OCCUPIED", "UNKNOWN"],
                "child_authority": "unit",
            },
            run_dir / "MODEL.pt",
        )
        rows: list[dict[str, Any]] = []
        predicted_occupied_parts = []
        target_occupied_parts = []
        keep_mass_parts = []
        project_mass_parts = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = m20_runner._energy_partition(
                    actor,
                    config["evaluation"],
                    float(config["model"]["anchor_scale_m"]),
                    device,
                )
                learned = _weighted_partition(authority, actor, config["evaluation"])
                logits = authority(actor["authority_anchor_features_t"])
                masses = torch.softmax(logits, dim=1)
                target_masses = actor["authority_target_masses_t"]
                provenance = actor["authority_provenance_t"]
                keep = provenance == 0
                project = provenance == 1
                predicted_occupied_parts.append(masses[:, 1].cpu().numpy())
                target_occupied_parts.append(target_masses[:, 1].cpu().numpy())
                if torch.any(keep):
                    keep_mass_parts.append(masses[keep].cpu().numpy())
                if torch.any(project):
                    project_mass_parts.append(masses[project].cpu().numpy())
                rows.append(
                    {
                        "scene_name": _scalar_text(actor["scene_name"]),
                        "track_id": _scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "anchor_count": int(len(masses)),
                        "child_count": int(len(actor["m8_children_t"])),
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        "learned_early_count": int(np.count_nonzero(learned["early"])),
                        "learned_hit_count": int(np.count_nonzero(learned["hit"])),
                        "learned_observable_count": int(np.count_nonzero(learned["observable"])),
                        "mean_anchor_masses": masses.mean(dim=0).cpu().tolist(),
                        "mean_keep_masses": (
                            masses[keep].mean(dim=0).cpu().tolist()
                            if torch.any(keep)
                            else None
                        ),
                        "mean_project_masses": (
                            masses[project].mean(dim=0).cpu().tolist()
                            if torch.any(project)
                            else None
                        ),
                        "target_mass_cross_entropy": float(
                            -torch.sum(
                                target_masses * torch.log_softmax(logits, dim=1), dim=1
                            ).mean()
                        ),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m34_holdout", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        predicted_occupied = np.concatenate(predicted_occupied_parts)
        target_occupied = np.concatenate(target_occupied_parts)
        occupied_correlation = float(
            np.corrcoef(predicted_occupied, target_occupied)[0, 1]
        )
        worst_stratum_delta = max(
            float(metrics["hazard"]["early_rate_delta"]),
            float(metrics["clear"]["early_rate_delta"]),
        )
        decisions = {
            "all_early_nonincrease": float(metrics["all"]["early_rate_delta"])
            <= float(config["decision"]["maximum_all_early_delta"]),
            "hazard_and_clear_early_nonincrease": worst_stratum_delta
            <= float(config["decision"]["maximum_worst_stratum_early_delta"]),
            "all_hit_retained": float(metrics["all"]["hit_rate_delta"])
            >= float(config["decision"]["minimum_all_hit_delta"]),
            "anchor_authority_identifiable": occupied_correlation
            >= float(config["decision"]["minimum_occupied_correlation"]),
        }
        passed = all(decisions.values())
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m34_development_passed" if passed else "m34_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "feature_names": FEATURE_NAMES,
            "training_history": history,
            "metrics": metrics,
            "decisions": decisions,
            "worst_stratum_early_delta": worst_stratum_delta,
            "holdout_predicted_target_occupied_correlation": occupied_correlation,
            "mean_holdout_target_mass_cross_entropy": float(
                np.mean([row["target_mass_cross_entropy"] for row in rows])
            ),
            "mean_holdout_anchor_masses": np.mean(
                [row["mean_anchor_masses"] for row in rows], axis=0
            ).tolist(),
            "mean_holdout_keep_masses": np.concatenate(keep_mass_parts).mean(axis=0).tolist(),
            "mean_holdout_project_masses": np.concatenate(project_mass_parts).mean(axis=0).tolist(),
            "geometry_centers_and_scales_frozen": True,
            "completion_child_authority": "unit",
            "geometry_state_retention": 1.0,
            "actor_and_hazard_state_retention": 1.0,
            "physics_supervision": "producer_build_inputs_to_heldout_native_lidar_evidence",
            "unknown_hard_mask_or_surface_filter": False,
            "binary_authority_threshold": False,
            "pretrained_holdout_exposure": True,
            "external_read": False,
            "m21_partial_quality_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "HOLDOUT_AUTHORITY_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m34", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
