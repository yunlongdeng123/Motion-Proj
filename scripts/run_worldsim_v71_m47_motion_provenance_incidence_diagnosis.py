"""Diagnose M46-vs-M39 errors by motion, provenance, and ray incidence."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m8_runner
import run_worldsim_v71_m11_exact_support_supervision as oriented_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as loader_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as anchor_runner
import run_worldsim_v71_m37_supervised_child_transmittance as child_runner
import run_worldsim_v71_m38_prehit_free_space_survival as authority_runner
import run_worldsim_v71_m39_categorical_authority_composition as composition_runner
import run_worldsim_v71_m45_oriented_categorical_surface_measure as m45_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import occupied_masses
from motion_proj.worldsim_v71.gaussian_anchor_relocation import OrientedGaussianSeedExpansionMLP


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _key(actor: Mapping[str, Any]) -> tuple[str, str]:
    return (
        anchor_runner._scalar_text(actor["scene_name"]),
        anchor_runner._scalar_text(actor["track_id"]),
    )


def _frame_ordinals(origins: np.ndarray) -> np.ndarray:
    mapping: dict[tuple[float, float, float], int] = {}
    output = []
    for origin in np.asarray(origins, dtype=np.float32).reshape(-1, 3):
        key = tuple(float(value) for value in origin)
        if key not in mapping:
            mapping[key] = len(mapping)
        output.append(mapping[key])
    return np.asarray(output, dtype=np.int64)


def _target_responsibility(
    actor: Mapping[str, Any],
    anchor_occupied: torch.Tensor,
    child_occupied: torch.Tensor,
    child_normals: torch.Tensor,
    child_thickness: torch.Tensor,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = actor["features"].device
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=device
    )
    directions = torch.nn.functional.normalize(targets - origins, dim=1)
    anchor_scales = actor["authority_scales_t"][: len(actor["anchors_t"])]
    incidence_rows = []
    child_mass_rows = []
    for start in range(0, len(targets), chunk_size):
        query = targets[start : start + chunk_size]
        direction = directions[start : start + chunk_size]
        anchor_distance = torch.cdist(query, actor["anchors_t"])
        anchor_components = (
            -0.5 * (anchor_distance / anchor_scales.reshape(1, -1)).square()
            + torch.log(anchor_occupied.clamp_min(1.0e-8)).reshape(1, -1)
        )
        displacement = query[:, None, :] - actor["m8_children_t"][None, :, :]
        normal_coordinate = torch.sum(displacement * child_normals[None, :, :], dim=-1)
        tangent_sq = (
            torch.sum(displacement.square(), dim=-1) - normal_coordinate.square()
        ).clamp_min(0.0)
        child_components = -0.5 * (
            tangent_sq / actor["m8_scales_t"].square().reshape(1, -1)
            + normal_coordinate.square() / child_thickness.square().reshape(1, -1)
        ) + torch.log(child_occupied.clamp_min(1.0e-8)).reshape(1, -1)
        component_mass = torch.softmax(
            torch.cat([anchor_components, child_components], dim=1), dim=1
        )
        child_mass = component_mass[:, len(actor["anchors_t"]) :]
        total_child_mass = child_mass.sum(dim=1)
        absolute_cosine = torch.abs(direction @ child_normals.T)
        weighted_incidence = (child_mass * absolute_cosine).sum(dim=1) / total_child_mass.clamp_min(
            1.0e-8
        )
        incidence_rows.append(weighted_incidence)
        child_mass_rows.append(total_child_mass)
    return (
        torch.cat(incidence_rows).cpu().numpy(),
        torch.cat(child_mass_rows).cpu().numpy(),
    )


def _empty_counts() -> dict[str, int]:
    return {
        "ray_count": 0,
        "m39_early_count": 0,
        "m46_early_count": 0,
        "m39_hit_count": 0,
        "m46_hit_count": 0,
        "added_early_count": 0,
        "removed_early_count": 0,
    }


def _update_counts(
    counts: dict[str, int],
    selected: np.ndarray,
    m39: Mapping[str, np.ndarray],
    m46: Mapping[str, np.ndarray],
) -> None:
    selected = np.asarray(selected, dtype=bool)
    m39_early = np.asarray(m39["early"], dtype=bool)
    m46_early = np.asarray(m46["early"], dtype=bool)
    counts["ray_count"] += int(np.count_nonzero(selected))
    counts["m39_early_count"] += int(np.count_nonzero(selected & m39_early))
    counts["m46_early_count"] += int(np.count_nonzero(selected & m46_early))
    counts["m39_hit_count"] += int(np.count_nonzero(selected & np.asarray(m39["hit"], dtype=bool)))
    counts["m46_hit_count"] += int(np.count_nonzero(selected & np.asarray(m46["hit"], dtype=bool)))
    counts["added_early_count"] += int(np.count_nonzero(selected & m46_early & ~m39_early))
    counts["removed_early_count"] += int(np.count_nonzero(selected & m39_early & ~m46_early))


def _rates(counts: Mapping[str, int]) -> dict[str, Any]:
    rays = int(counts["ray_count"])
    output: dict[str, Any] = dict(counts)
    if rays == 0:
        output.update(
            {
                "m39_early_rate": None,
                "m46_early_rate": None,
                "early_delta": None,
                "m39_hit_rate": None,
                "m46_hit_rate": None,
                "hit_delta": None,
                "added_early_rate": None,
                "removed_early_rate": None,
            }
        )
        return output
    for name in ("m39_early", "m46_early", "m39_hit", "m46_hit", "added_early", "removed_early"):
        output[f"{name}_rate"] = int(counts[f"{name}_count"]) / rays
    output["early_delta"] = output["m46_early_rate"] - output["m39_early_rate"]
    output["hit_delta"] = output["m46_hit_rate"] - output["m39_hit_rate"]
    return output


def _pearson(rows: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    pairs = [
        (float(row[field]), float(row["early_delta_rate"]))
        for row in rows
        if row.get(field) is not None
    ]
    if len(pairs) < 3:
        return {"actor_count": len(pairs), "correlation": None}
    values = np.asarray(pairs, dtype=np.float64)
    if np.std(values[:, 0]) == 0.0 or np.std(values[:, 1]) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    return {"actor_count": len(pairs), "correlation": correlation}


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    child_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M47 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = loader_runner._load_m8(
            config, device
        )
        actors = [
            actor
            for path in m0_runner._paths(
                Path(config["cache_root"]), int(config["maximum_training_actors"])
            )
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        with torch.inference_mode():
            for actor in actors:
                _, actor["m5_centers_t"] = m5_runner._move(base, actor, base_config)
                (
                    actor["m8_children_t"],
                    actor["m8_residuals_t"],
                    actor["m8_scales_t"],
                ) = m8_runner._predict(surface, actor, surface_config)
        holdout = []
        for index, actor in enumerate(actors):
            if index % int(config["holdout_stride"]) != 0:
                continue
            anchor_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["anchor_scale_m"]),
                config["features"],
                device,
            )
            actor["authority_child_features_t"] = child_runner._child_features(actor)
            holdout.append(actor)

        anchor_checkpoint = torch.load(Path(config["m35_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        anchor_authority = authority_runner._load_authority(anchor_checkpoint, device)
        anchor_authority.eval().requires_grad_(False)
        child_checkpoint = torch.load(Path(config["m38_run"]) / "CHILD_MODEL.pt", map_location=device, weights_only=False)
        child_authority = authority_runner._load_authority(child_checkpoint, device)
        child_authority.eval().requires_grad_(False)
        support_checkpoint = torch.load(Path(config["m46_run"]) / "MODEL.pt", map_location=device, weights_only=False)
        support_model = OrientedGaussianSeedExpansionMLP(
            int(support_checkpoint["input_dim"]),
            int(support_checkpoint["hidden_dim"]),
            int(support_checkpoint["branch_factor"]),
            int(support_checkpoint["slot_dim"]),
        ).to(device)
        support_model.load_state_dict(support_checkpoint["state_dict"])
        support_model.eval().requires_grad_(False)
        support_config = yaml.safe_load(
            (Path(config["m46_run"]) / "resolved.yaml").read_text(encoding="utf-8")
        )["model"]
        m31_lookup = {
            (str(row["scene_name"]), str(row["track_id"])): row
            for row in _load_jsonl(Path(config["m31_run"]) / "ANCHOR_ATTRIBUTION_ROWS.jsonl")
        }

        grouped: defaultdict[str, dict[str, int]] = defaultdict(_empty_counts)
        actor_rows = []
        missing_m31 = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout):
                key = _key(actor)
                motion = m31_lookup.get(key)
                if motion is None:
                    missing_m31.append({"scene_name": key[0], "track_id": key[1]})
                    continue
                anchor_occupied = occupied_masses(anchor_authority(actor["authority_anchor_features_t"]))
                child_occupied = occupied_masses(child_authority(actor["authority_child_features_t"]))
                m39 = composition_runner._categorical_partition(
                    actor, torch.cat([anchor_occupied, child_occupied]), config["evaluation"]
                )
                _, _, normals, thickness = oriented_runner._predict_support(
                    support_model, actor, support_config
                )
                m46 = m45_runner._partition(
                    actor,
                    anchor_occupied,
                    child_occupied,
                    normals,
                    thickness,
                    config["evaluation"],
                )
                incidence, child_mass = _target_responsibility(
                    actor,
                    anchor_occupied,
                    child_occupied,
                    normals,
                    thickness,
                    int(config["diagnosis"]["responsibility_chunk_size"]),
                )
                ray_count = len(incidence)
                all_rays = np.ones(ray_count, dtype=bool)
                frame_ordinals = _frame_ordinals(np.asarray(actor["target_sensor_origins"]))
                kept_early = int(motion["anchor_first_early_by_provenance"]["kept"])
                projected_early = int(motion["anchor_first_early_by_provenance"]["projected"])
                attributed = kept_early + projected_early
                projected_fraction = projected_early / attributed if attributed else None
                provenance = (
                    "none"
                    if attributed == 0
                    else "projected_dominant"
                    if projected_early > kept_early
                    else "kept_dominant"
                )
                actor_groups = [
                    "all",
                    "hazard" if bool(motion["hazardous"]) else "clear",
                    "moving" if bool(motion["moving"]) else "quasi_static",
                    ("hazard" if bool(motion["hazardous"]) else "clear")
                    + "_"
                    + ("moving" if bool(motion["moving"]) else "quasi_static"),
                    f"provenance_{provenance}",
                    f"category_{motion['category']}",
                ]
                for name in actor_groups:
                    _update_counts(grouped[name], all_rays, m39, m46)
                incidence_masks = {
                    "incidence_grazing": incidence < float(config["diagnosis"]["grazing_max_abs_cosine"]),
                    "incidence_oblique": (incidence >= float(config["diagnosis"]["grazing_max_abs_cosine"]))
                    & (incidence < float(config["diagnosis"]["normal_min_abs_cosine"])),
                    "incidence_normal": incidence >= float(config["diagnosis"]["normal_min_abs_cosine"]),
                }
                for name, selected in incidence_masks.items():
                    _update_counts(grouped[name], selected, m39, m46)
                for ordinal in np.unique(frame_ordinals):
                    _update_counts(grouped[f"target_frame_{int(ordinal)}"], frame_ordinals == ordinal, m39, m46)
                actor_count = _empty_counts()
                _update_counts(actor_count, all_rays, m39, m46)
                actor_rate = _rates(actor_count)
                anchor_early_rate = int(motion["surface_counts"]["anchors"]["early_count"]) / int(motion["ray_count"])
                actor_rows.append(
                    {
                        "scene_name": key[0],
                        "track_id": key[1],
                        "category": str(motion["category"]),
                        "hazardous": bool(motion["hazardous"]),
                        "moving": bool(motion["moving"]),
                        "trajectory_max_displacement_m": float(motion["trajectory_max_displacement_m"]),
                        "projected_early_fraction": projected_fraction,
                        "anchor_early_rate": anchor_early_rate,
                        "mean_target_incidence_abs_cosine": float(np.mean(incidence)),
                        "mean_target_child_responsibility": float(np.mean(child_mass)),
                        "mean_thickness_m": float(thickness.mean()),
                        "early_delta_rate": float(actor_rate["early_delta"]),
                        "hit_delta_rate": float(actor_rate["hit_delta"]),
                        "counts": actor_count,
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout):
                    print(json.dumps({"stage": "m47_diagnosis", "progress": f"{index + 1}/{len(holdout)}"}), flush=True)

        correlations = {
            field: _pearson(actor_rows, field)
            for field in (
                "trajectory_max_displacement_m",
                "projected_early_fraction",
                "anchor_early_rate",
                "mean_target_incidence_abs_cosine",
                "mean_target_child_responsibility",
            )
        }
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m46_error_factors_diagnosed",
            "holdout_actor_count": len(holdout),
            "matched_actor_count": len(actor_rows),
            "missing_m31_actor_count": len(missing_m31),
            "missing_m31_actors": missing_m31,
            "metrics": {name: _rates(counts) for name, counts in sorted(grouped.items())},
            "actor_correlations_with_m46_early_delta": correlations,
            "training": False,
            "model_selection": False,
            "decision_gate": False,
            "motion_or_hazard_input": False,
            "frozen_m39_and_m46": True,
            "posthoc_filter": False,
            "pretrained_holdout_exposure": True,
            "external_read": False,
            "m43_partial_quality_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        child_runner._write_jsonl(run_dir / "ACTOR_DIAGNOSIS_ROWS.jsonl", actor_rows)
        child_runner._write_json(run_dir / "summary.json", summary)
        child_runner._write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "diagnosis", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        child_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m47", "error": f"{type(error).__name__}: {error}"},
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
