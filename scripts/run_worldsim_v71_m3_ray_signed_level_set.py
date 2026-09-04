"""训练V7.1 M3单一ray-signed Actor level-set并做train-role holdout评估。"""

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

import run_worldsim_v71_m1_evidential_surface_field as m1_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.signed_surface_field import (
    RaySignedSurfaceField,
    extract_signed_zero_crossings,
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


def _normalized(query: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return query / (size.reshape(1, 3) * 0.5).clamp_min(1.0e-3)


def _signed_ray_supervision(
    actor: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = actor["target_t"]
    origins = actor["target_origins_t"]
    count = min(len(targets), int(config["maximum_target_rays"]))
    selected = torch.linspace(0, len(targets) - 1, steps=count, device=targets.device).long()
    targets = targets[selected]
    origins = origins[selected]
    direction = targets - origins
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    offsets = torch.as_tensor(config["signed_offsets_m"], dtype=targets.dtype, device=targets.device)
    ray_query = targets[:, None, :] + direction[:, None, :] * offsets[None, :, None]
    ray_sdf = -offsets[None, :].expand(count, -1)
    ray_query = ray_query.reshape(-1, 3)
    ray_sdf = ray_sdf.reshape(-1)
    half = actor["size_t"] * 0.5 + float(config["cuboid_padding_m"])
    inside = torch.all(ray_query.abs() <= half[None, :], dim=1)
    ray_query = ray_query[inside]
    ray_sdf = ray_sdf[inside]

    anchors = actor["anchors_t"]
    if len(anchors) > int(config["maximum_anchor_points"]):
        indices = torch.linspace(
            0,
            len(anchors) - 1,
            steps=int(config["maximum_anchor_points"]),
            device=anchors.device,
        ).long()
        anchors = anchors[indices]
    return (
        torch.cat([ray_query, anchors], dim=0),
        torch.cat([ray_sdf, torch.zeros(len(anchors), dtype=ray_sdf.dtype, device=ray_sdf.device)]),
    )


def _first_return_loss(
    model: RaySignedSurfaceField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    targets = actor["target_t"]
    origins = actor["target_origins_t"]
    count = min(len(targets), int(config["maximum_render_rays"]))
    selected = torch.linspace(0, len(targets) - 1, steps=count, device=targets.device).long()
    targets = targets[selected]
    origins = origins[selected]
    direction = targets - origins
    target_depth = torch.linalg.vector_norm(direction, dim=1)
    direction /= target_depth[:, None].clamp_min(1.0e-6)
    offsets = torch.linspace(
        -float(config["render_back_m"]),
        float(config["render_front_m"]),
        steps=int(config["render_samples"]),
        device=targets.device,
    )
    depths = (target_depth[:, None] + offsets[None, :]).clamp_min(0.05)
    query = origins[:, None, :] + direction[:, None, :] * depths[:, :, None]
    latent = model.encode(actor["evidence_t"])
    sdf = model.decode(
        latent,
        _normalized(query.reshape(-1, 3), actor["size_t"]),
        actor["size_t"],
    ).reshape(count, -1)
    density = torch.exp(-sdf.abs() / float(config["render_surface_sigma_m"]))
    alpha = 1.0 - torch.exp(-float(config["render_density_scale"]) * density)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((count, 1), device=targets.device), 1.0 - alpha + 1.0e-6], dim=1),
        dim=1,
    )
    weights = transmittance[:, :-1] * alpha
    fallback = target_depth + float(config["render_front_m"])
    predicted = (weights * depths).sum(dim=1) + transmittance[:, -1] * fallback
    return F.smooth_l1_loss(
        (predicted - target_depth) / float(config["first_return_scale_m"]),
        torch.zeros_like(target_depth),
    )


def _actor_loss(
    model: RaySignedSurfaceField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query, target_sdf = _signed_ray_supervision(actor, config)
    predicted = model(actor["evidence_t"], _normalized(query, actor["size_t"]), actor["size_t"])
    sdf_loss = F.smooth_l1_loss(
        predicted / float(config["maximum_distance_m"]),
        target_sdf / float(config["maximum_distance_m"]),
    )

    half = actor["size_t"] * 0.5 + float(config["cuboid_padding_m"])
    eikonal_query = ((torch.rand(int(config["eikonal_points"]), 3, device=query.device) * 2.0 - 1.0) * half).requires_grad_(True)
    eikonal_sdf = model(
        actor["evidence_t"], _normalized(eikonal_query, actor["size_t"]), actor["size_t"]
    )
    gradient = torch.autograd.grad(
        eikonal_sdf.sum(), eikonal_query, create_graph=True, retain_graph=True
    )[0]
    eikonal_loss = (torch.linalg.vector_norm(gradient, dim=1) - 1.0).square().mean()
    first_return_loss = _first_return_loss(model, actor, config)
    total = (
        float(config["sdf_loss_weight"]) * sdf_loss
        + float(config["eikonal_loss_weight"]) * eikonal_loss
        + float(config["first_return_loss_weight"]) * first_return_loss
    )
    return total, {"sdf": sdf_loss, "eikonal": eikonal_loss, "first_return": first_return_loss}


def _train(
    model: RaySignedSurfaceField,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(len(actors)).tolist()
        totals = {"total": 0.0, "sdf": 0.0, "eikonal": 0.0, "first_return": 0.0}
        for start in range(0, len(permutation), batch_size):
            batch = [_actor_loss(model, actors[index], config) for index in permutation[start : start + batch_size]]
            loss = torch.stack([item[0] for item in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["total"] += float(loss.detach()) * len(batch)
            for name in ("sdf", "eikonal", "first_return"):
                totals[name] += sum(float(item[1][name].detach()) for item in batch)
        row = {"epoch": epoch + 1, **{name: value / len(actors) for name, value in totals.items()}}
        history.append(row)
        print(json.dumps({"stage": "m3_signed_train", "actors": len(actors), **row}), flush=True)
    return history


def _grid_axes(size: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=np.float32) * 0.5 + float(config["cuboid_padding_m"])
    voxel = float(config["extraction_voxel_size_m"])
    axes = tuple(np.arange(-value, value + 0.5 * voxel, voxel, dtype=np.float32) for value in half)
    count = int(np.prod([len(axis) for axis in axes]))
    if count > int(config["maximum_extraction_points"]):
        voxel *= (count / float(config["maximum_extraction_points"])) ** (1.0 / 3.0)
        axes = tuple(np.arange(-value, value + 0.5 * voxel, voxel, dtype=np.float32) for value in half)
    return axes


def _extract(
    model: RaySignedSurfaceField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, int, float, float]:
    axes = _grid_axes(actor["size_lwh_m"], config)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    query = torch.as_tensor(grid, dtype=torch.float32, device=actor["size_t"].device)
    latent = model.encode(actor["evidence_t"])
    parts = []
    for start in range(0, len(query), int(config["extraction_chunk_size"])):
        parts.append(
            model.decode(
                latent,
                _normalized(query[start : start + int(config["extraction_chunk_size"])], actor["size_t"]),
                actor["size_t"],
            )
        )
    sdf = torch.cat(parts)
    crossings = extract_signed_zero_crossings(axes, sdf)
    merged = torch.cat([actor["anchors_t"], crossings], dim=0).cpu().numpy()
    output = _voxel_unique(merged, float(config["output_voxel_size_m"]))
    return output, len(crossings), float(sdf.min()), float(sdf.max())


def _decisions(summary: Mapping[str, Any], extraction_fraction: float, config: Mapping[str, Any]) -> dict[str, bool]:
    reduction = summary["hazard"]["relative_early_reduction"]
    return {
        "surface_extracted": extraction_fraction >= float(config["minimum_extraction_actor_fraction"]),
        "hazard_literal_first_return_relative_reduction": reduction is not None and float(reduction) >= float(config["minimum_hazard_literal_relative_reduction"]),
        "chamfer_non_degradation": float(summary["chamfer_delta_m"]) <= float(config["maximum_chamfer_delta_m"]),
        "target_hit_recall": float(summary["hit_recall_delta"]) >= float(config["minimum_hit_recall_delta"]),
        "actor_state_retention": float(summary["minimum_actor_state_retention"]) == float(config["required_actor_state_retention"]),
        "hazard_state_retention": float(summary["minimum_hazard_state_retention"]) == float(config["required_hazard_state_retention"]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading_train"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M3 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        paths = sorted(
            path
            for path in (Path(config["cache_root"]) / "train").glob("*/*.npz")
            if not path.name.endswith(".tmp.npz")
        )[: int(config["model"]["maximum_training_actors"])]
        payloads = [load_actor_cache(path) for path in paths]
        payloads = [
            payload
            for payload in payloads
            if len(payload["candidates"]) and len(payload["target"]) and len(payload["anchors"])
        ]
        stride = int(config["model"]["holdout_stride"])
        train_payloads = [payload for index, payload in enumerate(payloads) if index % stride != 0]
        holdout_payloads = [payload for index, payload in enumerate(payloads) if index % stride == 0]
        feature_arrays = [
            m1_runner._evidence_features(payload, int(config["model"]["maximum_evidence_points"]))
            for payload in train_payloads
        ]
        standardizer = FeatureStandardizer.fit(np.concatenate(feature_arrays, axis=0))
        train_actors = [
            m1_runner._actor_tensors(payload, standardizer, config["model"], device)
            for payload in train_payloads
        ]
        holdout_actors = [
            m1_runner._actor_tensors(payload, standardizer, config["model"], device)
            for payload in holdout_payloads
        ]
        model = RaySignedSurfaceField(
            train_actors[0]["evidence_t"].shape[1],
            latent_dim=int(config["model"]["latent_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            frequency_count=int(config["model"]["frequency_count"]),
            maximum_distance_m=float(config["model"]["maximum_distance_m"]),
        ).to(device)
        torch.nn.init.constant_(model.decoder[-1].bias, 0.1)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "signed_training"})
        history = _train(model, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "evidence_dim": int(train_actors[0]["evidence_t"].shape[1]),
                "latent_dim": int(config["model"]["latent_dim"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "frequency_count": int(config["model"]["frequency_count"]),
                "maximum_distance_m": float(config["model"]["maximum_distance_m"]),
                "seed": int(config["model"]["seed"]),
            },
            run_dir / "MODEL.pt",
        )

        _write_json(run_dir / "status.json", {"status": "running", "phase": "train_holdout"})
        rows: list[dict[str, Any]] = []
        actors_with_surface = 0
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                output, extracted, sdf_min, sdf_max = _extract(model, actor, config["extraction"])
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["extraction"]["output_voxel_size_m"]),
                )
                row = evaluate_actor_surface(
                    baseline,
                    output,
                    actor["target"],
                    actor["target_sensor_origins"],
                    hazardous=bool(actor["hazardous"]),
                    device=device,
                    lateral_tolerance_m=float(config["evaluation"]["literal_lateral_tolerance_m"]),
                    depth_tolerance_m=float(config["evaluation"]["literal_depth_tolerance_m"]),
                    distance_chunk_size=int(config["evaluation"]["distance_chunk_size"]),
                )
                if extracted:
                    actors_with_surface += 1
                row.update(
                    {
                        "track_id": str(actor["track_id"]),
                        "scene_name": str(actor["scene_name"]),
                        "extracted_field_points": int(extracted),
                        "sdf_min_m": sdf_min,
                        "sdf_max_m": sdf_max,
                    }
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "m3_train_holdout",
                            "progress": f"{index + 1}/{len(holdout_actors)}",
                            "actors_with_surface": actors_with_surface,
                        }
                    ),
                    flush=True,
                )
        metrics = summarize_surface_rows(rows)
        extraction_fraction = actors_with_surface / max(len(rows), 1)
        decisions = _decisions(metrics, extraction_fraction, config["decision"])
        passed = all(decisions.values())
        _write_jsonl(run_dir / "TRAIN_HOLDOUT_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m3_ray_signed_level_set.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m3_train_holdout_passed" if passed else "m3_train_holdout_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "actors_with_extracted_surface": actors_with_surface,
            "extraction_actor_fraction": extraction_fraction,
            "training_history": history,
            "train_holdout": metrics,
            "decisions": decisions,
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "train_holdout", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m3", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
