"""训练V7.1 M4平衡occupancy decision boundary并做开发holdout。"""

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
import run_worldsim_v71_m3_ray_signed_level_set as m3_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.signed_surface_field import RayOccupancyBoundaryField


def _ray_labels(
    actor: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = actor["target_t"]
    origins = actor["target_origins_t"]
    count = min(len(targets), int(config["maximum_target_rays"]))
    selected = torch.linspace(0, len(targets) - 1, steps=count, device=targets.device).long()
    targets = targets[selected]
    origins = origins[selected]
    direction = targets - origins
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    offsets = torch.as_tensor(config["signed_offsets_m"], dtype=targets.dtype, device=targets.device)
    query = (targets[:, None, :] + direction[:, None, :] * offsets[None, :, None]).reshape(-1, 3)
    labels = (offsets > 0.0).to(targets.dtype)[None, :].expand(count, -1).reshape(-1)
    half = actor["size_t"] * 0.5 + float(config["cuboid_padding_m"])
    inside = torch.all(query.abs() <= half[None, :], dim=1)
    query = query[inside]
    labels = labels[inside]

    anchors = actor["anchors_t"]
    boundary = torch.cat([targets, anchors], dim=0)
    if len(boundary) > int(config["maximum_boundary_points"]):
        indices = torch.linspace(
            0,
            len(boundary) - 1,
            steps=int(config["maximum_boundary_points"]),
            device=boundary.device,
        ).long()
        boundary = boundary[indices]
    return query, labels, boundary


def _first_return_loss(
    model: RayOccupancyBoundaryField,
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
    logits = model.decode(
        latent,
        m3_runner._normalized(query.reshape(-1, 3), actor["size_t"]),
        actor["size_t"],
    ).reshape(count, -1)
    density = torch.exp(-logits.abs() / float(config["render_logit_temperature"]))
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
    model: RayOccupancyBoundaryField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query, labels, boundary = _ray_labels(actor, config)
    logits = model(
        actor["evidence_t"], m3_runner._normalized(query, actor["size_t"]), actor["size_t"]
    )
    occupancy = F.binary_cross_entropy_with_logits(logits, labels)
    boundary_logits = model(
        actor["evidence_t"],
        m3_runner._normalized(boundary, actor["size_t"]),
        actor["size_t"],
    )
    boundary_loss = F.smooth_l1_loss(boundary_logits, torch.zeros_like(boundary_logits))
    first_return = _first_return_loss(model, actor, config)
    total = (
        float(config["occupancy_loss_weight"]) * occupancy
        + float(config["boundary_loss_weight"]) * boundary_loss
        + float(config["first_return_loss_weight"]) * first_return
    )
    return total, {"occupancy": occupancy, "boundary": boundary_loss, "first_return": first_return}


def _train(
    model: RayOccupancyBoundaryField,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(len(actors)).tolist()
        totals = {"total": 0.0, "occupancy": 0.0, "boundary": 0.0, "first_return": 0.0}
        for start in range(0, len(permutation), batch_size):
            batch = [_actor_loss(model, actors[index], config) for index in permutation[start : start + batch_size]]
            loss = torch.stack([item[0] for item in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["total"] += float(loss.detach()) * len(batch)
            for name in ("occupancy", "boundary", "first_return"):
                totals[name] += sum(float(item[1][name].detach()) for item in batch)
        row = {"epoch": epoch + 1, **{name: value / len(actors) for name, value in totals.items()}}
        history.append(row)
        print(json.dumps({"stage": "m4_occupancy_train", "actors": len(actors), **row}), flush=True)
    return history


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    m3_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "loading_train"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M4 requires CUDA")
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
        features = [
            m1_runner._evidence_features(payload, int(config["model"]["maximum_evidence_points"]))
            for payload in train_payloads
        ]
        standardizer = FeatureStandardizer.fit(np.concatenate(features, axis=0))
        train_actors = [
            m1_runner._actor_tensors(payload, standardizer, config["model"], device)
            for payload in train_payloads
        ]
        holdout_actors = [
            m1_runner._actor_tensors(payload, standardizer, config["model"], device)
            for payload in holdout_payloads
        ]
        model = RayOccupancyBoundaryField(
            train_actors[0]["evidence_t"].shape[1],
            latent_dim=int(config["model"]["latent_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            frequency_count=int(config["model"]["frequency_count"]),
        ).to(device)
        torch.nn.init.constant_(model.decoder[-1].bias, 0.0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        m3_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "occupancy_training"})
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
                "seed": int(config["model"]["seed"]),
                "surface": "zero_occupancy_logit",
            },
            run_dir / "MODEL.pt",
        )

        m3_runner._write_json(run_dir / "status.json", {"status": "running", "phase": "train_holdout"})
        rows: list[dict[str, Any]] = []
        actors_with_surface = 0
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                output, extracted, logit_min, logit_max = m3_runner._extract(
                    model, actor, config["extraction"]
                )
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
                        "logit_min": logit_min,
                        "logit_max": logit_max,
                    }
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "m4_train_holdout",
                            "progress": f"{index + 1}/{len(holdout_actors)}",
                            "actors_with_surface": actors_with_surface,
                        }
                    ),
                    flush=True,
                )
        metrics = summarize_surface_rows(rows)
        extraction_fraction = actors_with_surface / max(len(rows), 1)
        decisions = m3_runner._decisions(metrics, extraction_fraction, config["decision"])
        passed = all(decisions.values())
        m3_runner._write_jsonl(run_dir / "TRAIN_HOLDOUT_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m4_balanced_occupancy_boundary.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m4_development_passed" if passed else "m4_development_rejected",
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
        m3_runner._write_json(run_dir / "summary.json", summary)
        m3_runner._write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "train_holdout", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        m3_runner._write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m4", "error": f"{type(error).__name__}: {error}"},
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
