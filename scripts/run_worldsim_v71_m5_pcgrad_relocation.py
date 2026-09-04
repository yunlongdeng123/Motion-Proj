"""训练V7.1 M5冲突投影位移并做train-role development评估。"""

from __future__ import annotations

import argparse
import json
import math
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
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.evaluate_surface import (
    differentiable_symmetric_chamfer,
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.first_return_renderer import differentiable_first_return_depth
from motion_proj.worldsim_v71.ray_displacement import (
    RaySurfaceDisplacementMLP,
    RaySurfaceRelocationMLP,
    apply_predicted_relocation,
    initialize_relocation_from_m0,
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


def _move(
    model: RaySurfaceRelocationMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(actor["features"])
    moved = apply_predicted_relocation(
        actor["candidates_t"],
        actor["ray_directions_t"],
        actor["normals_t"],
        prediction,
        maximum_ray_displacement_m=float(config["maximum_ray_displacement_m"]),
        maximum_normal_displacement_m=float(config["maximum_normal_displacement_m"]),
        actor_half_size_m=actor["size_t"] * 0.5,
        cuboid_padding_m=float(config["cuboid_padding_m"]),
    )
    return prediction, moved


def _actor_losses(
    model: RaySurfaceRelocationMLP,
    teacher: RaySurfaceDisplacementMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    prediction, moved = _move(model, actor, config)
    anchors = m0_runner._limit_tensor(
        actor["anchors_t"], int(config["maximum_training_anchors"])
    )
    surface = torch.cat([anchors, moved], dim=0)
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=moved.device)
    origins = torch.as_tensor(
        actor["target_sensor_origins"], dtype=torch.float32, device=moved.device
    )
    if len(targets) > int(config["maximum_training_rays"]):
        indices = torch.linspace(
            0,
            len(targets) - 1,
            steps=int(config["maximum_training_rays"]),
            device=moved.device,
        ).long()
        targets = targets.index_select(0, indices)
        origins = origins.index_select(0, indices)
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1)
    with torch.no_grad():
        baseline = torch.cat([anchors, actor["candidates_t"]], dim=0)
        baseline_depth = differentiable_first_return_depth(
            baseline, origins, targets, **config["renderer"]
        )
        reference_first = F.smooth_l1_loss(baseline_depth, target_depth).clamp_min(1.0e-3)
        reference_surface = differentiable_symmetric_chamfer(
            baseline, targets
        ).clamp_min(1.0e-3)
        teacher_prediction = teacher(actor["features"])[:, :2]
    predicted_depth = differentiable_first_return_depth(
        surface, origins, targets, **config["renderer"]
    )
    first = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    surface_distance = differentiable_symmetric_chamfer(surface, targets) / reference_surface
    displacement = moved - actor["candidates_t"]
    anchor = displacement.square().mean() / max(
        float(config["maximum_ray_displacement_m"]) ** 2, 1.0e-6
    )
    if len(moved) > 1:
        nearest = torch.cdist(actor["candidates_t"], actor["candidates_t"]).topk(
            min(5, len(moved)), largest=False
        ).indices[:, 1:]
        center = displacement[:, None, :].expand(-1, nearest.shape[1], -1)
        smooth = (center - displacement[nearest]).square().mean() / max(
            float(config["maximum_ray_displacement_m"]) ** 2, 1.0e-6
        )
    else:
        smooth = torch.zeros((), dtype=moved.dtype, device=moved.device)
    teacher_loss = F.smooth_l1_loss(prediction, teacher_prediction)
    auxiliary = (
        float(config["anchor_loss_weight"]) * anchor
        + float(config["smooth_loss_weight"]) * smooth
        + float(config["teacher_consistency_weight"]) * teacher_loss
    )
    return {
        "first": first,
        "surface": surface_distance,
        "anchor": anchor,
        "smooth": smooth,
        "teacher": teacher_loss,
        "auxiliary": auxiliary,
    }


def _pcgrad_step(
    model: RaySurfaceRelocationMLP,
    optimizer: torch.optim.Optimizer,
    first: torch.Tensor,
    surface: torch.Tensor,
    auxiliary: torch.Tensor,
) -> tuple[bool, float]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    first_grads = torch.autograd.grad(first, parameters, retain_graph=True, allow_unused=True)
    surface_grads = torch.autograd.grad(surface, parameters, retain_graph=True, allow_unused=True)
    auxiliary_grads = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
    dot = sum(
        (first_grad * surface_grad).sum()
        for first_grad, surface_grad in zip(first_grads, surface_grads)
        if first_grad is not None and surface_grad is not None
    )
    first_sq = sum(
        first_grad.square().sum() for first_grad in first_grads if first_grad is not None
    )
    surface_sq = sum(
        surface_grad.square().sum() for surface_grad in surface_grads if surface_grad is not None
    )
    conflict = bool(float(dot.detach()) < 0.0)
    denominator = math.sqrt(float((first_sq * surface_sq).detach()))
    cosine = float(dot.detach()) / denominator if denominator > 0.0 else 0.0
    if conflict:
        first_scale = dot / surface_sq.clamp_min(1.0e-12)
        surface_scale = dot / first_sq.clamp_min(1.0e-12)
    optimizer.zero_grad(set_to_none=True)
    for parameter, first_grad, surface_grad, auxiliary_grad in zip(
        parameters, first_grads, surface_grads, auxiliary_grads
    ):
        if first_grad is None and surface_grad is None and auxiliary_grad is None:
            continue
        combined = torch.zeros_like(parameter)
        if first_grad is not None:
            combined.add_(
                first_grad - first_scale * surface_grad
                if conflict and surface_grad is not None
                else first_grad
            )
        if surface_grad is not None:
            combined.add_(
                surface_grad - surface_scale * first_grad
                if conflict and first_grad is not None
                else surface_grad
            )
        if auxiliary_grad is not None:
            combined.add_(auxiliary_grad)
        parameter.grad = combined
    optimizer.step()
    return conflict, cosine


def _train(
    model: RaySurfaceRelocationMLP,
    teacher: RaySurfaceDisplacementMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(len(actors)).tolist()
        totals = {name: 0.0 for name in ("first", "surface", "anchor", "smooth", "teacher", "auxiliary")}
        conflicts = 0
        cosines: list[float] = []
        batches = 0
        for start in range(0, len(permutation), batch_size):
            items = [
                _actor_losses(model, teacher, actors[index], config)
                for index in permutation[start : start + batch_size]
            ]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in totals
            }
            conflict, cosine = _pcgrad_step(
                model, optimizer, means["first"], means["surface"], means["auxiliary"]
            )
            conflicts += int(conflict)
            cosines.append(cosine)
            batches += 1
            for name in totals:
                totals[name] += float(means[name].detach()) * len(items)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / len(actors) for name, value in totals.items()},
            "conflict_batch_fraction": conflicts / max(batches, 1),
            "mean_preprojection_cosine": float(np.mean(cosines)),
        }
        history.append(row)
        print(json.dumps({"stage": "m5_pcgrad_train", "actors": len(actors), **row}), flush=True)
    return history


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading_train"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M5 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))
    started = time.monotonic()
    try:
        checkpoint = torch.load(
            Path(config["m0_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        teacher = RaySurfaceDisplacementMLP(
            int(checkpoint["input_dim"]), int(config["model"]["hidden_dim"])
        ).to(device)
        teacher.load_state_dict(checkpoint["state_dict"])
        teacher.eval()
        teacher.requires_grad_(False)
        model = RaySurfaceRelocationMLP(
            int(checkpoint["input_dim"]), int(config["model"]["hidden_dim"])
        ).to(device)
        initialize_relocation_from_m0(model, teacher)

        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["model"]["maximum_training_actors"])
        )
        actors = [
            actor
            for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        stride = int(config["model"]["holdout_stride"])
        train_actors = [actor for index, actor in enumerate(actors) if index % stride != 0]
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "pcgrad_training"})
        history = _train(model, teacher, train_actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "seed": int(config["model"]["seed"]),
                "deployment_surface": "anchors_plus_all_moved_candidates",
                "optimizer": "two_task_symmetric_pcgrad_plus_auxiliary",
            },
            run_dir / "MODEL.pt",
        )

        _write_json(run_dir / "status.json", {"status": "running", "phase": "train_holdout"})
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                _, moved = _move(model, actor, config["model"])
                output = _voxel_unique(
                    torch.cat([actor["anchors_t"], moved], dim=0).cpu().numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
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
                row.update(
                    {
                        "scene_name": str(actor["scene_name"]),
                        "track_id": str(actor["track_id"]),
                        "mean_displacement_m": float(
                            torch.linalg.vector_norm(moved - actor["candidates_t"], dim=1).mean()
                        ),
                    }
                )
                rows.append(row)
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {
                                "stage": "m5_train_holdout",
                                "progress": f"{index + 1}/{len(holdout_actors)}",
                            }
                        ),
                        flush=True,
                    )
        metrics = summarize_surface_rows(rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        passed = all(decisions.values())
        _write_jsonl(run_dir / "TRAIN_HOLDOUT_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m5_pcgrad_relocation.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m5_development_passed" if passed else "m5_development_rejected",
            "training_actor_count": len(train_actors),
            "holdout_actor_count": len(holdout_actors),
            "pretrained_holdout_exposure": True,
            "training_history": history,
            "train_holdout": metrics,
            "decisions": decisions,
            "m0_initialization": config["m0_run"],
            "diagnostic_c_source": config["diagnostic_c_run"],
            "deployment_surface": "anchors_plus_all_moved_candidates",
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
            {
                "status": "done",
                "phase": "train_holdout",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m5", "error": f"{type(error).__name__}: {error}"},
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
