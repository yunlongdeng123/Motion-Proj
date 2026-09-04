"""测量冻结M2中first-return与surface目标的真实梯度夹角。"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import differentiable_symmetric_chamfer
from motion_proj.worldsim_v71.first_return_renderer import differentiable_first_return_depth
from motion_proj.worldsim_v71.ray_displacement import (
    RaySurfaceRelocationMLP,
    apply_predicted_relocation,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _actor_objectives(
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
    predicted_depth = differentiable_first_return_depth(
        surface, origins, targets, **config["renderer"]
    )
    first = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    surface_distance = differentiable_symmetric_chamfer(surface, targets) / reference_surface
    return first, surface_distance


def _gradient_stats(
    first: torch.Tensor,
    surface: torch.Tensor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    parameters = tuple(parameter for _, parameter in named_parameters)
    first_grads = torch.autograd.grad(first, parameters, retain_graph=True, allow_unused=True)
    surface_grads = torch.autograd.grad(surface, parameters, allow_unused=True)

    def accumulate(prefix: str) -> tuple[float, float, float]:
        dot = 0.0
        first_sq = 0.0
        surface_sq = 0.0
        for (name, _), first_grad, surface_grad in zip(
            named_parameters, first_grads, surface_grads
        ):
            if not name.startswith(prefix) or first_grad is None or surface_grad is None:
                continue
            dot += float((first_grad * surface_grad).sum().detach())
            first_sq += float(first_grad.square().sum().detach())
            surface_sq += float(surface_grad.square().sum().detach())
        return dot, first_sq, surface_sq

    payload: dict[str, Any] = {
        "first_loss": float(first.detach()),
        "surface_loss": float(surface.detach()),
    }
    for label, prefix in (("all", ""), ("encoder", "point_encoder"), ("head", "head")):
        dot, first_sq, surface_sq = accumulate(prefix)
        denominator = math.sqrt(first_sq * surface_sq)
        payload[label] = {
            "cosine": dot / denominator if denominator > 0.0 else 0.0,
            "dot": dot,
            "first_norm": math.sqrt(first_sq),
            "surface_norm": math.sqrt(surface_sq),
            "first_to_surface_norm_ratio": math.sqrt(first_sq / surface_sq)
            if surface_sq > 0.0
            else 0.0,
        }
    return payload


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "negative_fraction": float(np.mean(array < 0.0)),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "gradient_diagnosis"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("gradient diagnosis requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    started = time.monotonic()
    try:
        checkpoint = torch.load(
            Path(config["m2_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = RaySurfaceRelocationMLP(
            int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.train()
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_training_actors"])
        )
        selected = torch.linspace(
            0,
            len(paths) - 1,
            steps=min(int(config["sample_actor_count"]), len(paths)),
        ).long().tolist()
        actors = [
            actor
            for index in selected
            if (actor := m0_runner._prepare_actor(paths[index], standardizer, device))
            is not None
        ]
        named_parameters = tuple(model.named_parameters())
        rows: list[dict[str, Any]] = []
        batch_size = int(config["actor_batch_size"])
        for start in range(0, len(actors), batch_size):
            objectives = [
                _actor_objectives(model, actor, config)
                for actor in actors[start : start + batch_size]
            ]
            first = torch.stack([value[0] for value in objectives]).mean()
            surface = torch.stack([value[1] for value in objectives]).mean()
            row = _gradient_stats(first, surface, named_parameters)
            row["batch_index"] = len(rows)
            row["actor_count"] = len(objectives)
            rows.append(row)
            print(json.dumps({"stage": "gradient_conflict", **row}), flush=True)
        metrics = {
            key: {
                "cosine": _summary([float(row[key]["cosine"]) for row in rows]),
                "first_to_surface_norm_ratio": _summary(
                    [float(row[key]["first_to_surface_norm_ratio"]) for row in rows]
                ),
            }
            for key in ("all", "encoder", "head")
        }
        summary = {
            "schema_version": "worldsim_v71.m2_gradient_conflict_diagnostic.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "diagnostic_only",
            "m2_checkpoint": config["m2_run"],
            "actor_count": len(actors),
            "batch_count": len(rows),
            "batch_size": batch_size,
            "metrics": metrics,
            "rows": rows,
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
                "phase": "gradient_diagnosis",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "gradient_diagnosis",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
