"""训练并一次性Source Final确认V7.1 M2部署一致位移模型。"""

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
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evaluate_surface import (
    differentiable_symmetric_chamfer,
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


def _moved(
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


def _physical_actor_loss(
    model: RaySurfaceRelocationMLP,
    teacher: RaySurfaceDisplacementMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    prediction, moved = _moved(model, actor, config)
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
        reference_surface = differentiable_symmetric_chamfer(baseline, targets).clamp_min(1.0e-3)
        teacher_prediction = teacher(actor["features"])[:, :2]
    predicted_depth = differentiable_first_return_depth(
        surface, origins, targets, **config["renderer"]
    )
    first_loss = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    surface_loss = differentiable_symmetric_chamfer(surface, targets) / reference_surface
    displacement = moved - actor["candidates_t"]
    anchor_loss = displacement.square().mean() / max(
        float(config["maximum_ray_displacement_m"]) ** 2, 1.0e-6
    )
    if len(moved) > 1:
        nearest = torch.cdist(actor["candidates_t"], actor["candidates_t"]).topk(
            min(5, len(moved)), largest=False
        ).indices[:, 1:]
        center = displacement[:, None, :].expand(-1, nearest.shape[1], -1)
        smooth_loss = (center - displacement[nearest]).square().mean() / max(
            float(config["maximum_ray_displacement_m"]) ** 2, 1.0e-6
        )
    else:
        smooth_loss = torch.zeros((), dtype=moved.dtype, device=moved.device)
    teacher_loss = F.smooth_l1_loss(prediction, teacher_prediction)
    return (
        first_loss
        + surface_loss
        + float(config["anchor_loss_weight"]) * anchor_loss
        + float(config["smooth_loss_weight"]) * smooth_loss
        + float(config["teacher_consistency_weight"]) * teacher_loss
    )


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
        total = 0.0
        for start in range(0, len(permutation), batch_size):
            losses = [
                _physical_actor_loss(model, teacher, actors[index], config)
                for index in permutation[start : start + batch_size]
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(losses)
        mean_loss = total / len(actors)
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss})
        print(
            json.dumps(
                {
                    "stage": "m2_deploy_consistent_train",
                    "epoch": epoch + 1,
                    "actors": len(actors),
                    "loss": mean_loss,
                }
            ),
            flush=True,
        )
    return history


class _M0EvaluationAdapter(torch.nn.Module):
    """为冻结 evaluator 提供永不UNKNOWN的第三个logit。"""

    def __init__(self, model: RaySurfaceRelocationMLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        displacement = self.model(features)
        known = torch.full(
            (len(displacement), 1), -100.0, dtype=displacement.dtype, device=displacement.device
        )
        return torch.cat([displacement, known], dim=1)


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    m0_runner._deep_update(compiler, config["compiler_overrides"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading_train"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M2 requires CUDA")
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(
            run_dir / "status.json", {"status": "running", "phase": "deploy_consistent_training"}
        )
        history = _train(model, teacher, actors, config["model"], optimizer)
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "seed": int(config["model"]["seed"]),
                "deployment_surface": "anchors_plus_all_moved_candidates",
            },
            run_dir / "MODEL.pt",
        )

        _write_json(run_dir / "status.json", {"status": "running", "phase": "source_final"})
        index = build_v71_index(Path(config["source"]["dataset_root"]), split)
        adapter = _M0EvaluationAdapter(model)
        evaluation_model = dict(config["model"])
        evaluation_model["unknown_threshold"] = 0.5
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for position, scene_name in enumerate(split["roles"]["source_final"]):
                bundles = compile_source_scene(
                    scene_name, index, config["actors"], compiler, device
                )
                for bundle in bundles:
                    row = m0_runner._evaluate_bundle(
                        bundle,
                        adapter,
                        standardizer,
                        evaluation_model,
                        config["evaluation"],
                        device,
                    )
                    if row is not None:
                        row["deployment_surface"] = "anchors_plus_all_moved_candidates"
                        rows.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "m2_source_final",
                            "progress": f"{position + 1}/{len(split['roles']['source_final'])}",
                            "actors": len(rows),
                        }
                    ),
                    flush=True,
                )
        metrics = summarize_surface_rows(rows)
        decisions = m0_runner._decisions(metrics, config["decision"])
        passed = all(decisions.values())
        _write_jsonl(run_dir / "SOURCE_FINAL_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m2_deploy_consistent_relocation.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m2_source_final_passed_frozen" if passed else "m2_source_final_rejected",
            "training_actor_count": len(actors),
            "source_final_actor_count": len(rows),
            "training_history": history,
            "source_final": metrics,
            "decisions": decisions,
            "m0_initialization": config["m0_run"],
            "diagnostic_a_hypothesis_source": config["diagnostic_a_run"],
            "deployment_surface": "anchors_plus_all_moved_candidates",
            "unknown_action_head": False,
            "selection_read": False,
            "source_final_read": True,
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
                "phase": "source_final",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m2", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
