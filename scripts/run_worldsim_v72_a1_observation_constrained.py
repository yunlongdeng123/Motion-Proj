"""在 AdaPoinTr 表面上加入同一几何产生的 held-out ray hit/free 监督。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
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

from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)
from scripts.run_worldsim_v72_g0_raw_fusion import _log_macro_summary, _moving, _stratum
from scripts.run_worldsim_v72_g3_adapointr import _build_model


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(config: Mapping[str, Any], checkpoint_path: Path) -> torch.nn.Module:
    build_config = {
        "pointr_root": config["pointr_root"],
        "pointr_config": config["pointr_config"],
        "input_point_count": config["input_point_count"],
        "output_point_count": config["output_point_count"],
        "initialization": "pretrained_adapted",
        "checkpoint": config["official_checkpoint"],
        "checkpoint_sha256": config["official_checkpoint_sha256"],
    }
    model, _ = _build_model(build_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"base checkpoint load mismatch: {missing}/{unexpected}")
    return model.cuda()


def _load_training_data(
    adapter_root: Path, cache_root: Path, rays_per_actor: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in (adapter_root / "SAMPLES.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == "train"]
    partials, targets, ray_points, ray_origins = [], [], [], []
    for row in rows:
        with np.load(adapter_root / row["relative_path"], allow_pickle=False) as payload:
            partials.append(np.asarray(payload["partial_normalized"], dtype=np.float32))
            targets.append(np.asarray(payload["target_normalized"], dtype=np.float32))
        scene_name, track_id = str(row["scene_name"]), str(row["track_id"])
        scale = float(row["scale_m"])
        with np.load(cache_root / scene_name / f"{track_id}.npz", allow_pickle=False) as payload:
            points = np.asarray(payload["target"], dtype=np.float32)
            origins = np.asarray(payload["target_sensor_origins"], dtype=np.float32)
        indices = np.linspace(0, len(points) - 1, num=rays_per_actor, dtype=np.int64)
        ray_points.append(points[indices] / scale)
        ray_origins.append(origins[indices] / scale)
    return (
        torch.from_numpy(np.stack(partials)),
        torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(ray_points)),
        torch.from_numpy(np.stack(ray_origins)),
        rows,
    )


def _load_clean_inputs(role_root: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in (role_root / "ACTORS.jsonl").read_text().splitlines()]
    inputs = []
    for row in rows:
        with np.load(role_root / row["relative_adapter"], allow_pickle=False) as payload:
            inputs.append(np.asarray(payload["partial_normalized"], dtype=np.float32))
    return torch.from_numpy(np.stack(inputs)), rows


def _predict(model: torch.nn.Module, inputs: torch.Tensor, batch_size: int, amp: bool) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            partial = inputs[start : start + batch_size].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                output = model(partial)[-1]
            predictions.extend(output.float().cpu().numpy())
    return predictions


def _ray_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    origins: torch.Tensor,
    ray_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    direction = target - origins
    target_depth = torch.linalg.vector_norm(direction, dim=-1).clamp_min(1.0e-6)
    direction = direction / target_depth[..., None]
    vectors = prediction[:, None, :, :] - origins[:, :, None, :]
    depths = torch.sum(vectors * direction[:, :, None, :], dim=-1)
    lateral = vectors - depths[..., None] * direction[:, :, None, :]
    lateral_sq = torch.sum(lateral.square(), dim=-1)
    temperature = float(ray_config["lateral_temperature_normalized"])
    weights = torch.softmax(-lateral_sq / (temperature * temperature), dim=-1)
    predicted_depth = torch.sum(weights * depths, dim=-1)
    hit = F.huber_loss(
        predicted_depth,
        target_depth,
        delta=float(ray_config["hit_huber_delta_normalized"]),
    )
    free = F.relu(
        target_depth - float(ray_config["free_margin_normalized"]) - predicted_depth
    ).mean()
    return hit, free


def _save_checkpoint(path: Path, epoch: int, model: torch.nn.Module) -> None:
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"epoch": epoch, "model": model.state_dict()}, temporary)
    temporary.replace(path)


def _train(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    ray_config = config["ray_supervision"]
    partial, surface_target, ray_target, ray_origins, _ = _load_training_data(
        Path(config["training_adapter_root"]),
        Path(config["training_cache_root"]),
        int(ray_config["rays_per_actor"]),
    )
    partial = partial.pin_memory().cuda(non_blocking=True)
    surface_target = surface_target.pin_memory().cuda(non_blocking=True)
    ray_target = ray_target.pin_memory().cuda(non_blocking=True)
    ray_origins = ray_origins.pin_memory().cuda(non_blocking=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    schedule = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: max(
            float(schedule["lr_decay"]) ** (epoch / float(schedule["decay_step"])),
            float(schedule["lowest_decay"]),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))
    history: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    model.train()
    for epoch in range(1, int(config["train_epochs"]) + 1):
        started = time.perf_counter()
        permutation = torch.randperm(len(partial), device="cuda")
        sums = defaultdict(float)
        count_sum = 0
        for start in range(0, len(partial), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["amp"])):
                outputs = model(partial[indices])
            outputs_float = tuple(value.float() for value in outputs)
            denoised, reconstruction = model.get_loss(
                outputs_float, surface_target[indices], epoch=epoch
            )
            hit, free = _ray_losses(
                outputs_float[-1], ray_target[indices], ray_origins[indices], ray_config
            )
            surface = denoised + reconstruction
            loss = surface + float(ray_config["lambda_hit"]) * hit + float(ray_config["lambda_known_free"]) * free
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = len(indices)
            count_sum += count
            for name, value in (("loss", loss), ("surface", surface), ("hit", hit), ("free", free)):
                sums[name] += float(value.detach()) * count
        scheduler.step()
        torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            **{name: value / count_sum for name, value in sums.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
            "wall_seconds": time.perf_counter() - started,
        }
        history.append(row)
        _write_jsonl(run_dir / "TRAIN.jsonl", history)
        if epoch == 1 or epoch % int(config["console_every_epochs"]) == 0:
            print(json.dumps({"stage": "a1_train", **row}), flush=True)
        if epoch % int(config["checkpoint_every_epochs"]) == 0:
            _save_checkpoint(run_dir / "latest.pt", epoch, model)
            _write_json(run_dir / "status.json", {"status": "running", "phase": "training", "epoch": epoch})
    _save_checkpoint(run_dir / "final.pt", int(config["train_epochs"]), model)
    return history


def _evaluate(
    role_root: Path,
    identities: list[Mapping[str, Any]],
    predictions: Mapping[str, list[np.ndarray]],
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budgets: list[int | None] = [int(value) for value in config["density_budgets"]]
    if config["include_native_density"]:
        budgets.append(None)
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        with np.load(role_root / identity["relative_adapter"], allow_pickle=False) as payload:
            canonical = np.asarray(payload["canonical"], dtype=np.float32)
            tsdf = np.asarray(payload["tsdf_surface"], dtype=np.float32)
            target = np.asarray(payload["target"], dtype=np.float32)
            origins = np.asarray(payload["target_sensor_origins"], dtype=np.float32)
            trajectory = np.asarray(payload["trajectory_xyz_m"], dtype=np.float32)
        scale = float(identity["scale_m"])
        methods = {"G0_raw": canonical, "G1_tsdf": tsdf}
        methods.update({name: values[index] * scale for name, values in predictions.items()})
        methods["A1_anchored_surface"] = np.concatenate(
            [tsdf, methods["A1_observation_constrained"]], axis=0
        )
        moving, displacement = _moving(trajectory)
        for method, native_surface in methods.items():
            for budget in budgets:
                if method == "A1_anchored_surface" and budget is not None:
                    tsdf_count = int(
                        round(
                            float(config["anchored_surface"]["tsdf_fraction_at_matched_budget"])
                            * budget
                        )
                    )
                    completion_count = int(budget) - tsdf_count
                    surface = np.concatenate(
                        [
                            deterministic_farthest_point_sample(tsdf, tsdf_count),
                            deterministic_farthest_point_sample(
                                methods["A1_observation_constrained"], completion_count
                            ),
                        ],
                        axis=0,
                    )
                else:
                    surface = (
                        native_surface
                        if budget is None
                        else deterministic_farthest_point_sample(native_surface, budget)
                    )
                metrics = evaluate_point_surface(
                    surface,
                    target,
                    origins,
                    fscore_threshold_m=float(config["evaluation"]["fscore_threshold_m"]),
                    lateral_tolerance_m=float(config["evaluation"]["literal_lateral_tolerance_m"]),
                    depth_tolerance_m=float(config["evaluation"]["literal_depth_tolerance_m"]),
                    distance_chunk_size=int(config["evaluation"]["distance_chunk_size"]),
                    ray_chunk_size=int(config["evaluation"]["ray_chunk_size"]),
                    point_chunk_size=int(config["evaluation"]["point_chunk_size"]),
                )
                rows.append(
                    {
                        "method": method,
                        "density_budget": "native" if budget is None else budget,
                        "native_point_count": len(native_surface),
                        "log_id": identity["log_id"],
                        "scene_name": identity["scene_name"],
                        "track_id": identity["track_id"],
                        "category": identity["category"],
                        "hazardous": identity["hazardous"],
                        "moving": moving,
                        "trajectory_max_displacement_m": displacement,
                        "target_used_by_surface_generation": False,
                        **metrics,
                    }
                )
        if (index + 1) % 10 == 0 or index + 1 == len(identities):
            print(json.dumps({"stage": "a1_evaluate", "progress": f"{index + 1}/{len(identities)}"}), flush=True)
    _write_jsonl(run_dir / "ACTORS.jsonl", rows)
    summary: dict[str, Any] = {}
    log_rows_all: list[dict[str, Any]] = []
    method_names = sorted({row["method"] for row in rows})
    for method_index, method in enumerate(method_names):
        summary[method] = {}
        for budget_index, budget in enumerate(budgets):
            label = "native" if budget is None else str(budget)
            selected = [row for row in rows if row["method"] == method and str(row["density_budget"]) == label]
            log_rows, macro = _log_macro_summary(
                selected,
                seed=int(config["bootstrap"]["seed"]) + 10 * method_index + budget_index,
                samples=int(config["bootstrap"]["samples"]),
            )
            log_rows_all.extend({"method": method, "density_budget": label, **row} for row in log_rows)
            summary[method][label] = {
                "all": _stratum(selected),
                "hazard": _stratum([row for row in selected if row["hazardous"]]),
                "clear": _stratum([row for row in selected if not row["hazardous"]]),
                "moving": _stratum([row for row in selected if row["moving"]]),
                "quasi_static": _stratum([row for row in selected if not row["moving"]]),
                "log_macro_bootstrap": macro,
            }
    _write_jsonl(run_dir / "LOGS.jsonl", log_rows_all)
    return rows, summary


def run(config_path: Path, role: str, run_id: str, candidate_checkpoint: Path | None) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if role not in {"train", "dev", "route_select"}:
        raise PermissionError("A1 只允许 train、dev 或 route_select")
    if config["source_test_read"] or config["external_test_read"]:
        raise PermissionError("A1 不允许读取最终角色")
    if role in {"dev", "route_select"} and candidate_checkpoint is None:
        raise ValueError("dev/route_select 必须显式绑定冻结 candidate checkpoint")
    if role == "train" and candidate_checkpoint is not None:
        raise ValueError("train 从冻结 base checkpoint 开始，不接受 candidate checkpoint")
    task_id = config["task_ids"][role]
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / task_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    role_root = Path(config["clean_data_root"]) / role if role != "train" else None
    started = time.monotonic()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    manifest = {
        "schema_version": "worldsim_v72.a1_observation_constrained.v1",
        "task_id": task_id,
        "run_id": run_id,
        "run_uri": f"run://worldsim_v72/{task_id}/{run_id}",
        "status": "running",
        "role": role,
        "training": role == "train",
        "candidate_checkpoint": str(candidate_checkpoint) if candidate_checkpoint else None,
        "git_commit": git_commit,
        "source_test_read": False,
        "external_test_read": False,
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": config["failure_ledger_delta"],
    }
    _write_json(run_dir / "manifest.json", manifest)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump({**config, "role": role, "run_id": run_id}, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "phase": "target_free_prediction"})
    try:
        base = _load_model(config, Path(config["base_checkpoint"]))
        if role == "train":
            _write_json(run_dir / "status.json", {"status": "running", "phase": "training"})
            history = _train(base, config, run_dir)
            candidate_checkpoint = run_dir / "final.pt"
            summary = {
                **manifest,
                "status": "done",
                "candidate_checkpoint": str(candidate_checkpoint),
                "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
                "train_actor_count": 593,
                "final_train": history[-1],
                "target_free_dev_prediction": True,
                "dev_target_read": False,
                "route_decision_allowed": False,
                "resources": {
                    "gpu": torch.cuda.get_device_name(0),
                    "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                    "wall_seconds": time.monotonic() - started,
                },
            }
            _write_json(run_dir / "summary.json", summary)
            _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "summary": "summary.json"})
            _write_json(run_dir / "status.json", {"status": "done", "phase": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
            return summary

        assert role_root is not None and candidate_checkpoint is not None
        clean_inputs, identities = _load_clean_inputs(role_root)
        base_predictions = _predict(base, clean_inputs.pin_memory(), int(config["batch_size"]), bool(config["amp"]))
        del base
        torch.cuda.empty_cache()
        candidate = _load_model(config, candidate_checkpoint)
        history = []
        candidate_predictions = _predict(candidate, clean_inputs.pin_memory(), int(config["batch_size"]), bool(config["amp"]))
        _write_json(run_dir / "status.json", {"status": "running", "phase": "evaluation"})
        actor_rows, metrics = _evaluate(
            role_root,
            identities,
            {"G3_adapointr": base_predictions, "A1_observation_constrained": candidate_predictions},
            config,
            run_dir,
        )
        summary = {
            **manifest,
            "status": "done",
            "candidate_checkpoint": str(candidate_checkpoint),
            "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
            "actor_count": len(identities),
            "log_count": len({row["log_id"] for row in identities}),
            "method_count": 5,
            "density_budgets": [*config["density_budgets"], "native"],
            "final_train": history[-1] if history else None,
            "metrics": metrics,
            "target_free_surface_generation": True,
            "route_decision_allowed": role == "route_select",
            "resources": {
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.role, args.run_id, args.candidate_checkpoint), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
