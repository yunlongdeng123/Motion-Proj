"""训练或评测 V7.2 AdaPoinTr G3，并接入统一点表面算子。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)
from scripts.run_worldsim_v72_g0_raw_fusion import (
    _log_macro_summary,
    _moving,
    _stratum,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _adapter_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_rows(root / "SAMPLES.jsonl")
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    if len(train) != 593 or len(holdout) != 66:
        raise RuntimeError(f"adapter split 不等于冻结 593/66: {len(train)}/{len(holdout)}")
    return train, holdout


def _load_arrays(
    root: Path, rows: list[Mapping[str, Any]], *, include_target: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    partials: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for row in rows:
        with np.load(root / str(row["relative_path"]), allow_pickle=False) as payload:
            partials.append(np.asarray(payload["partial_normalized"], dtype=np.float32))
            if include_target:
                targets.append(np.asarray(payload["target_normalized"], dtype=np.float32))
    partial = torch.from_numpy(np.stack(partials))
    target = torch.from_numpy(np.stack(targets)) if include_target else None
    return partial, target


def _import_pointr(root: Path):
    sys.path.insert(0, str(root))
    from models import build_model_from_cfg
    from utils.config import cfg_from_yaml_file

    return build_model_from_cfg, cfg_from_yaml_file


def _build_model(config: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    pointr_root = Path(config["pointr_root"])
    build_model_from_cfg, cfg_from_yaml_file = _import_pointr(pointr_root)
    model_config = cfg_from_yaml_file(str(pointr_root / str(config["pointr_config"]))).model
    model_config.num_points = int(config["output_point_count"])
    model_config.num_query = int(config["input_point_count"])
    model = build_model_from_cfg(model_config)
    initialization = str(config["initialization"])
    report: dict[str, Any] = {
        "initialization": initialization,
        "loaded_tensor_count": 0,
        "shape_mismatches": {},
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if initialization == "scratch":
        return model, report
    checkpoint_path = Path(str(config["checkpoint"]))
    if _sha256(checkpoint_path) != str(config["checkpoint_sha256"]):
        raise RuntimeError("AdaPoinTr checkpoint SHA-256 不匹配")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["base_model"]
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    mismatches = {
        key: {"checkpoint": list(value.shape), "model": list(current[key].shape)}
        for key, value in state.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if initialization == "pretrained_strict" and (mismatches or missing or unexpected):
        raise RuntimeError("官方 zero-shot checkpoint 未能 strict load")
    if initialization == "pretrained_adapted" and set(mismatches) != {
        "decode_head.layer.fc2.weight",
        "decode_head.layer.fc2.bias",
    }:
        raise RuntimeError(f"4096 decoder 的不兼容键超出冻结预期: {mismatches}")
    report.update(
        {
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "loaded_tensor_count": len(compatible),
            "shape_mismatches": mismatches,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    )
    return model, report


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> None:
    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _train(
    model: torch.nn.Module,
    inputs_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    config: Mapping[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    epochs = int(config["train_epochs"])
    if epochs == 0:
        return []
    inputs = inputs_cpu.pin_memory().cuda(non_blocking=True)
    targets = targets_cpu.pin_memory().cuda(non_blocking=True)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
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
    batch_size = int(config["batch_size"])
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        permutation = torch.randperm(len(inputs), device="cuda")
        loss_sum = 0.0
        denoised_sum = 0.0
        reconstruction_sum = 0.0
        sample_count = 0
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            partial = inputs[indices]
            target = targets[indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=bool(config["amp"])
            ):
                output = model(partial)
            output_float = tuple(value.float() for value in output)
            loss_denoised, loss_reconstruction = model.get_loss(
                output_float, target, epoch=epoch
            )
            loss = loss_denoised + loss_reconstruction
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = len(indices)
            sample_count += count
            loss_sum += float(loss.detach()) * count
            denoised_sum += float(loss_denoised.detach()) * count
            reconstruction_sum += float(loss_reconstruction.detach()) * count
        scheduler.step()
        torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            "samples": sample_count,
            "loss": loss_sum / sample_count,
            "loss_denoised": denoised_sum / sample_count,
            "loss_reconstruction": reconstruction_sum / sample_count,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "wall_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        _write_jsonl(run_dir / "TRAIN.jsonl", history)
        if epoch % int(config["console_every_epochs"]) == 0 or epoch == 1:
            print(json.dumps({"stage": "train", **row}), flush=True)
        if epoch % int(config["checkpoint_every_epochs"]) == 0:
            _save_training_checkpoint(
                run_dir / "latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
            )
            _write_json(
                run_dir / "status.json",
                {"status": "running", "phase": "training", "epoch": epoch},
            )
    _save_training_checkpoint(
        run_dir / "final.pt",
        epoch=epochs,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
    )
    return history


def _predict(
    model: torch.nn.Module,
    inputs_cpu: torch.Tensor,
    *,
    batch_size: int,
    amp: bool,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs_cpu), batch_size):
            partial = inputs_cpu[start : start + batch_size].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                prediction = model(partial)[-1]
            outputs.extend(prediction.float().cpu().numpy())
    return outputs


def _evaluate(
    predictions: list[np.ndarray],
    holdout_rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budgets: list[int | None] = [int(value) for value in config["density_budgets"]]
    if bool(config["include_native_density"]):
        budgets.append(None)
    evaluation = config["evaluation"]
    cache_root = Path(config["legacy_cache_root"])
    rows: list[dict[str, Any]] = []
    for index, (prediction_normalized, identity) in enumerate(
        zip(predictions, holdout_rows, strict=True)
    ):
        scene_name = str(identity["scene_name"])
        track_id = str(identity["track_id"])
        scale_m = float(identity["scale_m"])
        prediction = np.asarray(prediction_normalized, dtype=np.float32) * scale_m
        with np.load(
            cache_root / scene_name / f"{track_id}.npz", allow_pickle=False
        ) as payload:
            target = np.asarray(payload["target"], dtype=np.float32)
            origins = np.asarray(payload["target_sensor_origins"], dtype=np.float32)
            hazardous = bool(payload["hazardous"])
            category = str(payload["category"])
            trajectory = np.asarray(payload["trajectory_xyz_m"], dtype=np.float32)
            size_lwh_m = np.asarray(payload["size_lwh_m"], dtype=np.float32)
        moving, displacement = _moving(trajectory)
        for budget in budgets:
            surface = (
                prediction.copy()
                if budget is None
                else deterministic_farthest_point_sample(prediction, budget)
            )
            metrics = evaluate_point_surface(
                surface,
                target,
                origins,
                fscore_threshold_m=float(evaluation["fscore_threshold_m"]),
                lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
                depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
                distance_chunk_size=int(evaluation["distance_chunk_size"]),
                ray_chunk_size=int(evaluation["ray_chunk_size"]),
                point_chunk_size=int(evaluation["point_chunk_size"]),
            )
            rows.append(
                {
                    "scene_name": scene_name,
                    "log_id": str(identity["log_id"]),
                    "track_id": track_id,
                    "category": category,
                    "hazardous": hazardous,
                    "moving": moving,
                    "trajectory_max_displacement_m": displacement,
                    "size_lwh_m": size_lwh_m.tolist(),
                    "density_budget": "native" if budget is None else budget,
                    "native_output_point_count": len(prediction),
                    "surface_generation": "adapointr_prediction_fps" if budget is not None else "adapointr_prediction_native",
                    "target_used_by_surface_generation": False,
                    **metrics,
                }
            )
        if (index + 1) % 10 == 0 or index + 1 == len(holdout_rows):
            print(
                json.dumps(
                    {"stage": "evaluate", "progress": f"{index + 1}/{len(holdout_rows)}"}
                ),
                flush=True,
            )
    _write_jsonl(run_dir / "ACTORS.jsonl", rows)
    all_log_rows: list[dict[str, Any]] = []
    by_budget: dict[str, Any] = {}
    for offset, budget in enumerate(budgets):
        label = "native" if budget is None else str(budget)
        selected = [row for row in rows if str(row["density_budget"]) == label]
        log_rows, log_macro = _log_macro_summary(
            selected,
            seed=int(config["bootstrap"]["seed"]) + offset,
            samples=int(config["bootstrap"]["samples"]),
        )
        all_log_rows.extend({"density_budget": label, **row} for row in log_rows)
        by_budget[label] = {
            "all": _stratum(selected),
            "hazard": _stratum([row for row in selected if bool(row["hazardous"])]),
            "clear": _stratum([row for row in selected if not bool(row["hazardous"])]),
            "moving": _stratum([row for row in selected if bool(row["moving"])]),
            "quasi_static": _stratum([row for row in selected if not bool(row["moving"])]),
            "log_macro_bootstrap": log_macro,
        }
    _write_jsonl(run_dir / "LOGS.jsonl", all_log_rows)
    return rows, by_budget


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("data_role") != "legacy_diagnostic":
        raise ValueError("G3 当前只允许 legacy_diagnostic")
    if bool(config.get("source_test_read")) or bool(config.get("external_test_read")):
        raise PermissionError("G3 不允许读取 source/external final")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "setup"})
    started = time.monotonic()
    try:
        git_commit = _git_commit()
        adapter_root = Path(config["adapter_root"])
        adapter_manifest = json.loads(Path(config["adapter_manifest"]).read_text(encoding="utf-8"))
        train_rows, holdout_rows = _adapter_rows(adapter_root)
        frozen_cohort = {
            (str(row["scene_name"]), str(row["track_id"]))
            for row in _load_rows(Path(config["cohort_rows"]))
        }
        adapter_holdout = {
            (str(row["scene_name"]), str(row["track_id"])) for row in holdout_rows
        }
        if frozen_cohort != adapter_holdout:
            raise RuntimeError("G3 adapter holdout 与 G0/G1 cohort identity 不一致")
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        model, load_report = _build_model(config)
        model = model.cuda()
        train_inputs: torch.Tensor | None = None
        train_targets: torch.Tensor | None = None
        if int(config["train_epochs"]) > 0:
            train_inputs, train_targets = _load_arrays(
                adapter_root, train_rows, include_target=True
            )
            if train_targets is None:
                raise AssertionError("训练 target 不应为空")
        holdout_inputs, _ = _load_arrays(adapter_root, holdout_rows, include_target=False)
        resolved = {
            **config,
            "run_id": run_id,
            "git_commit": git_commit,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema_version": "worldsim_v72.run_manifest.v1",
            "task_id": config["task_id"],
            "run_id": run_id,
            "git_commit": git_commit,
            "data_role": config["data_role"],
            "initialization": config["initialization"],
            "training": int(config["train_epochs"]) > 0,
            "train_actor_count": len(train_rows),
            "holdout_actor_count": len(holdout_rows),
            "adapter_manifest_fingerprint": adapter_manifest["fingerprint"],
            "pretrained_holdout_exposure": bool(config["pretrained_holdout_exposure"]),
            "source_test_read": False,
            "external_test_read": False,
            "route_decision_allowed": False,
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
        }
        _write_json(run_dir / "manifest.json", manifest)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                    "git_commit": git_commit,
                    "adapter_fingerprint": adapter_manifest["fingerprint"],
                    "checkpoint_sha256": config.get("checkpoint_sha256"),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        _write_json(
            run_dir / "fingerprint.json",
            {"algorithm": "sha256", "value": fingerprint, "scope": "config+git+adapter+checkpoint"},
        )
        _write_json(run_dir / "CHECKPOINT_LOAD.json", load_report)
        _write_json(
            run_dir / "status.json", {"status": "running", "phase": "training"}
        )
        history = _train(
            model,
            train_inputs if train_inputs is not None else torch.empty(0),
            train_targets if train_targets is not None else torch.empty(0),
            config,
            run_dir,
        )
        _write_json(
            run_dir / "status.json", {"status": "running", "phase": "evaluation"}
        )
        predictions = _predict(
            model,
            holdout_inputs.pin_memory(),
            batch_size=int(config["batch_size"]),
            amp=bool(config["amp"]),
        )
        actor_rows, metrics = _evaluate(
            predictions, holdout_rows, config, run_dir
        )
        summary = {
            "schema_version": "worldsim_v72.g3_adapointr.v1",
            **manifest,
            "status": "done",
            "verdict": "legacy_diagnostic_complete_no_route_decision",
            "baseline_id": "G3",
            "baseline_name": "AdaPoinTr",
            "checkpoint_load": load_report,
            "input_point_count": int(config["input_point_count"]),
            "native_output_point_count": int(config["output_point_count"]),
            "train_epochs": int(config["train_epochs"]),
            "batch_size": int(config["batch_size"]),
            "amp": bool(config["amp"]),
            "tf32": bool(config["tf32"]),
            "final_train": history[-1] if history else None,
            "minimum_train_loss": min((row["loss"] for row in history), default=None),
            "actor_row_count": len(actor_rows),
            "metrics": metrics,
            "target_free_surface_generation": True,
            "full_return_metrics_supported": False,
            "full_return_metrics_reason": "legacy v1 cohort has positive held-out returns only",
            "resources": {
                "device": "cuda:0",
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"},
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
