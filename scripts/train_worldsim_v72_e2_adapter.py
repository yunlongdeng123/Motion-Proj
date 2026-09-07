"""Train E2 continuous evidence and actor-local surface adapters on frozen visual splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries
from motion_proj.worldsim_v72.eas_vggt.actor_visual_cache import load_actor_visual_cache
from motion_proj.worldsim_v72.eas_vggt.models import (
    EvidenceConditionedSurfaceAdapter,
    MatchedScalarSurfaceAdapter,
    trainable_parameter_count,
)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aggregate_visual(run_root: Path, backbone: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for path in sorted((Path(run_root) / "actor_cache" / backbone).glob("*/*.npz")):
        cache = load_actor_visual_cache(path)
        grouped[cache.track_id].append(cache)
    actors = []
    for track_id, caches in sorted(grouped.items()):
        first = caches[0]
        for cache in caches[1:]:
            if cache.candidates_actor_m.shape != first.candidates_actor_m.shape or not np.allclose(
                cache.candidates_actor_m, first.candidates_actor_m
            ):
                raise ValueError(f"candidate ordering changed for {track_id}")
        confidence = np.sum([cache.confidence_sum for cache in caches], axis=0)
        visual_sum = np.sum(
            [cache.visual_features * cache.confidence_sum[:, None] for cache in caches], axis=0
        )
        geometry_sum = np.sum(
            [cache.geometry_features * cache.confidence_sum[:, None] for cache in caches], axis=0
        )
        denominator = np.maximum(confidence, 1.0e-8)[:, None]
        visual = visual_sum / denominator
        geometry = geometry_sum / denominator
        observed = confidence > 0.0
        visual[~observed] = 0.0
        geometry[~observed] = 0.0
        actors.append(
            {
                "track_id": track_id,
                "scene_id": first.scene_id,
                "candidates": first.candidates_actor_m,
                "base_features": first.base_features,
                "input_evidence_fou": first.input_evidence_fou,
                "opportunity_count": first.opportunity_count,
                "visual_features": visual.astype(np.float32),
                "geometric_features": geometry.astype(np.float32),
                "observed": observed,
                "window_count": len(caches),
            }
        )
    return actors


def _prepare_dataset(
    actors: list[dict[str, Any]],
    corpus_root: Path,
    evidence_config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    rows = []
    for actor in actors:
        source = Path(corpus_root) / actor["scene_id"] / f"{actor['track_id']}.npz"
        with np.load(source, allow_pickle=False) as payload:
            target = payload["target"]
            target_origins = payload["target_sensor_origins"]
            oracle = payload["oracle_displacement"]
        if len(target) != len(target_origins):
            raise ValueError(f"target ray mismatch in {source}")
        supervision = build_evidential_queries(
            actor["candidates"],
            target_origins,
            target,
            beam_radius_m=float(evidence_config["beam_radius_m"]),
            endpoint_radius_m=float(evidence_config["endpoint_radius_m"]),
            query_chunk_size=int(evidence_config["query_chunk_size"]),
            device=device,
        )
        count = len(actor["candidates"])
        has_oracle = len(oracle) == count
        parts["base_features"].append(actor["base_features"])
        parts["canonical_xyz"].append(actor["candidates"])
        parts["evidence_fou"].append(actor["input_evidence_fou"])
        parts["opportunity_count"].append(actor["opportunity_count"])
        parts["visual_features"].append(actor["visual_features"])
        parts["geometric_features"].append(actor["geometric_features"])
        parts["visual_observed"].append(actor["observed"])
        parts["geometry_observed"].append(actor["observed"])
        parts["target_fou"].append(supervision.masses)
        parts["target_opportunity"].append(supervision.opportunity_count)
        parts["oracle_displacement"].append(oracle if has_oracle else np.zeros((count, 3), dtype=np.float32))
        parts["oracle_mask"].append(np.full(count, has_oracle, dtype=bool))
        rows.append(
            {
                "scene_id": actor["scene_id"],
                "track_id": actor["track_id"],
                "candidate_count": count,
                "observed_candidate_count": int(np.sum(actor["observed"])),
                "supervision_supported_count": int(np.sum(supervision.opportunity_count > 0)),
                "oracle_available": has_oracle,
                "window_count": actor["window_count"],
            }
        )
    arrays = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    tensors = {
        name: torch.from_numpy(value).to(device)
        for name, value in arrays.items()
    }
    return tensors, rows


def _model_inputs(data: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    visual = data["visual_features"]
    geometry = data["geometric_features"]
    visual_observed = data["visual_observed"]
    geometry_observed = data["geometry_observed"]
    if variant == "eas_no_appearance":
        visual = torch.zeros_like(visual)
        visual_observed = torch.zeros_like(visual_observed)
    if variant == "eas_no_geometry":
        geometry = torch.zeros_like(geometry)
        geometry_observed = torch.zeros_like(geometry_observed)
    return {
        "base_features": data["base_features"].float(),
        "canonical_xyz": data["canonical_xyz"].float(),
        "evidence_fou": data["evidence_fou"].float(),
        "opportunity_count": data["opportunity_count"],
        "visual_features": visual.float(),
        "visual_observed": visual_observed,
        "geometric_features": geometry.float(),
        "geometry_observed": geometry_observed,
    }


def _scalar_fou(response: torch.Tensor, input_fou: torch.Tensor) -> torch.Tensor:
    occupied = torch.sigmoid(response)
    non_occupied = input_fou[:, [0, 2]]
    denominator = non_occupied.sum(dim=-1, keepdim=True)
    ratio = torch.where(
        denominator > 1.0e-8,
        non_occupied / denominator.clamp_min(1.0e-8),
        torch.full_like(non_occupied, 0.5),
    )
    remaining = 1.0 - occupied
    return torch.stack([remaining * ratio[:, 0], occupied, remaining * ratio[:, 1]], dim=-1)


def _loss_and_prediction(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    variant: str,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = _model_inputs(data, variant)
    target = data["target_fou"].float()
    supported = data["target_opportunity"] > 0
    weights = torch.where(supported, torch.ones_like(target[:, 0]), torch.full_like(target[:, 0], float(config["unsupported_weight"])))
    if variant == "scalar_full":
        delta, response = model(**inputs)
        predicted = _scalar_fou(response, inputs["evidence_fou"])
        strength_loss = torch.zeros((), device=delta.device)
    else:
        output = model(**inputs)
        delta = output.surface_delta_actor_m
        predicted = output.evidence_fou
        strength_target = torch.log1p(1.0 + torch.log1p(data["target_opportunity"].float()))
        strength_loss = torch.nn.functional.smooth_l1_loss(
            torch.log1p(output.evidence_strength), strength_target
        )
    evidence_nll = -(target * torch.log(predicted.clamp_min(1.0e-8))).sum(dim=-1)
    evidence_loss = torch.sum(weights * evidence_nll) / weights.sum().clamp_min(1.0)
    oracle_mask = data["oracle_mask"].bool()
    if torch.any(oracle_mask):
        geometry_loss = torch.nn.functional.smooth_l1_loss(
            delta[oracle_mask], data["oracle_displacement"].float()[oracle_mask]
        )
    else:
        geometry_loss = delta.sum() * 0.0
    loss = (
        float(config["evidence_loss_weight"]) * evidence_loss
        + float(config["strength_loss_weight"]) * strength_loss
        + float(config["geometry_loss_weight"]) * geometry_loss
    )
    return loss, predicted, delta


def _metrics(predicted: torch.Tensor, delta: torch.Tensor, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    target = data["target_fou"].float()
    supported = data["target_opportunity"] > 0
    nll = -(target * torch.log(predicted.clamp_min(1.0e-8))).sum(dim=-1)
    brier = torch.sum((predicted - target) ** 2, dim=-1)
    oracle = data["oracle_mask"].bool()
    result = {
        "candidate_count": int(len(target)),
        "supported_candidate_count": int(supported.sum().item()),
        "fou_nll_all": float(nll.mean().cpu()),
        "fou_brier_all": float(brier.mean().cpu()),
        "occupied_mae_all": float((predicted[:, 1] - target[:, 1]).abs().mean().cpu()),
        "fou_nll_supported": float(nll[supported].mean().cpu()) if torch.any(supported) else None,
        "fou_brier_supported": float(brier[supported].mean().cpu()) if torch.any(supported) else None,
        "oracle_candidate_count": int(oracle.sum().item()),
    }
    if torch.any(oracle):
        residual = delta[oracle] - data["oracle_displacement"].float()[oracle]
        result["surface_displacement_rmse_m"] = float(torch.sqrt(torch.mean(residual.square())).cpu())
        result["surface_displacement_mae_m"] = float(residual.abs().mean().cpu())
    else:
        result["surface_displacement_rmse_m"] = None
        result["surface_displacement_mae_m"] = None
    return result


def _train_variant(
    fit: dict[str, torch.Tensor],
    development: dict[str, torch.Tensor],
    variant: str,
    model_config: dict[str, Any],
    optimization: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    kwargs = {
        "base_feature_dim": int(fit["base_features"].shape[-1]),
        "visual_feature_dim": int(fit["visual_features"].shape[-1]),
        "geometric_feature_dim": int(fit["geometric_features"].shape[-1]),
        **model_config,
    }
    if variant == "scalar_full":
        model = MatchedScalarSurfaceAdapter(**kwargs).to(fit["base_features"].device)
    else:
        model = EvidenceConditionedSurfaceAdapter(**kwargs).to(fit["base_features"].device)
        for parameter in model.return_head.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    history = []
    for step in range(int(optimization["steps"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _loss_and_prediction(model, fit, variant, optimization)
        loss.backward()
        optimizer.step()
        if step in {0, int(optimization["steps"]) // 2, int(optimization["steps"]) - 1}:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.inference_mode():
        _, fit_prediction, fit_delta = _loss_and_prediction(model, fit, variant, optimization)
        _, dev_prediction, dev_delta = _loss_and_prediction(model, development, variant, optimization)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_kwargs": kwargs, "variant": variant}, checkpoint_path)
    result = {
        "variant": variant,
        "trainable_parameter_count": trainable_parameter_count(model),
        "history": history,
        "fit": _metrics(fit_prediction, fit_delta, fit),
        "development": _metrics(dev_prediction, dev_delta, development),
        "checkpoint": str(checkpoint_path),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["task_id"] != "WS-V72-E2-LEARNED-VISUAL-EVIDENCE-01":
        raise ValueError("E2 task id mismatch")
    if config["data"]["selection_uses_quality"] or config["data"]["source_test_read"] or config["data"]["external_test_read"]:
        raise ValueError("E2 training role contract violated")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    manifest = {
        "schema_version": "worldsim_v72.e2_adapter_training.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "seed": int(config["seed"]),
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "selection_uses_quality": False,
        "supervision_access": True,
        "supervision_fields": list(config["data"]["supervision_fields"]),
        "raw_beam_outcome_supervision": False,
        "source_test_read": False,
        "external_test_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "dataset"})
    try:
        results = {}
        dataset_rows = {}
        for backbone in config["data"]["backbones"]:
            fit_actors = _aggregate_visual(Path(config["data"]["fit_run"]), backbone)
            dev_actors = _aggregate_visual(Path(config["data"]["development_run"]), backbone)
            fit, fit_rows = _prepare_dataset(
                fit_actors, Path(config["data"]["actor_corpus_root"]), config["supervision_evidence"], device
            )
            development, dev_rows = _prepare_dataset(
                dev_actors, Path(config["data"]["actor_corpus_root"]), config["supervision_evidence"], device
            )
            dataset_rows[backbone] = {"fit": fit_rows, "development": dev_rows}
            baseline_prediction = fit["evidence_fou"].float()
            zero_delta = torch.zeros_like(fit["canonical_xyz"].float())
            backbone_results = {
                "input_only_fit": _metrics(baseline_prediction, zero_delta, fit),
                "input_only_development": _metrics(
                    development["evidence_fou"].float(),
                    torch.zeros_like(development["canonical_xyz"].float()),
                    development,
                ),
            }
            for variant in config["variants"]:
                _write_json(run_dir / "status.json", {"status": "running", "phase": f"train_{backbone}_{variant}"})
                backbone_results[variant] = _train_variant(
                    fit,
                    development,
                    str(variant),
                    config["model"],
                    config["optimization"],
                    run_dir / "checkpoints" / f"{backbone}_{variant}.pt",
                )
            results[backbone] = backbone_results
            del fit, development
            torch.cuda.empty_cache()
        _write_json(run_dir / "dataset_rows.json", dataset_rows)
        summary = {
            **manifest,
            "status": "done",
            "failure_ledger_delta": "none",
            "results": results,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            },
            "interpretation": "Development result for surface/F-O-U learning; categorical beam outcome remains untested without native emission records.",
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "failure_ledger_delta": "none", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
