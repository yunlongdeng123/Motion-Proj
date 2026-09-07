"""Train appearance-only candidate colors from context windows and evaluate a heldout window."""

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

from motion_proj.worldsim_v72.eas_vggt.actor_visual_cache import load_actor_visual_cache
from motion_proj.worldsim_v72.eas_vggt.appearance import AnchoredAppearanceAdapter
from motion_proj.worldsim_v72.eas_vggt.appearance_cache import load_actor_appearance_cache


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_dataset(e2_run: Path, e3_run: Path, backbone: str, device: torch.device):
    visual = {
        (path.parent.name, path.stem): load_actor_visual_cache(path)
        for path in sorted((Path(e2_run) / "actor_cache" / backbone).glob("*/*.npz"))
    }
    grouped = defaultdict(list)
    for path in sorted((Path(e3_run) / "appearance_cache" / backbone).glob("*/*.npz")):
        cache = load_actor_appearance_cache(path)
        grouped[cache.track_id].append(cache)
    visual_parts = []
    target_parts = []
    target_mask_parts = []
    context_rgb_parts = []
    context_mask_parts = []
    rows = []
    for track_id, caches in sorted(grouped.items()):
        context = [cache for cache in caches if cache.provenance["payload_role"] == "context"]
        heldout = [cache for cache in caches if cache.provenance["payload_role"] == "heldout_rgb"]
        if not context or len(heldout) != 1:
            continue
        target = heldout[0]
        context_visual = [visual[(cache.window_id, track_id)] for cache in context if (cache.window_id, track_id) in visual]
        if not context_visual:
            continue
        confidence = np.sum([cache.confidence_sum for cache in context_visual], axis=0)
        feature_sum = np.sum(
            [cache.visual_features * cache.confidence_sum[:, None] for cache in context_visual], axis=0
        )
        features = feature_sum / np.maximum(confidence, 1.0e-8)[:, None]
        context_observed = confidence > 0.0
        features[~context_observed] = 0.0
        rgb_confidence = np.sum([cache.confidence_sum for cache in context], axis=0)
        rgb_sum = np.sum([cache.rgb * cache.confidence_sum[:, None] for cache in context], axis=0)
        context_rgb = rgb_sum / np.maximum(rgb_confidence, 1.0e-8)[:, None]
        context_rgb[rgb_confidence <= 0.0] = 0.0
        target_mask = target.observation_count > 0
        visual_parts.append(features.astype(np.float32))
        target_parts.append(target.rgb)
        target_mask_parts.append(target_mask)
        context_rgb_parts.append(context_rgb.astype(np.float32))
        context_mask_parts.append(context_observed & (rgb_confidence > 0.0))
        rows.append(
            {
                "track_id": track_id,
                "scene_id": target.scene_id,
                "candidate_count": len(target.rgb),
                "context_visible_count": int(context_observed.sum()),
                "heldout_visible_count": int(target_mask.sum()),
                "joint_visible_count": int(np.sum(context_observed & target_mask)),
            }
        )
    arrays = {
        "visual_features": np.concatenate(visual_parts),
        "target_rgb": np.concatenate(target_parts),
        "target_mask": np.concatenate(target_mask_parts),
        "context_rgb": np.concatenate(context_rgb_parts),
        "context_mask": np.concatenate(context_mask_parts),
    }
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}, rows


def _metrics(rgb: torch.Tensor, opacity: torch.Tensor, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    mask = data["target_mask"].bool() & data["context_mask"].bool()
    if not torch.any(mask):
        raise RuntimeError("no jointly visible heldout candidates")
    error = rgb[mask] - data["target_rgb"].float()[mask]
    mse = torch.mean(error.square())
    visibility_target = data["target_mask"].float()
    return {
        "candidate_count": int(len(rgb)),
        "joint_visible_count": int(mask.sum().item()),
        "rgb_mae": float(error.abs().mean().cpu()),
        "rgb_psnr": float((-10.0 * torch.log10(mse.clamp_min(1.0e-10))).cpu()),
        "visibility_brier": float(torch.mean((opacity - visibility_target) ** 2).cpu()),
    }


def _train(fit: dict[str, torch.Tensor], development: dict[str, torch.Tensor], config: dict[str, Any], checkpoint: Path):
    model = AnchoredAppearanceAdapter(
        int(fit["visual_features"].shape[-1]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        maximum_rgb_residual=float(config["model"]["maximum_rgb_residual"]),
    ).to(fit["visual_features"].device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    history = []
    observed = fit["context_mask"].bool()
    target_visible = fit["target_mask"].bool()
    color_mask = observed & target_visible
    for step in range(int(config["optimization"]["steps"])):
        output = model(
            fit["visual_features"].float(),
            observed,
            fit["context_rgb"].float(),
            fit["context_mask"].bool(),
        )
        color_loss = torch.nn.functional.smooth_l1_loss(
            output.rgb[color_mask], fit["target_rgb"].float()[color_mask]
        )
        opacity_loss = torch.nn.functional.binary_cross_entropy(
            output.opacity[observed], target_visible.float()[observed]
        )
        residual_penalty = torch.mean(
            torch.abs(output.rgb[color_mask] - fit["context_rgb"].float()[color_mask])
        )
        loss = (
            color_loss
            + float(config["optimization"]["opacity_loss_weight"]) * opacity_loss
            + float(config["optimization"]["residual_l1_weight"]) * residual_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, int(config["optimization"]["steps"]) // 2, int(config["optimization"]["steps"]) - 1}:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.inference_mode():
        fit_output = model(
            fit["visual_features"].float(),
            fit["context_mask"].bool(),
            fit["context_rgb"].float(),
            fit["context_mask"].bool(),
        )
        dev_output = model(
            development["visual_features"].float(),
            development["context_mask"].bool(),
            development["context_rgb"].float(),
            development["context_mask"].bool(),
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    result = {
        "variant": "canonical_rgb_plus_bounded_visual_residual",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "fit": _metrics(fit_output.rgb, fit_output.opacity, fit),
        "development": _metrics(dev_output.rgb, dev_output.opacity, development),
        "context_rgb_baseline_fit": _metrics(fit["context_rgb"].float(), fit["context_mask"].float(), fit),
        "context_rgb_baseline_development": _metrics(
            development["context_rgb"].float(), development["context_mask"].float(), development
        ),
        "checkpoint": str(checkpoint),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["task_id"] != "WS-V72-E3-DECOUPLED-APPEARANCE-01":
        raise ValueError("E3 task id mismatch")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device(config["device"])
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": "worldsim_v72.e3_appearance_training.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "seed": int(config["seed"]),
        "rgb_gradient_targets": ["appearance_adapter"],
        "physical_gradient_targets": [],
        "source_test_read": False,
        "external_test_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "dataset"})
    try:
        results = {}
        rows = {}
        for backbone in config["inputs"]["backbones"]:
            fit, fit_rows = _load_dataset(
                Path(config["inputs"]["fit_e2_run"]), Path(config["inputs"]["fit_e3_run"]), backbone, device
            )
            development, dev_rows = _load_dataset(
                Path(config["inputs"]["development_e2_run"]),
                Path(config["inputs"]["development_e3_run"]),
                backbone,
                device,
            )
            rows[backbone] = {"fit": fit_rows, "development": dev_rows}
            _write_json(run_dir / "status.json", {"status": "running", "phase": f"train_{backbone}"})
            results[backbone] = _train(fit, development, config, run_dir / "checkpoints" / f"{backbone}.pt")
        _write_json(run_dir / "dataset_rows.json", rows)
        summary = {
            **manifest,
            "status": "done",
            "failure_ledger_delta": "none",
            "results": results,
            "physical_parameter_update_count": 0,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            },
            "interpretation": "Candidate-level temporal appearance correspondence; not a full-frame rendering claim.",
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
