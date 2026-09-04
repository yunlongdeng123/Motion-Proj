"""Train the deployed return CDF on GT-defined not-early and hit events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m37_supervised_child_transmittance as m37_runner
import run_worldsim_v71_m41_conserved_surface_measure as m41_runner


def _losses(
    reference_anchor: torch.nn.Module,
    reference_child: torch.nn.Module,
    anchor_authority: torch.nn.Module,
    child_authority: torch.nn.Module,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    depths, target_depth, energy, anchor_logits, child_logits = m41_runner._ray_values(
        reference_anchor,
        reference_child,
        anchor_authority,
        child_authority,
        actor,
        config,
    )
    probabilities = torch.softmax(energy, dim=1)
    tolerance = float(config["interval_tolerance_m"])
    early_mask = depths < target_depth[:, None] - tolerance
    hit_mask = torch.abs(depths - target_depth[:, None]) <= tolerance
    early_probability = torch.sum(probabilities * early_mask, dim=1)
    hit_probability = torch.sum(probabilities * hit_mask, dim=1)
    not_early_nll = -torch.log(
        (1.0 - early_probability).clamp_min(1.0e-8)
    ).mean()
    hit_band_nll = -torch.log(hit_probability.clamp_min(1.0e-8)).mean()
    anchor_evidential = -torch.sum(
        actor["authority_target_masses_t"]
        * torch.log_softmax(anchor_logits, dim=1),
        dim=1,
    ).mean()
    child_evidential = -torch.sum(
        actor["authority_child_target_masses_t"]
        * torch.log_softmax(child_logits, dim=1),
        dim=1,
    ).mean()
    interval_nll = (
        float(config["not_early_weight"]) * not_early_nll
        + float(config["hit_band_weight"]) * hit_band_nll
    )
    loss = (
        interval_nll
        + float(config["anchor_evidential_mass_weight"]) * anchor_evidential
        + float(config["child_evidential_mass_weight"]) * child_evidential
    )
    return {
        "loss": loss,
        "interval_nll": interval_nll,
        "not_early_nll": not_early_nll,
        "hit_band_nll": hit_band_nll,
        "anchor_evidential_cross_entropy": anchor_evidential,
        "child_evidential_cross_entropy": child_evidential,
        "early_probability": early_probability.mean(),
        "hit_probability": hit_probability.mean(),
    }


def _train(
    reference_anchor: torch.nn.Module,
    reference_child: torch.nn.Module,
    anchor_authority: torch.nn.Module,
    child_authority: torch.nn.Module,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    names = (
        "loss",
        "interval_nll",
        "not_early_nll",
        "hit_band_nll",
        "anchor_evidential_cross_entropy",
        "child_evidential_cross_entropy",
        "early_probability",
        "hit_probability",
    )
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["epochs"])):
        totals = {name: 0.0 for name in names}
        permutation = torch.randperm(len(actors)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            items = [
                _losses(
                    reference_anchor,
                    reference_child,
                    anchor_authority,
                    child_authority,
                    actors[index],
                    config,
                )
                for index in indices
            ]
            means = {
                name: torch.stack([item[name] for item in items]).mean()
                for name in names
            }
            optimizer.zero_grad(set_to_none=True)
            means["loss"].backward()
            optimizer.step()
            for name in names:
                totals[name] += float(means[name].detach()) * len(indices)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: total / len(actors) for name, total in totals.items()},
        }
        history.append(row)
        print(json.dumps({"stage": "m42_interval_event_train", **row}), flush=True)
    return history


def _rename_candidate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for stratum, values in metrics.items():
        renamed[stratum] = {
            key.replace("m41", "m42"): value for key, value in values.items()
        }
    return renamed


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    m41_runner._train = _train
    summary = m41_runner.run(config_path, run_id)
    summary["metrics"] = _rename_candidate_metrics(summary["metrics"])
    summary["verdict"] = (
        "m42_development_passed"
        if all(summary["decisions"].values())
        else "m42_development_rejected"
    )
    summary["ray_supervision"] = {
        "not_early_event": "depth >= target_depth - interval_tolerance_m",
        "hit_event": "abs(depth - target_depth) <= interval_tolerance_m",
        "interval_tolerance_m": float(config["model"]["interval_tolerance_m"]),
        "point_bin_cross_entropy": False,
        "expected_depth_proxy": False,
    }
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    rows_path = run_dir / "HOLDOUT_CONSERVED_MEASURE_ROWS.jsonl"
    if rows_path.exists():
        rows_path.rename(run_dir / "HOLDOUT_INTERVAL_EVENT_ROWS.jsonl")
    m37_runner._write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
