"""Train three independent Actor Gaussian blocks with batched packed kernels."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import _actor_entries
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p133_batchensemble_actor_uncertainty import _trajectory_score


class PackedIndependentLinear(torch.nn.Module):
    def __init__(self, input_width: int, output_width: int, members: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(members, output_width, input_width))
        self.bias = torch.nn.Parameter(torch.zeros(members, output_width))
        for member in range(members):
            torch.nn.init.kaiming_uniform_(self.weight[member], a=5 ** 0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.bmm(value, self.weight.transpose(1, 2)) + self.bias[:, None, :]


class PackedIndependentActorGaussian(torch.nn.Module):
    def __init__(
        self, feature_count: int, hidden_dimensions: list[int], members: int,
    ) -> None:
        super().__init__()
        widths = [feature_count] + [int(x) for x in hidden_dimensions] + [4]
        self.layers = torch.nn.ModuleList([
            PackedIndependentLinear(widths[index], widths[index + 1], members)
            for index in range(len(widths) - 1)
        ])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = features
        for layer in self.layers[:-1]:
            value = functional.silu(layer(value))
        output = self.layers[-1](value)
        return output[..., :2], functional.softplus(output[..., 2:]) + 0.02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    torch.manual_seed(int(config["seed"]))
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    raw_features, raw_target, _ = _actor_entries(source)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = raw_target.mean(0)
    target_scale = raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    members = int(model_config["members"])
    model = PackedIndependentActorGaussian(
        features.shape[1], model_config["hidden_dimensions"], members,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    member_batch = int(model_config["member_batch_size"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        index = torch.randint(len(features), (members, member_batch), device="cuda")
        mean, scale = model(features[index])
        residual = (targets[index] - mean) / scale
        loss = (0.5 * residual.square() + torch.log(scale)).sum(dim=2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P134 packed independent step={step + 1} nll={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "members": members,
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, model, feature_mean, feature_scale, target_mean, target_scale, members,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P134 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "packed_selected_mean_cost": float(actual_cost[selected].mean()),
            "deep_ensemble_selected_mean_cost": float(reference["selected_cost"]),
            "packed_cost_spearman": model_spearman,
            "deep_ensemble_cost_spearman": float(reference["spearman"]),
            "spearman_difference_from_deep_ensemble": float(
                model_spearman - float(reference["spearman"])
            ),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    differences = [x["spearman_difference_from_deep_ensemble"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression_from_deep_ensemble": all(
            x["packed_selected_mean_cost"] <= x["deep_ensemble_selected_mean_cost"]
            for x in results.values()
        ),
        "mean_spearman_retention": float(np.mean(differences))
        >= float(config["decision"]["minimum_mean_spearman_difference_from_deep_ensemble"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "members": members,
                     "final_packed_nll": final_loss},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_difference_from_deep_ensemble": float(np.mean(differences)),
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
