"""Monotone quantile pooling for trajectory-visited state reliability."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch

from motion_proj.worldsim_v64.bounded_collision_critic import _action_rows
from motion_proj.worldsim_v64.native_voxel_uq import _native_unit_dir, _unit_dirs
from motion_proj.worldsim_v67.trajectory_reliability import _ranking_pairs
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _visited_masks
from scripts.run_worldsim_v65_p2v_visited_state_transfer import _frozen_q0_scores, _load_unit


class MonotoneQuantilePool(torch.nn.Module):
    """Convex quantile pool mixed conservatively with the frozen qmean reference."""

    def __init__(self, quantile_count: int, maximum_distribution_mix: float) -> None:
        super().__init__()
        self.quantile_logits = torch.nn.Parameter(torch.zeros(int(quantile_count)))
        self.mix_logit = torch.nn.Parameter(torch.tensor(-2.0))
        self.maximum_distribution_mix = float(maximum_distribution_mix)

    def forward(self, qmean: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.quantile_logits, dim=0)
        pooled = quantiles @ weights
        mix = self.maximum_distribution_mix * torch.sigmoid(self.mix_logit)
        return ((1.0 - mix) * qmean + mix * pooled).clamp(0.0, 1.0)


def materialize_quantiles(
    config: Mapping[str, Any], runs_root: Path, cache_path: Path
) -> dict[str, int]:
    inputs = config["inputs"]
    evidence_root = runs_root / str(inputs["evidence_run"])
    native_root = runs_root / str(inputs["native_run"])
    processed_root = Path(inputs["processed_root"])
    q0 = joblib.load(
        runs_root / str(inputs["risk_run"]) / str(inputs["risk_model_relative_path"])
    )
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    future_frames = int(config["trajectory"]["future_frame_count"])
    radius = float(config["trajectory"]["visited_corridor_radius_m"])
    minimum_visited = int(config["trajectory"]["minimum_visited_points_per_action"])
    point_limit = int(config["sampling"]["evaluation_points_per_unit"])
    quantile_levels = np.asarray(config["quantile_levels"], dtype=np.float64)
    rng = np.random.default_rng(int(config["sampling"]["seed"]))
    descriptors = []
    for scene_index, scene in enumerate(config["scenes"]):
        name = str(scene["name"])
        for unit_index, evidence_unit in enumerate(_unit_dirs(evidence_root, name)):
            descriptors.append(
                (
                    scene_index,
                    unit_index,
                    evidence_unit,
                    _native_unit_dir(
                        native_root,
                        name,
                        evidence_unit.name,
                        {name: str(inputs["native_partition"])},
                    ),
                    processed_root / f"{int(scene['processed_index']):03d}",
                )
            )
    fields: dict[str, list[Any]] = {
        name: []
        for name in (
            "qmean",
            "qstd",
            "target_cost",
            "unsafe",
            "visited_count",
            "hidden_free_count",
            "scene_index",
            "unit_index",
            "case_index",
            "action_index",
            "progress_ratio",
            "lateral_offset_m",
        )
    }
    quantile_rows = []
    source_action_count = 0
    excluded_action_count = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def submit(row: tuple[Any, ...]):
        return executor.submit(
            _load_unit,
            row[2],
            row[3],
            row[4],
            origin=origin,
            voxel_size=voxel_size,
            future_frame_count=future_frames,
        )

    future = submit(descriptors[0])
    try:
        for case_index, descriptor in enumerate(descriptors):
            features, centers, labels, logged_route = future.result()
            if case_index + 1 < len(descriptors):
                future = submit(descriptors[case_index + 1])
            if len(features) > point_limit:
                chosen = rng.choice(len(features), size=point_limit, replace=False)
                features, centers, labels = features[chosen], centers[chosen], labels[chosen]
            scores = _frozen_q0_scores(q0, features)
            actions = [
                row
                for row in _action_rows(logged_route, config["action_lattice"])
                if row["source_role"] != "stop"
            ]
            paths = np.stack([np.asarray(row["path"], dtype=np.float32) for row in actions])
            visited_masks = _visited_masks(centers, paths, radius)
            source_action_count += len(actions)
            eligible_here = 0
            for action_index, (action, visited) in enumerate(zip(actions, visited_masks)):
                visited_count = int(np.count_nonzero(visited))
                if visited_count < minimum_visited:
                    excluded_action_count += 1
                    continue
                eligible_here += 1
                visited_scores = np.asarray(scores[visited], dtype=np.float32)
                hidden_free_count = int(np.count_nonzero(labels[visited]))
                fields["qmean"].append(float(visited_scores.mean()))
                fields["qstd"].append(float(visited_scores.std()))
                fields["target_cost"].append(hidden_free_count / visited_count)
                fields["unsafe"].append(hidden_free_count > 0)
                fields["visited_count"].append(visited_count)
                fields["hidden_free_count"].append(hidden_free_count)
                fields["scene_index"].append(descriptor[0])
                fields["unit_index"].append(descriptor[1])
                fields["case_index"].append(case_index)
                fields["action_index"].append(action_index)
                fields["progress_ratio"].append(float(action["progress_ratio"]))
                fields["lateral_offset_m"].append(float(action["lateral_offset_m"]))
                quantile_rows.append(np.quantile(visited_scores, quantile_levels))
            print(
                f"TQ data {case_index + 1}/{len(descriptors)} scene={descriptor[0]} "
                f"unit={descriptor[1]} eligible={eligible_here}/{len(actions)}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
    payload = {
        "qmean": np.asarray(fields["qmean"], dtype=np.float32),
        "qstd": np.asarray(fields["qstd"], dtype=np.float32),
        "quantiles": np.asarray(quantile_rows, dtype=np.float32),
        "quantile_levels": quantile_levels.astype(np.float32),
        "target_cost": np.asarray(fields["target_cost"], dtype=np.float32),
        "unsafe": np.asarray(fields["unsafe"], dtype=bool),
        "visited_count": np.asarray(fields["visited_count"], dtype=np.int32),
        "hidden_free_count": np.asarray(fields["hidden_free_count"], dtype=np.int32),
        "scene_index": np.asarray(fields["scene_index"], dtype=np.int64),
        "unit_index": np.asarray(fields["unit_index"], dtype=np.int64),
        "case_index": np.asarray(fields["case_index"], dtype=np.int64),
        "action_index": np.asarray(fields["action_index"], dtype=np.int64),
        "progress_ratio": np.asarray(fields["progress_ratio"], dtype=np.float32),
        "lateral_offset_m": np.asarray(fields["lateral_offset_m"], dtype=np.float32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)
    return {
        "source_case_count": len(descriptors),
        "source_action_count": source_action_count,
        "excluded_action_count": excluded_action_count,
        "eligible_action_count": len(payload["qmean"]),
    }


def train_quantile_pool(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, Any], seed: int
) -> tuple[MonotoneQuantilePool, dict[str, Any]]:
    target_np = np.asarray(arrays["target_cost"], dtype=np.float32)
    cases_np = np.asarray(arrays["case_index"], dtype=np.int64)
    domains_np = np.asarray(arrays["domain_index"], dtype=np.int64)
    target = torch.from_numpy(target_np).cuda()
    unsafe = torch.from_numpy(np.asarray(arrays["unsafe"], dtype=np.float32)).cuda()
    qmean = torch.from_numpy(np.asarray(arrays["qmean"], dtype=np.float32)).cuda()
    quantiles = torch.from_numpy(np.asarray(arrays["quantiles"], dtype=np.float32)).cuda()
    domains = torch.from_numpy(domains_np).cuda()
    left_np, right_np, signs_np = _ranking_pairs(
        target_np, cases_np, float(config["pairwise_minimum_target_gap"])
    )
    left, right, signs = (
        torch.from_numpy(left_np).cuda(),
        torch.from_numpy(right_np).cuda(),
        torch.from_numpy(signs_np).cuda(),
    )
    torch.manual_seed(int(seed))
    model = MonotoneQuantilePool(
        quantiles.shape[1], float(config["maximum_distribution_mix"])
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    final = {}
    for _ in range(int(config["epochs"])):
        prediction = model(qmean, quantiles)
        regression_elements = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=float(config["huber_beta"]), reduction="none"
        )
        domain_loss = torch.stack(
            [regression_elements[domains == domain].mean() for domain in torch.unique(domains)]
        )
        regression = domain_loss.mean()
        variance = domain_loss.var(unbiased=False)
        unsafe_loss = torch.nn.functional.binary_cross_entropy(
            prediction.clamp(1e-5, 1 - 1e-5), unsafe
        )
        pair_delta = (prediction[left] - prediction[right]) * signs
        ranking = torch.nn.functional.softplus(
            -pair_delta / float(config["ranking_temperature"])
        ).mean()
        mix = model.maximum_distribution_mix * torch.sigmoid(model.mix_logit)
        loss = (
            float(config["regression_weight"]) * regression
            + float(config["unsafe_weight"]) * unsafe_loss
            + float(config["ranking_weight"]) * ranking
            + float(config["domain_loss_variance_weight"]) * variance
            + float(config["mix_regularization_weight"]) * mix.square()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "total_loss": float(loss.detach().cpu()),
            "regression_loss": float(regression.detach().cpu()),
            "unsafe_loss": float(unsafe_loss.detach().cpu()),
            "ranking_loss": float(ranking.detach().cpu()),
            "domain_loss_variance": float(variance.detach().cpu()),
        }
    with torch.inference_mode():
        weights = torch.softmax(model.quantile_logits, dim=0).cpu().numpy()
        mix = model.maximum_distribution_mix * torch.sigmoid(model.mix_logit)
    final.update(
        train_row_count=int(len(target_np)),
        pair_count=int(len(left_np)),
        distribution_mix=float(mix.cpu()),
        quantile_weights=[float(value) for value in weights],
    )
    return model.eval(), final


def score_quantile_pool(
    model: MonotoneQuantilePool, arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    with torch.inference_mode():
        return model(
            torch.from_numpy(np.asarray(arrays["qmean"], dtype=np.float32)).cuda(),
            torch.from_numpy(np.asarray(arrays["quantiles"], dtype=np.float32)).cuda(),
        ).float().cpu().numpy()
