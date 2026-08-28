"""Cross-case calibration for adaptive fixed-total action budgets."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


FEATURE_NAMES = (
    "compiled_mean",
    "compiled_std",
    "compiled_selected_mean",
    "compiled_unselected_gap",
    "selected_point_score_std",
    "selected_quantile_span",
    "log_selected_visited_count",
)


class BoundedCaseOffset(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimension: int, maximum_offset: float) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(feature_count), int(hidden_dimension)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dimension), 1),
        )
        self.maximum_offset = float(maximum_offset)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(features).squeeze(-1)) * self.maximum_offset


def case_offset_dataset(
    arrays: Mapping[str, np.ndarray], compiled_scores: np.ndarray, selected_fraction: float
) -> dict[str, np.ndarray]:
    cases = np.asarray(arrays["case_index"], dtype=np.int64)
    target = np.asarray(arrays["target_cost"], dtype=np.float32)
    domains = np.asarray(arrays.get("domain_index", np.zeros(len(cases))), dtype=np.int64)
    qstd = np.asarray(arrays["qstd"], dtype=np.float32)
    quantiles = np.asarray(arrays["quantiles"], dtype=np.float32)
    visited = np.asarray(arrays["visited_count"], dtype=np.float32)
    features, offsets, case_ids, scene_ids, domain_ids, action_members = [], [], [], [], [], []
    scenes = np.asarray(arrays["scene_index"], dtype=np.int64)
    for case in np.unique(cases):
        members = np.flatnonzero(cases == case)
        if len(members) < 2:
            continue
        count = max(1, int(np.floor(float(selected_fraction) * len(members))))
        order = members[np.argsort(compiled_scores[members], kind="stable")]
        selected, unselected = order[:count], order[count:]
        selected_mean = float(compiled_scores[selected].mean())
        features.append(
            np.asarray(
                [
                    float(compiled_scores[members].mean()),
                    float(compiled_scores[members].std()),
                    selected_mean,
                    float(compiled_scores[unselected].mean()) - selected_mean,
                    float(qstd[selected].mean()),
                    float((quantiles[selected, -2] - quantiles[selected, 1]).mean()),
                    float(np.log1p(visited[selected].mean())),
                ],
                dtype=np.float32,
            )
        )
        offsets.append(float(target[selected].mean()) - selected_mean)
        case_ids.append(int(case))
        scene_ids.append(int(scenes[members[0]]))
        domain_ids.append(int(domains[members[0]]))
        action_members.append(members)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "target_offset": np.asarray(offsets, dtype=np.float32),
        "case_index": np.asarray(case_ids, dtype=np.int64),
        "scene_index": np.asarray(scene_ids, dtype=np.int64),
        "domain_index": np.asarray(domain_ids, dtype=np.int64),
        "action_indices": np.asarray(action_members, dtype=object),
    }


def train_case_offset(
    dataset: Mapping[str, np.ndarray], config: Mapping[str, Any], seed: int
) -> tuple[BoundedCaseOffset, np.ndarray, np.ndarray, dict[str, Any]]:
    features_np = np.asarray(dataset["features"], dtype=np.float32)
    target_np = np.asarray(dataset["target_offset"], dtype=np.float32)
    domains_np = np.asarray(dataset["domain_index"], dtype=np.int64)
    mean = features_np.mean(axis=0)
    scale = features_np.std(axis=0).clip(min=1e-4)
    features = torch.from_numpy(((features_np - mean) / scale).astype(np.float32)).cuda()
    target = torch.from_numpy(target_np).cuda()
    domains = torch.from_numpy(domains_np).cuda()
    torch.manual_seed(int(seed))
    model = BoundedCaseOffset(
        features.shape[1], int(config["hidden_dimension"]), float(config["maximum_case_offset"])
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    final = {}
    for _ in range(int(config["epochs"])):
        prediction = model(features)
        elements = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=float(config["huber_beta"]), reduction="none"
        )
        domain_losses = torch.stack(
            [elements[domains == domain].mean() for domain in torch.unique(domains)]
        )
        regression = domain_losses.mean()
        variance = domain_losses.var(unbiased=False)
        offset_penalty = prediction.square().mean()
        loss = regression + float(config["domain_loss_variance_weight"]) * variance + float(config["offset_regularization_weight"]) * offset_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {"total_loss": float(loss.detach().cpu()), "regression_loss": float(regression.detach().cpu()),
                 "domain_loss_variance": float(variance.detach().cpu()), "offset_rms": float(torch.sqrt(offset_penalty).detach().cpu())}
    final.update(train_case_count=int(len(target_np)), development_domain_count=int(len(np.unique(domains_np))))
    return model.eval(), mean.astype(np.float32), scale.astype(np.float32), final


def score_case_offset(model: BoundedCaseOffset, dataset: Mapping[str, np.ndarray], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    features = (np.asarray(dataset["features"], dtype=np.float32) - mean) / scale
    with torch.inference_mode():
        return model(torch.from_numpy(features.astype(np.float32)).cuda()).cpu().numpy()


def adaptive_fixed_total_selection(
    arrays: Mapping[str, np.ndarray], compiled_scores: np.ndarray, case_offsets: np.ndarray,
    selected_fraction: float, maximum_actions_per_case: int,
) -> dict[str, Any]:
    cases = np.asarray(arrays["case_index"], dtype=np.int64)
    scenes = np.asarray(arrays["scene_index"], dtype=np.int64)
    target = np.asarray(arrays["target_cost"], dtype=np.float32)
    unique_cases = np.asarray(
        [case for case in np.unique(cases) if np.count_nonzero(cases == case) >= 2],
        dtype=np.int64,
    )
    orders = []
    fixed_total = 0
    selected = []
    candidates = []
    for row, case in enumerate(unique_cases):
        members = np.flatnonzero(cases == case)
        order = members[np.argsort(compiled_scores[members], kind="stable")]
        orders.append(order)
        fixed_total += max(1, int(np.floor(float(selected_fraction) * len(members))))
        selected.append(int(order[0]))
        for rank, action in enumerate(order[1 : int(maximum_actions_per_case)], start=2):
            candidates.append((float(compiled_scores[action] + case_offsets[row]), row, rank, int(action)))
    remaining = fixed_total - len(selected)
    chosen_candidates = sorted(candidates, key=lambda item: item[0])[:remaining]
    selected.extend(item[3] for item in chosen_candidates)
    selected_array = np.asarray(selected, dtype=np.int64)
    evaluable_indices = np.concatenate(orders).astype(np.int64)
    all_cost = float(target[evaluable_indices].mean())
    selected_cost = float(target[selected_array].mean())
    counts = np.asarray([1] * len(unique_cases), dtype=np.int64)
    for _, row, _, _ in chosen_candidates:
        counts[row] += 1
    scene_rows = []
    for scene in np.unique(scenes[evaluable_indices]):
        scene_all = evaluable_indices[scenes[evaluable_indices] == scene]
        scene_selected = selected_array[scenes[selected_array] == scene]
        if not len(scene_selected):
            continue
        scene_all_cost = float(target[scene_all].mean())
        scene_selected_cost = float(target[scene_selected].mean())
        scene_rows.append({"scene_index": int(scene), "all_mean_cost": scene_all_cost,
                           "selected_mean_cost": scene_selected_cost, "delta": scene_selected_cost - scene_all_cost})
    return {
        "evaluable_case_count": int(len(unique_cases)), "fixed_total_action_budget": int(fixed_total),
        "selected_action_count": int(len(selected_array)), "minimum_actions_per_case": int(counts.min()),
        "maximum_actions_per_case": int(counts.max()), "mean_actions_per_case": float(counts.mean()),
        "all_mean_cost": all_cost, "selected_mean_cost": selected_cost,
        "relative_cost_reduction": float((all_cost - selected_cost) / all_cost if all_cost else 0.0),
        "scene_nonincreasing_count": int(sum(row["delta"] <= 0 for row in scene_rows)), "scene_rows": scene_rows,
    }
