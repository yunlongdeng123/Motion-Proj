"""Case-level selective authority over a frozen trajectory qmean ranking."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


FEATURE_NAMES = (
    "qmean_all_minus_selected",
    "qmean_unselected_minus_selected",
    "qmean_standard_deviation",
    "qmean_median_minus_selected",
    "negative_selected_point_score_std",
    "negative_selected_quantile_span",
    "log_selected_visited_count",
)


class MonotoneBenefitHead(torch.nn.Module):
    """Positive linear evidence pool; it cannot change the underlying action order."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.weight_logits = torch.nn.Parameter(torch.zeros(int(feature_count)))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ torch.nn.functional.softplus(self.weight_logits) + self.bias


def case_dataset(
    arrays: Mapping[str, np.ndarray], selected_fraction: float
) -> dict[str, np.ndarray]:
    qmean = np.asarray(arrays["qmean"], dtype=np.float32)
    target = np.asarray(arrays["target_cost"], dtype=np.float32)
    cases = np.asarray(arrays["case_index"], dtype=np.int64)
    scenes = np.asarray(arrays["scene_index"], dtype=np.int64)
    qstd = np.asarray(arrays["qstd"], dtype=np.float32)
    quantiles = np.asarray(arrays["quantiles"], dtype=np.float32)
    visited = np.asarray(arrays["visited_count"], dtype=np.float32)
    domains = np.asarray(
        arrays.get("domain_index", np.zeros(len(qmean), dtype=np.int64)), dtype=np.int64
    )
    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "features",
            "benefit",
            "relative_reduction",
            "case_index",
            "scene_index",
            "domain_index",
            "all_action_indices",
            "selected_action_indices",
        )
    }
    for case in np.unique(cases):
        members = np.flatnonzero(cases == case)
        if members.size < 2:
            continue
        selected_count = max(1, int(np.floor(float(selected_fraction) * members.size)))
        order = members[np.argsort(qmean[members], kind="stable")]
        selected = order[:selected_count]
        unselected = order[selected_count:]
        selected_qmean = float(qmean[selected].mean())
        all_target = float(target[members].mean())
        selected_target = float(target[selected].mean())
        quantile_span = quantiles[selected, -2] - quantiles[selected, 1]
        features = np.asarray(
            [
                float(qmean[members].mean()) - selected_qmean,
                float(qmean[unselected].mean()) - selected_qmean,
                float(qmean[members].std()),
                float(np.median(qmean[members])) - selected_qmean,
                -float(qstd[selected].mean()),
                -float(quantile_span.mean()),
                float(np.log1p(visited[selected].mean())),
            ],
            dtype=np.float32,
        )
        rows["features"].append(features)
        rows["benefit"].append(all_target - selected_target)
        rows["relative_reduction"].append(
            (all_target - selected_target) / all_target if all_target > 0 else 0.0
        )
        rows["case_index"].append(int(case))
        rows["scene_index"].append(int(scenes[members[0]]))
        rows["domain_index"].append(int(domains[members[0]]))
        rows["all_action_indices"].append(members)
        rows["selected_action_indices"].append(selected)
    return {
        "features": np.asarray(rows["features"], dtype=np.float32),
        "benefit": np.asarray(rows["benefit"], dtype=np.float32),
        "relative_reduction": np.asarray(rows["relative_reduction"], dtype=np.float32),
        "case_index": np.asarray(rows["case_index"], dtype=np.int64),
        "scene_index": np.asarray(rows["scene_index"], dtype=np.int64),
        "domain_index": np.asarray(rows["domain_index"], dtype=np.int64),
        "all_action_indices": np.asarray(rows["all_action_indices"], dtype=object),
        "selected_action_indices": np.asarray(rows["selected_action_indices"], dtype=object),
    }


def _case_ranking_pairs(
    benefit: np.ndarray, domains: np.ndarray, minimum_gap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right, signs = [], [], []
    for domain in np.unique(domains):
        members = np.flatnonzero(domains == domain)
        for offset, first in enumerate(members):
            rest = members[offset + 1 :]
            delta = benefit[first] - benefit[rest]
            valid = np.abs(delta) >= float(minimum_gap)
            left.extend([int(first)] * int(np.count_nonzero(valid)))
            right.extend(rest[valid].tolist())
            signs.extend(np.sign(delta[valid]).tolist())
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(signs, dtype=np.float32),
    )


def train_benefit_head(
    cases: Mapping[str, np.ndarray], config: Mapping[str, Any], seed: int
) -> tuple[MonotoneBenefitHead, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_np = np.asarray(cases["features"], dtype=np.float32)
    benefit_np = np.asarray(cases["benefit"], dtype=np.float32)
    domain_np = np.asarray(cases["domain_index"], dtype=np.int64)
    mean = feature_np.mean(axis=0)
    scale = feature_np.std(axis=0).clip(min=1e-4)
    features = torch.from_numpy(((feature_np - mean) / scale).astype(np.float32)).cuda()
    benefit = torch.from_numpy(benefit_np).cuda()
    domains = torch.from_numpy(domain_np).cuda()
    left_np, right_np, signs_np = _case_ranking_pairs(
        benefit_np, domain_np, float(config["pairwise_minimum_benefit_gap"])
    )
    left = torch.from_numpy(left_np).cuda()
    right = torch.from_numpy(right_np).cuda()
    signs = torch.from_numpy(signs_np).cuda()
    torch.manual_seed(int(seed))
    model = MonotoneBenefitHead(features.shape[1]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    final = {}
    for _ in range(int(config["epochs"])):
        prediction = model(features)
        elements = torch.nn.functional.smooth_l1_loss(
            prediction, benefit, beta=float(config["huber_beta"]), reduction="none"
        )
        domain_losses = torch.stack(
            [elements[domains == domain].mean() for domain in torch.unique(domains)]
        )
        regression = domain_losses.mean()
        domain_variance = domain_losses.var(unbiased=False)
        pair_delta = (prediction[left] - prediction[right]) * signs
        ranking = torch.nn.functional.softplus(
            -pair_delta / float(config["ranking_temperature"])
        ).mean()
        weights = torch.nn.functional.softplus(model.weight_logits)
        loss = (
            regression
            + float(config["ranking_weight"]) * ranking
            + float(config["domain_loss_variance_weight"]) * domain_variance
            + float(config["weight_regularization_weight"]) * weights.square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "total_loss": float(loss.detach().cpu()),
            "regression_loss": float(regression.detach().cpu()),
            "ranking_loss": float(ranking.detach().cpu()),
            "domain_loss_variance": float(domain_variance.detach().cpu()),
        }
    with torch.inference_mode():
        weights = torch.nn.functional.softplus(model.weight_logits).cpu().numpy()
    final.update(
        train_case_count=int(len(benefit_np)),
        pair_count=int(len(left_np)),
        feature_weights={name: float(value) for name, value in zip(FEATURE_NAMES, weights)},
    )
    return model.eval(), mean, scale, final


def score_benefit_head(
    model: MonotoneBenefitHead,
    cases: Mapping[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    features = (np.asarray(cases["features"], dtype=np.float32) - mean) / scale
    with torch.inference_mode():
        return model(torch.from_numpy(features.astype(np.float32)).cuda()).cpu().numpy()


def authority_metrics(
    action_arrays: Mapping[str, np.ndarray],
    cases: Mapping[str, np.ndarray],
    authority_scores: np.ndarray,
    authority_fraction: float,
) -> dict[str, Any]:
    case_count = len(authority_scores)
    authority_count = max(1, int(np.floor(float(authority_fraction) * case_count)))
    authorized = np.argsort(-authority_scores, kind="stable")[:authority_count]
    target = np.asarray(action_arrays["target_cost"], dtype=np.float32)
    all_indices = np.concatenate(cases["all_action_indices"][authorized]).astype(np.int64)
    selected_indices = np.concatenate(cases["selected_action_indices"][authorized]).astype(np.int64)
    all_cost = float(target[all_indices].mean())
    selected_cost = float(target[selected_indices].mean())
    scene_rows = []
    for scene in np.unique(cases["scene_index"][authorized]):
        scene_cases = authorized[cases["scene_index"][authorized] == scene]
        scene_all = np.concatenate(cases["all_action_indices"][scene_cases]).astype(np.int64)
        scene_selected = np.concatenate(cases["selected_action_indices"][scene_cases]).astype(np.int64)
        scene_all_cost = float(target[scene_all].mean())
        scene_selected_cost = float(target[scene_selected].mean())
        scene_rows.append(
            {
                "scene_index": int(scene),
                "authorized_case_count": int(len(scene_cases)),
                "all_mean_cost": scene_all_cost,
                "selected_mean_cost": scene_selected_cost,
                "delta": scene_selected_cost - scene_all_cost,
            }
        )
    return {
        "case_count": int(case_count),
        "authorized_case_count": int(authority_count),
        "authority_fraction": float(authority_count / case_count),
        "all_mean_cost": all_cost,
        "selected_mean_cost": selected_cost,
        "relative_cost_reduction": float((all_cost - selected_cost) / all_cost if all_cost else 0.0),
        "authorized_positive_benefit_rate": float(
            np.mean(np.asarray(cases["benefit"])[authorized] >= 0.0)
        ),
        "authorized_mean_case_benefit": float(np.mean(np.asarray(cases["benefit"])[authorized])),
        "scene_support_count": int(len(scene_rows)),
        "scene_nonincreasing_count": int(sum(row["delta"] <= 0 for row in scene_rows)),
        "scene_rows": scene_rows,
        "authorized_case_indices": [int(value) for value in authorized],
    }
