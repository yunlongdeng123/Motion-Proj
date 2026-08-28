"""Trajectory-conditioned reliability learning for future actor states."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ACTOR_FEATURE_NAMES = (
    "actor_speed_mps", "actor_acceleration_longitudinal_mps2",
    "actor_acceleration_lateral_mps2", "actor_yaw_rate_rps",
    "actor_heading_cos_ego", "actor_heading_sin_ego",
    "relative_position_longitudinal_m", "relative_position_lateral_m",
    "relative_velocity_longitudinal_mps", "relative_velocity_lateral_mps",
    "ego_speed_mps", "actor_length_m", "actor_width_m", "observed_age_s",
    "horizon_seconds", "class_vehicle", "class_pedestrian", "class_cycle", "class_other",
)
QUERY_FEATURE_NAMES = (
    "ego_progress_ratio", "ego_lateral_offset_m", "predicted_minimum_separation_m",
    "predicted_terminal_separation_m", "predicted_closing_speed_mps",
)
FEATURE_NAMES = ACTOR_FEATURE_NAMES + QUERY_FEATURE_NAMES


def _wrapped_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _class_features(class_name: str) -> tuple[float, float, float, float]:
    name = class_name.lower()
    if "vehicle" in name:
        return 1.0, 0.0, 0.0, 0.0
    if "pedestrian" in name:
        return 0.0, 1.0, 0.0, 0.0
    if "bicycle" in name or "motorcycle" in name or "cycle" in name:
        return 0.0, 0.0, 1.0, 0.0
    return 0.0, 0.0, 0.0, 1.0


def _pose_table(scene_dir: Path) -> dict[int, np.ndarray]:
    return {
        int(path.stem): np.loadtxt(path, dtype=np.float64)
        for path in sorted((scene_dir / "lidar_pose").glob("*.txt"))
    }


def _actor_table(row: Mapping[str, Any]) -> dict[int, tuple[np.ndarray, float, np.ndarray]]:
    annotations = row["frame_annotations"]
    table: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
    for frame, transform, box in zip(
        annotations["frame_idx"], annotations["obj_to_world"], annotations["box_size"]
    ):
        matrix = np.asarray(transform, dtype=np.float64)
        table[int(frame)] = (
            matrix[:2, 3].copy(), float(math.atan2(matrix[1, 0], matrix[0, 0])),
            np.asarray(box[:2], dtype=np.float64),
        )
    return table


def _candidate_path(
    ego_position: np.ndarray, ego_heading: np.ndarray, ego_speed: float, horizon: float,
    progress_ratio: float, lateral_offset: float, point_count: int,
) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    left = np.asarray([-ego_heading[1], ego_heading[0]], dtype=np.float64)
    terminal = ego_heading * ego_speed * horizon * progress_ratio
    return ego_position[None, :] + progress[:, None] * (
        terminal[None, :] + left[None, :] * lateral_offset
    )


def materialize_actor_query_rows(
    scene_dirs: Sequence[Path], horizons_seconds: Sequence[float], data_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    history_far = int(data_config["history_frame_offsets"][0])
    history_near = int(data_config["history_frame_offsets"][1])
    stride = int(data_config["anchor_stride_frames"])
    point_count = int(data_config["path_point_count"])
    radius = float(data_config["maximum_predicted_query_distance_m"])
    exposure_scale = float(data_config["exposure_distance_scale_m"])
    progress_ratios = [float(x) for x in data_config["ego_progress_ratios"]]
    lateral_offsets = [float(x) for x in data_config["ego_lateral_offsets_m"]]
    features: list[list[float]] = []
    target_cost: list[float] = []
    raw_error: list[float] = []
    separation: list[float] = []
    scene_index: list[int] = []
    horizon_values: list[float] = []
    actor_ids: list[int] = []
    query_ids: list[int] = []
    anchor_frames: list[int] = []
    for scene_dir in scene_dirs:
        poses = _pose_table(scene_dir)
        if not poses:
            continue
        actors = json.loads((scene_dir / "instances" / "instances_info.json").read_text(encoding="utf-8"))
        numeric_scene = int(scene_dir.name)
        for actor_id_text, actor in actors.items():
            actor_table = _actor_table(actor)
            if not actor_table:
                continue
            first_frame = min(actor_table)
            for horizon_seconds in horizons_seconds:
                horizon_frames = int(round(float(horizon_seconds) * 10.0))
                anchors = sorted(
                    frame for frame in actor_table
                    if frame % stride == 0 and frame - history_far in actor_table
                    and frame - history_near in actor_table and frame + horizon_frames in actor_table
                    and frame in poses and frame - history_near in poses
                )
                for frame in anchors:
                    past_far, yaw_far, _ = actor_table[frame - history_far]
                    past_near, yaw_near, _ = actor_table[frame - history_near]
                    current, yaw, box = actor_table[frame]
                    future, _, _ = actor_table[frame + horizon_frames]
                    near_dt = history_near * 0.1
                    far_dt = (history_far - history_near) * 0.1
                    velocity = (current - past_near) / near_dt
                    previous_velocity = (past_near - past_far) / far_dt
                    acceleration = (velocity - previous_velocity) / max(near_dt, 1e-6)
                    actor_speed = float(np.linalg.norm(velocity))
                    actor_direction = velocity / max(actor_speed, 1e-6)
                    actor_left = np.asarray([-actor_direction[1], actor_direction[0]])
                    acceleration_longitudinal = float(np.dot(acceleration, actor_direction))
                    acceleration_lateral = float(np.dot(acceleration, actor_left))
                    yaw_rate = _wrapped_angle(yaw - yaw_near) / near_dt
                    ego_pose = poses[frame]
                    ego_previous_pose = poses[frame - history_near]
                    ego_position = ego_pose[:2, 3]
                    ego_previous = ego_previous_pose[:2, 3]
                    ego_speed = float(np.linalg.norm(ego_position - ego_previous) / near_dt)
                    ego_heading = ego_pose[:2, 0].copy()
                    ego_heading /= max(float(np.linalg.norm(ego_heading)), 1e-6)
                    ego_left = np.asarray([-ego_heading[1], ego_heading[0]])
                    relative_position = current - ego_position
                    ego_velocity = ego_heading * ego_speed
                    relative_velocity = velocity - ego_velocity
                    predicted_future = current + velocity * float(horizon_seconds)
                    endpoint_error = min(
                        float(np.linalg.norm(future - predicted_future)),
                        float(data_config["maximum_actor_state_error_m"]),
                    )
                    times = np.linspace(0.0, float(horizon_seconds), point_count)
                    predicted_actor_path = current[None, :] + times[:, None] * velocity[None, :]
                    heading_delta = _wrapped_angle(yaw - math.atan2(ego_heading[1], ego_heading[0]))
                    base = [
                        actor_speed, acceleration_longitudinal, acceleration_lateral, yaw_rate,
                        math.cos(heading_delta), math.sin(heading_delta),
                        float(np.dot(relative_position, ego_heading)), float(np.dot(relative_position, ego_left)),
                        float(np.dot(relative_velocity, ego_heading)), float(np.dot(relative_velocity, ego_left)),
                        ego_speed, float(box[0]), float(box[1]), (frame - first_frame) * 0.1,
                        float(horizon_seconds), *_class_features(str(actor["class_name"])),
                    ]
                    for query_id, (progress_ratio, lateral_offset) in enumerate(
                        (p, l) for p in progress_ratios for l in lateral_offsets
                    ):
                        ego_path = _candidate_path(
                            ego_position, ego_heading, ego_speed, float(horizon_seconds),
                            progress_ratio, lateral_offset, point_count,
                        )
                        distances = np.linalg.norm(predicted_actor_path - ego_path, axis=1)
                        minimum_distance = float(np.min(distances))
                        if minimum_distance > radius:
                            continue
                        terminal_distance = float(distances[-1])
                        relative_terminal = predicted_actor_path[-1] - ego_path[-1]
                        initial_distance = max(float(np.linalg.norm(relative_position)), 1e-6)
                        closing_speed = (initial_distance - terminal_distance) / float(horizon_seconds)
                        exposure = math.exp(-minimum_distance / exposure_scale)
                        features.append(base + [
                            progress_ratio, lateral_offset, minimum_distance, terminal_distance, closing_speed,
                        ])
                        target_cost.append(endpoint_error * exposure)
                        raw_error.append(endpoint_error)
                        separation.append(minimum_distance)
                        scene_index.append(numeric_scene)
                        horizon_values.append(float(horizon_seconds))
                        actor_ids.append(int(actor_id_text))
                        query_ids.append(query_id)
                        anchor_frames.append(frame)
        print(f"actor-query materialization scene={scene_dir.name} rows={len(features)}", flush=True)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "target_cost": np.asarray(target_cost, dtype=np.float32),
        "raw_actor_state_error_m": np.asarray(raw_error, dtype=np.float32),
        "predicted_minimum_separation_m": np.asarray(separation, dtype=np.float32),
        "scene_index": np.asarray(scene_index, dtype=np.int32),
        "horizon_seconds": np.asarray(horizon_values, dtype=np.float32),
        "actor_id": np.asarray(actor_ids, dtype=np.int32),
        "query_id": np.asarray(query_ids, dtype=np.int16),
        "anchor_frame": np.asarray(anchor_frames, dtype=np.int16),
    }


class ReliabilityMLP(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.encoder = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(width, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.head(self.encode(features)).squeeze(-1))


class QuantileReliabilityMLP(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.encoder = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(width, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(features))
        lower = torch.nn.functional.softplus(raw[:, 0])
        median = lower + torch.nn.functional.softplus(raw[:, 1])
        upper = median + torch.nn.functional.softplus(raw[:, 2])
        return torch.stack((lower, median, upper), dim=1)


class BinaryReliabilityMLP(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def train_binary_reliability_models(
    arrays: Mapping[str, np.ndarray], model_config: Mapping[str, Any], evaluation_config: Mapping[str, Any], seed: int,
) -> tuple[ReliabilityMLP, BinaryReliabilityMLP, BinaryReliabilityMLP, np.ndarray, np.ndarray, dict[str, float]]:
    raw_features = np.asarray(arrays["features"], dtype=np.float32)
    mean = raw_features.mean(axis=0)
    scale = raw_features.std(axis=0).clip(min=1e-4)
    features = torch.from_numpy((raw_features - mean) / scale).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    continuous_target = torch.from_numpy(
        np.log1p(np.asarray(arrays["target_cost"], dtype=np.float32))
    ).cuda()
    binary_labels_np = (
        (np.asarray(arrays["raw_actor_state_error_m"]) > float(evaluation_config["unreliable_actor_state_error_m"]))
        & (np.asarray(arrays["predicted_minimum_separation_m"]) <= float(evaluation_config["unreliable_exposure_radius_m"]))
    ).astype(np.float32)
    binary_labels = torch.from_numpy(binary_labels_np).cuda()
    positive_count = max(int(np.count_nonzero(binary_labels_np)), 1)
    negative_count = max(int(len(binary_labels_np) - positive_count), 1)
    positive_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32, device="cuda")
    torch.manual_seed(int(seed))
    continuous_model = ReliabilityMLP(features.shape[1], model_config["hidden_dimensions"]).cuda()
    binary_query_model = BinaryReliabilityMLP(features.shape[1], model_config["hidden_dimensions"]).cuda()
    binary_actor_model = BinaryReliabilityMLP(actor_features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        list(continuous_model.parameters()) + list(binary_query_model.parameters())
        + list(binary_actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    final_continuous = final_query = final_actor = 0.0
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        continuous_prediction = continuous_model(features)
        query_logits = binary_query_model(features)
        actor_logits = binary_actor_model(actor_features)
        continuous_loss = torch.nn.functional.smooth_l1_loss(
            continuous_prediction, continuous_target, beta=float(model_config["huber_beta"])
        )
        query_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            query_logits, binary_labels, pos_weight=positive_weight
        )
        actor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            actor_logits, binary_labels, pos_weight=positive_weight
        )
        (continuous_loss + query_loss + actor_loss).backward()
        optimizer.step()
        final_continuous = float(continuous_loss.detach().cpu())
        final_query = float(query_loss.detach().cpu())
        final_actor = float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(
                f"binary actor reliability epoch={epoch + 1} continuous={final_continuous:.6f} "
                f"query_bce={final_query:.6f} actor_bce={final_actor:.6f}", flush=True,
            )
    return continuous_model.eval(), binary_query_model.eval(), binary_actor_model.eval(), mean, scale, {
        "continuous_final_loss": final_continuous, "binary_query_final_loss": final_query,
        "binary_actor_only_final_loss": final_actor, "training_row_count": int(len(raw_features)),
        "training_unreliable_row_count": positive_count,
        "positive_class_weight": float(negative_count / positive_count),
    }


def predict_binary_reliability(
    model: BinaryReliabilityMLP, raw_features: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    actor_only: bool = False,
) -> np.ndarray:
    normalized = (np.asarray(raw_features, dtype=np.float32) - mean) / scale
    if actor_only:
        normalized = normalized[:, :len(ACTOR_FEATURE_NAMES)]
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(normalized).cuda())).cpu().numpy()


def train_reliability_models(
    arrays: Mapping[str, np.ndarray], model_config: Mapping[str, Any], seed: int,
) -> tuple[ReliabilityMLP, ReliabilityMLP, np.ndarray, np.ndarray, dict[str, float]]:
    raw_features = np.asarray(arrays["features"], dtype=np.float32)
    mean = raw_features.mean(axis=0)
    scale = raw_features.std(axis=0).clip(min=1e-4)
    normalized = (raw_features - mean) / scale
    features = torch.from_numpy(normalized).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    target = torch.from_numpy(np.log1p(np.asarray(arrays["target_cost"], dtype=np.float32))).cuda()
    torch.manual_seed(int(seed))
    if "quantile_levels" in model_config:
        query_model = QuantileReliabilityMLP(
            features.shape[1], model_config["hidden_dimensions"]
        ).cuda()
    else:
        query_model = ReliabilityMLP(features.shape[1], model_config["hidden_dimensions"]).cuda()
    actor_model = ReliabilityMLP(actor_features.shape[1], model_config["hidden_dimensions"]).cuda()
    if "quantile_levels" in model_config:
        optimizer = torch.optim.AdamW(
            list(query_model.parameters()) + list(actor_model.parameters()),
            lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
        )
        levels = torch.tensor(model_config["quantile_levels"], dtype=target.dtype, device=target.device)
        final_quantile_loss = final_actor_loss = 0.0
        for epoch in range(int(model_config["epochs"])):
            optimizer.zero_grad(set_to_none=True)
            quantiles = query_model(features)
            errors = target[:, None] - quantiles
            quantile_loss = torch.maximum((levels - 1.0) * errors, levels * errors).mean()
            actor_prediction = actor_model(actor_features)
            actor_loss = torch.nn.functional.smooth_l1_loss(
                actor_prediction, target, beta=float(model_config["huber_beta"])
            )
            (quantile_loss + actor_loss).backward()
            optimizer.step()
            final_quantile_loss = float(quantile_loss.detach().cpu())
            final_actor_loss = float(actor_loss.detach().cpu())
            if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
                print(
                    f"actor reliability quantile epoch={epoch + 1} "
                    f"quantile_loss={final_quantile_loss:.6f} actor_only_loss={final_actor_loss:.6f}",
                    flush=True,
                )
        return query_model.eval(), actor_model.eval(), mean, scale, {
            "query_conditioned_quantile_loss": final_quantile_loss,
            "actor_only_final_loss": final_actor_loss,
            "quantile_levels": [float(value) for value in model_config["quantile_levels"]],
            "training_row_count": int(len(raw_features)),
        }
    if "rank_contrastive_pretrain_epochs" in model_config:
        actor_optimizer = torch.optim.AdamW(
            actor_model.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        encoder_optimizer = torch.optim.AdamW(
            query_model.encoder.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        final_contrastive_loss = final_actor_loss = 0.0
        shifts = [int(value) for value in model_config["rank_contrastive_pair_shifts"]]
        for epoch in range(int(model_config["rank_contrastive_pretrain_epochs"])):
            encoder_optimizer.zero_grad(set_to_none=True)
            actor_optimizer.zero_grad(set_to_none=True)
            embedding = torch.nn.functional.normalize(query_model.encode(features), dim=1)
            candidate_distances = torch.stack([
                (embedding - torch.roll(embedding, shift, dims=0)).square().sum(dim=1)
                for shift in shifts
            ])
            target_distances = torch.stack([
                (target - torch.roll(target, shift, dims=0)).abs() for shift in shifts
            ])
            positive_index = target_distances.argmin(dim=0, keepdim=True)
            negative_index = target_distances.argmax(dim=0, keepdim=True)
            positive_distance = candidate_distances.gather(0, positive_index).squeeze(0)
            negative_distance = candidate_distances.gather(0, negative_index).squeeze(0)
            valid = (
                target_distances.gather(0, negative_index).squeeze(0)
                - target_distances.gather(0, positive_index).squeeze(0)
            ) >= float(model_config["rank_contrastive_minimum_target_gap"])
            contrastive_loss = torch.nn.functional.softplus(
                (positive_distance - negative_distance)
                / float(model_config["rank_contrastive_temperature"])
            )[valid].mean()
            actor_prediction = actor_model(actor_features)
            actor_loss = torch.nn.functional.smooth_l1_loss(
                actor_prediction, target, beta=float(model_config["huber_beta"])
            )
            (contrastive_loss + actor_loss).backward()
            encoder_optimizer.step()
            actor_optimizer.step()
            final_contrastive_loss = float(contrastive_loss.detach().cpu())
            final_actor_loss = float(actor_loss.detach().cpu())
            if epoch % 100 == 0 or epoch + 1 == int(model_config["rank_contrastive_pretrain_epochs"]):
                print(
                    f"actor reliability contrastive epoch={epoch + 1} "
                    f"contrastive_loss={final_contrastive_loss:.6f} actor_only_loss={final_actor_loss:.6f}",
                    flush=True,
                )
        for parameter in query_model.encoder.parameters():
            parameter.requires_grad_(False)
        head_optimizer = torch.optim.AdamW(
            query_model.head.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        final_query_regression = 0.0
        for epoch in range(int(model_config["regression_head_epochs"])):
            head_optimizer.zero_grad(set_to_none=True)
            actor_optimizer.zero_grad(set_to_none=True)
            query_prediction = query_model(features)
            actor_prediction = actor_model(actor_features)
            query_regression = torch.nn.functional.smooth_l1_loss(
                query_prediction, target, beta=float(model_config["huber_beta"])
            )
            actor_loss = torch.nn.functional.smooth_l1_loss(
                actor_prediction, target, beta=float(model_config["huber_beta"])
            )
            (query_regression + actor_loss).backward()
            head_optimizer.step()
            actor_optimizer.step()
            final_query_regression = float(query_regression.detach().cpu())
            final_actor_loss = float(actor_loss.detach().cpu())
            if epoch % 250 == 0 or epoch + 1 == int(model_config["regression_head_epochs"]):
                print(
                    f"actor reliability frozen-encoder epoch={epoch + 1} "
                    f"query_loss={final_query_regression:.6f} actor_only_loss={final_actor_loss:.6f}",
                    flush=True,
                )
        return query_model.eval(), actor_model.eval(), mean, scale, {
            "query_conditioned_final_loss": final_query_regression,
            "query_conditioned_regression_loss": final_query_regression,
            "query_conditioned_rank_contrastive_loss": final_contrastive_loss,
            "actor_only_final_loss": final_actor_loss,
            "rank_contrastive_pretrain_epochs": int(model_config["rank_contrastive_pretrain_epochs"]),
            "regression_head_epochs": int(model_config["regression_head_epochs"]),
            "training_row_count": int(len(raw_features)),
        }
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    final_query_loss = final_query_regression = final_query_ranking = final_actor_loss = 0.0
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        query_prediction = query_model(features)
        actor_prediction = actor_model(actor_features)
        query_regression = torch.nn.functional.smooth_l1_loss(
            query_prediction, target, beta=float(model_config["huber_beta"])
        )
        actor_loss = torch.nn.functional.smooth_l1_loss(
            actor_prediction, target, beta=float(model_config["huber_beta"])
        )
        query_ranking = query_regression.new_zeros(())
        ranking_weight = float(model_config.get("ranking_weight", 0.0))
        if ranking_weight > 0.0:
            ranking_terms = []
            for shift in model_config["ranking_pair_shifts"]:
                shifted_target = torch.roll(target, int(shift))
                target_delta = target - shifted_target
                valid = target_delta.abs() >= float(model_config["pairwise_minimum_target_gap"])
                prediction_delta = query_prediction - torch.roll(query_prediction, int(shift))
                terms = torch.nn.functional.softplus(
                    -(prediction_delta * torch.sign(target_delta))
                    / float(model_config["ranking_temperature"])
                )
                ranking_terms.append(terms[valid].mean())
            query_ranking = torch.stack(ranking_terms).mean()
        query_loss = query_regression + ranking_weight * query_ranking
        loss = query_loss + actor_loss
        loss.backward()
        optimizer.step()
        final_query_loss = float(query_loss.detach().cpu())
        final_query_regression = float(query_regression.detach().cpu())
        final_query_ranking = float(query_ranking.detach().cpu())
        final_actor_loss = float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(
                f"actor reliability epoch={epoch + 1} query_loss={final_query_loss:.6f} "
                f"actor_only_loss={final_actor_loss:.6f}", flush=True,
            )
    return query_model.eval(), actor_model.eval(), mean, scale, {
        "query_conditioned_final_loss": final_query_loss,
        "query_conditioned_regression_loss": final_query_regression,
        "query_conditioned_pairwise_ranking_loss": final_query_ranking,
        "actor_only_final_loss": final_actor_loss,
        "training_row_count": int(len(raw_features)),
    }


def predict_reliability(
    model: torch.nn.Module, raw_features: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    actor_only: bool = False,
) -> np.ndarray:
    normalized = (np.asarray(raw_features, dtype=np.float32) - mean) / scale
    if actor_only:
        normalized = normalized[:, :len(ACTOR_FEATURE_NAMES)]
    with torch.no_grad():
        prediction = model(torch.from_numpy(normalized).cuda()).cpu().numpy()
    return np.expm1(prediction).clip(min=0.0)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result
    if len(left) < 2:
        return float("nan")
    return float(np.corrcoef(ranks(np.asarray(left)), ranks(np.asarray(right)))[0, 1])


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positive = int(np.count_nonzero(labels))
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def evaluate_reliability(
    arrays: Mapping[str, np.ndarray], query_prediction: np.ndarray, actor_prediction: np.ndarray,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    target = np.asarray(arrays["target_cost"], dtype=np.float64)
    raw_error = np.asarray(arrays["raw_actor_state_error_m"], dtype=np.float64)
    distance = np.asarray(arrays["predicted_minimum_separation_m"], dtype=np.float64)
    unreliable = (
        (raw_error > float(evaluation_config["unreliable_actor_state_error_m"]))
        & (distance <= float(evaluation_config["unreliable_exposure_radius_m"]))
    )
    query_spearman = spearman_correlation(query_prediction, target)
    actor_spearman = spearman_correlation(actor_prediction, target)
    query_mae = float(np.mean(np.abs(query_prediction - target)))
    actor_mae = float(np.mean(np.abs(actor_prediction - target)))
    scene_rows = []
    for scene in np.unique(arrays["scene_index"]):
        inside = np.asarray(arrays["scene_index"]) == scene
        scene_rows.append({
            "scene_index": int(scene), "row_count": int(np.count_nonzero(inside)),
            "query_spearman": spearman_correlation(query_prediction[inside], target[inside]),
            "actor_only_spearman": spearman_correlation(actor_prediction[inside], target[inside]),
            "query_mae": float(np.mean(np.abs(query_prediction[inside] - target[inside]))),
            "actor_only_mae": float(np.mean(np.abs(actor_prediction[inside] - target[inside]))),
        })
    return {
        "row_count": int(len(target)), "scene_count": int(len(scene_rows)),
        "unreliable_row_count": int(np.count_nonzero(unreliable)),
        "query_conditioned_spearman": query_spearman,
        "actor_only_spearman": actor_spearman,
        "spearman_delta_over_actor_only": query_spearman - actor_spearman,
        "query_conditioned_mae": query_mae, "actor_only_mae": actor_mae,
        "mae_reduction_over_actor_only": (actor_mae - query_mae) / max(actor_mae, 1e-12),
        "query_conditioned_unreliable_auroc": binary_auroc(unreliable, query_prediction),
        "actor_only_unreliable_auroc": binary_auroc(unreliable, actor_prediction),
        "scene_query_spearman_noninferior_count": int(sum(
            row["query_spearman"] >= row["actor_only_spearman"] for row in scene_rows
            if np.isfinite(row["query_spearman"]) and np.isfinite(row["actor_only_spearman"])
        )),
        "scene_rows": scene_rows,
    }
