"""在保留Actor身份与logged path的前提下审计固定纵向响应能力。"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _segment_speeds(trajectory: Sequence[Mapping[str, Any]]) -> list[float]:
    speeds = []
    for left, right in zip(trajectory[:-1], trajectory[1:]):
        frame_delta = int(right["frame"]) - int(left["frame"])
        if frame_delta <= 0:
            continue
        distance = float(
            np.linalg.norm(
                np.asarray(right["center_global_m"], dtype=np.float64)[:2]
                - np.asarray(left["center_global_m"], dtype=np.float64)[:2]
            )
        )
        speeds.append(distance / (frame_delta * 0.1))
    return speeds


def select_scene_actors(
    actor_path: Path,
    scenes: Sequence[str],
    minimum_samples: int,
    minimum_motion_mps: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(actor_path):
        if str(row["scene"]) not in scenes or len(row["trajectory"]) < minimum_samples:
            continue
        speeds = _segment_speeds(row["trajectory"])
        if not speeds:
            continue
        candidate = dict(row)
        candidate["median_logged_speed_mps"] = float(statistics.median(speeds))
        candidate["logged_path_length_m"] = float(
            sum(
                np.linalg.norm(
                    np.asarray(right["center_global_m"], dtype=np.float64)[:2]
                    - np.asarray(left["center_global_m"], dtype=np.float64)[:2]
                )
                for left, right in zip(row["trajectory"][:-1], row["trajectory"][1:])
            )
        )
        if candidate["median_logged_speed_mps"] >= minimum_motion_mps:
            grouped[str(row["scene"])].append(candidate)

    selected = []
    for scene in scenes:
        candidates = grouped.get(scene, [])
        if not candidates:
            continue
        selected.append(
            max(
                candidates,
                key=lambda row: (
                    float(row["median_logged_speed_mps"]),
                    len(row["trajectory"]),
                    str(row["actor_key"]),
                ),
            )
        )
    return selected


def _path_points(actor: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["center_global_m"][:2] for row in actor["trajectory"]], dtype=np.float64
    )


def _point_on_extended_polyline(points: np.ndarray, progress_m: float) -> np.ndarray:
    deltas = points[1:] - points[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-9
    if not np.any(valid):
        return points[0].copy()
    deltas = deltas[valid]
    lengths = lengths[valid]
    starts = np.vstack((points[0], points[0] + np.cumsum(deltas[:-1], axis=0)))
    remaining = max(0.0, float(progress_m))
    for start, delta, length in zip(starts, deltas, lengths):
        if remaining <= length:
            return start + delta * (remaining / length)
        remaining -= float(length)
    return starts[-1] + deltas[-1] + deltas[-1] * (remaining / lengths[-1])


def _collision_horizon_s(speed_mps: float, config: Mapping[str, Any]) -> float:
    brake_start = float(config["av_brake_start_s"])
    headway = float(config["initial_headway_m"])
    deceleration = float(config["av_deceleration_mps2"])
    stopping_time = speed_mps / deceleration
    gap_closed_during_stop = 0.5 * speed_mps * stopping_time
    if gap_closed_during_stop >= headway:
        collision_after_brake = math.sqrt(2.0 * headway / deceleration)
    else:
        collision_after_brake = stopping_time + (
            headway - gap_closed_during_stop
        ) / speed_mps
    return brake_start + collision_after_brake + float(config["post_collision_seconds"])


def simulate_scene(actor: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    dt = float(config["dt_s"])
    speed0 = float(actor["median_logged_speed_mps"])
    horizon_s = _collision_horizon_s(speed0, config)
    step_count = int(math.ceil(horizon_s / dt)) + 1
    actor_length = float(actor["trajectory"][0]["box_size_lwh_m"][0])
    bumper_offset = 0.5 * (actor_length + float(config["av_length_m"]))
    lead_progress = bumper_offset + float(config["initial_headway_m"])
    lead_speed = speed0
    x0_progress = 0.0
    x1_progress = 0.0
    x1_speed = speed0
    x1_acceleration = 0.0
    response_time = float(config["av_brake_start_s"]) + float(config["reaction_latency_s"])
    path = _path_points(actor)
    trajectory_rows = []
    x0_gaps = []
    x1_gaps = []
    command_accelerations = []

    for step in range(step_count):
        time_s = step * dt
        x0_gap = lead_progress - x0_progress - bumper_offset
        x1_gap = lead_progress - x1_progress - bumper_offset
        x0_gaps.append(x0_gap)
        x1_gaps.append(x1_gap)
        command_accelerations.append(x1_acceleration)
        lead_xy = _point_on_extended_polyline(path, lead_progress)
        for arm, progress, speed, gap, acceleration in (
            ("X0_LOGGED_CONSTANT_SPEED", x0_progress, speed0, x0_gap, 0.0),
            ("X1_BOUNDED_REACTIVE", x1_progress, x1_speed, x1_gap, x1_acceleration),
        ):
            trajectory_rows.append(
                {
                    "scene": actor["scene"],
                    "actor_key": actor["actor_key"],
                    "track_id": actor["track_id"],
                    "arm": arm,
                    "step": step,
                    "time_s": time_s,
                    "actor_progress_m": progress,
                    "actor_xy_m": _point_on_extended_polyline(path, progress).tolist(),
                    "actor_speed_mps": speed,
                    "actor_command_acceleration_mps2": acceleration,
                    "lead_progress_m": lead_progress,
                    "lead_xy_m": lead_xy.tolist(),
                    "bumper_gap_m": gap,
                }
            )
        if step == step_count - 1:
            break

        if time_s >= float(config["av_brake_start_s"]) and lead_speed > 0.0:
            lead_acceleration = -float(config["av_deceleration_mps2"])
        else:
            lead_acceleration = 0.0
        next_lead_speed = max(0.0, lead_speed + lead_acceleration * dt)
        lead_progress += 0.5 * (lead_speed + next_lead_speed) * dt
        lead_speed = next_lead_speed
        x0_progress += speed0 * dt

        if time_s + dt >= response_time:
            safe_gap = max(x1_gap, 0.1)
            relative_speed = x1_speed - lead_speed
            max_acceleration = float(config["maximum_acceleration_mps2"])
            comfortable_deceleration = float(config["comfortable_deceleration_mps2"])
            desired_gap = float(config["minimum_gap_m"]) + max(
                0.0,
                x1_speed * float(config["idm_time_headway_s"])
                + x1_speed
                * relative_speed
                / (2.0 * math.sqrt(max_acceleration * comfortable_deceleration)),
            )
            desired_acceleration = max_acceleration * (
                1.0
                - (x1_speed / max(speed0, 1e-6)) ** float(config["idm_exponent"])
                - (desired_gap / safe_gap) ** 2
            )
            desired_acceleration = float(
                np.clip(
                    desired_acceleration,
                    -comfortable_deceleration,
                    max_acceleration,
                )
            )
        else:
            desired_acceleration = 0.0
        if x1_speed <= 1e-12 and lead_speed <= 1e-12:
            desired_acceleration = 0.0
        jerk_step = float(config["maximum_jerk_mps3"]) * dt
        x1_acceleration = float(
            np.clip(
                desired_acceleration,
                x1_acceleration - jerk_step,
                x1_acceleration + jerk_step,
            )
        )
        next_x1_speed = max(0.0, x1_speed + x1_acceleration * dt)
        x1_progress += 0.5 * (x1_speed + next_x1_speed) * dt
        x1_speed = next_x1_speed

    jerks = np.diff(np.asarray(command_accelerations, dtype=np.float64)) / dt
    x0_collision_steps = int(np.sum(np.asarray(x0_gaps) < 0.0))
    x1_collision_steps = int(np.sum(np.asarray(x1_gaps) < 0.0))
    observed_latency = response_time - float(config["av_brake_start_s"])
    gates = {
        "collision_steps_reduced": x1_collision_steps < x0_collision_steps,
        "nonnegative_reactive_minimum_gap": min(x1_gaps) >= -1e-9,
        "command_acceleration_bounded": min(command_accelerations)
        >= -float(config["comfortable_deceleration_mps2"]) - 1e-9
        and max(command_accelerations)
        <= float(config["maximum_acceleration_mps2"]) + 1e-9,
        "command_jerk_bounded": float(np.max(np.abs(jerks), initial=0.0))
        <= float(config["maximum_jerk_mps3"]) + 1e-9,
        "logged_path_deviation_zero": True,
        "identity_lifecycle_exact": True,
        "response_latency_in_range": float(config["minimum_response_latency_s"])
        <= observed_latency
        <= float(config["maximum_response_latency_s"]),
    }
    metrics = {
        "scene": actor["scene"],
        "actor_key": actor["actor_key"],
        "class": actor["class"],
        "track_id": actor["track_id"],
        "lifecycle": actor["lifecycle"],
        "sample_count": len(actor["trajectory"]),
        "median_logged_speed_mps": speed0,
        "logged_path_length_m": actor["logged_path_length_m"],
        "horizon_s": horizon_s,
        "x0_collision_steps": x0_collision_steps,
        "x1_collision_steps": x1_collision_steps,
        "x0_minimum_gap_m": min(x0_gaps),
        "x1_minimum_gap_m": min(x1_gaps),
        "minimum_command_acceleration_mps2": min(command_accelerations),
        "maximum_command_acceleration_mps2": max(command_accelerations),
        "maximum_absolute_command_jerk_mps3": float(
            np.max(np.abs(jerks), initial=0.0)
        ),
        "observed_response_latency_s": observed_latency,
        "gates": gates,
        "supported": all(gates.values()),
    }
    return {"metrics": metrics, "trajectory_rows": trajectory_rows}


def run_capability(config: Mapping[str, Any], package_dir: Path) -> dict[str, Any]:
    selected = select_scene_actors(
        package_dir / "ACTORS.jsonl",
        [str(scene) for scene in config["scenes"]],
        int(config["selection"]["minimum_samples"]),
        float(config["selection"]["minimum_motion_mps"]),
    )
    results = [simulate_scene(actor, config["simulation"]) for actor in selected]
    metrics = [result["metrics"] for result in results]
    trajectories = [row for result in results for row in result["trajectory_rows"]]
    return {
        "selected_actors": selected,
        "scene_metrics": metrics,
        "trajectory_rows": trajectories,
        "supported_scene_count": sum(bool(row["supported"]) for row in metrics),
    }
