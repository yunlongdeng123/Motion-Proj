"""Natural-event 海选使用的关键帧运动学与连续中心线几何。

本模块只从 nuScenes 原始 2 Hz annotation keyframe 派生速度、加速度和横向运动。
10 Hz 插值轨迹可用于地图匹配与可视化，但不得作为更高频物理观测。
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def angle_error(left: float, right: float) -> float:
    """返回两个弧度角之间的最小绝对差。"""
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def circular_mean(angles: Iterable[float]) -> float:
    values = list(angles)
    if not values:
        raise ValueError("angles 不能为空")
    vector = np.asarray(
        [sum(math.cos(value) for value in values), sum(math.sin(value) for value in values)]
    )
    if float(np.linalg.norm(vector)) < 1e-9:
        return float(values[0])
    return float(math.atan2(vector[1], vector[0]))


def project_to_polyline(
    xy: Iterable[float],
    centerline: np.ndarray,
    arc_lengths: np.ndarray,
) -> dict:
    """把点连续投影到折线段，返回 s、带符号横向距离与局部航向。

    ``centerline`` 至少含 x/y；若第三列存在，仅用于退化单点折线的航向。
    横向距离按局部切向量左侧为正。
    """
    point = np.asarray(list(xy), dtype=float)[:2]
    line = np.asarray(centerline, dtype=float)
    arcs = np.asarray(arc_lengths, dtype=float)
    if line.ndim != 2 or line.shape[0] == 0 or line.shape[1] < 2:
        raise ValueError("centerline 必须是非空 Nx2/Nx3 数组")
    if arcs.shape != (line.shape[0],):
        raise ValueError("arc_lengths 与 centerline 长度不一致")
    if line.shape[0] == 1:
        delta = point - line[0, :2]
        heading = float(line[0, 2]) if line.shape[1] >= 3 else 0.0
        tangent = np.asarray([math.cos(heading), math.sin(heading)])
        signed = float(tangent[0] * delta[1] - tangent[1] * delta[0])
        return {
            "s_m": float(arcs[0]),
            "distance_m": float(np.linalg.norm(delta)),
            "signed_lateral_m": signed,
            "heading_rad": heading,
            "segment_index": 0,
            "segment_fraction": 0.0,
            "projected_xy": line[0, :2].astype(float).tolist(),
        }

    starts = line[:-1, :2]
    segments = line[1:, :2] - starts
    lengths_sq = np.einsum("ij,ij->i", segments, segments)
    valid = lengths_sq > 1e-12
    fractions = np.zeros(len(segments), dtype=float)
    fractions[valid] = np.clip(
        np.einsum("ij,ij->i", point - starts, segments)[valid] / lengths_sq[valid],
        0.0,
        1.0,
    )
    projections = starts + fractions[:, None] * segments
    deltas = point - projections
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    index = int(np.argmin(distances_sq))
    segment = segments[index]
    length = float(np.linalg.norm(segment))
    if length <= 1e-12:
        heading = (
            float(line[index, 2])
            if line.shape[1] >= 3
            else 0.0
        )
        tangent = np.asarray([math.cos(heading), math.sin(heading)])
    else:
        tangent = segment / length
        heading = float(math.atan2(tangent[1], tangent[0]))
    delta = deltas[index]
    signed = float(tangent[0] * delta[1] - tangent[1] * delta[0])
    s_m = float(arcs[index] + fractions[index] * length)
    return {
        "s_m": s_m,
        "distance_m": float(math.sqrt(max(0.0, distances_sq[index]))),
        "signed_lateral_m": signed,
        "heading_rad": heading,
        "segment_index": index,
        "segment_fraction": float(fractions[index]),
        "projected_xy": projections[index].astype(float).tolist(),
    }


def annotation_keyframes(
    rows: list[dict],
    start_frame: int,
    end_frame: int,
    stride: int,
) -> list[dict]:
    """只取原始标注关键帧；不把插值帧当作物理观测。"""
    return [
        row
        for row in rows
        if start_frame <= int(row["frame_index"]) <= end_frame
        and int(row["frame_index"]) % stride == 0
    ]


def _course_heading(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    delta = np.asarray(rows[-1]["xy"], dtype=float) - np.asarray(
        rows[0]["xy"], dtype=float
    )
    if float(np.linalg.norm(delta)) < 1e-6:
        return None
    return float(math.atan2(delta[1], delta[0]))


def _median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if array.size == 0 else float(np.median(array))


def _max_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if array.size == 0 else float(np.max(array))


def _polyline_distance_rows(rows: list[dict], lane_index, token: str) -> list[float]:
    line = lane_index.centerlines[token]
    arcs = lane_index.arc_lengths[token]
    return [
        float(project_to_polyline(row["xy"], line, arcs)["distance_m"])
        for row in rows
    ]


def _heading_at_s(lane_index, token: str, s_m: float) -> float:
    arcs = lane_index.arc_lengths[token]
    line = lane_index.centerlines[token]
    index = int(np.argmin(np.abs(arcs - float(s_m))))
    return float(line[index, 2])


def _join_geometry(
    lane_index,
    source_token: str,
    target_token: str,
    branch_lookback_m: float,
) -> dict:
    connectivity = lane_index.nmap.connectivity
    direct = target_token in connectivity.get(source_token, {}).get("outgoing", [])
    source_line = lane_index.centerlines[source_token]
    target_line = lane_index.centerlines[target_token]
    source_heading = float(source_line[-1, 2])
    target_heading = float(target_line[0, 2])
    join_error = math.degrees(angle_error(source_heading, target_heading))
    source_length = float(lane_index.arc_lengths[source_token][-1])
    target_length = float(lane_index.arc_lengths[target_token][-1])
    source_approach_heading = _heading_at_s(
        lane_index, source_token, max(0.0, source_length - branch_lookback_m)
    )
    target_departure_heading = _heading_at_s(
        lane_index, target_token, min(target_length, branch_lookback_m)
    )
    approach_error = math.degrees(
        angle_error(source_approach_heading, target_departure_heading)
    )
    endpoint_gap = float(
        np.linalg.norm(source_line[-1, :2] - target_line[0, :2])
    )
    incoming = [
        token
        for token in connectivity.get(target_token, {}).get("incoming", [])
        if token in lane_index.centerlines
    ]
    incoming_errors = {}
    for token in incoming:
        length = float(lane_index.arc_lengths[token][-1])
        approach = _heading_at_s(
            lane_index, token, max(0.0, length - branch_lookback_m)
        )
        incoming_errors[token] = math.degrees(
            angle_error(approach, target_departure_heading)
        )
    ranked = sorted(incoming_errors, key=lambda token: (incoming_errors[token], token))
    source_rank = ranked.index(source_token) + 1 if source_token in ranked else None
    alternative_errors = [
        value for token, value in incoming_errors.items() if token != source_token
    ]
    return {
        "directed_connected": direct,
        "source_target_join_heading_error_deg": join_error,
        "source_target_approach_heading_error_deg": approach_error,
        "source_target_endpoint_gap_m": endpoint_gap,
        "branch_lookback_m": branch_lookback_m,
        "target_incoming_count": len(incoming),
        "source_incoming_alignment_rank": source_rank,
        "best_alternative_incoming_heading_error_deg": (
            min(alternative_errors) if alternative_errors else None
        ),
        "source_is_best_aligned_incoming": source_rank == 1,
        "incoming_heading_errors_deg": incoming_errors,
    }


def motion_features(
    rows: list[dict],
    source_run: dict,
    target_run: dict,
    lane_index,
    topology: dict,
    config: dict,
    frame_times_s: dict[int, float] | None = None,
) -> dict:
    """计算 source→target 转换的关键帧物理证据与逐项门禁。"""
    stride = int(config["annotation_keyframe_stride"])
    period = float(config["dense_frame_period_s"])
    count = int(config["keyframes_each_side"])
    pre_all = annotation_keyframes(
        rows,
        int(source_run["start_frame"]),
        int(source_run["end_frame"]),
        stride,
    )
    post_all = annotation_keyframes(
        rows,
        int(target_run["start_frame"]),
        int(target_run["end_frame"]),
        stride,
    )
    pre = pre_all[-count:]
    post = post_all[:count]
    source_token = str(source_run["token"])
    target_token = str(target_run["token"])
    join = _join_geometry(
        lane_index,
        source_token,
        target_token,
        float(config["branch_lookback_m"]),
    )

    base = {
        "schema_version": "event-kinematics-v1",
        "source_token": source_token,
        "target_token": target_token,
        "pre_keyframe_indices": [int(row["frame_index"]) for row in pre],
        "post_keyframe_indices": [int(row["frame_index"]) for row in post],
        "pre_keyframe_count": len(pre),
        "post_keyframe_count": len(post),
        "uses_interpolated_derivatives": False,
        "time_source": (
            "nuscenes_sample_timestamp"
            if frame_times_s is not None
            else "dense_frame_period_fallback"
        ),
        "join_geometry": join,
    }
    support = len(pre) >= count and len(post) >= count
    if not support:
        checks = {
            "keyframe_support": False,
            "minimum_motion_speed": False,
            "not_normal_turn": False,
            "lane_preference_flip": False,
            "global_lateral_motion": False,
            "parallel_lane_change_evidence": False,
            "branch_merge_evidence": False,
            "subject_maneuver": False,
            "kinematic_sanity": False,
        }
        return {
            **base,
            "status": "UNKNOWN",
            "reason": "insufficient_annotation_keyframes",
            "checks": checks,
            "physical_motion_pass": False,
        }

    combined = sorted(pre + post, key=lambda row: int(row["frame_index"]))
    pre_course = _course_heading(pre)
    post_course = _course_heading(post)
    if pre_course is None or post_course is None:
        checks = {
            "keyframe_support": True,
            "minimum_motion_speed": False,
            "not_normal_turn": False,
            "lane_preference_flip": False,
            "global_lateral_motion": False,
            "parallel_lane_change_evidence": False,
            "branch_merge_evidence": False,
            "subject_maneuver": False,
            "kinematic_sanity": False,
        }
        return {
            **base,
            "status": "UNKNOWN",
            "reason": "stationary_or_degenerate_course",
            "checks": checks,
            "physical_motion_pass": False,
        }

    reference_heading = circular_mean([pre_course, post_course])
    tangent = np.asarray(
        [math.cos(reference_heading), math.sin(reference_heading)], dtype=float
    )
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    positions = np.asarray([row["xy"] for row in combined], dtype=float)
    lateral = positions @ normal
    longitudinal = positions @ tangent
    pre_lateral = positions[: len(pre)] @ normal
    post_lateral = positions[len(pre) :] @ normal
    signed_lateral_change = float(np.median(post_lateral) - np.median(pre_lateral))
    lateral_span = float(np.max(lateral) - np.min(lateral))
    longitudinal_progress = float(np.median(longitudinal[len(pre) :]) - np.median(longitudinal[: len(pre)]))

    segment_lateral_speeds = []
    speeds = []
    velocity_vectors = []
    velocity_mid_times = []
    course_segments = []
    for left, right in zip(combined, combined[1:]):
        left_frame = int(left["frame_index"])
        right_frame = int(right["frame_index"])
        if frame_times_s is None:
            left_time = left_frame * period
            right_time = right_frame * period
        else:
            left_time = float(frame_times_s[left_frame])
            right_time = float(frame_times_s[right_frame])
        dt = right_time - left_time
        if dt <= 0:
            continue
        delta = np.asarray(right["xy"], dtype=float) - np.asarray(left["xy"], dtype=float)
        velocity = delta / dt
        velocity_vectors.append(velocity)
        velocity_mid_times.append((left_time + right_time) / 2.0)
        speeds.append(float(np.linalg.norm(velocity)))
        segment_lateral_speeds.append(float(np.dot(velocity, normal)))
        if float(np.linalg.norm(delta)) > 1e-6:
            course_segments.append(float(math.atan2(delta[1], delta[0])))
    accelerations = []
    for index, (left, right) in enumerate(
        zip(velocity_vectors, velocity_vectors[1:])
    ):
        dt = velocity_mid_times[index + 1] - velocity_mid_times[index]
        if dt > 0:
            accelerations.append(float(np.linalg.norm(right - left) / dt))
    yaw_rates = []
    for left, right in zip(combined, combined[1:]):
        left_frame = int(left["frame_index"])
        right_frame = int(right["frame_index"])
        if frame_times_s is None:
            dt = (right_frame - left_frame) * period
        else:
            dt = float(frame_times_s[right_frame]) - float(
                frame_times_s[left_frame]
            )
        if dt > 0:
            yaw_rates.append(
                math.degrees(angle_error(float(left["yaw"]), float(right["yaw"]))) / dt
            )

    direction = 1.0 if signed_lateral_change >= 0 else -1.0
    directed = [direction * value for value in np.diff(lateral)]
    total_lateral_variation = float(sum(abs(value) for value in directed))
    direction_consistency = (
        float(sum(max(0.0, value) for value in directed) / total_lateral_variation)
        if total_lateral_variation > 1e-9
        else 0.0
    )

    pre_source = _polyline_distance_rows(pre, lane_index, source_token)
    pre_target = _polyline_distance_rows(pre, lane_index, target_token)
    post_source = _polyline_distance_rows(post, lane_index, source_token)
    post_target = _polyline_distance_rows(post, lane_index, target_token)
    pre_source_distance = float(np.median(pre_source))
    pre_target_distance = float(np.median(pre_target))
    post_source_distance = float(np.median(post_source))
    post_target_distance = float(np.median(post_target))
    pre_preference = pre_target_distance - pre_source_distance
    post_preference = post_source_distance - post_target_distance

    net_course_change = math.degrees(angle_error(pre_course, post_course))
    net_yaw_change = math.degrees(
        angle_error(
            circular_mean([float(row["yaw"]) for row in pre]),
            circular_mean([float(row["yaw"]) for row in post]),
        )
    )
    max_course_deviation = (
        max(
            math.degrees(angle_error(value, reference_heading))
            for value in course_segments
        )
        if course_segments
        else None
    )
    median_speed = _median(speeds)
    peak_lateral_speed = _max_or_none(abs(value) for value in segment_lateral_speeds)
    max_acceleration = _max_or_none(accelerations)
    max_yaw_rate = _max_or_none(yaw_rates)

    minimum_motion_speed = (
        median_speed is not None
        and median_speed >= float(config["min_median_speed_mps"])
    )
    not_normal_turn = (
        net_course_change <= float(config["max_net_course_change_deg"])
        and net_yaw_change <= float(config["max_net_yaw_change_deg"])
        and (
            max_course_deviation is None
            or max_course_deviation <= float(config["max_course_deviation_deg"])
        )
        and join["source_target_approach_heading_error_deg"]
        <= float(config["max_lane_join_heading_error_deg"])
    )
    global_lateral_motion = (
        abs(signed_lateral_change)
        >= float(config["min_lateral_displacement_m"])
        and lateral_span >= float(config["min_lateral_span_m"])
        and peak_lateral_speed is not None
        and peak_lateral_speed >= float(config["min_peak_lateral_speed_mps"])
        and peak_lateral_speed <= float(config["max_peak_lateral_speed_mps"])
        and direction_consistency
        >= float(config["min_lateral_direction_consistency"])
        and longitudinal_progress >= float(config["min_longitudinal_progress_m"])
    )
    lane_preference_flip = (
        pre_source_distance <= float(config["max_source_distance_m"])
        and post_target_distance <= float(config["max_target_distance_m"])
        and pre_preference >= float(config["min_lane_preference_margin_m"])
        and post_preference >= float(config["min_lane_preference_margin_m"])
    )
    topology_type = str(topology.get("type", ""))
    parallel_lane_change = (
        topology_type == "lane_change"
        and not join["directed_connected"]
        and lane_preference_flip
    )
    alternative_error = join["best_alternative_incoming_heading_error_deg"]
    alignment_advantage = (
        join["source_target_approach_heading_error_deg"] - alternative_error
        if alternative_error is not None
        else None
    )
    branch_merge = (
        topology_type == "merge"
        and join["directed_connected"]
        and join["target_incoming_count"]
        >= int(config["merge_min_target_incoming_lanes"])
        and alignment_advantage is not None
        and alignment_advantage
        >= float(config["min_merge_branch_alignment_disadvantage_deg"])
        and join["source_target_approach_heading_error_deg"]
        >= float(config["min_merge_branch_convergence_angle_deg"])
        and join["source_target_approach_heading_error_deg"]
        <= float(config["max_lane_join_heading_error_deg"])
        and pre_source_distance <= float(config["max_source_distance_m"])
        and post_target_distance <= float(config["max_target_distance_m"])
    )
    kinematic_sanity = (
        (
            max_acceleration is None
            or max_acceleration <= float(config["max_acceleration_mps2"])
        )
        and (
            max_yaw_rate is None
            or max_yaw_rate <= float(config["max_yaw_rate_deg_s"])
        )
    )
    checks = {
        "keyframe_support": support,
        "minimum_motion_speed": minimum_motion_speed,
        "not_normal_turn": not_normal_turn,
        "lane_preference_flip": lane_preference_flip,
        # 全局横移只作独立诊断：弯曲/收缩道路上的真实 road-relative lane crossing
        # 不一定表现为世界坐标中的横移。
        "global_lateral_motion": global_lateral_motion,
        "parallel_lane_change_evidence": parallel_lane_change,
        "branch_merge_evidence": branch_merge,
        "subject_maneuver": parallel_lane_change or branch_merge,
        "kinematic_sanity": kinematic_sanity,
    }
    required_checks = (
        "keyframe_support",
        "minimum_motion_speed",
        "not_normal_turn",
        "subject_maneuver",
        "kinematic_sanity",
    )
    physical_motion_pass = all(checks[name] for name in required_checks)
    if parallel_lane_change:
        maneuver_mode = "parallel_lane_change"
    elif branch_merge:
        maneuver_mode = "converging_branch_merge"
    else:
        maneuver_mode = "no_supported_lateral_maneuver"
    return {
        **base,
        "status": "PASS" if physical_motion_pass else "FAIL",
        "maneuver_mode": maneuver_mode,
        "pre_course_heading_deg": math.degrees(pre_course),
        "post_course_heading_deg": math.degrees(post_course),
        "reference_heading_deg": math.degrees(reference_heading),
        "net_course_change_deg": net_course_change,
        "net_yaw_change_deg": net_yaw_change,
        "max_course_deviation_deg": max_course_deviation,
        "signed_lateral_displacement_m": signed_lateral_change,
        "lateral_span_m": lateral_span,
        "lateral_direction_consistency": direction_consistency,
        "peak_lateral_speed_mps": peak_lateral_speed,
        "median_speed_mps": median_speed,
        "max_acceleration_mps2": max_acceleration,
        "max_yaw_rate_deg_s": max_yaw_rate,
        "longitudinal_progress_m": longitudinal_progress,
        "pre_source_distance_m": pre_source_distance,
        "pre_target_distance_m": pre_target_distance,
        "post_source_distance_m": post_source_distance,
        "post_target_distance_m": post_target_distance,
        "pre_lane_preference_margin_m": pre_preference,
        "post_lane_preference_margin_m": post_preference,
        "merge_branch_alignment_disadvantage_deg": alignment_advantage,
        "segment_lateral_speeds_mps": segment_lateral_speeds,
        "checks": checks,
        "required_checks": list(required_checks),
        "physical_motion_pass": physical_motion_pass,
    }


def lane_keeping_features(
    rows: list[dict],
    start_frame: int,
    end_frame: int,
    lane_index,
    lane_token: str,
    config: dict,
    frame_times_s: dict[int, float] | None = None,
) -> dict:
    """验证 same-actor negative 窗口确实是有运动的稳定 lane keeping。"""
    stride = int(config["annotation_keyframe_stride"])
    period = float(config["dense_frame_period_s"])
    keyframes = annotation_keyframes(rows, start_frame, end_frame, stride)
    minimum = int(config["negative_min_keyframes"])
    if len(keyframes) < minimum:
        return {
            "status": "UNKNOWN",
            "reason": "insufficient_annotation_keyframes",
            "keyframe_indices": [int(row["frame_index"]) for row in keyframes],
            "checks": {"keyframe_support": False},
        }
    projections = [
        project_to_polyline(
            row["xy"],
            lane_index.centerlines[lane_token],
            lane_index.arc_lengths[lane_token],
        )
        for row in keyframes
    ]
    speeds = []
    accelerations = []
    velocities = []
    mid_times = []
    for left, right in zip(keyframes, keyframes[1:]):
        left_frame = int(left["frame_index"])
        right_frame = int(right["frame_index"])
        if frame_times_s is None:
            left_time = left_frame * period
            right_time = right_frame * period
        else:
            left_time = float(frame_times_s[left_frame])
            right_time = float(frame_times_s[right_frame])
        dt = right_time - left_time
        if dt <= 0:
            continue
        velocity = (
            np.asarray(right["xy"], dtype=float)
            - np.asarray(left["xy"], dtype=float)
        ) / dt
        velocities.append(velocity)
        mid_times.append((left_time + right_time) / 2.0)
        speeds.append(float(np.linalg.norm(velocity)))
    for index, (left, right) in enumerate(zip(velocities, velocities[1:])):
        dt = mid_times[index + 1] - mid_times[index]
        if dt > 0:
            accelerations.append(float(np.linalg.norm(right - left) / dt))
    signed = [float(value["signed_lateral_m"]) for value in projections]
    distances = [float(value["distance_m"]) for value in projections]
    heading_errors = [
        math.degrees(angle_error(float(row["yaw"]), float(projection["heading_rad"])))
        for row, projection in zip(keyframes, projections)
    ]
    lateral_span = max(signed) - min(signed)
    median_speed = _median(speeds)
    max_acceleration = _max_or_none(accelerations)
    checks = {
        "keyframe_support": len(keyframes) >= minimum,
        "moving": (
            median_speed is not None
            and median_speed >= float(config["min_median_speed_mps"])
        ),
        "centerline_distance": max(distances)
        <= float(config["negative_max_centerline_distance_m"]),
        "lateral_stability": lateral_span
        <= float(config["negative_max_lateral_span_m"]),
        "heading_stability": max(heading_errors)
        <= float(config["negative_max_heading_error_deg"]),
        "kinematic_sanity": (
            max_acceleration is None
            or max_acceleration <= float(config["max_acceleration_mps2"])
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "keyframe_indices": [int(row["frame_index"]) for row in keyframes],
        "median_speed_mps": median_speed,
        "max_acceleration_mps2": max_acceleration,
        "lateral_span_m": lateral_span,
        "max_centerline_distance_m": max(distances),
        "max_heading_error_deg": max(heading_errors),
        "checks": checks,
    }
