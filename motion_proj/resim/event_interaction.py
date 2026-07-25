"""分支安全、跨关键帧持续的 target-corridor front/rear 关系。"""
from __future__ import annotations

import math

import numpy as np

from motion_proj.resim.event_kinematics import angle_error, project_to_polyline


def _edge_geometry(lane_index, left: str, right: str) -> dict:
    left_line = lane_index.centerlines[left]
    right_line = lane_index.centerlines[right]
    return {
        "heading_error_deg": math.degrees(
            angle_error(float(left_line[-1, 2]), float(right_line[0, 2]))
        ),
        "endpoint_gap_m": float(
            np.linalg.norm(left_line[-1, :2] - right_line[0, :2])
        ),
    }


def _extend_chain(
    lane_index,
    token: str,
    direction: str,
    hops: int,
    max_heading_error_deg: float,
    max_endpoint_gap_m: float,
) -> tuple[list[str], list[dict]]:
    connectivity = lane_index.nmap.connectivity
    chain: list[str] = []
    edges: list[dict] = []
    current = token
    visited = {token}
    for _ in range(hops):
        neighbors = [
            value
            for value in connectivity.get(current, {}).get(direction, [])
            if value in lane_index.centerlines and value not in visited
        ]
        scored = []
        for neighbor in neighbors:
            left, right = (
                (current, neighbor)
                if direction == "outgoing"
                else (neighbor, current)
            )
            geometry = _edge_geometry(lane_index, left, right)
            if (
                geometry["heading_error_deg"] <= max_heading_error_deg
                and geometry["endpoint_gap_m"] <= max_endpoint_gap_m
            ):
                scored.append((geometry["heading_error_deg"], geometry["endpoint_gap_m"], neighbor, geometry))
        if not scored:
            break
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        _, _, best, geometry = scored[0]
        runner_up_error = scored[1][0] if len(scored) > 1 else None
        edge = {
            "from_token": best if direction == "incoming" else current,
            "to_token": current if direction == "incoming" else best,
            **geometry,
            "candidate_count": len(scored),
            "runner_up_heading_error_deg": runner_up_error,
            "heading_margin_to_runner_up_deg": (
                runner_up_error - geometry["heading_error_deg"]
                if runner_up_error is not None
                else None
            ),
        }
        chain.append(best)
        edges.append(edge)
        visited.add(best)
        current = best
    return chain, edges


def build_branch_safe_corridor(lane_index, target_token: str, config: dict) -> dict:
    """沿最连续的单一有向分支建立 target-centered s 轴。"""
    hops = int(config["graph_hops"])
    max_heading = float(config["max_edge_heading_error_deg"])
    max_gap = float(config["max_edge_endpoint_gap_m"])
    incoming, incoming_edges = _extend_chain(
        lane_index,
        target_token,
        "incoming",
        hops,
        max_heading,
        max_gap,
    )
    outgoing, outgoing_edges = _extend_chain(
        lane_index,
        target_token,
        "outgoing",
        hops,
        max_heading,
        max_gap,
    )
    offsets = {target_token: 0.0}
    cursor = 0.0
    for token in incoming:
        length = float(lane_index.arc_lengths[token][-1])
        cursor -= length
        offsets[token] = cursor
    cursor = float(lane_index.arc_lengths[target_token][-1])
    for token in outgoing:
        offsets[token] = cursor
        cursor += float(lane_index.arc_lengths[token][-1])
    return {
        "target_token": target_token,
        "incoming_tokens_nearest_first": incoming,
        "outgoing_tokens_nearest_first": outgoing,
        "offsets_m": offsets,
        "incoming_edges": incoming_edges,
        "outgoing_edges": outgoing_edges,
    }


def _corridor_position(row: dict, lane_index, offsets: dict[str, float]) -> dict | None:
    token = row.get("lane_token")
    if token not in offsets:
        return None
    projection = project_to_polyline(
        row["xy"], lane_index.centerlines[token], lane_index.arc_lengths[token]
    )
    return {
        "token": token,
        "s_m": float(offsets[token] + projection["s_m"]),
        "lane_heading_rad": float(projection["heading_rad"]),
        "lane_distance_m": float(projection["distance_m"]),
    }


def _longitudinal_half_extent(row: dict, lane_heading_rad: float) -> float:
    length, width = (float(value) for value in row["dimensions_lwh"][:2])
    delta = angle_error(float(row["yaw"]), lane_heading_rad)
    return 0.5 * (
        length * abs(math.cos(delta)) + width * abs(math.sin(delta))
    )


def _frame_relation(
    actor_id: int,
    frame: int,
    matches_by_actor: dict[int, dict[int, dict]],
    lane_index,
    corridor: dict,
    config: dict,
) -> dict:
    offsets = corridor["offsets_m"]
    subject_row = matches_by_actor.get(actor_id, {}).get(frame)
    if subject_row is None:
        return {"frame": frame, "status": "UNKNOWN", "reason": "subject_missing"}
    subject = _corridor_position(subject_row, lane_index, offsets)
    if subject is None:
        return {
            "frame": frame,
            "status": "UNKNOWN",
            "reason": "subject_outside_target_corridor",
        }
    max_heading = float(config["max_actor_heading_error_deg"])
    subject_extent = _longitudinal_half_extent(
        subject_row, subject["lane_heading_rad"]
    )
    neighbors = []
    for other_id, rows in sorted(matches_by_actor.items()):
        if other_id == actor_id:
            continue
        row = rows.get(frame)
        if row is None:
            continue
        position = _corridor_position(row, lane_index, offsets)
        if position is None:
            continue
        heading_error = math.degrees(
            angle_error(float(row["yaw"]), position["lane_heading_rad"])
        )
        if heading_error > max_heading:
            continue
        delta_s = position["s_m"] - subject["s_m"]
        other_extent = _longitudinal_half_extent(
            row, position["lane_heading_rad"]
        )
        bumper_gap = abs(delta_s) - subject_extent - other_extent
        neighbors.append(
            {
                "actor_id": int(other_id),
                "lane_token": position["token"],
                "center_delta_s_m": float(delta_s),
                "bumper_gap_m": float(bumper_gap),
                "heading_error_deg": heading_error,
                "same_exact_target_token": (
                    position["token"] == corridor["target_token"]
                ),
            }
        )
    fronts = [row for row in neighbors if row["center_delta_s_m"] > 0]
    rears = [row for row in neighbors if row["center_delta_s_m"] < 0]
    front = min(fronts, key=lambda row: row["center_delta_s_m"], default=None)
    rear = max(rears, key=lambda row: row["center_delta_s_m"], default=None)
    return {
        "frame": frame,
        "status": "OBSERVED",
        "subject_lane_token": subject["token"],
        "subject_s_m": subject["s_m"],
        "front": front,
        "rear": rear,
        "neighbor_count": len(neighbors),
    }


def _speed_along_lane(
    actor_id: int,
    frame: int,
    matches_by_actor: dict[int, dict[int, dict]],
    frame_times_s: dict[int, float],
    lane_heading_rad: float,
    stride: int,
) -> float | None:
    rows = matches_by_actor.get(actor_id, {})
    left = rows.get(frame - stride)
    right = rows.get(frame + stride)
    if left is None or right is None:
        return None
    if frame - stride not in frame_times_s or frame + stride not in frame_times_s:
        return None
    dt = frame_times_s[frame + stride] - frame_times_s[frame - stride]
    if dt <= 0:
        return None
    velocity = (
        np.asarray(right["xy"], dtype=float) - np.asarray(left["xy"], dtype=float)
    ) / dt
    tangent = np.asarray(
        [math.cos(lane_heading_rad), math.sin(lane_heading_rad)]
    )
    return float(np.dot(velocity, tangent))


def temporal_relation(
    actor_id: int,
    relation_frame: int,
    target_token: str,
    matches_by_actor: dict[int, dict[int, dict]],
    lane_index,
    frame_times_s: dict[int, float],
    config: dict,
) -> dict:
    """要求 front/rear 身份和物理 gap 在多个 2 Hz 关键帧上持续。"""
    stride = int(config["annotation_keyframe_stride"])
    frame_count = int(config["temporal_keyframe_count"])
    if frame_count % 2 != 1:
        raise ValueError("temporal_keyframe_count 必须为奇数")
    center = int(round(relation_frame / stride) * stride)
    radius = frame_count // 2
    frames = [center + offset * stride for offset in range(-radius, radius + 1)]
    corridor = build_branch_safe_corridor(lane_index, target_token, config)
    per_frame = [
        _frame_relation(
            actor_id,
            frame,
            matches_by_actor,
            lane_index,
            corridor,
            config,
        )
        for frame in frames
    ]
    center_row = next((row for row in per_frame if row["frame"] == center), None)
    if (
        center_row is None
        or center_row["status"] != "OBSERVED"
        or center_row.get("subject_lane_token") != target_token
    ):
        return {
            "status": "UNKNOWN",
            "reason": "subject_not_on_exact_target_at_center_keyframe",
            "center_frame": center,
            "frames": frames,
            "corridor": corridor,
            "per_frame": per_frame,
        }
    front_id = (
        int(center_row["front"]["actor_id"]) if center_row.get("front") else None
    )
    rear_id = int(center_row["rear"]["actor_id"]) if center_row.get("rear") else None
    if front_id is None or rear_id is None:
        return {
            "status": "FAIL",
            "reason": "missing_front_or_rear_at_center_keyframe",
            "center_frame": center,
            "frames": frames,
            "corridor": corridor,
            "per_frame": per_frame,
        }

    supporting = []
    for row in per_frame:
        front = row.get("front")
        rear = row.get("rear")
        if (
            row.get("status") == "OBSERVED"
            and front is not None
            and rear is not None
            and int(front["actor_id"]) == front_id
            and int(rear["actor_id"]) == rear_id
        ):
            supporting.append(row)
    min_frames = int(config["min_identity_support_keyframes"])
    support_fraction = len(supporting) / len(frames)
    min_fraction = float(config["min_identity_support_fraction"])
    min_gap = float(config["min_bumper_gap_m"])
    max_gap = float(config["max_bumper_gap_m"])
    gaps_pass = all(
        min_gap <= float(row["front"]["bumper_gap_m"]) <= max_gap
        and min_gap <= float(row["rear"]["bumper_gap_m"]) <= max_gap
        for row in supporting
    )
    persistence_pass = (
        len(supporting) >= min_frames and support_fraction >= min_fraction
    )

    subject_row = matches_by_actor[actor_id][center]
    subject_position = _corridor_position(
        subject_row, lane_index, corridor["offsets_m"]
    )
    assert subject_position is not None
    subject_speed = _speed_along_lane(
        actor_id,
        center,
        matches_by_actor,
        frame_times_s,
        subject_position["lane_heading_rad"],
        stride,
    )
    front_speed = _speed_along_lane(
        front_id,
        center,
        matches_by_actor,
        frame_times_s,
        subject_position["lane_heading_rad"],
        stride,
    )
    rear_speed = _speed_along_lane(
        rear_id,
        center,
        matches_by_actor,
        frame_times_s,
        subject_position["lane_heading_rad"],
        stride,
    )
    front_gap = float(center_row["front"]["bumper_gap_m"])
    rear_gap = float(center_row["rear"]["bumper_gap_m"])
    front_closing = (
        subject_speed - front_speed
        if subject_speed is not None and front_speed is not None
        else None
    )
    rear_closing = (
        rear_speed - subject_speed
        if rear_speed is not None and subject_speed is not None
        else None
    )
    front_ttc = (
        front_gap / front_closing
        if front_closing is not None and front_closing > 1e-3
        else None
    )
    rear_ttc = (
        rear_gap / rear_closing
        if rear_closing is not None and rear_closing > 1e-3
        else None
    )
    checks = {
        "center_front_and_rear": True,
        "same_identity_persistence": persistence_pass,
        "bumper_gaps_in_range": gaps_pass,
        "subject_exact_target_at_center": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "reason": None if all(checks.values()) else "temporal_or_gap_gate_failed",
        "center_frame": center,
        "frames": frames,
        "front_actor_id": front_id,
        "rear_actor_id": rear_id,
        "identity_support_keyframes": len(supporting),
        "identity_support_fraction": support_fraction,
        "front_center_gap_m": float(center_row["front"]["center_delta_s_m"]),
        "rear_center_gap_m": -float(center_row["rear"]["center_delta_s_m"]),
        "front_bumper_gap_m": front_gap,
        "rear_bumper_gap_m": rear_gap,
        "subject_longitudinal_speed_mps": subject_speed,
        "front_longitudinal_speed_mps": front_speed,
        "rear_longitudinal_speed_mps": rear_speed,
        "front_closing_speed_mps": front_closing,
        "rear_closing_speed_mps": rear_closing,
        "front_ttc_s": front_ttc,
        "rear_ttc_s": rear_ttc,
        "corridor": corridor,
        "per_frame": per_frame,
        "checks": checks,
    }
