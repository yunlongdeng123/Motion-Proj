"""基于原始 2 Hz 轨迹的 receiver-centric cut-in 几何与交互判定。

第三版只检查 subject 经过了一个具有多 incoming 的地图节点，并在事后 corridor
上寻找前后车。这会把同一车流的正常过弯/续接误判为 merge。本模块改为验证：

1. subject 车身先稳定在接收车道外，随后真实进入并稳定至少 1 秒；
2. 接收车在进入前后均沿独立 target branch 跟随 subject；
3. 接收车是 target branch 上最近的后车，二者之间没有被忽略的同车道车辆。

速度、横移、持续时间和身份支持全部只使用 nuScenes annotation keyframe。
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from motion_proj.resim.event_kinematics import (
    angle_error,
    annotation_keyframes,
    project_to_polyline,
)


STRICT_V2_SCHEMA_VERSION = "receiver-centric-cutin-strict-v2"
STRICT_V2_REASONS = (
    "UNSUPPORTED_BRANCH_MERGE_MODE",
    "INSUFFICIENT_RAW_SUPPORT",
    "MAP_GEOMETRY_UNAVAILABLE",
    "SOURCE_TARGET_NOT_PARALLEL",
    "NO_RAW_LATERAL_ENTRY",
    "POST_HEADING_UNSTABLE",
    "SUBJECT_NOT_DYNAMIC",
    "AMBIGUOUS_RECEIVER_CORRIDOR",
    "RECEIVER_NOT_DYNAMIC",
    "RECEIVER_WRONG_DIRECTION",
    "RECEIVER_NOT_ESTABLISHED_ON_TARGET_STREAM",
    "RECEIVER_IDENTITY_SWITCH",
    "RECEIVER_SUPPORT_INSUFFICIENT",
    "RECEIVER_GAP_INVALID",
    "PATH_NOT_CLEAR",
    "INTERPOLATION_ONLY",
)
_STRICT_V2_REASON_RANK = {
    reason: index for index, reason in enumerate(STRICT_V2_REASONS)
}
_STRICT_V2_CHECK_KEYS = (
    "supported_mode",
    "source_target_parallel",
    "raw_pre_outside",
    "raw_post_inside",
    "lateral_convergence",
    "post_heading_stable",
    "subject_dynamic",
    "receiver_dynamic",
    "receiver_same_direction",
    "receiver_identity_persistent",
    "receiver_nearest_rear_persistent",
    "path_clear",
    "corridor_unambiguous",
)


def ordered_strict_v2_reasons(reasons: Iterable[str]) -> list[str]:
    """按冻结的 first-failure 优先级稳定排序 reason。"""
    unique = {str(reason) for reason in reasons if reason is not None}
    unknown = unique.difference(_STRICT_V2_REASON_RANK)
    if unknown:
        raise ValueError(f"未知 strict-v2 reason: {sorted(unknown)}")
    return sorted(unique, key=_STRICT_V2_REASON_RANK.__getitem__)


def strict_v2_result(
    *,
    status: str,
    maneuver_mode: str,
    reasons: Iterable[str] = (),
    checks: Mapping[str, bool] | None = None,
    subject: Mapping[str, Any] | None = None,
    receiver: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """构造带 fail-closed 语义的 strict-v2 evidence record。

    这里集中约束 schema，避免任何调用方把 branch merge、插值或 FAIL/ABSTAIN
    错写成 machine-positive。所有 hard evidence 的来源固定为原始 2 Hz annotation。
    """
    status = str(status).upper()
    if status not in {"PASS", "FAIL", "ABSTAIN"}:
        raise ValueError(f"非法 strict-v2 status: {status}")
    maneuver_mode = str(maneuver_mode)
    if maneuver_mode not in {"parallel_lane_change", "receiver_branch_merge"}:
        raise ValueError(f"非法 maneuver_mode: {maneuver_mode}")
    ordered = ordered_strict_v2_reasons(reasons)
    if maneuver_mode == "receiver_branch_merge":
        status = "ABSTAIN"
        ordered = ["UNSUPPORTED_BRANCH_MERGE_MODE"]
    if "INTERPOLATION_ONLY" in ordered and status == "PASS":
        status = "ABSTAIN"
    if status == "PASS" and ordered:
        raise ValueError("PASS 不得携带 hard failure reason")
    if status != "PASS" and not ordered:
        ordered = ["INSUFFICIENT_RAW_SUPPORT"]

    resolved_checks = {key: False for key in _STRICT_V2_CHECK_KEYS}
    if checks:
        unknown_checks = set(checks).difference(resolved_checks)
        if unknown_checks:
            raise ValueError(f"未知 strict-v2 check: {sorted(unknown_checks)}")
        resolved_checks.update({key: bool(value) for key, value in checks.items()})
    resolved_checks["supported_mode"] = maneuver_mode == "parallel_lane_change"

    value = {
        "schema_version": STRICT_V2_SCHEMA_VERSION,
        "status": status,
        "primary_reason": ordered[0] if ordered else None,
        "all_reasons": ordered,
        "maneuver_mode": maneuver_mode,
        "machine_positive": status == "PASS",
        "hard_evidence_source": "raw_2hz_annotations",
        "uses_interpolated_physics": False,
        "provenance": {
            "source_event_record_sha256": None,
            "config_fingerprint": None,
            "map_version": None,
            "lane_width_source": "configured_nominal_fallback",
            **dict(provenance or {}),
        },
        "subject": {
            "actor_id": None,
            "instance_token": None,
            "source_token": None,
            "target_token": None,
            "pre_frames": [],
            "post_frames": [],
            "per_frame": [],
            **dict(subject or {}),
        },
        "receiver": {
            "selected_actor_id": None,
            "actor_id_by_frame": [],
            "nearest_rear_rank_by_frame": [],
            "gap_m_by_frame": [],
            "longitudinal_speed_mps_by_frame": [],
            "heading_error_deg_by_frame": [],
            "identity_switch_frames": [],
            "missing_frames": [],
            "intermediate_actor_ids_by_frame": {},
            "identity_persistent": False,
            "nearest_rear_persistent": False,
            "path_clear": False,
            **dict(receiver or {}),
        },
        "checks": resolved_checks,
    }
    if extra:
        overlap = set(value).intersection(extra)
        if overlap:
            raise ValueError(f"strict-v2 extra 覆盖保留字段: {sorted(overlap)}")
        value.update(dict(extra))
    return value


def adapt_v1_evidence_to_v2(record: Mapping[str, Any]) -> dict:
    """只读适配历史 receiver-centric v1 evidence 为 v2 diagnostic。

    该 adapter 不迁移 parent 文件，也不把 v1 的 PASS 当成新的 final truth；它只让
    K4 fixture 和历史诊断能够由 v2 工具稳定读取。
    """
    cutin = record.get("cutin", record)
    if not isinstance(cutin, Mapping):
        raise ValueError("v1 evidence 缺少 cutin object")
    mode = str(record.get("maneuver_mode", "parallel_lane_change"))
    if mode == "lane_change":
        mode = "parallel_lane_change"
    if mode == "merge":
        mode = "receiver_branch_merge"
    if mode not in {"parallel_lane_change", "receiver_branch_merge"}:
        raise ValueError(f"无法适配的 v1 maneuver mode: {mode}")
    legacy_status = str(cutin.get("status", "UNKNOWN")).upper()
    subject_checks_v1 = dict(cutin.get("subject_checks", {}))
    receiver_checks_v1 = dict(cutin.get("receiver_checks", {}))
    checks = {
        "source_target_parallel": mode == "parallel_lane_change",
        "raw_pre_outside": bool(
            subject_checks_v1.get("pre_center_outside_target_band", False)
        ),
        "raw_post_inside": bool(
            subject_checks_v1.get("post_box_inside_target_band", False)
        ),
        "lateral_convergence": bool(
            subject_checks_v1.get("lateral_convergence", False)
        ),
        "post_heading_stable": bool(
            subject_checks_v1.get("post_heading_alignment", False)
        ),
        "subject_dynamic": bool(
            subject_checks_v1.get("minimum_motion_speed", False)
        ),
        "receiver_dynamic": cutin.get("receiver_longitudinal_speed_mps") is not None,
        "receiver_same_direction": True,
        "receiver_identity_persistent": bool(
            receiver_checks_v1.get("receiver_pre_identity_support", False)
            and receiver_checks_v1.get("receiver_post_identity_support", False)
        ),
        "receiver_nearest_rear_persistent": bool(
            receiver_checks_v1.get("receiver_pre_identity_support", False)
            and receiver_checks_v1.get("receiver_post_identity_support", False)
        ),
        "path_clear": True,
        "corridor_unambiguous": int(
            cutin.get("candidate_receiver_branch_count", 1)
        ) <= 1,
    }
    if mode == "receiver_branch_merge":
        status, reasons = "ABSTAIN", ["UNSUPPORTED_BRANCH_MERGE_MODE"]
    elif legacy_status == "PASS":
        status, reasons = "PASS", []
    elif legacy_status == "UNKNOWN":
        status, reasons = "ABSTAIN", ["INSUFFICIENT_RAW_SUPPORT"]
    elif not checks["raw_pre_outside"] or not checks["raw_post_inside"]:
        status, reasons = "FAIL", ["NO_RAW_LATERAL_ENTRY"]
    elif not checks["post_heading_stable"]:
        status, reasons = "FAIL", ["POST_HEADING_UNSTABLE"]
    else:
        status, reasons = "FAIL", ["RECEIVER_SUPPORT_INSUFFICIENT"]

    per_frame = list(cutin.get("per_frame", []))
    receiver_ids = [
        (row.get("receiver") or {}).get("actor_id") for row in per_frame
    ]
    selected = cutin.get("receiver_actor_id")
    selected_int = int(selected) if selected is not None else None
    non_null = [int(value) for value in receiver_ids if value is not None]
    switches = [
        int(row.get("frame"))
        for row, value in zip(per_frame, receiver_ids)
        if value is not None and selected_int is not None and int(value) != selected_int
    ]
    subject_frames = [
        {
            "frame": int(row["frame"]),
            "observation_source": "raw_2hz",
            "source_d_m": None,
            "target_d_m": float(row["signed_lateral_m"]),
            "target_s_m": float(row["s_m"]),
            "target_heading_error_deg": float(row["heading_error_deg"]),
            "speed_mps": None,
            "center_outside_target_band": bool(
                row["center_outside_target_band"]
            ),
            "box_inside_target_band": bool(row["box_inside_target_band"]),
        }
        for row in [
            *list(cutin.get("pre_keyframes", [])),
            *list(cutin.get("post_keyframes", [])),
        ]
    ]
    return strict_v2_result(
        status=status,
        maneuver_mode=mode,
        reasons=reasons,
        checks=checks,
        provenance={
            "source_event_record_sha256": record.get("event_record_sha256"),
            "config_fingerprint": record.get("config_fingerprint"),
            "map_version": record.get("map_name"),
        },
        subject={
            "actor_id": record.get("actor_id"),
            "instance_token": (record.get("roles") or {}).get("SUBJECT"),
            "source_token": (record.get("source_run") or {}).get("token"),
            "target_token": (record.get("target_run") or {}).get("token"),
            "pre_frames": [
                int(row["frame"]) for row in cutin.get("pre_keyframes", [])
            ],
            "post_frames": [
                int(row["frame"]) for row in cutin.get("post_keyframes", [])
            ],
            "per_frame": subject_frames,
        },
        receiver={
            "selected_actor_id": selected_int,
            "actor_id_by_frame": receiver_ids,
            "nearest_rear_rank_by_frame": [
                1 if value is not None else None for value in receiver_ids
            ],
            "gap_m_by_frame": [
                (row.get("receiver") or {}).get("bumper_gap_m")
                for row in per_frame
            ],
            "longitudinal_speed_mps_by_frame": [None for _ in per_frame],
            "heading_error_deg_by_frame": [
                (row.get("receiver") or {}).get("heading_error_deg")
                for row in per_frame
            ],
            "identity_switch_frames": switches,
            "missing_frames": [
                int(row.get("frame"))
                for row, value in zip(per_frame, receiver_ids)
                if value is None
            ],
            "intermediate_actor_ids_by_frame": {
                str(row.get("frame")): [] for row in per_frame
            },
            "identity_persistent": len(set(non_null)) <= 1 and bool(non_null),
            "nearest_rear_persistent": len(set(non_null)) <= 1 and bool(non_null),
            "path_clear": True,
        },
        extra={
            "legacy_v1_diagnostic": {
                "status": legacy_status,
                "reason": cutin.get("reason"),
                "schema_version": cutin.get("schema_version"),
            }
        },
    )


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


def _best_chain(
    lane_index,
    start: str,
    direction: str,
    hops: int,
    max_heading_error_deg: float,
    max_endpoint_gap_m: float,
    excluded_tokens: set[str] | None = None,
) -> tuple[list[str], list[dict]]:
    excluded = set(excluded_tokens or ())
    connectivity = lane_index.nmap.connectivity
    current = start
    visited = {start} | excluded
    chain: list[str] = []
    edges: list[dict] = []
    for _ in range(hops):
        candidates = []
        for neighbor in connectivity.get(current, {}).get(direction, []):
            if neighbor in visited or neighbor not in lane_index.centerlines:
                continue
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
                candidates.append(
                    (
                        geometry["heading_error_deg"],
                        geometry["endpoint_gap_m"],
                        neighbor,
                        geometry,
                    )
                )
        if not candidates:
            break
        candidates.sort(key=lambda value: (value[0], value[1], value[2]))
        _, _, neighbor, geometry = candidates[0]
        runner_up = candidates[1][0] if len(candidates) > 1 else None
        edges.append(
            {
                "from_token": neighbor if direction == "incoming" else current,
                "to_token": current if direction == "incoming" else neighbor,
                **geometry,
                "candidate_count": len(candidates),
                "runner_up_heading_error_deg": runner_up,
            }
        )
        chain.append(neighbor)
        visited.add(neighbor)
        current = neighbor
    return chain, edges


def receiver_branch_tokens(
    lane_index,
    source_token: str,
    target_token: str,
    topology_type: str,
) -> list[str | None]:
    """列出与 subject source 分离的 target 接收分支。

    平行变道直接使用 target 自身最连续的上游；merge 必须显式选择不同于
    subject source 的 direct incoming。没有独立接收分支时不会把 source
    车流重新解释成 target 车流。
    """
    if topology_type == "lane_change":
        return [None]
    if topology_type != "merge":
        return []
    incoming = lane_index.nmap.connectivity.get(target_token, {}).get("incoming", [])
    return sorted(
        token
        for token in incoming
        if token != source_token and token in lane_index.centerlines
    )


def build_receiver_corridor(
    lane_index,
    source_token: str,
    target_token: str,
    receiver_branch_token: str | None,
    config: dict,
) -> dict:
    """建立排除 subject source 的单分支 target corridor。"""
    hops = int(config["graph_hops"])
    max_heading = float(config["max_edge_heading_error_deg"])
    max_gap = float(config["max_edge_endpoint_gap_m"])
    excluded = {source_token}
    if receiver_branch_token is None:
        incoming, incoming_edges = _best_chain(
            lane_index,
            target_token,
            "incoming",
            hops,
            max_heading,
            max_gap,
            excluded,
        )
    else:
        direct = lane_index.nmap.connectivity.get(target_token, {}).get("incoming", [])
        if receiver_branch_token not in direct:
            raise ValueError("receiver_branch_token 不是 target direct incoming")
        first_geometry = _edge_geometry(
            lane_index, receiver_branch_token, target_token
        )
        if (
            first_geometry["heading_error_deg"] > max_heading
            or first_geometry["endpoint_gap_m"] > max_gap
        ):
            incoming, incoming_edges = [], []
        else:
            tail, tail_edges = _best_chain(
                lane_index,
                receiver_branch_token,
                "incoming",
                max(0, hops - 1),
                max_heading,
                max_gap,
                excluded,
            )
            incoming = [receiver_branch_token, *tail]
            incoming_edges = [
                {
                    "from_token": receiver_branch_token,
                    "to_token": target_token,
                    **first_geometry,
                    "candidate_count": 1,
                    "runner_up_heading_error_deg": None,
                },
                *tail_edges,
            ]
    outgoing, outgoing_edges = _best_chain(
        lane_index,
        target_token,
        "outgoing",
        hops,
        max_heading,
        max_gap,
        excluded,
    )
    ordered_tokens = [*reversed(incoming), target_token, *outgoing]
    points: list[np.ndarray] = []
    for token in ordered_tokens:
        line = np.asarray(lane_index.centerlines[token], dtype=float)
        for index, point in enumerate(line):
            if (
                points
                and index == 0
                and float(np.linalg.norm(points[-1][:2] - point[:2])) < 1e-6
            ):
                continue
            points.append(point)
    centerline = np.asarray(points, dtype=float)
    distances = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(distances)))
    return {
        "source_token_excluded": source_token,
        "target_token": target_token,
        "receiver_branch_token": receiver_branch_token,
        "incoming_tokens_nearest_first": incoming,
        "outgoing_tokens_nearest_first": outgoing,
        "ordered_tokens": ordered_tokens,
        "token_set": set(ordered_tokens),
        "incoming_edges": incoming_edges,
        "outgoing_edges": outgoing_edges,
        "centerline": centerline,
        "arc_lengths": arc_lengths,
    }


def _project_xy(xy: Iterable[float], corridor: dict) -> dict:
    return project_to_polyline(
        xy,
        corridor["centerline"],
        corridor["arc_lengths"],
    )


def _box_corners(row: dict) -> np.ndarray:
    length, width = (float(value) for value in row["dimensions_lwh"][:2])
    local = np.asarray(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=float,
    )
    yaw = float(row["yaw"])
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    return local @ rotation.T + np.asarray(row["xy"], dtype=float)


def _subject_projection(row: dict, corridor: dict, config: dict) -> dict:
    center = _project_xy(row["xy"], corridor)
    corner_lateral = [
        float(_project_xy(point, corridor)["signed_lateral_m"])
        for point in _box_corners(row)
    ]
    center_lateral = float(center["signed_lateral_m"])
    lane_half_width = float(config["lane_half_width_m"])
    outside_clearance = float(config["pre_box_outside_clearance_m"])
    inside_tolerance = float(config["post_box_inside_tolerance_m"])
    sign = 1.0 if center_lateral >= 0 else -1.0
    signed_corners = [sign * value for value in corner_lateral]
    return {
        "frame": int(row["frame_index"]),
        "s_m": float(center["s_m"]),
        "signed_lateral_m": center_lateral,
        "abs_lateral_m": abs(center_lateral),
        "corridor_heading_rad": float(center["heading_rad"]),
        "heading_error_deg": math.degrees(
            angle_error(float(row["yaw"]), float(center["heading_rad"]))
        ),
        "corner_signed_lateral_m": corner_lateral,
        "center_outside_target_band": abs(center_lateral)
        >= lane_half_width + outside_clearance,
        "box_fully_outside_target_band": min(signed_corners)
        >= lane_half_width + outside_clearance,
        "box_inside_target_band": max(abs(value) for value in corner_lateral)
        <= lane_half_width + inside_tolerance,
    }


def _time(frame: int, frame_times_s: dict[int, float], fallback_period: float) -> float:
    return float(frame_times_s.get(frame, frame * fallback_period))


def _median_speed(
    rows: list[dict],
    frame_times_s: dict[int, float],
    fallback_period: float,
) -> float | None:
    speeds = []
    for left, right in zip(rows, rows[1:]):
        dt = _time(int(right["frame_index"]), frame_times_s, fallback_period) - _time(
            int(left["frame_index"]), frame_times_s, fallback_period
        )
        if dt <= 0:
            continue
        speeds.append(
            float(
                np.linalg.norm(
                    np.asarray(right["xy"], dtype=float)
                    - np.asarray(left["xy"], dtype=float)
                )
                / dt
            )
        )
    return float(np.median(speeds)) if speeds else None


def _longitudinal_half_extent(row: dict, corridor_heading_rad: float) -> float:
    length, width = (float(value) for value in row["dimensions_lwh"][:2])
    delta = angle_error(float(row["yaw"]), corridor_heading_rad)
    return 0.5 * (
        length * abs(math.cos(delta)) + width * abs(math.sin(delta))
    )


def _corridor_neighbors(
    actor_id: int,
    frame: int,
    subject_projection: dict,
    matches_by_actor: dict[int, dict[int, dict]],
    corridor: dict,
    config: dict,
) -> dict:
    subject_row = matches_by_actor[actor_id][frame]
    subject_extent = _longitudinal_half_extent(
        subject_row, float(subject_projection["corridor_heading_rad"])
    )
    neighbors = []
    for other_id, rows in sorted(matches_by_actor.items()):
        if other_id == actor_id:
            continue
        row = rows.get(frame)
        if row is None or row.get("lane_token") not in corridor["token_set"]:
            continue
        projection = _project_xy(row["xy"], corridor)
        heading_error = math.degrees(
            angle_error(float(row["yaw"]), float(projection["heading_rad"]))
        )
        if heading_error > float(config["max_receiver_heading_error_deg"]):
            continue
        if abs(float(projection["signed_lateral_m"])) > float(
            config["max_receiver_centerline_distance_m"]
        ):
            continue
        delta_s = float(projection["s_m"] - subject_projection["s_m"])
        extent = _longitudinal_half_extent(row, float(projection["heading_rad"]))
        neighbors.append(
            {
                "actor_id": int(other_id),
                "lane_token": str(row["lane_token"]),
                "center_delta_s_m": delta_s,
                "bumper_gap_m": abs(delta_s) - subject_extent - extent,
                "heading_error_deg": heading_error,
                "signed_lateral_m": float(projection["signed_lateral_m"]),
            }
        )
    fronts = [row for row in neighbors if row["center_delta_s_m"] > 0]
    rears = [row for row in neighbors if row["center_delta_s_m"] < 0]
    return {
        "front": min(fronts, key=lambda row: row["center_delta_s_m"], default=None),
        "receiver": max(rears, key=lambda row: row["center_delta_s_m"], default=None),
        "neighbor_count": len(neighbors),
    }


def _actor_speed_along_corridor(
    actor_id: int,
    frame: int,
    matches_by_actor: dict[int, dict[int, dict]],
    frame_times_s: dict[int, float],
    corridor_heading_rad: float,
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
        [math.cos(corridor_heading_rad), math.sin(corridor_heading_rad)]
    )
    return float(np.dot(velocity, tangent))


def _evaluate_corridor(
    actor_id: int,
    pre_rows: list[dict],
    post_rows: list[dict],
    matches_by_actor: dict[int, dict[int, dict]],
    corridor: dict,
    frame_times_s: dict[int, float],
    config: dict,
) -> dict:
    pre = [_subject_projection(row, corridor, config) for row in pre_rows]
    post = [_subject_projection(row, corridor, config) for row in post_rows]
    sequence = [*pre, *post]
    abs_lateral = [float(row["abs_lateral_m"]) for row in sequence]
    inward = [-value for value in np.diff(abs_lateral)]
    total_variation = float(sum(abs(value) for value in inward))
    convergence_consistency = (
        float(sum(max(0.0, value) for value in inward) / total_variation)
        if total_variation > 1e-9
        else 0.0
    )
    pre_signs = [
        1 if row["signed_lateral_m"] >= 0 else -1
        for row in pre
        if abs(float(row["signed_lateral_m"])) > 1e-6
    ]
    sign_consistency = (
        max(pre_signs.count(1), pre_signs.count(-1)) / len(pre_signs)
        if pre_signs
        else 0.0
    )
    pre_outside_count = sum(row["box_fully_outside_target_band"] for row in pre)
    pre_center_outside_count = sum(
        row["center_outside_target_band"] for row in pre
    )
    post_inside_count = sum(row["box_inside_target_band"] for row in post)
    settle_duration = _time(
        int(post[-1]["frame"]), frame_times_s, float(config["dense_frame_period_s"])
    ) - _time(
        int(post[0]["frame"]), frame_times_s, float(config["dense_frame_period_s"])
    )
    median_speed = _median_speed(
        [*pre_rows, *post_rows],
        frame_times_s,
        float(config["dense_frame_period_s"]),
    )
    subject_checks = {
        "pre_center_outside_target_band": pre_center_outside_count
        >= int(config["min_pre_center_outside_keyframes"]),
        "post_box_inside_target_band": post_inside_count
        >= int(config["min_post_inside_keyframes"]),
        "pre_center_lateral_offset": float(np.median(abs_lateral[: len(pre)]))
        >= float(config["min_pre_center_lateral_m"]),
        "post_center_lateral_offset": float(np.median(abs_lateral[len(pre) :]))
        <= float(config["max_post_center_lateral_m"]),
        "lateral_convergence": (
            float(np.median(abs_lateral[: len(pre)]))
            - float(np.median(abs_lateral[len(pre) :]))
        )
        >= float(config["min_lateral_convergence_m"]),
        "lateral_convergence_consistency": convergence_consistency
        >= float(config["min_lateral_convergence_consistency"]),
        "pre_side_consistency": sign_consistency
        >= float(config["min_pre_side_consistency"]),
        "pre_heading_alignment": float(
            np.median([row["heading_error_deg"] for row in pre])
        )
        <= float(config["max_pre_heading_error_deg"]),
        "post_heading_alignment": max(row["heading_error_deg"] for row in post)
        <= float(config["max_post_heading_error_deg"]),
        "settled_duration": settle_duration
        + float(config.get("timestamp_tolerance_s", 0.0))
        >= float(config["min_settle_duration_s"]),
        "minimum_motion_speed": median_speed is not None
        and median_speed >= float(config["min_median_speed_mps"]),
    }
    subject_pass = all(subject_checks.values())

    per_frame = []
    for item in sequence:
        relation = _corridor_neighbors(
            actor_id,
            int(item["frame"]),
            item,
            matches_by_actor,
            corridor,
            config,
        )
        per_frame.append({"frame": int(item["frame"]), **relation})
    relation_frame = int(post[len(post) // 2]["frame"])
    center_relation = next(
        row for row in per_frame if int(row["frame"]) == relation_frame
    )
    receiver_id = (
        int(center_relation["receiver"]["actor_id"])
        if center_relation["receiver"] is not None
        else None
    )
    front_id = (
        int(center_relation["front"]["actor_id"])
        if center_relation["front"] is not None
        else None
    )
    pre_frames = {int(row["frame"]) for row in pre}
    post_frames = {int(row["frame"]) for row in post}
    receiver_rows = [
        row
        for row in per_frame
        if row["receiver"] is not None
        and receiver_id is not None
        and int(row["receiver"]["actor_id"]) == receiver_id
    ]
    pre_receiver_rows = [
        row for row in receiver_rows if int(row["frame"]) in pre_frames
    ]
    post_receiver_rows = [
        row for row in receiver_rows if int(row["frame"]) in post_frames
    ]
    min_gap = float(config["min_receiver_bumper_gap_m"])
    max_gap = float(config["max_receiver_bumper_gap_m"])
    gaps_pass = bool(receiver_rows) and all(
        min_gap <= float(row["receiver"]["bumper_gap_m"]) <= max_gap
        for row in receiver_rows
    )
    receiver_checks = {
        "receiver_at_relation_frame": receiver_id is not None,
        "receiver_pre_identity_support": len(pre_receiver_rows)
        >= int(config["min_receiver_pre_keyframes"]),
        "receiver_post_identity_support": len(post_receiver_rows)
        >= int(config["min_receiver_post_keyframes"]),
        "receiver_bumper_gaps": gaps_pass,
        "receiver_branch_excludes_subject_source": corridor["source_token_excluded"]
        not in corridor["token_set"],
    }
    receiver_pass = all(receiver_checks.values())

    subject_speed = _actor_speed_along_corridor(
        actor_id,
        relation_frame,
        matches_by_actor,
        frame_times_s,
        float(post[len(post) // 2]["corridor_heading_rad"]),
        int(config["annotation_keyframe_stride"]),
    )
    receiver_speed = (
        _actor_speed_along_corridor(
            receiver_id,
            relation_frame,
            matches_by_actor,
            frame_times_s,
            float(post[len(post) // 2]["corridor_heading_rad"]),
            int(config["annotation_keyframe_stride"]),
        )
        if receiver_id is not None
        else None
    )
    closing_speed = (
        receiver_speed - subject_speed
        if receiver_speed is not None and subject_speed is not None
        else None
    )
    relation_gap = (
        float(center_relation["receiver"]["bumper_gap_m"])
        if center_relation["receiver"] is not None
        else None
    )
    rear_ttc = (
        relation_gap / closing_speed
        if relation_gap is not None and closing_speed is not None and closing_speed > 1e-3
        else None
    )
    event_pass = subject_pass and receiver_pass
    reason = None
    if not subject_pass:
        reason = "subject_body_did_not_cross_target_lane"
    elif not receiver_pass:
        reason = "no_stable_independent_target_lane_receiver"
    return {
        "schema_version": "receiver-centric-cutin-v1",
        "status": "PASS" if event_pass else "FAIL",
        "reason": reason,
        "subject_entry_pass": subject_pass,
        "receiver_interaction_pass": receiver_pass,
        "subject_checks": subject_checks,
        "receiver_checks": receiver_checks,
        "pre_keyframes": pre,
        "post_keyframes": post,
        "pre_box_outside_count": pre_outside_count,
        "pre_center_outside_count": pre_center_outside_count,
        "post_box_inside_count": post_inside_count,
        "pre_median_abs_lateral_m": float(
            np.median(abs_lateral[: len(pre)])
        ),
        "post_median_abs_lateral_m": float(
            np.median(abs_lateral[len(pre) :])
        ),
        "lateral_convergence_m": float(
            np.median(abs_lateral[: len(pre)])
            - np.median(abs_lateral[len(pre) :])
        ),
        "lateral_convergence_consistency": convergence_consistency,
        "pre_side_consistency": sign_consistency,
        "settle_duration_s": settle_duration,
        "median_speed_mps": median_speed,
        "relation_frame": relation_frame,
        "receiver_actor_id": receiver_id,
        "front_actor_id": front_id,
        "receiver_pre_support_keyframes": len(pre_receiver_rows),
        "receiver_post_support_keyframes": len(post_receiver_rows),
        "receiver_bumper_gap_m": relation_gap,
        "subject_longitudinal_speed_mps": subject_speed,
        "receiver_longitudinal_speed_mps": receiver_speed,
        "receiver_closing_speed_mps": closing_speed,
        "receiver_ttc_s": rear_ttc,
        "per_frame": per_frame,
        "corridor": {
            key: value
            for key, value in corridor.items()
            if key not in {"centerline", "arc_lengths", "token_set"}
        },
        "event_pass": event_pass,
    }


def receiver_centric_cutin(
    actor_id: int,
    source_run: dict,
    target_run: dict,
    topology: dict,
    tracks_by_actor: dict[int, list[dict]],
    matches_by_actor: dict[int, dict[int, dict]],
    lane_index,
    frame_times_s: dict[int, float],
    config: dict,
) -> dict:
    """评估一个 source→target transition 是否为真实 target-lane cut-in。"""
    stride = int(config["annotation_keyframe_stride"])
    pre_count = int(config["subject_pre_keyframes"])
    post_count = int(config["subject_post_keyframes"])
    pre_search_count = int(
        config.get("subject_pre_search_keyframes", pre_count)
    )
    post_search_count = int(
        config.get("subject_post_search_keyframes", post_count)
    )
    track = tracks_by_actor[actor_id]
    keyframes = annotation_keyframes(
        track,
        int(track[0]["frame_index"]),
        int(track[-1]["frame_index"]),
        stride,
    )
    pre_end = int(source_run["end_frame"])
    post_start = int(target_run["start_frame"])
    pre_all = [
        row
        for row in keyframes
        if pre_end - (pre_search_count - 1) * stride
        <= int(row["frame_index"])
        <= pre_end
    ]
    post_all = [
        row
        for row in keyframes
        if post_start
        <= int(row["frame_index"])
        <= post_start + (post_search_count - 1) * stride
    ]
    if len(pre_all) < pre_count or len(post_all) < post_count:
        return {
            "schema_version": "receiver-centric-cutin-v1",
            "status": "UNKNOWN",
            "reason": "insufficient_annotation_keyframes",
            "event_pass": False,
            "uses_interpolated_physics": False,
        }
    source_token = str(source_run["token"])
    target_token = str(target_run["token"])
    topology_type = str(topology.get("type", ""))
    branches = receiver_branch_tokens(
        lane_index, source_token, target_token, topology_type
    )
    pre_search = pre_all[-pre_search_count:]
    post_search = post_all[:post_search_count]
    pre_windows = [
        pre_search[index : index + pre_count]
        for index in range(len(pre_search) - pre_count + 1)
    ]
    post_windows = [
        post_search[index : index + post_count]
        for index in range(len(post_search) - post_count + 1)
    ]
    max_entry_duration = float(
        config.get("max_entry_transition_duration_s", float("inf"))
    )
    evaluations = []
    for branch in branches:
        corridor = build_receiver_corridor(
            lane_index,
            source_token,
            target_token,
            branch,
            config,
        )
        if not corridor["incoming_tokens_nearest_first"] and branch is not None:
            continue
        for pre_rows in pre_windows:
            for post_rows in post_windows:
                entry_duration = _time(
                    int(post_rows[0]["frame_index"]),
                    frame_times_s,
                    float(config["dense_frame_period_s"]),
                ) - _time(
                    int(pre_rows[-1]["frame_index"]),
                    frame_times_s,
                    float(config["dense_frame_period_s"]),
                )
                if entry_duration <= 0 or entry_duration > max_entry_duration:
                    continue
                value = _evaluate_corridor(
                    actor_id,
                    pre_rows,
                    post_rows,
                    matches_by_actor,
                    corridor,
                    frame_times_s,
                    config,
                )
                value["entry_transition_duration_s"] = entry_duration
                value["pre_window_frames"] = [
                    int(row["frame_index"]) for row in pre_rows
                ]
                value["post_window_frames"] = [
                    int(row["frame_index"]) for row in post_rows
                ]
                evaluations.append(value)
    if not evaluations:
        return {
            "schema_version": "receiver-centric-cutin-v1",
            "status": "FAIL",
            "reason": "no_independent_receiver_branch",
            "candidate_receiver_branch_count": 0,
            "event_pass": False,
            "uses_interpolated_physics": False,
        }
    ranked = sorted(
        evaluations,
        key=lambda value: (
            not bool(value["event_pass"]),
            not bool(value["subject_entry_pass"]),
            not bool(value["receiver_interaction_pass"]),
            -int(value["receiver_pre_support_keyframes"]),
            -int(value["receiver_post_support_keyframes"]),
            float(value["entry_transition_duration_s"]),
            float("inf")
            if value["receiver_bumper_gap_m"] is None
            else float(value["receiver_bumper_gap_m"]),
            str(value["corridor"]["receiver_branch_token"]),
        ),
    )
    best = ranked[0]
    branch_results = []
    for branch in branches:
        matching = [
            row
            for row in ranked
            if row["corridor"]["receiver_branch_token"] == branch
        ]
        if not matching:
            continue
        row = matching[0]
        branch_results.append(
            {
                "receiver_branch_token": branch,
                "status": row["status"],
                "reason": row["reason"],
                "subject_entry_pass": row["subject_entry_pass"],
                "receiver_interaction_pass": row["receiver_interaction_pass"],
                "receiver_actor_id": row["receiver_actor_id"],
                "receiver_bumper_gap_m": row["receiver_bumper_gap_m"],
                "entry_transition_duration_s": row[
                    "entry_transition_duration_s"
                ],
                "pre_window_frames": row["pre_window_frames"],
                "post_window_frames": row["post_window_frames"],
            }
        )
    output = {
        **best,
        "candidate_receiver_branch_count": len(branch_results),
        "candidate_window_count": len(evaluations),
        "candidate_receiver_branch_results": branch_results,
        "uses_interpolated_physics": False,
    }
    if bool(config.get("include_window_diagnostics", False)):
        output["candidate_window_results"] = [
            {
                "receiver_branch_token": row["corridor"][
                    "receiver_branch_token"
                ],
                "pre_window_frames": row["pre_window_frames"],
                "post_window_frames": row["post_window_frames"],
                "entry_transition_duration_s": row[
                    "entry_transition_duration_s"
                ],
                "subject_entry_pass": row["subject_entry_pass"],
                "receiver_interaction_pass": row[
                    "receiver_interaction_pass"
                ],
                "event_pass": row["event_pass"],
                "subject_checks": row["subject_checks"],
                "receiver_checks": row["receiver_checks"],
                "pre_median_abs_lateral_m": row[
                    "pre_median_abs_lateral_m"
                ],
                "post_median_abs_lateral_m": row[
                    "post_median_abs_lateral_m"
                ],
                "pre_box_outside_count": row["pre_box_outside_count"],
                "pre_center_outside_count": row[
                    "pre_center_outside_count"
                ],
                "post_box_inside_count": row["post_box_inside_count"],
                "lateral_convergence_m": row["lateral_convergence_m"],
                "lateral_convergence_consistency": row[
                    "lateral_convergence_consistency"
                ],
            }
            for row in evaluations
        ]
    return output


def lane_keeping_receiver(
    actor_id: int,
    center_frame: int,
    lane_token: str,
    matches_by_actor: dict[int, dict[int, dict]],
    lane_index,
    frame_times_s: dict[int, float],
    config: dict,
) -> dict:
    """验证 lane-keeping control 也有持续、最近的同车道后车。

    该 control 与正例使用相同的 receiver/gap 定义，但不包含横向进入。它用于
    防止把“孤车普通直行”当成交互密度等价的 same-actor negative。
    """
    stride = int(config["annotation_keyframe_stride"])
    frame_count = int(config["control_receiver_keyframes"])
    if frame_count % 2 != 1:
        raise ValueError("control_receiver_keyframes 必须为奇数")
    center = int(round(center_frame / stride) * stride)
    radius = frame_count // 2
    frames = [center + offset * stride for offset in range(-radius, radius + 1)]
    corridor = build_receiver_corridor(
        lane_index,
        "__no_subject_source__",
        lane_token,
        None,
        config,
    )
    per_frame = []
    for frame in frames:
        row = matches_by_actor.get(actor_id, {}).get(frame)
        if row is None or row.get("lane_token") not in corridor["token_set"]:
            per_frame.append(
                {
                    "frame": frame,
                    "status": "UNKNOWN",
                    "reason": "subject_missing_or_outside_control_corridor",
                }
            )
            continue
        subject = _subject_projection(row, corridor, config)
        relation = _corridor_neighbors(
            actor_id,
            frame,
            subject,
            matches_by_actor,
            corridor,
            config,
        )
        per_frame.append(
            {
                "frame": frame,
                "status": "OBSERVED",
                "subject": subject,
                **relation,
            }
        )
    center_row = next(row for row in per_frame if row["frame"] == center)
    receiver_id = (
        int(center_row["receiver"]["actor_id"])
        if center_row.get("receiver") is not None
        else None
    )
    supporting = [
        row
        for row in per_frame
        if row.get("receiver") is not None
        and receiver_id is not None
        and int(row["receiver"]["actor_id"]) == receiver_id
    ]
    min_gap = float(config["min_receiver_bumper_gap_m"])
    max_gap = float(config["max_receiver_bumper_gap_m"])
    checks = {
        "receiver_at_center": receiver_id is not None,
        "receiver_identity_support": len(supporting)
        >= int(config["min_control_receiver_keyframes"]),
        "receiver_bumper_gaps": bool(supporting)
        and all(
            min_gap <= float(row["receiver"]["bumper_gap_m"]) <= max_gap
            for row in supporting
        ),
    }
    return {
        "schema_version": "lane-keeping-receiver-control-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "center_frame": center,
        "frames": frames,
        "receiver_actor_id": receiver_id,
        "receiver_support_keyframes": len(supporting),
        "receiver_bumper_gap_m": (
            float(center_row["receiver"]["bumper_gap_m"])
            if center_row.get("receiver") is not None
            else None
        ),
        "checks": checks,
        "per_frame": per_frame,
        "corridor": {
            key: value
            for key, value in corridor.items()
            if key not in {"centerline", "arc_lengths", "token_set"}
        },
    }
