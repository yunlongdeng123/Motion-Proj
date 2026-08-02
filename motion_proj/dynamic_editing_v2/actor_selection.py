"""只基于输入支持度的确定性 actor 选择。"""
from __future__ import annotations

import math
from typing import Any


def evaluate_support(actor: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["thresholds"]
    annotations = actor["raw_annotations"]
    observations = actor["camera_observations"]
    reasons = []
    if not actor["category_name"].startswith(config["category_prefix"]):
        reasons.append("category_prefix")
    if len(annotations) < int(thresholds["min_raw_annotations"]):
        reasons.append("raw_annotation_count")
    point_supported = sum(
        int(row["num_lidar_pts"]) + int(row["num_radar_pts"]) > 0 for row in annotations
    )
    if point_supported < int(thresholds["min_point_supported_annotations"]):
        reasons.append("point_supported_annotation_count")
    center_inside = sum(row["projection"]["center_inside_image"] for row in observations if row["projection"]["valid"])
    if center_inside < int(thresholds["min_center_inside_observations"]):
        reasons.append("center_inside_observation_count")
    areas = sorted(
        float(row["projection"]["visible_area_px"])
        for row in observations
        if row["projection"]["valid"]
    )
    median_area = areas[len(areas) // 2] if areas else 0.0
    if median_area < float(thresholds["min_median_visible_area_px"]):
        reasons.append("median_visible_area")
    point_sum = sum(int(row["num_lidar_pts"]) + int(row["num_radar_pts"]) for row in annotations)
    weights = config["score_weights"]
    score = (
        float(weights["raw_annotation_count"]) * len(annotations)
        + float(weights["valid_camera_observation_count"]) * sum(row["projection"]["valid"] for row in observations)
        + float(weights["log1p_lidar_radar_point_sum"]) * math.log1p(point_sum)
        + float(weights["log1p_median_visible_area_px"]) * math.log1p(median_area)
    )
    return {
        "eligible": not reasons,
        "failure_reasons": reasons,
        "raw_annotation_count": len(annotations),
        "point_supported_annotation_count": point_supported,
        "lidar_radar_point_sum": point_sum,
        "valid_camera_observation_count": sum(row["projection"]["valid"] for row in observations),
        "center_inside_observation_count": center_inside,
        "median_visible_area_px": median_area,
        "support_score": score,
    }


def select_cohort(actors: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    evaluated = []
    for actor in actors:
        support = evaluate_support(actor, config)
        actor["support_summary"] = support
        evaluated.append(actor)
    eligible = [actor for actor in evaluated if actor["support_summary"]["eligible"]]
    eligible.sort(key=lambda actor: (-actor["support_summary"]["support_score"], actor["instance_token"]))
    selected = []
    if eligible:
        selected.append({"role": "high-support", "instance_token": eligible[0]["instance_token"]})
    if len(eligible) >= 2:
        low = sorted(eligible, key=lambda actor: (actor["support_summary"]["support_score"], actor["instance_token"]))[0]
        selected.append({"role": "boundary-support", "instance_token": low["instance_token"]})
    return {
        "actors": evaluated,
        "eligible_instance_tokens": [actor["instance_token"] for actor in eligible],
        "selected": selected,
        "slot_coverage": len(selected) / 2.0,
    }
