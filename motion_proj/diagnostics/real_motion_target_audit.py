"""RP-A0：真实 ego–actor motion target 合法性审计。

真实 annotation、ego pose 与 LiDAR 只用于 real-training representation target。
RAFT 在本 gate 中只审计真实背景方向一致性；自由生成 evaluator 不读取任何 future GT。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from ..auditor.flow_raft import RAFTFlow
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..data.real_motion_targets import (
    REAL_TARGET_SCOPE,
    assert_target_scope,
    binary_roc_auc,
    boxes_background_mask,
    build_actor_residual_targets,
    flow_direction_agreement,
    sparse_ego_flow_target,
    spearman_correlation,
    timestamps_to_seconds,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video


REQUIRED_MOTION_BOX_FIELDS = {
    "annotation_token",
    "instance_token",
    "category",
    "attributes",
    "visibility",
    "xyxy",
    "center_cam",
    "corners_cam",
    "center_depth",
    "size3d",
    "velocity_global",
}


class RealMotionAuditError(RuntimeError):
    """A0 provenance、schema、单位或正式 gate 不成立。"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(values: Sequence[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def finite_summary(values: Sequence[float | None]) -> dict[str, Any]:
    data = np.asarray(_finite(values), dtype=np.float64)
    if data.size == 0:
        return {"status": "invalid", "count": 0}
    return {
        "status": "valid",
        "count": int(data.size),
        "min": float(data.min()),
        "p05": float(np.quantile(data, 0.05)),
        "median": float(np.quantile(data, 0.5)),
        "mean": float(data.mean()),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(data.max()),
    }


def select_scene_distinct_records(
    records: Sequence[Mapping[str, Any]], *, count: int,
) -> list[dict[str, Any]]:
    indexed = [dict(row, dataset_index=index) for index, row in enumerate(records)]
    indexed.sort(
        key=lambda row: (
            str(row.get("scene_name", "")),
            int(row.get("start_index", -1)),
            str(row.get("sample_id", "")),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in indexed:
        scene = str(row.get("scene_token") or row.get("scene_name") or "")
        if not scene or scene in seen:
            continue
        seen.add(scene)
        selected.append(row)
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RealMotionAuditError(f"scene-distinct clips 不足: required={count}, actual={len(selected)}")
    return selected


def _copy_data_config(data_cfg: Any) -> Any:
    copied = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    copied.split = "train"
    copied.use_lidar_depth = True
    return copied


def _schema_metrics(sample: Mapping[str, Any], *, min_visibility: int) -> dict[str, Any]:
    timestamps_to_seconds(sample["timestamps"])
    boxes = [box for frame in sample["boxes"] for box in frame]
    missing = [
        {"instance_token": str(box.get("instance_token", "")), "missing": sorted(REQUIRED_MOTION_BOX_FIELDS - set(box))}
        for box in boxes
        if REQUIRED_MOTION_BOX_FIELDS - set(box)
    ]
    visibility_violations = sum(int(box.get("visibility", 0)) < int(min_visibility) for box in boxes)
    count = int(torch.as_tensor(sample["frames"]).shape[0])
    camera = torch.as_tensor(sample["cam2ego_frames"], dtype=torch.float64)
    intrinsics = torch.as_tensor(sample["intrinsics_frames"], dtype=torch.float64)
    camera_drift = float((camera - camera[0]).abs().max())
    intrinsics_drift = float((intrinsics - intrinsics[0]).abs().max())
    lidar = torch.as_tensor(sample["lidar_depth"])
    lidar_counts = [int((torch.isfinite(frame) & (frame > 0)).sum()) for frame in lidar]
    return {
        "frame_count": count,
        "box_count": len(boxes),
        "missing_schema_count": len(missing),
        "missing_schema_preview": missing[:5],
        "visibility_violation_count": int(visibility_violations),
        "cam2ego_max_abs_drift": camera_drift,
        "intrinsics_max_abs_drift": intrinsics_drift,
        "lidar_point_counts": lidar_counts,
        "minimum_lidar_points": min(lidar_counts) if lidar_counts else 0,
        "timestamp_unit": "microseconds",
    }


def _background_pair_rows(
    sample: Mapping[str, Any],
    raft_flow: torch.Tensor,
    raft_confidence: torch.Tensor,
    *,
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]]]:
    frames = torch.as_tensor(sample["frames"])
    count, _, height, width = frames.shape
    intrinsics = torch.as_tensor(sample["intrinsics_frames"], dtype=torch.float64)
    cam2ego = torch.as_tensor(sample["cam2ego_frames"], dtype=torch.float64)
    ego2global = torch.as_tensor(sample["ego2global"], dtype=torch.float64)
    lidar = torch.as_tensor(sample["lidar_depth"], dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    panel_data: list[dict[str, torch.Tensor]] = []
    for frame_index in range(count - 1):
        target, lidar_valid = sparse_ego_flow_target(
            lidar[frame_index], intrinsics[frame_index], intrinsics[frame_index + 1],
            cam2ego[frame_index], cam2ego[frame_index + 1],
            ego2global[frame_index], ego2global[frame_index + 1],
        )
        background = boxes_background_mask(
            height, width, sample["boxes"][frame_index],
            dilation_px=int(thresholds["box_dilation_px"]),
        )
        # 对下一帧同一像素位置也做保守排除，降低 box 边界和遮挡污染。
        background &= boxes_background_mask(
            height, width, sample["boxes"][frame_index + 1],
            dilation_px=int(thresholds["box_dilation_px"]),
        )
        valid = (
            lidar_valid
            & background
            & torch.isfinite(raft_confidence[frame_index])
            & (raft_confidence[frame_index] >= float(thresholds["minimum_raft_confidence"]))
        )
        agreement = flow_direction_agreement(
            target,
            raft_flow[frame_index],
            valid,
            minimum_magnitude_px=float(thresholds["minimum_flow_magnitude_px"]),
            maximum_angle_deg=float(thresholds["maximum_direction_angle_deg"]),
        )
        rows.append(
            {
                "frame_index": frame_index,
                "next_frame_index": frame_index + 1,
                "lidar_valid_count": int(lidar_valid.sum()),
                "confident_background_count": int(valid.sum()),
                "direction": agreement,
                "target_scope": REAL_TARGET_SCOPE,
            }
        )
        panel_data.append(
            {
                "target": target.detach().cpu(),
                "raft": raft_flow[frame_index].detach().cpu(),
                "valid": valid.detach().cpu(),
            }
        )
    return rows, panel_data


def aggregate_a0_metrics(
    actor_rows: Sequence[Mapping[str, Any]],
    background_rows: Sequence[Mapping[str, Any]],
    schema_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    finite_actor = [row for row in actor_rows if bool(row.get("finite"))]
    moving = [float(row["residual_speed_px_per_s"]) for row in finite_actor if row.get("motion_label") == "moving"]
    stationary = [float(row["residual_speed_px_per_s"]) for row in finite_actor if row.get("motion_label") == "stationary"]
    projection_pairs = [
        (bool(row.get(f"center_projection_eligible_{suffix}")), bool(row.get(f"center_projection_in_box_{suffix}")))
        for row in actor_rows
        for suffix in ("t", "tp1")
    ]
    projection_values = [inside for eligible, inside in projection_pairs if eligible]
    velocity_cosines = _finite([row.get("velocity_direction_cosine") for row in finite_actor])
    residual = [float(row["residual_speed_px_per_s"]) for row in finite_actor]
    ego = [float(row["ego_translation_speed_mps"]) for row in finite_actor]
    direction_count = sum(int(row.get("direction", {}).get("count", 0)) for row in background_rows)
    direction_pass = sum(
        int(row.get("direction", {}).get("count", 0))
        * float(row.get("direction", {}).get("agreement_fraction", 0.0))
        for row in background_rows
        if row.get("direction", {}).get("status") == "valid"
    )
    unique_tracks = {
        (str(row.get("sample_id", "")), str(row.get("instance_token", "")))
        for row in finite_actor
    }
    return {
        "actor_pair_count": len(actor_rows),
        "finite_actor_pair_count": len(finite_actor),
        "finite_target_fraction": len(finite_actor) / max(len(actor_rows), 1),
        "valid_paired_actor_track_count": len(unique_tracks),
        "moving_pair_count": len(moving),
        "stationary_pair_count": len(stationary),
        "moving_residual_speed_px_per_s": finite_summary(moving),
        "stationary_residual_speed_px_per_s": finite_summary(stationary),
        "moving_vs_stationary_residual_auc": binary_roc_auc(moving, stationary),
        "center_projection_in_box_fraction": (
            sum(projection_values) / len(projection_values) if projection_values else None
        ),
        "center_projection_eligible_count": len(projection_values),
        "offscreen_visible_center_count": sum(not eligible for eligible, _ in projection_pairs),
        "velocity_direction_pair_count": len(velocity_cosines),
        "velocity_direction_positive_fraction": (
            sum(value > 0.0 for value in velocity_cosines) / len(velocity_cosines)
            if velocity_cosines else None
        ),
        "velocity_direction_cosine": finite_summary(velocity_cosines),
        "residual_vs_ego_speed_spearman": spearman_correlation(residual, ego),
        "background_pair_count": len(background_rows),
        "background_direction_point_count": direction_count,
        "background_ego_vs_raft_angular_agreement": (
            direction_pass / direction_count if direction_count else None
        ),
        "minimum_lidar_points_per_frame": min(
            (int(row["minimum_lidar_points"]) for row in schema_rows), default=0,
        ),
        "missing_schema_count": sum(int(row["missing_schema_count"]) for row in schema_rows),
        "visibility_violation_count": sum(int(row["visibility_violation_count"]) for row in schema_rows),
        "maximum_cam2ego_drift": max(
            (float(row["cam2ego_max_abs_drift"]) for row in schema_rows), default=float("inf"),
        ),
        "maximum_intrinsics_drift": max(
            (float(row["intrinsics_max_abs_drift"]) for row in schema_rows), default=float("inf"),
        ),
    }


def decide_a0_gate(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    def finite_number(name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    auc = finite_number("moving_vs_stationary_residual_auc")
    projection = finite_number("center_projection_in_box_fraction")
    finite_fraction = finite_number("finite_target_fraction")
    background = finite_number("background_ego_vs_raft_angular_agreement")
    velocity = finite_number("velocity_direction_positive_fraction")
    correlation = finite_number("residual_vs_ego_speed_spearman")
    checks = {
        "moving_stationary_support": (
            int(metrics.get("moving_pair_count", 0)) >= int(thresholds["minimum_labeled_actor_pairs_each"])
            and int(metrics.get("stationary_pair_count", 0)) >= int(thresholds["minimum_labeled_actor_pairs_each"])
        ),
        "moving_stationary_auc": auc is not None and auc >= float(thresholds["minimum_moving_stationary_auc"]),
        "center_projection": projection is not None and projection >= float(thresholds["minimum_projection_in_box_fraction"]),
        "actor_track_count": int(metrics.get("valid_paired_actor_track_count", 0)) >= int(thresholds["minimum_valid_actor_tracks"]),
        "finite_targets": finite_fraction is not None and finite_fraction >= float(thresholds["minimum_finite_target_fraction"]),
        "background_support": int(metrics.get("background_direction_point_count", 0)) >= int(thresholds["minimum_background_direction_points"]),
        "background_direction": background is not None and background >= float(thresholds["minimum_background_angular_agreement"]),
        "velocity_support": int(metrics.get("velocity_direction_pair_count", 0)) >= int(thresholds["minimum_velocity_direction_pairs"]),
        "velocity_direction": velocity is not None and velocity >= float(thresholds["minimum_velocity_direction_positive_fraction"]),
        "ego_disentanglement": correlation is not None and abs(correlation) <= float(thresholds["maximum_abs_residual_ego_spearman"]),
        "schema_complete": int(metrics.get("missing_schema_count", 0)) == 0,
        "visibility_filter": int(metrics.get("visibility_violation_count", 0)) == 0,
        "cam2ego_constant": float(metrics.get("maximum_cam2ego_drift", float("inf"))) <= float(thresholds["maximum_calibration_drift"]),
        "intrinsics_constant": float(metrics.get("maximum_intrinsics_drift", float("inf"))) <= float(thresholds["maximum_calibration_drift"]),
        "lidar_support": int(metrics.get("minimum_lidar_points_per_frame", 0)) >= int(thresholds["minimum_lidar_points_per_frame"]),
    }
    machine_pass = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "machine_pass" if machine_pass else "machine_rejected",
        "machine_pass": machine_pass,
        "checks": checks,
        "failed_checks": failed,
        "next_gate": "RP-A1-SCAN-04A" if machine_pass else "RP-B0-05",
    }


def _draw_boxes(draw: ImageDraw.ImageDraw, boxes: Sequence[Mapping[str, Any]], *, color: str) -> None:
    for box in boxes:
        xyxy = [float(value) for value in box["xyxy"]]
        draw.rectangle(xyxy, outline=color, width=2)
        label = f"{str(box.get('category', '')).split('.')[-1]} v{box.get('visibility', 0)}"
        draw.text((xyxy[0] + 2, max(xyxy[1] - 10, 0)), label, fill=color)


def _flow_overlay(
    frame: np.ndarray,
    flow: torch.Tensor,
    valid: torch.Tensor,
    *,
    color: str,
    maximum_arrows: int = 180,
) -> Image.Image:
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    indices = torch.nonzero(valid, as_tuple=False)
    if int(indices.shape[0]) > maximum_arrows:
        chosen = torch.linspace(0, indices.shape[0] - 1, maximum_arrows).round().long()
        indices = indices[chosen]
    for v, u in indices.tolist():
        vector = flow[v, u]
        if not bool(torch.isfinite(vector).all()):
            continue
        end = (float(u + vector[0]), float(v + vector[1]))
        draw.line([(float(u), float(v)), end], fill=color, width=1)
        draw.ellipse([end[0] - 1, end[1] - 1, end[0] + 1, end[1] + 1], fill=color)
    return image


def render_review_panel(
    sample: Mapping[str, Any],
    actor_rows: Sequence[Mapping[str, Any]],
    background_panels: Sequence[Mapping[str, torch.Tensor]],
    *,
    case_id: str,
    sample_id: str,
    scene_name: str,
    destination: Path,
) -> dict[str, Any]:
    frames = to_uint8_video(torch.as_tensor(sample["frames"]))
    finite_rows = [row for row in actor_rows if bool(row.get("finite"))]
    chosen = max(
        finite_rows,
        key=lambda row: (row.get("motion_label") == "moving", float(row["residual_speed_px_per_s"])),
        default=None,
    )
    frame_index = int(chosen["frame_index"]) if chosen is not None else 0
    frame_index = min(frame_index, len(background_panels) - 1)
    first = Image.fromarray(frames[frame_index].copy())
    second = Image.fromarray(frames[frame_index + 1].copy())
    _draw_boxes(ImageDraw.Draw(first), sample["boxes"][frame_index], color="lime")
    _draw_boxes(ImageDraw.Draw(second), sample["boxes"][frame_index + 1], color="lime")
    if chosen is not None:
        draw_first, draw_second = ImageDraw.Draw(first), ImageDraw.Draw(second)
        p0 = tuple(float(value) for value in chosen["actual_uv_t"])
        actual = tuple(float(value) for value in chosen["actual_uv_tp1"])
        static = tuple(float(value) for value in chosen["static_uv_tp1"])
        draw_first.ellipse([p0[0] - 4, p0[1] - 4, p0[0] + 4, p0[1] + 4], fill="cyan")
        draw_second.line([static, actual], fill="red", width=3)
        draw_second.ellipse([actual[0] - 4, actual[1] - 4, actual[0] + 4, actual[1] + 4], fill="cyan")
        draw_second.ellipse([static[0] - 4, static[1] - 4, static[0] + 4, static[1] + 4], fill="orange")
    background = background_panels[frame_index]
    ego_panel = _flow_overlay(
        frames[frame_index], background["target"], background["valid"], color="yellow",
    )
    raft_panel = _flow_overlay(
        frames[frame_index], background["raft"], background["valid"], color="cyan",
    )
    panels = [np.asarray(first), np.asarray(second), np.asarray(ego_panel), np.asarray(raft_panel)]
    canvas = Image.fromarray(np.concatenate(panels, axis=1))
    output = Image.new("RGB", (canvas.width, canvas.height + 52), "black")
    output.paste(canvas, (0, 0))
    text = f"{case_id} {scene_name} {sample_id} pair={frame_index}->{frame_index + 1}"
    if chosen is not None:
        text += (
            f" actor={chosen['category']} attr={','.join(chosen['attributes']) or '-'}"
            f" vis={chosen['visibility_t']}/{chosen['visibility_tp1']}"
            f" residual={chosen['residual_speed_px_per_s']:.2f}px/s dt={chosen['dt_s']:.3f}s"
        )
    draw = ImageDraw.Draw(output)
    draw.text((8, canvas.height + 5), text, fill="white")
    draw.text(
        (8, canvas.height + 25),
        "panels: RGB_t boxes | RGB_t+1 cyan=actual orange=static red=residual | sparse ego flow | RAFT",
        fill="white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)
    return {
        "case_id": case_id,
        "panel_path": str(destination),
        "sample_id": sample_id,
        "scene_name": scene_name,
        "frame_index": frame_index,
        "actor_instance_token": str(chosen["instance_token"]) if chosen is not None else None,
        "actor_category": str(chosen["category"]) if chosen is not None else None,
        "actor_attributes": list(chosen["attributes"]) if chosen is not None else [],
        "timestamps_us": [int(value) for value in torch.as_tensor(sample["timestamps"]).tolist()],
    }


def _make_review_material(work_dir: Path, panel_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    review_dir = work_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=False)
    public_rows = []
    templates = []
    for row in panel_rows:
        panel_path = Path(str(row["panel_path"]))
        if panel_path.is_absolute():
            panel_path = panel_path.relative_to(work_dir)
        public_rows.append(
            {
                "case_id": row["case_id"],
                "panel_path": str(panel_path),
                "question": "真实几何、actor residual 与背景 ego flow 是否在可见区域内一致且可解释？",
            }
        )
        templates.append(
            {
                "case_id": row["case_id"],
                "box_projection": None,
                "actor_actual_vs_static": None,
                "background_ego_vs_raft": None,
                "attribute_visibility": None,
                "overall_target_validity": None,
                "notes": "",
            }
        )
    _write_jsonl(review_dir / "cases.jsonl", public_rows)
    _write_jsonl(review_dir / "reviews.template.jsonl", templates)
    prompt = """# A0 真实运动 target 人工复核

每个 panel 从左到右为：当前 RGB + 3D box 投影、下一帧 RGB、稀疏 LiDAR ego-induced flow、RAFT flow。
下一帧中 cyan 点是真实 annotation center，orange 点是“actor 世界静止”假设位置，red 是 actor residual。

请逐项检查：

1. 3D box 与 center 是否落在对应可见 actor 上；
2. actual/static/residual 的方向是否符合两帧内容与 ego motion；
3. GT box 外稀疏 ego flow 与 RAFT 主方向是否一致；
4. category、attribute、visibility/occlusion 是否与画面相容；
5. 任一关键项无法判断时填 uncertain，不得为了通过 gate 猜测。

前四项与 overall 只允许 `pass / fail / uncertain`。模板中的 null 必须由人工填写；不得自动代填。
这些 target 只用于真实训练视频 representation，不是生成时 future condition，也不是 generated evaluator truth。
"""
    atomic_write_text(str(review_dir / "REVIEW_PROMPT.md"), prompt)
    return {
        "status": "awaiting_reviews",
        "case_count": len(panel_rows),
        "cases_path": "review/cases.jsonl",
        "template_path": "review/reviews.template.jsonl",
        "prompt_path": "review/REVIEW_PROMPT.md",
    }


def preflight_real_motion_target_audit(cfg: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": str(cfg.a0.task_id),
        "status": "ready",
        "uses_gpu": True,
        "target_scope": REAL_TARGET_SCOPE,
        "uses_future_gt_for_generated_evaluation": False,
        "blockers": [],
    }
    try:
        dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data))
        selected = select_scene_distinct_records(dataset.clip_records, count=int(cfg.a0.clip_count))
        result["data"] = {
            "clip_count": len(dataset),
            "selected_scene_count": len(selected),
            "sample_ids": [str(row["sample_id"]) for row in selected],
            "selection_fingerprint": sha256_json([str(row["sample_id"]) for row in selected]),
        }
    except Exception as exc:
        result["blockers"].append({"kind": "nuscenes", "error": repr(exc)})
    checkpoint = Path(str(cfg.a0.raft.checkpoint_path))
    raft = {
        "checkpoint_path": str(checkpoint),
        "available": checkpoint.is_file(),
        "checkpoint_sha256": file_fingerprint(str(checkpoint)) if checkpoint.is_file() else None,
        "checkpoint_sha256_expected": str(cfg.a0.raft.checkpoint_sha256),
    }
    if raft["checkpoint_sha256"] != raft["checkpoint_sha256_expected"]:
        raft["available"] = False
        result["blockers"].append({"kind": "raft", "detail": raft})
    result["raft"] = raft
    usage = shutil.disk_usage(Path(str(cfg.work_dir)).parent)
    free_gb = float(usage.free) / float(1024**3)
    result["disk"] = {"free_gb": free_gb, "minimum_free_gb": float(cfg.a0.minimum_free_disk_gb)}
    if free_gb < float(cfg.a0.minimum_free_disk_gb):
        result["blockers"].append({"kind": "disk", "free_gb": free_gb})
    if result["blockers"]:
        result["status"] = "blocked"
    return result


def _validate_protocol(cfg: Any) -> None:
    if str(cfg.a0.task_id) != "RP-A0-03":
        raise RealMotionAuditError("A0 task_id 必须为 RP-A0-03")
    if int(cfg.a0.clip_count) != 16 or int(cfg.a0.review_panel_count) != 12:
        raise RealMotionAuditError("A0 必须使用 16 clips / 12 review panels")
    if int(cfg.data.num_frames) != 8 or int(cfg.model.num_frames) != 8:
        raise RealMotionAuditError("A0 必须使用 8 frames")
    if not bool(cfg.data.use_lidar_depth) or int(cfg.data.min_box_visibility) < 2:
        raise RealMotionAuditError("A0 必须启用 LiDAR depth 且 min_box_visibility>=2")
    if int(cfg.model.generation.fps) != 7:
        raise RealMotionAuditError("A0 必须继承 R1 冻结的 generation.fps=7")
    assert_target_scope(str(cfg.a0.target_scope))


def run_real_motion_target_audit(cfg: Any) -> dict[str, Any]:
    _validate_protocol(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise RealMotionAuditError("正式 A0 拒绝在 dirty worktree 上运行")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"A0 run directory 已存在: {work_dir}")
    preflight = preflight_real_motion_target_audit(cfg)
    if preflight["status"] != "ready":
        raise RealMotionAuditError(f"A0 preflight blocked: {preflight['blockers']}")

    config_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "panels").mkdir()
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint=str(preflight["data"]["selection_fingerprint"]),
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split="nuScenes official train / 16 scene-distinct real clips",
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.a0.task_id),
        "preflight": preflight,
        "target_scope": REAL_TARGET_SCOPE,
        "uses_future_gt_for_real_training_target": True,
        "uses_future_gt_for_generated_evaluation": False,
        "uses_future_gt_as_inference_condition": False,
        "generation_fps_frozen_by_r1": int(cfg.model.generation.fps),
    }
    atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics_log = JsonlMetrics(str(work_dir / "metrics.jsonl"))

    try:
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data))
        selected = select_scene_distinct_records(dataset.clip_records, count=int(cfg.a0.clip_count))
        raft = RAFTFlow(device=str(cfg.a0.raft.device))
        actor_rows: list[dict[str, Any]] = []
        background_rows: list[dict[str, Any]] = []
        schema_rows: list[dict[str, Any]] = []
        clip_rows: list[dict[str, Any]] = []
        panel_rows: list[dict[str, Any]] = []
        thresholds = dict(cfg.a0.thresholds)

        for clip_index, record in enumerate(selected):
            sample = dataset[int(record["dataset_index"])]
            sample_id = str(record["sample_id"])
            scene_name = str(record["scene_name"])
            schema = _schema_metrics(sample, min_visibility=int(cfg.data.min_box_visibility))
            schema.update({"sample_id": sample_id, "scene_name": scene_name})
            schema_rows.append(schema)

            clip_actor = build_actor_residual_targets(
                sample, min_visibility=int(cfg.data.min_box_visibility),
            )
            for row in clip_actor:
                row.update({"sample_id": sample_id, "scene_name": scene_name})
            actor_rows.extend(clip_actor)

            with torch.no_grad():
                raft_flow, raft_confidence = raft.flow_with_confidence(torch.as_tensor(sample["frames"]))
            raft_flow = raft_flow.detach().cpu()
            raft_confidence = raft_confidence.detach().cpu()
            clip_background, background_panels = _background_pair_rows(
                sample, raft_flow, raft_confidence, thresholds=thresholds,
            )
            for row in clip_background:
                row.update({"sample_id": sample_id, "scene_name": scene_name})
            background_rows.extend(clip_background)

            clip_row = {
                "sample_id": sample_id,
                "scene_name": scene_name,
                "scene_token": str(record["scene_token"]),
                "dataset_index": int(record["dataset_index"]),
                "actor_pair_count": len(clip_actor),
                "finite_actor_pair_count": sum(bool(row["finite"]) for row in clip_actor),
                "background_pair_count": len(clip_background),
                "schema": schema,
            }
            clip_rows.append(clip_row)
            metrics_log.append(
                clip_index + 1,
                {
                    "event": "clip_audited",
                    "sample_id": sample_id,
                    "scene_name": scene_name,
                    "actor_pair_count": len(clip_actor),
                    "finite_actor_pair_count": clip_row["finite_actor_pair_count"],
                    "background_direction_points": sum(
                        int(row["direction"].get("count", 0)) for row in clip_background
                    ),
                },
            )
            if len(panel_rows) < int(cfg.a0.review_panel_count):
                case_id = f"a0-review-{len(panel_rows) + 1:03d}"
                panel = render_review_panel(
                    sample,
                    clip_actor,
                    background_panels,
                    case_id=case_id,
                    sample_id=sample_id,
                    scene_name=scene_name,
                    destination=work_dir / "panels" / f"{case_id}.png",
                )
                panel["panel_path"] = str(Path(panel["panel_path"]).relative_to(work_dir))
                panel_rows.append(panel)

        _write_jsonl(work_dir / "clips.jsonl", clip_rows)
        _write_jsonl(work_dir / "actor_residual_targets.jsonl", actor_rows)
        _write_jsonl(work_dir / "background_ego_flow_audit.jsonl", background_rows)
        _write_jsonl(work_dir / "schema_audit.jsonl", schema_rows)
        _write_jsonl(work_dir / "panel_manifest.jsonl", panel_rows)
        metrics = aggregate_a0_metrics(actor_rows, background_rows, schema_rows)
        decision = decide_a0_gate(metrics, thresholds)
        review = _make_review_material(work_dir, panel_rows)
        result = {
            "task_id": str(cfg.a0.task_id),
            "target_scope": REAL_TARGET_SCOPE,
            "uses_future_gt_for_generated_evaluation": False,
            "clip_count": len(clip_rows),
            "scene_count": len({row["scene_name"] for row in clip_rows}),
            "metrics": metrics,
            "decision": decision,
            "review": review,
        }
        atomic_write_json(str(work_dir / "result.json"), _json_safe(result))
        research_status = "awaiting_reviews" if bool(decision["machine_pass"]) else "rejected"
        summary = {
            "status": research_status,
            "task_id": str(cfg.a0.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": config_fp,
            "selection_fingerprint": str(preflight["data"]["selection_fingerprint"]),
            "clip_count": len(clip_rows),
            "scene_count": len({row["scene_name"] for row in clip_rows}),
            "machine_pass": bool(decision["machine_pass"]),
            "failed_checks": decision["failed_checks"],
            "review_status": review["status"],
            "review_case_count": review["case_count"],
            "uses_future_gt_for_generated_evaluation": False,
            "result_fingerprint": sha256_json(_json_safe(result)),
            "next_gate": decision["next_gate"],
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update(
            {
                "status": "completed",
                "ended_at": utc_now(),
                "exit_reason": research_status,
                "machine_pass": bool(decision["machine_pass"]),
            }
        )
        atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
        return summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "task_id": str(cfg.a0.task_id),
            "run_id": str(cfg.run_id),
            "error": repr(exc),
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="audit real ego-actor motion target legality")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    result = preflight_real_motion_target_audit(cfg) if args.preflight else run_real_motion_target_audit(cfg)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
