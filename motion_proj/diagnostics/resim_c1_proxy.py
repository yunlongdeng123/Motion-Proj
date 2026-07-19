"""C1B-01：冻结 ReSim scenes，并在真实 nuScenes future 上校准本地 ego-motion proxy。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from ..auditor.flow_raft import RAFTFlow
from ..auditor.generated_geometry import fit_affine_background_flow
from ..runtime.atomic import atomic_write_json, atomic_write_text


CLASSES = ("stationary", "forward", "left", "right")
MOVING_CLASSES = ("forward", "left", "right")
FEATURE_NAMES = (
    "sum_tx", "sum_ty", "sum_divergence", "sum_curl", "sum_anisotropy", "sum_shear",
    "median_tx", "median_ty", "median_divergence", "median_curl",
    "median_affine_energy", "median_fit_residual", "median_fit_confidence", "valid_pair_fraction",
    # RF-17 reopen：显式 yaw / 分段侧向，供 kinematic lateral 分类器使用
    "sum_yaw", "abs_sum_yaw", "median_yaw",
    "sum_tx_early", "sum_tx_late", "sum_yaw_early", "sum_yaw_late",
    "delta_sum_tx", "delta_sum_yaw", "tx_over_ty", "yaw_over_trans",
)
PROXY_PROTOCOL = "local-ego-motion-proxy-v2-kinematic-lateral"
STATIONARY_DISP_GRID = tuple(round(0.3 + 0.1 * index, 4) for index in range(18))
LATERAL_TURN_GRID = tuple(round(0.3 + 0.1 * index, 4) for index in range(28))


class ProxyCalibrationError(RuntimeError):
    """C1B-01 的 provenance、数据或 proxy 契约不成立。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_snapshot(root: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--short", "--branch"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary"])
    return {"root": str(root), "head": head, "status": status, "diff_sha256": hashlib.sha256(diff).hexdigest()}


def _require_clean(snapshot: Mapping[str, Any], label: str) -> None:
    lines = str(snapshot["status"]).splitlines()
    if any(line and not line.startswith("##") for line in lines):
        raise ProxyCalibrationError(f"{label} 正式运行要求 clean worktree: {snapshot['status']}")


def trajectory_at_horizon(
    trajectory: Sequence[Sequence[float]], *, horizon_seconds: float, waypoint_hz: float
) -> list[float]:
    """把 2 Hz future waypoints 线性插值到视频 proxy 的固定 horizon。"""
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("trajectory 必须是 [N,D] 且 D>=2")
    if not 0 < horizon_seconds <= values.shape[0] / waypoint_hz:
        raise ValueError("horizon_seconds 超出 trajectory 支持")
    times = np.arange(1, values.shape[0] + 1, dtype=np.float64) / float(waypoint_hz)
    times = np.concatenate([[0.0], times])
    values = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.float64), values], axis=0)
    return [float(np.interp(horizon_seconds, times, values[:, dim])) for dim in range(values.shape[1])]


def action_class(command: str, displacement: float, *, stationary_max: float, moving_min: float) -> str | None:
    if displacement <= stationary_max:
        return "stationary"
    mapping = {"Moving_Forward": "forward", "Turning_Left": "left", "Turning_Right": "right"}
    if displacement < moving_min:
        return None
    return mapping.get(command)


def load_clip_records(source_json: Path, nuscenes_root: Path, selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    version = nuscenes_root / "v1.0-trainval"
    sample_data = json.loads((version / "sample_data.json").read_text(encoding="utf-8"))
    samples = json.loads((version / "sample.json").read_text(encoding="utf-8"))
    scenes = json.loads((version / "scene.json").read_text(encoding="utf-8"))
    annotations = json.loads((version / "sample_annotation.json").read_text(encoding="utf-8"))
    sd_by_filename = {row["filename"]: row for row in sample_data}
    sample_by_token = {row["token"]: row for row in samples}
    scene_by_token = {row["token"]: row for row in scenes}
    annotation_count = Counter(row["sample_token"] for row in annotations)
    excluded = tuple(str(word).lower() for word in selection["excluded_scene_keywords"])
    output = []
    for clip_index, clip in enumerate(payload["clips"]):
        sequence = list(clip.get("img_seq") or clip.get("img_seq_his", []) + clip.get("img_seq_fut", []))
        if len(sequence) < int(selection["source_frames"]):
            continue
        sd = sd_by_filename.get(sequence[0])
        if sd is None:
            continue
        sample = sample_by_token[sd["sample_token"]]
        scene = scene_by_token[sample["scene_token"]]
        description = str(scene.get("description", ""))
        if any(word in description.lower() for word in excluded):
            continue
        trajectory = [list(map(float, point)) for point in clip["traj_fut"][:8]]
        try:
            source_timestamps_us = [int(sd_by_filename[name]["timestamp"]) for name in sequence]
        except KeyError as error:
            raise ProxyCalibrationError(f"source frame 缺少 sample_data timestamp: {error}") from error
        if any(next_value <= value for value, next_value in zip(source_timestamps_us, source_timestamps_us[1:])):
            raise ProxyCalibrationError(f"clip {clip_index} source timestamps 非严格递增")
        horizon = trajectory_at_horizon(
            trajectory,
            horizon_seconds=float(selection["horizon_seconds"]),
            waypoint_hz=float(selection["waypoint_hz"]),
        )
        displacement = float(math.hypot(horizon[0], horizon[1]))
        class_name = action_class(
            str(clip.get("cmd", "")), displacement,
            stationary_max=float(selection["stationary_max_displacement_m"]),
            moving_min=float(selection["moving_min_displacement_m"]),
        )
        if class_name is None or annotation_count[sample["token"]] < int(selection["minimum_annotations"]):
            continue
        output.append({
            "clip_index": clip_index,
            "clip_token": str(clip.get("token", clip_index)),
            "lidar_pc_token": str(clip.get("lidar_pc_token", clip.get("token", clip_index))),
            "sample_data_token": sd["token"],
            "sample_token": sample["token"],
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "scene_description": description,
            "command": str(clip.get("cmd", "")),
            "action_class": class_name,
            "trajectory": trajectory,
            "trajectory_at_horizon": horizon,
            "target_displacement_m": displacement,
            "target_lateral_m": float(horizon[1]),
            "annotation_count": int(annotation_count[sample["token"]]),
            "source_frames": sequence,
            "source_timestamps_us": source_timestamps_us,
        })
    return output


def _stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode("utf-8")).hexdigest()


def _scene_representatives(records: Sequence[Mapping[str, Any]], class_name: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in records if row["action_class"] == class_name]
    median = float(np.median([row["target_displacement_m"] for row in selected]))
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_scene[str(row["scene_token"])].append(row)
    representatives = []
    for values in by_scene.values():
        values.sort(key=lambda row: (
            abs(float(row["target_displacement_m"]) - median),
            -int(row["annotation_count"]), int(row["clip_index"]),
        ))
        representatives.append(values[0])
    return representatives


def select_scene_sets(records: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(selection["selection_seed"])
    used_scenes: set[str] = set()
    screen: list[dict[str, Any]] = []
    representatives = {name: _scene_representatives(records, name) for name in CLASSES}
    screen_counts = {str(key): int(value) for key, value in selection["screen_counts"].items()}
    order = ("left", "right", "stationary", "forward")
    for class_name in order:
        candidates = sorted(
            representatives[class_name],
            key=lambda row: (
                -int(row["annotation_count"]),
                _stable_key(seed, "screen", class_name, str(row["scene_token"])),
            ),
        )
        chosen = [row for row in candidates if row["scene_token"] not in used_scenes][:screen_counts[class_name]]
        if len(chosen) != screen_counts[class_name]:
            raise ProxyCalibrationError(f"screen {class_name} scene 不足")
        for index, row in enumerate(chosen):
            used_scenes.add(str(row["scene_token"]))
            screen.append({
                **row,
                "context_id": f"screen-{class_name}-{index:02d}",
                "seed": int(selection["screen_seed_base"]) + len(screen),
                "selection_reason": "source metadata only: scene-disjoint, non-night/rain, annotation-rich",
            })

    calibration: list[dict[str, Any]] = []
    per_class = int(selection["calibration_per_class"])
    fit_per_class = int(selection["calibration_fit_per_class"])
    if not 0 < fit_per_class < per_class:
        raise ValueError("calibration_fit_per_class 必须在 (0, calibration_per_class)")
    for class_name in order:
        candidates = [row for row in representatives[class_name] if row["scene_token"] not in used_scenes]
        candidates.sort(key=lambda row: _stable_key(seed, "calibration", class_name, str(row["scene_token"])))
        chosen = candidates[:per_class]
        if len(chosen) != per_class:
            raise ProxyCalibrationError(f"calibration {class_name} scene 不足")
        for index, row in enumerate(chosen):
            used_scenes.add(str(row["scene_token"]))
            calibration.append({
                **row,
                "calibration_id": f"cal-{class_name}-{index:02d}",
                "calibration_split": "fit" if index < fit_per_class else "eval",
                "selection_reason": "stable scene hash; no generated future inspected",
            })

    screen_scenes = {row["scene_token"] for row in screen}
    calibration_scenes = {row["scene_token"] for row in calibration}
    if len(screen_scenes) != len(screen) or len(calibration_scenes) != len(calibration):
        raise ProxyCalibrationError("screen/calibration 内部 scene 不唯一")
    if screen_scenes & calibration_scenes:
        raise ProxyCalibrationError("screen 与 calibration scene 重叠")
    return {
        "screen": sorted(screen, key=lambda row: row["context_id"]),
        "calibration": sorted(calibration, key=lambda row: row["calibration_id"]),
        "counts": {
            "screen": Counter(row["action_class"] for row in screen),
            "calibration": Counter(row["action_class"] for row in calibration),
            "calibration_fit": Counter(row["action_class"] for row in calibration if row["calibration_split"] == "fit"),
            "calibration_eval": Counter(row["action_class"] for row in calibration if row["calibration_split"] == "eval"),
        },
    }


def build_asset_plan(selection_result: Mapping[str, Any], nuscenes_root: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    required: set[str] = set()
    for row in selection_result["screen"]:
        required.update(row["source_frames"][: int(selection["source_frames"])])
    for row in selection_result["calibration"]:
        required.update(row["source_frames"][: int(selection["calibration_rgb_frames"])])
    required_sorted = sorted(required)
    missing = [name for name in required_sorted if not (nuscenes_root / name).is_file()]
    existing_sizes = [(nuscenes_root / name).stat().st_size for name in required_sorted if (nuscenes_root / name).is_file()]
    average = int(np.median(existing_sizes)) if existing_sizes else 200_000
    return {
        "protocol": "resim-c1b01-exact-assets-v1",
        "nuscenes_root": str(nuscenes_root),
        "required": required_sorted,
        "missing": missing,
        "required_count": len(required_sorted),
        "missing_count": len(missing),
        "estimated_missing_bytes": int(len(missing) * average),
        "selection_fingerprint": _sha256_json({
            "screen": selection_result["screen"], "calibration": selection_result["calibration"]
        }),
    }


def _load_frames(paths: Sequence[Path], height: int, width: int) -> torch.Tensor:
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            arrays.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1))
    frames = torch.stack(arrays).float() / 255.0
    source_h, source_w = frames.shape[-2:]
    scale = max(height / source_h, width / source_w)
    resized_h, resized_w = math.ceil(source_h * scale), math.ceil(source_w * scale)
    frames = F.interpolate(frames, size=(resized_h, resized_w), mode="bicubic", align_corners=False, antialias=True)
    top, left = (resized_h - height) // 2, (resized_w - width) // 2
    return frames[:, :, top : top + height, left : left + width].clamp(0, 1)


def source_quality(frames: torch.Tensor) -> dict[str, Any]:
    history = frames[:9].float()
    temporal = (history[1:] - history[:-1]).abs().mean() if history.shape[0] > 1 else torch.tensor(0.0)
    gray = history.mean(dim=1)
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    return {
        "mean": float(history.mean()), "std": float(history.std()),
        "black_fraction": float((history <= 0.01).float().mean()),
        "white_fraction": float((history >= 0.99).float().mean()),
        "temporal_abs_mean": float(temporal),
        "gradient_abs_mean": float((dx.abs().mean() + dy.abs().mean()) * 0.5),
    }


@torch.no_grad()
def flow_with_confidence_chunked(
    provider: RAFTFlow, frames: torch.Tensor, *, pair_batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """按连续 pair 小批量运行 RAFT，避免 24-pair future 一次性占满 4090。"""
    if frames.ndim != 4 or frames.shape[0] < 2 or pair_batch_size <= 0:
        raise ValueError("frames 必须是 [T,3,H,W] 且 pair_batch_size>0")
    forward, confidence = [], []
    for start in range(0, frames.shape[0] - 1, pair_batch_size):
        end = min(start + pair_batch_size, frames.shape[0] - 1)
        source, target = frames[start:end], frames[start + 1 : end + 1]
        fwd = provider.flow(source, target)
        bwd = provider.flow(target, source)
        forward.append(fwd)
        confidence.append(provider._fb_consistency(fwd, bwd))
    return torch.cat(forward, dim=0), torch.cat(confidence, dim=0)


def resample_future_indices(
    timestamps_us: Sequence[int], *, start_index: int, horizon_seconds: float, fps: float
) -> list[int]:
    """按真实 source timestamp 单调最近邻重采样 0..horizon。

    nuScenes CAM_FRONT 间隔常混有 0.05/0.10/0.15/0.20 s；独立最近邻会在缺口处撞到同一帧。
    因此每个目标时刻只在上一选中帧之后的候选中取最近邻，以保持严格递增。
    """
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    if timestamps.ndim != 1 or not 0 <= start_index < len(timestamps) or horizon_seconds <= 0 or fps <= 0:
        raise ValueError("timestamp resampling 参数无效")
    count = int(round(horizon_seconds * fps)) + 1
    targets = timestamps[start_index] + np.arange(count, dtype=np.float64) * (1_000_000.0 / fps)
    if targets[-1] > timestamps[-1] + 0.5 * (1_000_000.0 / fps):
        raise ProxyCalibrationError("source sequence 不覆盖 proxy future horizon")
    indices: list[int] = []
    previous = start_index - 1
    for target in targets:
        candidates = np.arange(max(previous + 1, start_index), len(timestamps))
        if candidates.size == 0:
            raise ProxyCalibrationError("timestamp 单调最近邻重采样耗尽候选帧")
        best = int(candidates[np.argmin(np.abs(timestamps[candidates] - target))])
        indices.append(best)
        previous = best
    if any(next_index <= index for index, next_index in zip(indices, indices[1:])):
        raise ProxyCalibrationError("timestamp 最近邻重采样产生重复/逆序 frame")
    return indices


def quality_passes(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    return (
        float(thresholds["minimum_mean"]) <= float(metrics["mean"]) <= float(thresholds["maximum_mean"])
        and float(metrics["std"]) >= float(thresholds["minimum_std"])
        and float(metrics["black_fraction"]) <= float(thresholds["maximum_black_fraction"])
        and float(metrics["white_fraction"]) <= float(thresholds["maximum_white_fraction"])
        and float(metrics["gradient_abs_mean"]) >= float(thresholds["minimum_gradient_abs_mean"])
    )


def affine_proxy_features(diagnostics: Mapping[str, Any], *, height: int, width: int) -> dict[str, Any]:
    pair_rows = list(diagnostics.get("pairs", []))
    values = []
    residuals = []
    confidences = []
    for row in pair_rows:
        if not row.get("valid"):
            continue
        theta = np.asarray(row["coefficients"], dtype=np.float64)
        if theta.shape != (3, 2) or not np.isfinite(theta).all():
            continue
        tx, ty = theta[0, 0] / width, theta[0, 1] / height
        dxx, dyx = theta[1, 0] / width, theta[1, 1] / height
        dxy, dyy = theta[2, 0] / width, theta[2, 1] / height
        divergence, curl = dxx + dyy, dyx - dxy
        anisotropy, shear = dxx - dyy, dyx + dxy
        yaw = math.atan2(curl, 2.0 + dxx + dyy)
        energy = math.sqrt(tx * tx + ty * ty + dxx * dxx + dyx * dyx + dxy * dxy + dyy * dyy)
        values.append([tx, ty, divergence, curl, anisotropy, shear, energy, yaw])
        residuals.append(float(row["residual_median_px"]))
        confidences.append(float(row["confidence_mean"]))
    if not values:
        return {"valid": False, "reason": "no_valid_affine_pairs", "features": None, "valid_pair_count": 0}
    matrix = np.asarray(values, dtype=np.float64)
    mid = max(1, matrix.shape[0] // 2)
    sum_tx, sum_ty = float(matrix[:, 0].sum()), float(matrix[:, 1].sum())
    sum_yaw = float(matrix[:, 7].sum())
    sum_tx_early, sum_tx_late = float(matrix[:mid, 0].sum()), float(matrix[mid:, 0].sum())
    sum_yaw_early, sum_yaw_late = float(matrix[:mid, 7].sum()), float(matrix[mid:, 7].sum())
    features = np.asarray([
        *matrix[:, :6].sum(axis=0).tolist(),
        *np.median(matrix[:, :4], axis=0).tolist(),
        float(np.median(matrix[:, 6])), float(np.median(residuals)), float(np.median(confidences)),
        len(values) / max(len(pair_rows), 1),
        sum_yaw, abs(sum_yaw), float(np.median(matrix[:, 7])),
        sum_tx_early, sum_tx_late, sum_yaw_early, sum_yaw_late,
        sum_tx_late - sum_tx_early, sum_yaw_late - sum_yaw_early,
        sum_tx / (abs(sum_ty) + 1e-3),
        sum_yaw / (abs(sum_ty) + abs(sum_tx) + 1e-3),
    ], dtype=np.float64)
    return {
        "valid": bool(np.isfinite(features).all()),
        "features": features.tolist(),
        "feature_names": list(FEATURE_NAMES),
        "valid_pair_count": len(values),
        "pair_count": len(pair_rows),
    }


def kinematic_class_from_motion(
    displacement_m: float, lateral_m: float, *, stationary_disp_m: float, lateral_turn_m: float
) -> str:
    """由预测位移/侧向导出 class；阈值只允许在 fit 上选择。"""
    if displacement_m < stationary_disp_m:
        return "stationary"
    if lateral_m > lateral_turn_m:
        return "left"
    if lateral_m < -lateral_turn_m:
        return "right"
    return "forward"


def select_kinematic_thresholds(
    actual_class: Sequence[str],
    predicted_displacement: np.ndarray,
    predicted_lateral: np.ndarray,
) -> dict[str, float]:
    """仅在 fit split 上网格搜索，最大化 moving BA + turn-sign。"""
    best: dict[str, float] | None = None
    for stationary_disp in STATIONARY_DISP_GRID:
        for lateral_turn in LATERAL_TURN_GRID:
            predicted = [
                kinematic_class_from_motion(
                    float(displacement), float(lateral),
                    stationary_disp_m=stationary_disp, lateral_turn_m=lateral_turn,
                )
                for displacement, lateral in zip(predicted_displacement, predicted_lateral)
            ]
            moving_indices = [index for index, name in enumerate(actual_class) if name in MOVING_CLASSES]
            turning_indices = [index for index, name in enumerate(actual_class) if name in ("left", "right")]
            moving_ba = _balanced_accuracy(
                [actual_class[index] for index in moving_indices],
                [predicted[index] for index in moving_indices],
                MOVING_CLASSES,
            )
            turn_acc = (
                sum(predicted[index] == actual_class[index] for index in turning_indices) / len(turning_indices)
                if turning_indices else 0.0
            )
            if moving_ba is None:
                continue
            score = float(moving_ba) + float(turn_acc)
            candidate = {
                "score": score,
                "stationary_disp_m": float(stationary_disp),
                "lateral_turn_m": float(lateral_turn),
                "fit_moving_balanced_accuracy": float(moving_ba),
                "fit_turn_sign_accuracy": float(turn_acc),
            }
            if best is None or candidate["score"] > best["score"] + 1e-12:
                best = candidate
            elif best is not None and abs(candidate["score"] - best["score"]) <= 1e-12:
                # 并列时取更保守（更大）阈值，降低假转向
                if (
                    candidate["lateral_turn_m"], candidate["stationary_disp_m"]
                ) > (
                    best["lateral_turn_m"], best["stationary_disp_m"]
                ):
                    best = candidate
    if best is None:
        raise ProxyCalibrationError("kinematic 阈值网格未找到合法 fit 解")
    return best


def _ridge(x: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    augmented = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1) @ weights


def _balanced_accuracy(actual: Sequence[str], predicted: Sequence[str], classes: Sequence[str]) -> float | None:
    recalls = []
    for name in classes:
        indices = [index for index, value in enumerate(actual) if value == name]
        if not indices:
            return None
        recalls.append(sum(predicted[index] == name for index in indices) / len(indices))
    return float(np.mean(recalls))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def calibrate_proxy(rows: Sequence[Mapping[str, Any]], proxy_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = [dict(row) for row in rows if row.get("proxy_valid")]
    fit = [row for row in valid if row["calibration_split"] == "fit"]
    evaluation = [row for row in valid if row["calibration_split"] == "eval"]
    minimum = int(proxy_cfg["minimum_valid_per_split_class"])
    support = {
        split: Counter(row["action_class"] for row in values)
        for split, values in (("fit", fit), ("eval", evaluation))
    }
    if any(support[split][name] < minimum for split in ("fit", "eval") for name in CLASSES):
        raise ProxyCalibrationError(f"proxy scene support 不足: {support}")
    x_fit = np.asarray([row["proxy_features"] for row in fit], dtype=np.float64)
    x_eval = np.asarray([row["proxy_features"] for row in evaluation], dtype=np.float64)
    mean, scale = x_fit.mean(axis=0), x_fit.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z_fit, z_eval = (x_fit - mean) / scale, (x_eval - mean) / scale
    alpha = float(proxy_cfg["ridge_alpha"])
    displacement_weights = _ridge(
        z_fit, np.asarray([row["target_displacement_m"] for row in fit], dtype=np.float64), alpha
    )
    lateral_weights = _ridge(
        z_fit, np.asarray([row["target_lateral_m"] for row in fit], dtype=np.float64), alpha
    )
    fit_displacement = np.maximum(0.0, _predict(z_fit, displacement_weights))
    fit_lateral = _predict(z_fit, lateral_weights)
    threshold_selection = select_kinematic_thresholds(
        [row["action_class"] for row in fit], fit_displacement, fit_lateral
    )
    stationary_disp_m = float(threshold_selection["stationary_disp_m"])
    lateral_turn_m = float(threshold_selection["lateral_turn_m"])
    predicted_displacement = np.maximum(0.0, _predict(z_eval, displacement_weights))
    predicted_lateral = _predict(z_eval, lateral_weights)
    predicted_class = [
        kinematic_class_from_motion(
            float(displacement), float(lateral),
            stationary_disp_m=stationary_disp_m, lateral_turn_m=lateral_turn_m,
        )
        for displacement, lateral in zip(predicted_displacement, predicted_lateral)
    ]
    actual_class = [row["action_class"] for row in evaluation]
    actual_displacement = np.asarray([row["target_displacement_m"] for row in evaluation], dtype=np.float64)
    actual_lateral = np.asarray([row["target_lateral_m"] for row in evaluation], dtype=np.float64)
    moving_indices = [index for index, name in enumerate(actual_class) if name in MOVING_CLASSES]
    turning_indices = [index for index, name in enumerate(actual_class) if name in ("left", "right")]
    constant_class = Counter(row["action_class"] for row in fit).most_common(1)[0][0]
    command_to_class = {"Moving_Forward": "forward", "Turning_Left": "left", "Turning_Right": "right"}
    constant_displacement = float(np.mean([row["target_displacement_m"] for row in fit]))
    command_means = {
        command: float(np.mean([row["target_displacement_m"] for row in fit if row["command"] == command]))
        for command in sorted({row["command"] for row in fit})
    }
    command_prediction = np.asarray([command_means[row["command"]] for row in evaluation])
    constant_prediction = np.full(len(evaluation), constant_displacement)
    stationary_predictions = predicted_displacement[
        np.asarray([name == "stationary" for name in actual_class], dtype=bool)
    ]
    metrics = {
        "support": {key: dict(value) for key, value in support.items()},
        "proxy_protocol": PROXY_PROTOCOL,
        "threshold_selection": threshold_selection,
        "moving_balanced_accuracy": _balanced_accuracy(
            [actual_class[i] for i in moving_indices], [predicted_class[i] for i in moving_indices], MOVING_CLASSES
        ),
        "four_class_balanced_accuracy": _balanced_accuracy(actual_class, predicted_class, CLASSES),
        "turn_sign_accuracy": (
            sum(predicted_class[i] == actual_class[i] for i in turning_indices) / len(turning_indices)
            if turning_indices else None
        ),
        "displacement_spearman": _spearman(predicted_displacement, actual_displacement),
        "lateral_spearman": _spearman(predicted_lateral, actual_lateral),
        "displacement_mae_m": float(np.mean(np.abs(predicted_displacement - actual_displacement))),
        "lateral_mae_m": float(np.mean(np.abs(predicted_lateral - actual_lateral))),
        "constant_displacement_mae_m": float(np.mean(np.abs(constant_prediction - actual_displacement))),
        "command_only_displacement_mae_m": float(np.mean(np.abs(command_prediction - actual_displacement))),
        "constant_moving_balanced_accuracy": _balanced_accuracy(
            [actual_class[i] for i in moving_indices], [constant_class for _ in moving_indices], MOVING_CLASSES
        ),
        "command_only_four_class_balanced_accuracy": _balanced_accuracy(
            actual_class, [command_to_class[row["command"]] for row in evaluation], CLASSES
        ),
        "command_only_classification_note": "request-label leakage baseline; reported but never required to beat",
        "stationary_predicted_displacement_m": {
            "count": int(len(stationary_predictions)),
            "median": float(np.median(stationary_predictions)) if len(stationary_predictions) else 0.0,
            "p95": float(np.quantile(stationary_predictions, 0.95)) if len(stationary_predictions) else 0.0,
            "max": float(np.max(stationary_predictions)) if len(stationary_predictions) else 0.0,
        },
        "evaluation_predictions": [
            {
                "calibration_id": row["calibration_id"], "scene_name": row["scene_name"],
                "actual_class": actual_name, "predicted_class": predicted_name,
                "actual_displacement_m": float(actual_disp),
                "predicted_displacement_m": float(predicted_disp),
                "actual_lateral_m": float(actual_lat),
                "predicted_lateral_m": float(predicted_lat),
            }
            for row, actual_name, predicted_name, actual_disp, predicted_disp, actual_lat, predicted_lat in zip(
                evaluation, actual_class, predicted_class, actual_displacement.tolist(),
                predicted_displacement.tolist(), actual_lateral.tolist(), predicted_lateral.tolist(),
            )
        ],
    }
    model = {
        "protocol": PROXY_PROTOCOL,
        "feature_names": list(FEATURE_NAMES), "classes": list(CLASSES),
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "displacement_weights": displacement_weights.tolist(),
        "lateral_weights": lateral_weights.tolist(),
        "stationary_disp_m": stationary_disp_m,
        "lateral_turn_m": lateral_turn_m,
        "threshold_selection": threshold_selection,
        "ridge_alpha": alpha, "command_displacement_means": command_means,
        "constant_displacement_mean": constant_displacement,
        "stationary_false_motion_p95_m": metrics["stationary_predicted_displacement_m"]["p95"],
    }
    for row, predicted_name, predicted_disp, predicted_lat in zip(
        evaluation, predicted_class, predicted_displacement.tolist(), predicted_lateral.tolist()
    ):
        row["predicted_class"] = predicted_name
        row["predicted_displacement_m"] = float(predicted_disp)
        row["predicted_lateral_m"] = float(predicted_lat)
    return model, metrics


def predict_proxy(model: Mapping[str, Any], features: Sequence[float]) -> dict[str, Any]:
    """使用冻结的 C1B-01 kinematic lateral 校准器推理，供 C1B-02 生成视频复用。"""
    feature = np.asarray(features, dtype=np.float64)
    names = tuple(model["feature_names"])
    if names != FEATURE_NAMES or feature.shape != (len(FEATURE_NAMES),):
        raise ValueError("proxy feature schema 不匹配")
    if str(model.get("protocol")) != PROXY_PROTOCOL:
        raise ValueError(f"proxy protocol 不匹配: {model.get('protocol')}")
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    z = ((feature - mean) / scale)[None]
    displacement = float(max(0.0, _predict(z, np.asarray(model["displacement_weights"], dtype=np.float64))[0]))
    lateral = float(_predict(z, np.asarray(model["lateral_weights"], dtype=np.float64))[0])
    predicted_class = kinematic_class_from_motion(
        displacement, lateral,
        stationary_disp_m=float(model["stationary_disp_m"]),
        lateral_turn_m=float(model["lateral_turn_m"]),
    )
    return {
        "predicted_class": predicted_class,
        "predicted_displacement_m": displacement,
        "predicted_lateral_m": lateral,
    }


def calibration_checks(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    moving = metrics.get("moving_balanced_accuracy")
    sign = metrics.get("turn_sign_accuracy")
    correlation = metrics.get("displacement_spearman")
    mae = float(metrics["displacement_mae_m"])
    return {
        "moving_balanced_accuracy": moving is not None and moving >= float(thresholds["minimum_balanced_accuracy"]),
        "turn_sign_accuracy": sign is not None and sign >= float(thresholds["minimum_turn_sign_accuracy"]),
        "displacement_spearman": correlation is not None and correlation >= float(thresholds["minimum_displacement_spearman"]),
        "beats_constant_displacement": mae < float(metrics["constant_displacement_mae_m"]),
        "beats_command_only_displacement": mae < float(metrics["command_only_displacement_mae_m"]),
    }


def prepare_assets(config_path: Path, output_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(config_path))
    records = load_clip_records(Path(str(cfg.paths.source_json)), Path(str(cfg.paths.nuscenes_root)), cfg.selection)
    selected = select_scene_sets(records, cfg.selection)
    plan = build_asset_plan(selected, Path(str(cfg.paths.nuscenes_root)), cfg.selection)
    plan["screen"] = selected["screen"]
    plan["calibration"] = selected["calibration"]
    plan["selection_counts"] = selected["counts"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(output_path), plan)
    return plan


def _run(config_path: Path) -> tuple[Path, dict[str, Any]]:
    cfg = OmegaConf.load(str(config_path))
    motion_root = Path(str(cfg.paths.motion_proj_root)).resolve()
    resim_root = Path(str(cfg.paths.resim_root)).resolve()
    motion_git, resim_git = _git_snapshot(motion_root), _git_snapshot(resim_root)
    _require_clean(motion_git, "motion_proj")
    _require_clean(resim_git, "ReSim")
    run_dir = Path(str(cfg.paths.output_root)).resolve() / str(cfg.task_id) / str(cfg.run_id)
    if run_dir.exists():
        raise FileExistsError(f"正式 run ID 不可复用: {run_dir}")
    run_dir.mkdir(parents=True)
    summary: dict[str, Any] = {
        "task_id": str(cfg.task_id), "run_id": str(cfg.run_id), "status": "running", "started_at": _utc_now(),
        "seed": int(cfg.selection.selection_seed), "git": {"motion_proj": motion_git, "resim": resim_git},
    }
    try:
        disk = shutil.disk_usage(Path(str(cfg.paths.disk_root)))
        projected_free = disk.free - int(cfg.disk.estimated_peak_bytes)
        summary["disk_before"] = {
            "root": str(cfg.paths.disk_root), "free_bytes": disk.free,
            "estimated_peak_bytes": int(cfg.disk.estimated_peak_bytes),
            "minimum_free_bytes": int(cfg.disk.minimum_free_bytes), "projected_free_bytes": projected_free,
        }
        if projected_free < int(cfg.disk.minimum_free_bytes):
            raise ProxyCalibrationError("C1B-01 磁盘安全门禁失败")
        records = load_clip_records(Path(str(cfg.paths.source_json)), Path(str(cfg.paths.nuscenes_root)), cfg.selection)
        selected = select_scene_sets(records, cfg.selection)
        asset_plan = build_asset_plan(selected, Path(str(cfg.paths.nuscenes_root)), cfg.selection)
        if asset_plan["missing"]:
            raise FileNotFoundError(f"C1B-01 精确资产缺失 {len(asset_plan['missing'])} 个")
        source_path = Path(str(cfg.paths.source_json))
        protocol = {
            "task_id": str(cfg.task_id), "selection": OmegaConf.to_container(cfg.selection, resolve=True),
            "proxy": OmegaConf.to_container(cfg.proxy, resolve=True),
            "thresholds": OmegaConf.to_container(cfg.thresholds, resolve=True),
            "screen_selection_fingerprint": asset_plan["selection_fingerprint"],
        }
        summary.update({
            "source_json_sha256": _file_sha256(source_path),
            "selection_fingerprint": asset_plan["selection_fingerprint"],
            "protocol_fingerprint": _sha256_json(protocol), "asset_plan": asset_plan,
        })
        atomic_write_json(str(run_dir / "manifest.json"), summary)
        atomic_write_json(str(run_dir / "frozen_protocol.json"), protocol)
        atomic_write_text(str(run_dir / "frozen_protocol.sha256"), summary["protocol_fingerprint"] + "\n")
        _write_jsonl(run_dir / "screen_contexts.jsonl", selected["screen"])
        _write_jsonl(run_dir / "calibration_scenes.jsonl", selected["calibration"])
        atomic_write_json(str(run_dir / "asset_plan.json"), asset_plan)
        atomic_write_text(
            str(run_dir / "command.sh"),
            f"python -m motion_proj.diagnostics.resim_c1_proxy --config {config_path}\n",
        )

        height, width = map(int, cfg.proxy.video_size)
        root = Path(str(cfg.paths.nuscenes_root))
        quality_rows = []
        for row in selected["screen"]:
            frames = _load_frames([root / name for name in row["source_frames"][:9]], height, width)
            metrics = source_quality(frames)
            passed = quality_passes(metrics, cfg.source_quality)
            quality_rows.append({"scope": "screen", "id": row["context_id"], "scene_name": row["scene_name"], "passed": passed, **metrics})
        if not all(row["passed"] for row in quality_rows):
            raise ProxyCalibrationError("screen source-history quality gate 失败")

        torch.manual_seed(int(cfg.selection.selection_seed))
        np.random.seed(int(cfg.selection.selection_seed) % (2**32 - 1))
        torch.backends.cudnn.benchmark = False
        raft = RAFTFlow(device=str(cfg.proxy.raft_device))
        proxy_rows = []
        for index, row in enumerate(selected["calibration"]):
            paths = [root / name for name in row["source_frames"][: int(cfg.selection.calibration_rgb_frames)]]
            frames = _load_frames(paths, height, width)
            quality = source_quality(frames)
            quality_ok = quality_passes(quality, cfg.source_quality)
            quality_rows.append({
                "scope": "calibration", "id": row["calibration_id"], "scene_name": row["scene_name"],
                "passed": quality_ok, **quality,
            })
            if not quality_ok:
                proxy_row = {**row, "proxy_valid": False, "proxy_reason": "source_quality", "source_quality": quality}
            else:
                indices = resample_future_indices(
                    row["source_timestamps_us"],
                    start_index=int(cfg.proxy.future_start_frame),
                    horizon_seconds=float(cfg.selection.horizon_seconds),
                    fps=float(cfg.proxy.source_resample_fps),
                )
                proxy_frames = frames[indices].mul(2).sub(1)
                observed, confidence = flow_with_confidence_chunked(
                    raft, proxy_frames.to(str(cfg.proxy.raft_device)),
                    pair_batch_size=int(cfg.proxy.pair_batch_size),
                )
                estimate = fit_affine_background_flow(
                    observed, confidence, **OmegaConf.to_container(cfg.proxy.affine_fit, resolve=True)
                )
                feature_result = affine_proxy_features(estimate.diagnostics, height=height, width=width)
                proxy_row = {
                    **row, "proxy_valid": bool(feature_result["valid"]),
                    "proxy_reason": None if feature_result["valid"] else feature_result.get("reason"),
                    "proxy_features": feature_result.get("features"),
                    "feature_names": feature_result.get("feature_names"),
                    "valid_pair_count": feature_result.get("valid_pair_count"),
                    "pair_count": feature_result.get("pair_count"),
                    "source_resample_indices": indices,
                    "source_resample_timestamps_us": [row["source_timestamps_us"][i] for i in indices],
                    "source_quality": quality,
                    "affine_diagnostics": estimate.diagnostics,
                }
            proxy_rows.append(proxy_row)
            _write_jsonl(run_dir / "proxy_features.jsonl", proxy_rows)
            print(f"C1B-01 proxy {index + 1}/{len(selected['calibration'])}: {row['calibration_id']}", flush=True)
            torch.cuda.empty_cache()
        _write_jsonl(run_dir / "source_quality.jsonl", quality_rows)
        model, metrics = calibrate_proxy(proxy_rows, cfg.proxy)
        checks = calibration_checks(metrics, cfg.thresholds)
        metrics["checks"] = checks
        metrics["passed"] = all(checks.values())
        atomic_write_json(str(run_dir / "proxy_model.json"), model)
        atomic_write_json(str(run_dir / "calibration_metrics.json"), metrics)
        _write_jsonl(run_dir / "proxy_features.jsonl", proxy_rows)
        summary.update({
            "proxy_model_fingerprint": _sha256_json(model), "calibration_metrics": metrics,
            "selected_screen_resolution": [height, width], "ended_at": _utc_now(),
        })
        if metrics["passed"]:
            summary.update({"status": "completed", "exit_reason": "c1b01_passed"})
            atomic_write_text(str(run_dir / "PASSED"), _sha256_json(summary) + "\n")
        else:
            summary.update({"status": "blocked", "exit_reason": "proxy_not_identifiable"})
            atomic_write_text(str(run_dir / "BLOCKED"), _sha256_json(summary) + "\n")
    except Exception as error:
        summary.update({
            "status": "failed", "ended_at": _utc_now(), "exit_reason": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        atomic_write_text(str(run_dir / "FAILED"), _sha256_json(summary) + "\n")
        atomic_write_json(str(run_dir / "summary.json"), summary)
        raise
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(
        str(run_dir / "RUN_PROVENANCE.md"),
        "\n".join([
            "# C1B-01 Run Provenance", "", f"- run_id: `{summary['run_id']}`",
            f"- status: `{summary['status']}`", f"- motion_proj_head: `{motion_git['head']}`",
            f"- resim_head: `{resim_git['head']}`", f"- selection_fingerprint: `{summary.get('selection_fingerprint')}`",
            f"- protocol_fingerprint: `{summary.get('protocol_fingerprint')}`",
            "- generated_future_inspected_for_selection: `false`", "",
        ]),
    )
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-assets", default=None)
    args = parser.parse_args()
    if args.prepare_assets:
        result = prepare_assets(Path(args.config), Path(args.prepare_assets))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    run_dir, summary = _run(Path(args.config))
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
