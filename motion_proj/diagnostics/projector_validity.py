"""P0：不改动正式 projector 的 point-track tube physical-validity 诊断。

本模块刻意只读取冻结的 Base replay RGB，并在内存中比较四个候选：

* ``P-ID``：identity；
* ``P-CUR``：现有 ``smooth_track``（作为已知风险基线）；
* ``P-CON``：端点/visibility/support 受约束的 robust smoother；
* ``P-UNC``：在 P-CON 之上按 tracker uncertainty 接受修正。

它不是正式 ``DynamicsProjector`` 的替代实现，也不写 replay cache。所有“object”
表述均指 generated point-track tube component，而非 dataset instance supervision。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from omegaconf import OmegaConf

from ..auditor.generated_tracks import RAFTChainGeneratedTrackProvider
from ..auditor.state import Track
from ..cache.dataset import ProjectionCacheDataset
from ..config import config_fingerprint, get_paths, load_config, save_resolved_config
from ..projector.smoothing import smooth_track
from ..projector.support import classify_support
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video


PROTOCOL_VERSION = "autoresearch-p0-projector-validity-v1"
CANDIDATES = ("P-ID", "P-CUR", "P-CON", "P-UNC")
PRIMARY_STRATA = {"dynamic_residual", "foreground_candidate"}
REVIEW_VALUES = {"valid", "invalid", "uncertain"}


@dataclass
class CandidateTracks:
    """一个候选的轨迹与逐点 uncertainty/SNR 审计数据。"""

    name: str
    tracks: list[Track]
    uncertainty: list[torch.Tensor]
    correction_snr: list[torch.Tensor]
    corrected: list[torch.Tensor]
    allowed: list[torch.Tensor]


def _clone_track_cpu(track: Track) -> Track:
    return Track(
        instance_token=str(track.instance_token),
        category=str(track.category),
        xyxy=track.xyxy.detach().cpu().clone(),
        depth=track.depth.detach().cpu().clone(),
        present=track.present.detach().cpu().bool().clone(),
    )


def _track_label(track: Track) -> str:
    return str(track.category).rsplit("/", 1)[-1]


def _track_query_index(track: Track) -> int | None:
    match = re.search(r"_(\d+)$", str(track.instance_token))
    return int(match.group(1)) if match else None


def _confidence_for_track(track: Track, confidence: torch.Tensor) -> torch.Tensor:
    """从 provider query 索引恢复某一保留 track 的逐帧 confidence。"""
    index = _track_query_index(track)
    if index is None or index < 0 or index >= confidence.shape[0]:
        return torch.zeros_like(track.present, dtype=torch.float32)
    return confidence[index].detach().cpu().float().clamp(0.0, 1.0)


def _second_difference(length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if length < 3:
        return torch.empty(0, length, dtype=dtype, device=device)
    matrix = torch.zeros(length - 2, length, dtype=dtype, device=device)
    rows = torch.arange(length - 2, device=device)
    matrix[rows, rows] = 1.0
    matrix[rows, rows + 1] = -2.0
    matrix[rows, rows + 2] = 1.0
    return matrix


def _third_difference(length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if length < 4:
        return torch.empty(0, length, dtype=dtype, device=device)
    matrix = torch.zeros(length - 3, length, dtype=dtype, device=device)
    rows = torch.arange(length - 3, device=device)
    matrix[rows, rows] = -1.0
    matrix[rows, rows + 1] = 3.0
    matrix[rows, rows + 2] = -3.0
    matrix[rows, rows + 3] = 1.0
    return matrix


def _consecutive_segments(visible: torch.Tensor) -> list[tuple[int, int]]:
    """返回 ``[start, end]``，绝不跨越 tracker 的不可见空洞平滑。"""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for time, value in enumerate(visible.detach().cpu().bool().tolist()):
        if value and start is None:
            start = time
        if start is not None and (not value or time == len(visible) - 1):
            end = time if value else time - 1
            segments.append((start, end))
            start = None
    return segments


def constrained_smooth_points(
    points: torch.Tensor,
    visible: torch.Tensor,
    confidence: torch.Tensor,
    allowed: torch.Tensor,
    *,
    lambda_acceleration: float,
    lambda_jerk: float,
    robust_delta_px: float,
    robust_iterations: int,
) -> torch.Tensor:
    """对可修正可见段做 Huber-IRLS 平滑，并硬冻结所有不允许的点。

    每个连续可见段的首尾均固定，因而不会藉由改动端点改变该段净位移；第 0 帧
    也被额外固定。 ``allowed=False`` 的 frame 是严格 identity，既不产生新点，也
    不向未支持的 reappearance 传播修正。
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points 必须是 [K,2]")
    if visible.shape != points.shape[:1] or confidence.shape != points.shape[:1] or allowed.shape != points.shape[:1]:
        raise ValueError("visible/confidence/allowed 必须是 [K]")
    if lambda_acceleration < 0 or lambda_jerk < 0 or robust_delta_px <= 0 or robust_iterations <= 0:
        raise ValueError("constrained smoother 参数无效")

    output = points.detach().clone()
    visible = visible.detach().bool()
    allowed = allowed.detach().bool() & visible
    confidence = confidence.detach().float().clamp(0.0, 1.0)
    # 缺席点维持 NaN；只有连续可见段进入线性求解。
    for start, end in _consecutive_segments(visible):
        length = end - start + 1
        if length < 3:
            continue
        data = points[start:end + 1].detach().double()
        can_correct = allowed[start:end + 1].clone()
        # 每段边界固定；frame 0 的 exactness 在此被显式而非“渲染后补丁”保证。
        can_correct[0] = False
        can_correct[-1] = False
        if start == 0:
            can_correct[0] = False
        if not bool(can_correct.any()):
            continue

        conf = confidence[start:end + 1].double().clamp_min(1.0e-3)
        d2 = _second_difference(length, dtype=torch.float64, device=data.device)
        d3 = _third_difference(length, dtype=torch.float64, device=data.device)
        roughness = lambda_acceleration * (d2.T @ d2) + lambda_jerk * (d3.T @ d3)
        # IRLS 只降低异常观测的 fidelity；冻结点仍以精确赋值处理。
        robust = torch.ones(length, dtype=torch.float64, device=data.device)
        estimate = data.clone()
        for _ in range(int(robust_iterations)):
            weights = conf * robust
            weights[~can_correct] = 1.0e8
            system = torch.diag(weights) + roughness + 1.0e-10 * torch.eye(length, dtype=torch.float64)
            rhs = weights[:, None] * data
            estimate = torch.linalg.solve(system, rhs)
            estimate[~can_correct] = data[~can_correct]
            residual = torch.linalg.vector_norm(estimate - data, dim=-1)
            robust = torch.where(
                residual <= robust_delta_px,
                torch.ones_like(residual),
                robust_delta_px / residual.clamp_min(1.0e-12),
            )
            robust[~can_correct] = 1.0
        output[start:end + 1][can_correct] = estimate[can_correct].to(output)
    output[~visible] = float("nan")
    return output


def _translate_track(track: Track, projected_points: torch.Tensor) -> Track:
    """保持点方框尺度/深度，只平移 center；缺席帧严格保留 NaN。"""
    result = _clone_track_cpu(track)
    original = result.center
    visible = result.present
    delta = projected_points - original
    finite = visible & torch.isfinite(delta).all(dim=-1)
    if bool(finite.any()):
        shift = torch.stack([delta[:, 0], delta[:, 1], delta[:, 0], delta[:, 1]], dim=-1)
        result.xyxy[finite] = result.xyxy[finite] + shift[finite]
    result.xyxy[~visible] = float("nan")
    return result


def _uncertainty(confidence: torch.Tensor, settings: Mapping[str, Any]) -> torch.Tensor:
    floor = float(settings["uncertainty_floor_px"])
    scale = float(settings["uncertainty_confidence_scale_px"])
    if floor <= 0 or scale < 0:
        raise ValueError("uncertainty 参数必须为非负且 floor 为正")
    return floor + scale * (1.0 - confidence.float().clamp(0.0, 1.0))


def build_candidate_tracks(
    tracks: list[Track],
    provider_confidence: torch.Tensor,
    image_hw: tuple[int, int],
    settings: Mapping[str, Any],
) -> dict[str, CandidateTracks]:
    """构造 P-ID/P-CUR/P-CON/P-UNC，且不修改输入 track。"""
    names = tuple(str(value) for value in settings.get("candidates", CANDIDATES))
    if names != CANDIDATES:
        raise ValueError(f"P0 candidates 必须严格为 {CANDIDATES}")
    tracks = [_clone_track_cpu(track) for track in tracks]
    support = classify_support(tracks, image_hw)
    constrained = settings["constrained"]
    confidence_floor = float(constrained["confidence_floor"])
    snr_threshold = float(constrained["snr_threshold"])
    if not 0 <= confidence_floor <= 1 or snr_threshold <= 0:
        raise ValueError("confidence_floor/SNR threshold 无效")

    identity = [_clone_track_cpu(track) for track in tracks]
    current = [smooth_track(track, lam=float(settings["current_smooth_lambda"])) for track in tracks]
    con_tracks: list[Track] = []
    unc_tracks: list[Track] = []
    zero = []
    con_uncertainty: list[torch.Tensor] = []
    con_snr: list[torch.Tensor] = []
    con_corrected: list[torch.Tensor] = []
    con_allowed: list[torch.Tensor] = []
    unc_snr: list[torch.Tensor] = []
    unc_corrected: list[torch.Tensor] = []

    for track in tracks:
        visible = track.present.bool()
        observed = track.center
        confidence = _confidence_for_track(track, provider_confidence)
        uncertainty = _uncertainty(confidence, constrained)
        stratum = _track_label(track)
        # background 是 preservation/negative relation，绝不作为 positive projection。
        allowed = (
            visible
            & support[track.instance_token].bool()
            & (confidence >= confidence_floor)
            & torch.tensor(stratum in PRIMARY_STRATA, dtype=torch.bool)
        )
        if allowed.numel():
            allowed[0] = False
        projected = constrained_smooth_points(
            observed,
            visible,
            confidence,
            allowed,
            lambda_acceleration=float(constrained["lambda_acceleration"]),
            lambda_jerk=float(constrained["lambda_jerk"]),
            robust_delta_px=float(constrained["robust_delta_px"]),
            robust_iterations=int(constrained["robust_iterations"]),
        )
        con_delta = torch.linalg.vector_norm(projected - observed, dim=-1)
        con_delta[~visible] = 0.0
        snr = con_delta / uncertainty.clamp_min(1.0e-8)
        accept = allowed & (snr >= snr_threshold)
        unc_points = observed.clone()
        unc_points[accept] = projected[accept]
        unc_points[~visible] = float("nan")
        con_tracks.append(_translate_track(track, projected))
        unc_tracks.append(_translate_track(track, unc_points))
        zero.append(torch.zeros_like(uncertainty))
        con_uncertainty.append(uncertainty)
        con_snr.append(snr)
        con_corrected.append((con_delta > 1.0e-7) & visible)
        con_allowed.append(allowed)
        unc_snr.append(snr)
        unc_corrected.append(accept)

    current_uncertainty = [_uncertainty(_confidence_for_track(track, provider_confidence), constrained) for track in tracks]
    current_snr = []
    current_corrected = []
    for original, projected, uncertainty in zip(tracks, current, current_uncertainty):
        common = original.present & projected.present
        delta = torch.zeros_like(uncertainty)
        if bool(common.any()):
            delta[common] = torch.linalg.vector_norm(projected.center[common] - original.center[common], dim=-1)
        current_snr.append(delta / uncertainty.clamp_min(1.0e-8))
        current_corrected.append(delta > 1.0e-7)

    identity_allowed = [torch.zeros_like(track.present, dtype=torch.bool) for track in tracks]
    current_allowed = [support[track.instance_token].bool() for track in tracks]
    return {
        "P-ID": CandidateTracks("P-ID", identity, zero, zero, identity_allowed, identity_allowed),
        "P-CUR": CandidateTracks("P-CUR", current, current_uncertainty, current_snr, current_corrected, current_allowed),
        "P-CON": CandidateTracks("P-CON", con_tracks, con_uncertainty, con_snr, con_corrected, con_allowed),
        "P-UNC": CandidateTracks("P-UNC", unc_tracks, con_uncertainty, unc_snr, unc_corrected, con_allowed),
    }


def _valid_velocity(points: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = points[1:] - points[:-1]
    valid = present[1:] & present[:-1]
    return velocity, valid


def _dynamics(points: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity, valid_velocity = _valid_velocity(points, present)
    acceleration = velocity[1:] - velocity[:-1]
    valid_acceleration = valid_velocity[1:] & valid_velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]
    valid_jerk = valid_acceleration[1:] & valid_acceleration[:-1]
    return velocity[valid_velocity], acceleration[valid_acceleration], jerk[valid_jerk]


def _rms(vectors: torch.Tensor) -> float | None:
    if not int(vectors.numel()):
        return None
    finite = vectors[torch.isfinite(vectors)]
    return float(finite.square().mean().sqrt()) if int(finite.numel()) else None


def _turn_sign(points: torch.Tensor, present: torch.Tensor) -> int:
    velocity = points[1:] - points[:-1]
    valid_velocity = present[1:] & present[:-1]
    valid_turn = valid_velocity[1:] & valid_velocity[:-1]
    if not bool(valid_turn.any()):
        return 0
    cross = velocity[:-1, 0] * velocity[1:, 1] - velocity[:-1, 1] * velocity[1:, 0]
    value = float(cross[valid_turn].sum())
    return 1 if value > 1.0e-8 else (-1 if value < -1.0e-8 else 0)


def _finite_quantiles(values: Iterable[float]) -> dict[str, float | None]:
    tensor = torch.tensor([float(value) for value in values if math.isfinite(float(value))], dtype=torch.float64)
    if not int(tensor.numel()):
        return {"mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "max": float(tensor.max()),
    }


def _track_row(
    *,
    candidate: CandidateTracks,
    original: Track,
    projected: Track,
    track_index: int,
    dataset_index: int,
    sample_id: str,
    support: torch.Tensor,
) -> dict[str, Any]:
    observed_points = original.center
    projected_points = projected.center
    observed_present = original.present.bool()
    projected_present = projected.present.bool()
    common = observed_present & projected_present
    correction = torch.zeros_like(observed_present, dtype=torch.float32)
    if bool(common.any()):
        correction[common] = torch.linalg.vector_norm(
            projected_points[common] - observed_points[common], dim=-1
        )
    # 写出 hard-invariant 违例，而非将其掩盖在聚合平均中。
    absent_generated = projected_present & ~observed_present
    valid_index_changed = int((projected_present != observed_present).sum())
    support_violation = common & ~support.bool() & (correction > 1.0e-7)
    visible_indices = torch.nonzero(observed_present, as_tuple=False).flatten()
    first, last = int(visible_indices[0]), int(visible_indices[-1])
    observed_net = observed_points[last] - observed_points[first]
    projected_net = projected_points[last] - projected_points[first]
    observed_norm = float(torch.linalg.vector_norm(observed_net))
    projected_norm = float(torch.linalg.vector_norm(projected_net))
    direction = float(torch.dot(observed_net, projected_net) / max(observed_norm * projected_norm, 1.0e-8))
    observed_velocity, observed_acceleration, observed_jerk = _dynamics(observed_points, observed_present)
    projected_velocity, projected_acceleration, projected_jerk = _dynamics(projected_points, projected_present)
    observed_acceleration_rms = _rms(observed_acceleration)
    projected_acceleration_rms = _rms(projected_acceleration)
    observed_jerk_rms = _rms(observed_jerk)
    projected_jerk_rms = _rms(projected_jerk)
    observed_turn = _turn_sign(observed_points, observed_present)
    projected_turn = _turn_sign(projected_points, projected_present)
    snr = candidate.correction_snr[track_index]
    uncertainty = candidate.uncertainty[track_index]
    corrected = candidate.corrected[track_index] & observed_present
    snr_values = snr[corrected]
    return {
        "candidate": candidate.name,
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "track_index": track_index,
        "track_token": original.instance_token,
        "stratum": _track_label(original),
        "point_track_tube_component": True,
        "present_count": int(observed_present.sum()),
        "projected_present_count": int(projected_present.sum()),
        "valid_time_index_changed_count": valid_index_changed,
        "visibility_expansion_count": int(absent_generated.sum()),
        "support_violation_count": int(support_violation.sum()),
        "frame0_correction_px": float(correction[0]) if bool(observed_present[0] & projected_present[0]) else None,
        "correction_px_mean": float(correction[common].mean()) if bool(common.any()) else None,
        "correction_px_max": float(correction[common].max()) if bool(common.any()) else None,
        "corrected_point_count": int(corrected.sum()),
        "allowed_point_count": int(candidate.allowed[track_index].sum()),
        "uncertainty_px_median": float(uncertainty[observed_present].median()) if bool(observed_present.any()) else None,
        "correction_snr_median": float(snr_values.median()) if int(snr_values.numel()) else None,
        "correction_snr_min": float(snr_values.min()) if int(snr_values.numel()) else None,
        "net_displacement_observed_px": observed_norm,
        "net_displacement_projected_px": projected_norm,
        "net_displacement_ratio": projected_norm / max(observed_norm, 1.0e-8),
        "direction_cosine": direction,
        "turn_direction_observed": observed_turn,
        "turn_direction_projected": projected_turn,
        "turn_direction_preserved": observed_turn == projected_turn,
        "velocity_rms_observed_px": _rms(observed_velocity),
        "velocity_rms_projected_px": _rms(projected_velocity),
        "dynamic_degree_observed_px": observed_acceleration_rms,
        "dynamic_degree_projected_px": projected_acceleration_rms,
        "dynamic_degree_ratio": (
            projected_acceleration_rms / max(observed_acceleration_rms, 1.0e-8)
            if observed_acceleration_rms is not None and projected_acceleration_rms is not None else None
        ),
        "acceleration_rms_observed_px": observed_acceleration_rms,
        "acceleration_rms_projected_px": projected_acceleration_rms,
        "jerk_rms_observed_px": observed_jerk_rms,
        "jerk_rms_projected_px": projected_jerk_rms,
    }


def _aggregate_track_rows(rows: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        primary = [row for row in candidate_rows if row["stratum"] in PRIMARY_STRATA]
        corrections = [row for row in primary if int(row["corrected_point_count"]) > 0]
        ratios = [float(row["net_displacement_ratio"]) for row in candidate_rows]
        directions = [float(row["direction_cosine"]) for row in candidate_rows]
        dynamic_ratios = [float(row["dynamic_degree_ratio"]) for row in primary if row["dynamic_degree_ratio"] is not None]
        snrs = [float(row["correction_snr_median"]) for row in corrections if row["correction_snr_median"] is not None]
        frame0 = [float(row["frame0_correction_px"]) for row in candidate_rows if row["frame0_correction_px"] is not None]
        turn_rows = [row for row in candidate_rows if int(row["turn_direction_observed"]) != 0]
        result[candidate] = {
            "track_count": len(candidate_rows),
            "primary_track_count": len(primary),
            "primary_corrected_track_count": len(corrections),
            "corrected_point_count": sum(int(row["corrected_point_count"]) for row in primary),
            "frame0_correction_px": _finite_quantiles(frame0),
            "visibility_expansion_count": sum(int(row["visibility_expansion_count"]) for row in candidate_rows),
            "valid_time_index_changed_count": sum(int(row["valid_time_index_changed_count"]) for row in candidate_rows),
            "support_violation_count": sum(int(row["support_violation_count"]) for row in candidate_rows),
            "net_displacement_ratio": _finite_quantiles(ratios),
            "direction_cosine": _finite_quantiles(directions),
            "turn_preservation_fraction": (
                sum(bool(row["turn_direction_preserved"]) for row in turn_rows) / len(turn_rows)
                if turn_rows else 1.0
            ),
            "dynamic_degree_ratio": _finite_quantiles(dynamic_ratios),
            "correction_snr": _finite_quantiles(snrs),
            "all_primary_corrections_above_snr_threshold": bool(snrs) and min(snrs) >= float(thresholds["snr_threshold"]),
        }
    return result


def _clean_track(points: torch.Tensor, visible: torch.Tensor) -> Track:
    half = 2.0
    xyxy = torch.stack([
        points[:, 0] - half, points[:, 1] - half,
        points[:, 0] + half, points[:, 1] + half,
    ], dim=-1)
    xyxy[~visible] = float("nan")
    return Track("generated_dynamic_residual_000", "generated_point/dynamic_residual", xyxy,
                 torch.ones(points.shape[0]), visible)


def _synthetic_paths(length: int) -> dict[str, torch.Tensor]:
    time = torch.arange(length, dtype=torch.float32)
    return {
        "constant_velocity": torch.stack([8.0 + 2.2 * time, 18.0 + 0.2 * time], dim=-1),
        "constant_acceleration": torch.stack([8.0 + 1.1 * time + 0.17 * time.square(), 22.0 + 0.1 * time], dim=-1),
        "brake": torch.stack([7.0 + 3.2 * time - 0.22 * time.square(), 28.0 + 0.15 * time], dim=-1),
        "smooth_turn": torch.stack([30.0 + 2.1 * time, 30.0 + 0.22 * time.square()], dim=-1),
        "lane_change": torch.stack([10.0 + 2.4 * time, 44.0 + 5.0 * torch.tanh((time - 3.5) / 1.5)], dim=-1),
    }


def synthetic_calibration(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """预注册 synthetic 集合：运动模式、jitter、单帧 outlier 与遮挡恢复。"""
    length = int(settings["synthetic"]["frame_count"])
    if length < 6:
        raise ValueError("synthetic frame_count 至少为 6")
    rows: list[dict[str, Any]] = []
    modes = _synthetic_paths(length)
    jitter = torch.stack([
        0.55 * torch.sin(torch.arange(length, dtype=torch.float32) * 1.73),
        0.45 * torch.cos(torch.arange(length, dtype=torch.float32) * 1.11),
    ], dim=-1)
    for motion, clean in modes.items():
        variants: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "clean": (clean.clone(), torch.ones(length, dtype=torch.bool)),
            "tracker_jitter": (clean + jitter, torch.ones(length, dtype=torch.bool)),
            "single_frame_outlier": (clean.clone(), torch.ones(length, dtype=torch.bool)),
        }
        variants["single_frame_outlier"][0][length // 2] += torch.tensor([3.6, -2.8])
        if motion == "lane_change":
            visible = torch.ones(length, dtype=torch.bool)
            visible[length // 2 - 1:length // 2 + 1] = False
            variants["occlusion_recovery"] = (clean + 0.35 * jitter, visible)
        for corruption, (observed, visible) in variants.items():
            track = _clean_track(observed, visible)
            # Confidence reflects a confidence-aware tracker, not oracle clean data.
            confidence = torch.full((1, length), 0.90)
            if corruption == "single_frame_outlier":
                confidence[:, length // 2] = 0.58
            if corruption == "occlusion_recovery":
                confidence[:, ~visible] = 0.0
                confidence[:, length // 2 + 1] = 0.62
            candidates = build_candidate_tracks([track], confidence, (96, 96), settings)
            for candidate in CANDIDATES:
                projected = candidates[candidate].tracks[0].center
                valid = visible & torch.isfinite(projected).all(dim=-1)
                before = torch.linalg.vector_norm(observed[valid] - clean[valid], dim=-1)
                after = torch.linalg.vector_norm(projected[valid] - clean[valid], dim=-1)
                rows.append({
                    "candidate": candidate,
                    "motion": motion,
                    "corruption": corruption,
                    "visible_count": int(valid.sum()),
                    "rmse_before_px": float(before.square().mean().sqrt()) if int(before.numel()) else None,
                    "rmse_after_px": float(after.square().mean().sqrt()) if int(after.numel()) else None,
                    "improvement_px": (
                        float(before.square().mean().sqrt() - after.square().mean().sqrt())
                        if int(before.numel()) else None
                    ),
                    "frame0_correction_px": float(torch.linalg.vector_norm(projected[0] - observed[0])),
                    "visibility_expansion_count": int((candidates[candidate].tracks[0].present & ~visible).sum()),
                })
    return rows


def _aggregate_synthetic(rows: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        clean = [row for row in selected if row["corruption"] == "clean"]
        noisy = [row for row in selected if row["corruption"] != "clean"]
        high_snr_outlier = [row for row in selected if row["corruption"] == "single_frame_outlier"]
        subuncertainty_jitter = [row for row in selected if row["corruption"] == "tracker_jitter"]
        improvements = [float(row["improvement_px"]) for row in noisy if row["improvement_px"] is not None]
        outlier_improvements = [float(row["improvement_px"]) for row in high_snr_outlier if row["improvement_px"] is not None]
        jitter_improvements = [float(row["improvement_px"]) for row in subuncertainty_jitter if row["improvement_px"] is not None]
        result[candidate] = {
            "case_count": len(selected),
            "clean_rmse_after_px": _finite_quantiles(
                [float(row["rmse_after_px"]) for row in clean if row["rmse_after_px"] is not None]
            ),
            "noisy_improvement_px": _finite_quantiles(improvements),
            "noisy_improvement_fraction": (
                sum(value > float(thresholds["minimum_noisy_improvement_px"]) for value in improvements) / len(improvements)
                if improvements else 0.0
            ),
            "high_snr_outlier_improvement_px": _finite_quantiles(outlier_improvements),
            "high_snr_outlier_improvement_fraction": (
                sum(value > float(thresholds["minimum_noisy_improvement_px"]) for value in outlier_improvements) / len(outlier_improvements)
                if outlier_improvements else 0.0
            ),
            # P-UNC 应拒绝小于 uncertainty 的 jitter；此处只检查它没有把 jitter 放大。
            "subuncertainty_jitter_improvement_px": _finite_quantiles(jitter_improvements),
            "subuncertainty_jitter_max_worsening_px": max([-value for value in jitter_improvements] + [0.0]),
            "frame0_max_correction_px": max(float(row["frame0_correction_px"]) for row in selected),
            "visibility_expansion_count": sum(int(row["visibility_expansion_count"]) for row in selected),
        }
    return result


def _candidate_machine_check(
    candidate: str,
    tracks: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    real = tracks[candidate]
    synth = synthetic[candidate]
    def ge(value: float | None, limit: float) -> bool:
        return value is not None and value >= limit
    def between(value: float | None, lo: float, hi: float) -> bool:
        return value is not None and lo <= value <= hi

    clean_median = synth["clean_rmse_after_px"]["median"]
    checks = {
        "nonidentity_primary_correction": int(real["corrected_point_count"]) > 0,
        "synthetic_clean_preserved": (
            clean_median is not None and clean_median <= float(thresholds["maximum_clean_rmse_px"])
        ),
        "synthetic_high_snr_outlier_improved": (
            float(synth["high_snr_outlier_improvement_fraction"])
            >= float(thresholds["minimum_high_snr_outlier_improvement_fraction"])
        ),
        "synthetic_subuncertainty_jitter_not_amplified": (
            float(synth["subuncertainty_jitter_max_worsening_px"])
            <= float(thresholds["maximum_subuncertainty_jitter_worsening_px"])
        ),
        "frame0_exact": (
            (real["frame0_correction_px"]["max"] or 0.0) <= float(thresholds["frame0_max_error_px"])
            and float(synth["frame0_max_correction_px"]) <= float(thresholds["frame0_max_error_px"])
        ),
        "visibility_preserved": int(real["visibility_expansion_count"]) == 0 and int(real["valid_time_index_changed_count"]) == 0 and int(synth["visibility_expansion_count"]) == 0,
        "support_preserved": int(real["support_violation_count"]) == 0,
        "net_displacement_preserved": between(
            real["net_displacement_ratio"]["median"],
            float(thresholds["net_displacement_median_min"]),
            float(thresholds["net_displacement_median_max"]),
        ) and ge(real["net_displacement_ratio"]["p10"], float(thresholds["net_displacement_p10_min"])),
        "direction_preserved": ge(real["direction_cosine"]["median"], float(thresholds["direction_median_min"])),
        "turn_preserved": float(real["turn_preservation_fraction"]) >= float(thresholds["turn_preservation_min"]),
        "dynamic_degree_preserved": between(
            real["dynamic_degree_ratio"]["median"],
            float(thresholds["dynamic_degree_median_min"]),
            float(thresholds["dynamic_degree_median_max"]),
        ),
        "correction_above_uncertainty": bool(real["all_primary_corrections_above_snr_threshold"]),
    }
    # P-ID 并非 positive projector；P-CUR 是对照，不能以它作为 rollout candidate。
    eligible_kind = candidate in {"P-CON", "P-UNC"}
    return {"candidate": candidate, "eligible_kind": eligible_kind, "checks": checks, "machine_pass": eligible_kind and all(checks.values())}


def _model_fingerprint(pretrained: str) -> str:
    root = Path(pretrained)
    if not root.is_dir():
        return sha256_json({"pretrained": pretrained})
    candidates = [
        root / "model_index.json", root / "unet" / "config.json", root / "vae" / "config.json",
        root / "scheduler" / "scheduler_config.json", root / "image_encoder" / "config.json",
    ]
    return sha256_json([
        (str(path.relative_to(root)), file_fingerprint(str(path)))
        for path in candidates if path.is_file()
    ])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _draw_panel(
    path: Path,
    base_rgb: torch.Tensor,
    original: Track,
    projected: Track,
    *,
    candidate: CandidateTracks,
    track_index: int,
    dataset_index: int,
    sample_id: str,
) -> None:
    """单张 PNG 检查 Base、轨迹、uncertainty 与局部 dynamics。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video = to_uint8_video(base_rgb)
    observed = original.center.detach().cpu()
    target = projected.center.detach().cpu()
    present = original.present.detach().cpu().bool()
    projected_present = projected.present.detach().cpu().bool()
    stratum = _track_label(original)
    palette = {"background": "#3d8ed0", "dynamic_residual": "#e74c3c", "foreground_candidate": "#2ca25f"}
    color = palette.get(stratum, "#777777")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, time, title in ((axes[0, 0], 0, "frame 0: exactness"), (axes[0, 1], len(video) - 1, "last frame")):
        axis.imshow(video[time])
        if bool(present[time]):
            axis.scatter(float(observed[time, 0]), float(observed[time, 1]), c=color, marker="o", s=55, label="observed")
        if bool(projected_present[time]):
            axis.scatter(float(target[time, 0]), float(target[time, 1]), c="white", edgecolors="black", marker="x", s=65, label="projected")
        axis.set_title(title)
        axis.axis("off")
    valid_observed = present & torch.isfinite(observed).all(dim=-1)
    valid_target = projected_present & torch.isfinite(target).all(dim=-1)
    axes[1, 0].plot(observed[valid_observed, 0], observed[valid_observed, 1], "o-", color=color, label="observed")
    axes[1, 0].plot(target[valid_target, 0], target[valid_target, 1], "x--", color="black", label="projected")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title(f"point-track tube: {stratum}")
    axes[1, 0].set_xlabel("x (px)")
    axes[1, 0].set_ylabel("y (px)")
    axes[1, 0].legend(fontsize=8)
    velocity_before, acceleration_before, jerk_before = _dynamics(observed, present)
    velocity_after, acceleration_after, jerk_after = _dynamics(target, projected_present)
    summary = [
        f"candidate: {candidate.name}",
        f"sample: {sample_id} (cache index {dataset_index})",
        f"confidence-derived uncertainty median: {float(candidate.uncertainty[track_index][present].median()):.3f}px",
        f"correction SNR median: {float(candidate.correction_snr[track_index][candidate.corrected[track_index]].median()):.3f}" if bool(candidate.corrected[track_index].any()) else "correction SNR median: n/a (identity/gated)",
        f"velocity RMS before/after: {_rms(velocity_before)} / {_rms(velocity_after)}",
        f"accel RMS before/after: {_rms(acceleration_before)} / {_rms(acceleration_after)}",
        f"jerk RMS before/after: {_rms(jerk_before)} / {_rms(jerk_after)}",
        "white x = projected; coloured circle = observed",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.98, "\n".join(summary), va="top", family="monospace", fontsize=9)
    figure.suptitle("P0 physical-validity review panel", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_review_package(run_dir: Path, panels: list[dict[str, Any]]) -> None:
    template = run_dir / "reviews.template.jsonl"
    if not template.exists():
        lines = []
        for panel in panels:
            lines.append(json.dumps({
                "case_id": panel["case_id"],
                "verdict": "pending",
                "reviewer": "human",
                "notes": "",
                "rubric": "轨迹是否仍贴合可见图像局部、保留合理加减速/转弯，且没有无支持跳变？",
            }, ensure_ascii=False))
        atomic_write_text(str(template), "\n".join(lines) + "\n")
    readme = run_dir / "REVIEW_README.md"
    if not readme.exists():
        atomic_write_text(
            str(readme),
            "# P0 point-track tube 人工复核\n\n"
            "每张 `panels/*.png` 显示 Base rollout 的 frame-0/末帧、observed/projected 轨迹、"
            "confidence 推导的 uncertainty 及 before/after velocity、acceleration、jerk。\n\n"
            "1. 复制 `reviews.template.jsonl` 为 `reviews.jsonl`。\n"
            "2. 对每个 case 将 `verdict` 填为 `valid`、`invalid` 或 `uncertain`，并补充 notes。\n"
            "3. 重新执行同一命令并增加 `--aggregate-only`。\n\n"
            "`valid` 仅表示该 point-track tube 的运动修正物理上可辨识；它不表示 dataset object-instance"
            "监督，也不允许根据人工复核改写缓存。\n",
        )


def _reviews_summary(run_dir: Path, review_cases: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    review_path = run_dir / "reviews.jsonl"
    review_by_case: dict[str, dict[str, Any]] = {}
    if review_path.is_file():
        for line in review_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            verdict = str(row.get("verdict", ""))
            if verdict in REVIEW_VALUES:
                review_by_case[str(row.get("case_id", ""))] = row
    valid_reviews = [review_by_case[panel["case_id"]] for panel in review_cases if panel["case_id"] in review_by_case]
    decisive = [row for row in valid_reviews if row["verdict"] != "uncertain"]
    valid = sum(row["verdict"] == "valid" for row in decisive)
    rate = valid / len(decisive) if decisive else None
    required = int(settings["review"]["required_panels"])
    complete = len(valid_reviews) >= required
    human_pass = bool(
        complete and rate is not None and rate >= float(settings["review"]["minimum_valid_rate"])
    )
    return {
        "required": required,
        "panel_count": len(review_cases),
        "completed": len(valid_reviews),
        "decisive": len(decisive),
        "valid": valid,
        "invalid": sum(row["verdict"] == "invalid" for row in decisive),
        "valid_rate": rate,
        "minimum_valid_rate": float(settings["review"]["minimum_valid_rate"]),
        "status": "pass" if human_pass else "awaiting_reviews",
        "human_pass": human_pass,
    }


def _clean_status_markers(run_dir: Path) -> None:
    for name in ("COMPLETE", "FAILED", "awaiting_reviews"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def _update_with_reviews(run_dir: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    machine_path = run_dir / "machine_summary.json"
    panels_path = run_dir / "review_cases.json"
    if not machine_path.is_file() or not panels_path.is_file():
        raise FileNotFoundError("aggregate-only requires machine_summary.json and review_cases.json")
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    panels = json.loads(panels_path.read_text(encoding="utf-8"))
    reviews = _reviews_summary(run_dir, panels, settings)
    machine_eligible = machine["machine_eligible"]
    if machine_eligible:
        status = "pass" if reviews["human_pass"] else "awaiting_reviews"
    else:
        status = "fail"
    summary = {**machine, "status": status, "human_review": reviews}
    atomic_write_json(str(run_dir / "summary.json"), summary)
    _clean_status_markers(run_dir)
    if status == "awaiting_reviews":
        atomic_write_text(str(run_dir / "awaiting_reviews"), sha256_json(summary) + "\n")
    else:
        atomic_write_text(str(run_dir / "COMPLETE"), sha256_json(summary) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": "human_review"})
    atomic_write_json(str(manifest_path), manifest)
    return summary


def _validate_replay_metadata(metadata: Mapping[str, Any], index: int) -> None:
    required = {
        "source": "replay_v2",
        "parent_kind": "base",
        "adapter_loaded": False,
        "uses_future_gt_ego": False,
        "uses_future_gt_track": False,
    }
    mismatch = {key: {"expected": value, "actual": metadata.get(key)} for key, value in required.items() if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"P0 index {index} is not leakage-free frozen-Base replay: {mismatch}")


def _validate_reconstruction(index: int, metadata: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> None:
    cached = metadata["projector_diagnostics"]["generated_tracks"]
    keys = ("provider", "uses_future_gt", "query_count", "stratum_query_count", "valid_track_count")
    mismatch = {key: {"cached": cached.get(key), "actual": diagnostics.get(key)} for key in keys if cached.get(key) != diagnostics.get(key)}
    if mismatch:
        raise RuntimeError(f"P0 generated-track reconstruction mismatch for index {index}: {mismatch}")


def _select_panel_rows(
    seen: dict[str, int],
    required: int,
    payload: dict[str, Any],
    candidate: CandidateTracks,
    panel_dir: Path,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    # 首先每 strata 至少四张；再以固定 track 顺序补足，总数至少 12。
    for track_index, (original, projected) in enumerate(zip(payload["tracks"], candidate.tracks)):
        if sum(seen.values()) >= required:
            break
        stratum = _track_label(original)
        quota = max(1, required // 3)
        if seen.get(stratum, 0) >= quota and sum(seen.values()) < required:
            continue
        case_id = f"p0-i{payload['dataset_index']:03d}-t{track_index:03d}-{candidate.name.lower()}"
        panel_path = panel_dir / f"{case_id}.png"
        _draw_panel(
            panel_path, payload["base_rgb"], original, projected,
            candidate=candidate, track_index=track_index,
            dataset_index=payload["dataset_index"], sample_id=payload["sample_id"],
        )
        selected.append({
            "case_id": case_id,
            "panel_path": str(panel_path),
            "dataset_index": payload["dataset_index"],
            "sample_id": payload["sample_id"],
            "track_index": track_index,
            "track_token": original.instance_token,
            "stratum": stratum,
            "candidate": candidate.name,
        })
        seen[stratum] = seen.get(stratum, 0) + 1
    return selected


def run_projector_validity(cfg: Any, *, aggregate_only: bool = False) -> dict[str, Any]:
    settings = OmegaConf.to_container(cfg.p0, resolve=True)
    assert isinstance(settings, dict)
    indices = [int(value) for value in settings["dataset_indices"]]
    if not 1 <= len(indices) <= 8 or len(indices) != len(set(indices)):
        raise ValueError("P0 requires 1-8 unique frozen replay indices")
    if tuple(settings["candidates"]) != CANDIDATES:
        raise ValueError(f"P0 requires exactly {CANDIDATES}")
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("formal P0 refuses to run in a dirty worktree")
    run_dir = Path(str(cfg.work_dir))
    if aggregate_only:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"P0 run directory does not exist: {run_dir}")
        return _update_with_reviews(run_dir, settings)
    if run_dir.exists():
        raise RuntimeError(f"P0 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "panels").mkdir()
    paths = get_paths(cfg)
    dataset = ProjectionCacheDataset(str(paths.cache_dir), expected_fingerprint=str(settings["cache_fingerprint"]))
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("P0 replay index is outside cache")
    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=config_fp,
        cache_fingerprint=str(settings["cache_fingerprint"]), seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.data.split),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(settings["task_id"]),
        "protocol": PROTOCOL_VERSION,
        "dataset_indices": indices,
        "base_model_fingerprint": _model_fingerprint(str(cfg.model.pretrained)),
        "base_generation_adapter_loaded": False,
        "uses_future_gt": False,
        "candidate_names": list(CANDIDATES),
        "preregistration": settings["thresholds"],
        "review_requirements": settings["review"],
    }
    atomic_write_json(str(run_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(run_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(run_dir / "metrics.jsonl"))
    try:
        # 现有 RAFT provider 的三帧 median CUDA kernel 不支持 strict deterministic
        # indices；复现仅在 cached diagnostics exact-match 后接受，且不含随机采样。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed_everything(int(cfg.seed), deterministic=False)
        options = dict(OmegaConf.to_container(cfg.auditor.generated_tracks, resolve=True))
        options.pop("provider", None)
        provider = RAFTChainGeneratedTrackProvider(device=str(cfg.device), **options)
        all_track_rows: list[dict[str, Any]] = []
        reconstructions: list[dict[str, Any]] = []
        panel_cases: list[dict[str, Any]] = []
        panel_seen: dict[str, int] = {}
        required_panels = int(settings["review"]["required_panels"])
        for index in indices:
            item = dataset[index]
            metadata = item["metadata"]
            _validate_replay_metadata(metadata, index)
            if "base_rgb" not in item:
                raise RuntimeError(f"P0 index {index} misses base_rgb")
            state = provider.track(item["base_rgb"])
            if state.uses_future_gt:
                raise RuntimeError("P0 generated track provider reported future-GT use")
            _validate_reconstruction(index, metadata, state.diagnostics)
            tracks = [_clone_track_cpu(track) for track in state.tracks]
            if not tracks:
                raise RuntimeError(f"P0 index {index} reconstructs zero tracks")
            confidence = state.confidence.detach().cpu()
            height, width = item["base_rgb"].shape[-2:]
            candidates = build_candidate_tracks(tracks, confidence, (height, width), settings)
            support = classify_support(tracks, (height, width))
            payload = {
                "dataset_index": index,
                "sample_id": str(metadata["sample_id"]),
                "tracks": tracks,
                "base_rgb": item["base_rgb"].detach().cpu(),
            }
            per_candidate = []
            for name in CANDIDATES:
                candidate = candidates[name]
                rows = [
                    _track_row(
                        candidate=candidate, original=original, projected=projected,
                        track_index=track_index, dataset_index=index, sample_id=payload["sample_id"],
                        support=support[original.instance_token].detach().cpu(),
                    )
                    for track_index, (original, projected) in enumerate(zip(tracks, candidate.tracks))
                ]
                all_track_rows.extend(rows)
                candidate_summary = _aggregate_track_rows(rows, settings["thresholds"])[name]
                per_candidate.append({"candidate": name, **candidate_summary})
                metrics.append(index, {"phase": "generated_track_candidate", **per_candidate[-1]})
            # P-UNC 是正式 uncertainty-gated candidate；若其没有有效点，panel 仍保留
            # 以让 reviewer 看见“被 gate 成 identity”的事实。
            if len(panel_cases) < required_panels:
                panel_cases.extend(_select_panel_rows(panel_seen, required_panels, payload, candidates["P-UNC"], run_dir / "panels"))
            reconstruction = {
                "dataset_index": index,
                "sample_id": payload["sample_id"],
                "track_count": len(tracks),
                "category_counts": {label: sum(_track_label(track) == label for track in tracks) for label in sorted({_track_label(track) for track in tracks})},
                "provider_diagnostics": state.diagnostics,
                "cache_match": True,
            }
            reconstructions.append(reconstruction)
            metrics.append(index, {"phase": "track_reconstruction", **reconstruction})
        del provider
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(panel_cases) < required_panels:
            raise RuntimeError(f"P0 requires {required_panels} review panels, only exported {len(panel_cases)}")
        synthetic_rows = synthetic_calibration(settings)
        for row in synthetic_rows:
            metrics.append(-1, {"phase": "synthetic_calibration", **row})
        track_summary = _aggregate_track_rows(all_track_rows, settings["thresholds"])
        synthetic_summary = _aggregate_synthetic(synthetic_rows, settings["thresholds"])
        machine_checks = {
            candidate: _candidate_machine_check(candidate, track_summary, synthetic_summary, settings["thresholds"])
            for candidate in CANDIDATES
        }
        eligible = [candidate for candidate, row in machine_checks.items() if row["machine_pass"]]
        machine = {
            "task_id": str(settings["task_id"]),
            "protocol": PROTOCOL_VERSION,
            "dataset_indices": indices,
            "sample_count": len(indices),
            "uses_future_gt": False,
            "base_generation_adapter_loaded": False,
            "track_reconstruction": reconstructions,
            "generated_track_audit": track_summary,
            "synthetic_calibration": synthetic_summary,
            "machine_checks": machine_checks,
            "machine_eligible": eligible,
            "machine_gate": "pass" if eligible else "fail",
            "precondition_for_p1": bool(eligible),
            "experiment_fingerprint": sha256_json({
                "config": config_fp, "track_summary": track_summary,
                "synthetic_summary": synthetic_summary, "machine_checks": machine_checks,
            }),
        }
        _write_csv(run_dir / "track_rows.csv", all_track_rows)
        _write_csv(run_dir / "synthetic_rows.csv", synthetic_rows)
        atomic_write_text(str(run_dir / "track_reconstruction.jsonl"), "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in reconstructions
        ))
        atomic_write_json(str(run_dir / "review_cases.json"), panel_cases)
        _write_review_package(run_dir, panel_cases)
        atomic_write_json(str(run_dir / "machine_summary.json"), machine)
        # 未造 review verdict；只有 machine-eligible 才需要 P0 human gate 才能晋级。
        summary = _update_with_reviews(run_dir, settings)
        return summary
    except Exception as exc:
        atomic_write_json(str(run_dir / "summary.json"), {"status": "failed", "error": repr(exc)})
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(run_dir / "manifest.json"), manifest_data)
        _clean_status_markers(run_dir)
        atomic_write_text(str(run_dir / "FAILED"), repr(exc) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    result = run_projector_validity(
        load_config(args.config, list(args.overrides)), aggregate_only=bool(args.aggregate_only)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
