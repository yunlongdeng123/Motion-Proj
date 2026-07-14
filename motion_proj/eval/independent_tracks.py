"""冻结的 CoTracker3 独立 point-track evaluator。

它不读取训练 RAFT-chain 的轨迹、cache projector output 或任一 future-GT 字段。query
来自 evaluator 自身的 first-frame grid；background/dynamic/foreground strata 仅由 CoTracker
输出和首帧图像梯度重建，因而不能被当作训练 target 的重用。
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


STRATA = ("background", "dynamic_residual", "foreground_candidate")


@dataclass
class IndependentTrackState:
    """独立 evaluator 的完整输出；无有效 track 时 ``valid=False``。"""

    points: torch.Tensor               # [N,T,2] image pixels, invalid entries are NaN
    visibility: torch.Tensor           # [N,T] bool
    labels: list[str]                 # N labels from evaluator-only stratification
    query_points: torch.Tensor         # [N,2], first-frame grid
    affine_background: torch.Tensor    # [T-1,2,3], NaN for unfit pair
    diagnostics: dict[str, Any]
    valid: bool


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalise_cotracker_outputs(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将官方 predictor 的 ``[1,T,N,2]`` / ``[1,T,N]`` 输出转成 ``[N,T,*]``。

    某些旧 wrapper 会把 visibility 保留成末尾 singleton 维；接受这一无信息维，
    其余形状一律显式报错，避免把 tracker API 变化静默解释成可用轨迹。
    """
    if tracks.ndim != 4 or tracks.shape[0] != 1 or tracks.shape[-1] != 2:
        raise ValueError(f"unexpected CoTracker track shape: {tuple(tracks.shape)}")
    if visibility.ndim == 4 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]
    if visibility.ndim != 3 or visibility.shape[0] != 1:
        raise ValueError(f"unexpected CoTracker visibility shape: {tuple(visibility.shape)}")
    if tuple(visibility.shape[1:]) != tuple(tracks.shape[1:3]):
        raise ValueError(
            "CoTracker track/visibility time-point dimensions disagree: "
            f"{tuple(tracks.shape)} vs {tuple(visibility.shape)}"
        )
    points = tracks[0].detach().float().cpu().permute(1, 0, 2).contiguous()
    visible = visibility[0].detach().bool().cpu().permute(1, 0).contiguous()
    return points, visible


def _sample_scalar(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """bilinear sample ``[H,W]`` field at ``[N,2]`` image-pixel points."""
    height, width = field.shape
    grid = torch.stack([
        2.0 * points[:, 0] / max(width - 1, 1) - 1.0,
        2.0 * points[:, 1] / max(height - 1, 1) - 1.0,
    ], dim=-1).reshape(1, 1, -1, 2)
    values = F.grid_sample(
        field[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return values[0, 0, 0]


def first_frame_gradient(frames: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """只从 first generated RGB frame 取 query texture score。"""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must be [T,3,H,W]")
    image = frames[0].float().mean(dim=0)
    dx = torch.zeros_like(image)
    dy = torch.zeros_like(image)
    dx[:, :-1] = (image[:, 1:] - image[:, :-1]).abs()
    dy[:-1, :] = (image[1:, :] - image[:-1, :]).abs()
    return _sample_scalar(dx + dy, points)


def fit_affine_background(
    points: torch.Tensor,
    visibility: torch.Tensor,
    *,
    max_iterations: int = 4,
    huber_delta_px: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 evaluator 自身 tracks 逐帧 IRLS 拟合 image-plane affine camera motion。

    返回 ``affine[T-1,2,3]`` 与 pair-valid mask；这不是 ego GT 或训练 auditor 的 flow。
    """
    if points.ndim != 3 or points.shape[-1] != 2 or visibility.shape != points.shape[:2]:
        raise ValueError("points/visibility shapes must be [N,T,2]/[N,T]")
    count, time_count = points.shape[:2]
    affine = torch.full((max(time_count - 1, 0), 2, 3), float("nan"), dtype=torch.float32)
    valid_pairs = torch.zeros(time_count - 1, dtype=torch.bool)
    for time in range(time_count - 1):
        valid = visibility[:, time] & visibility[:, time + 1]
        source, destination = points[valid, time].float(), points[valid, time + 1].float()
        if source.shape[0] < 3:
            continue
        design = torch.cat([source, torch.ones(source.shape[0], 1)], dim=-1)
        weights = torch.ones(source.shape[0], dtype=torch.float32)
        solved = None
        for _ in range(max_iterations):
            root_weight = weights.sqrt()
            weighted_design = design * root_weight[:, None]
            normal = weighted_design.T @ weighted_design + 1.0e-5 * torch.eye(3)
            rhs = weighted_design.T @ (destination * root_weight[:, None])
            solved = torch.linalg.solve(normal, rhs).T
            residual = torch.linalg.vector_norm((design @ solved.T) - destination, dim=-1)
            weights = torch.where(
                residual <= huber_delta_px, torch.ones_like(residual),
                huber_delta_px / residual.clamp_min(1.0e-6),
            )
        assert solved is not None
        affine[time] = solved
        valid_pairs[time] = True
    return affine, valid_pairs


def camera_compensated_velocity(
    points: torch.Tensor,
    visibility: torch.Tensor,
    affine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 camera-compensated image-plane velocity ``[N,T-1,2]`` 与 valid mask。"""
    count, time_count = visibility.shape
    velocity = torch.full((count, max(time_count - 1, 0), 2), float("nan"), dtype=torch.float32)
    valid = torch.zeros((count, max(time_count - 1, 0)), dtype=torch.bool)
    for time in range(time_count - 1):
        if not bool(torch.isfinite(affine[time]).all()):
            continue
        usable = visibility[:, time] & visibility[:, time + 1]
        if not bool(usable.any()):
            continue
        source = points[usable, time].float()
        prediction = torch.cat([source, torch.ones(source.shape[0], 1)], dim=-1) @ affine[time].T
        camera_flow = prediction - source
        velocity[usable, time] = points[usable, time + 1].float() - source - camera_flow
        valid[usable, time] = True
    return velocity, valid


def _rms(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = value[mask]
    selected = selected[torch.isfinite(selected)]
    return float(selected.square().mean().sqrt()) if int(selected.numel()) else None


def _track_strata(
    points: torch.Tensor,
    visibility: torch.Tensor,
    affine: torch.Tensor,
    frames: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    """仅从独立 tracks+first RGB 层化；不访问 RAFT 或 P0/P1 输出。"""
    velocity, valid_velocity = camera_compensated_velocity(points, visibility, affine)
    residual = torch.full((points.shape[0],), float("nan"))
    for index in range(points.shape[0]):
        if bool(valid_velocity[index].any()):
            residual[index] = torch.linalg.vector_norm(velocity[index, valid_velocity[index]], dim=-1).median()
    finite = residual[torch.isfinite(residual)]
    if not int(finite.numel()):
        return ["background"] * points.shape[0], residual
    low, high = torch.quantile(finite, 0.5), torch.quantile(finite, 0.75)
    texture = first_frame_gradient(frames, points[:, 0])
    texture_threshold = torch.quantile(texture[torch.isfinite(texture)], 0.60)
    labels = []
    for index in range(points.shape[0]):
        value = residual[index]
        if not bool(torch.isfinite(value)) or value <= low:
            labels.append("background")
        elif value >= high:
            labels.append("dynamic_residual")
        elif texture[index] >= texture_threshold:
            labels.append("foreground_candidate")
        else:
            labels.append("background")
    return labels, residual


def summarize_camera_compensated_dynamics(state: IndependentTrackState) -> dict[str, Any]:
    """按 strata 输出 camera-compensated image-plane acceleration/jerk，空集合显式 invalid。"""
    velocity, valid_velocity = camera_compensated_velocity(state.points, state.visibility, state.affine_background)
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    valid_acceleration = valid_velocity[:, 1:] & valid_velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]
    valid_jerk = valid_acceleration[:, 1:] & valid_acceleration[:, :-1]
    output: dict[str, Any] = {}
    for stratum in STRATA:
        mask = torch.tensor([label == stratum for label in state.labels], dtype=torch.bool)
        if not bool(mask.any()):
            output[stratum] = {"status": "invalid", "track_count": 0, "valid_velocity_count": 0}
            continue
        selected_velocity = valid_velocity[mask]
        selected_acceleration = valid_acceleration[mask]
        selected_jerk = valid_jerk[mask]
        velocity_count = int(selected_velocity.sum())
        if velocity_count == 0:
            output[stratum] = {"status": "invalid", "track_count": int(mask.sum()), "valid_velocity_count": 0}
            continue
        output[stratum] = {
            "status": "valid",
            "track_count": int(mask.sum()),
            "valid_velocity_count": velocity_count,
            "survival_rate": float(state.visibility[mask, -1].float().mean()),
            "camera_compensated_image_plane_velocity_rms_px": _rms(velocity[mask], selected_velocity),
            "camera_compensated_image_plane_acceleration_rms_px": _rms(acceleration[mask], selected_acceleration),
            "camera_compensated_image_plane_jerk_rms_px": _rms(jerk[mask], selected_jerk),
        }
    all_valid = [row for row in output.values() if row["status"] == "valid"]
    output["all"] = {
        "status": "valid" if all_valid else "invalid",
        "valid_strata": [name for name in STRATA if output[name]["status"] == "valid"],
        "camera_model": "per-pair evaluator-only robust affine image-plane fit",
    }
    return output


def aggregate_dynamics(summary: Mapping[str, Any]) -> dict[str, float] | None:
    """为 repeatability/sweep 提取不将 invalid 当 0 的 aggregate。"""
    rows = [summary[name] for name in STRATA if summary.get(name, {}).get("status") == "valid"]
    if not rows:
        return None
    weights = [int(row["valid_velocity_count"]) for row in rows]
    total = sum(weights)
    if total == 0:
        return None
    values = {}
    for field in (
        "survival_rate", "camera_compensated_image_plane_velocity_rms_px",
        "camera_compensated_image_plane_acceleration_rms_px", "camera_compensated_image_plane_jerk_rms_px",
    ):
        usable = [(float(row[field]), weight) for row, weight in zip(rows, weights) if row.get(field) is not None]
        values[field] = sum(value * weight for value, weight in usable) / sum(weight for _, weight in usable) if usable else None
    return values


class CoTracker3IndependentEvaluator:
    """官方 CoTracker3 offline checkpoint；缺失时明确不可用，绝不回退 RAFT。"""

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.repo = Path(str(self.settings["repository_path"])).resolve()
        self.checkpoint = Path(str(self.settings["checkpoint_path"])).resolve()
        self.device = str(self.settings.get("device", "cuda"))
        self.grid_size = int(self.settings.get("grid_size", 16))
        self._model = None
        if self.grid_size <= 1:
            raise ValueError("CoTracker grid_size must exceed 1")

    def preflight(self) -> dict[str, Any]:
        status = {"provider": "cotracker3_offline", "available": False, "reasons": []}
        if not (self.repo / "hubconf.py").is_file():
            status["reasons"].append(f"official repository missing: {self.repo}")
        commit = None
        git_dir = self.repo / ".git"
        if git_dir.exists():
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                status["reasons"].append("official repository commit unavailable")
        expected_commit = self.settings.get("repository_commit")
        if expected_commit and commit != str(expected_commit):
            status["reasons"].append("repository commit mismatch")
        actual = None
        expected = self.settings.get("checkpoint_sha256")
        if not self.checkpoint.is_file():
            status["reasons"].append(f"official checkpoint missing: {self.checkpoint}")
        else:
            actual = file_sha256(self.checkpoint)
            if expected and str(expected) != actual:
                status["reasons"].append("checkpoint sha256 mismatch")
        status.update({
            "available": not status["reasons"], "repository_path": str(self.repo), "repository_commit": commit,
            "repository_commit_expected": expected_commit,
            "checkpoint_path": str(self.checkpoint), "checkpoint_sha256": actual,
            "checkpoint_sha256_expected": expected,
            "checkpoint_url": self.settings.get("checkpoint_url"),
        })
        return status

    def _load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        preflight = self.preflight()
        if not preflight["available"]:
            raise RuntimeError("CoTracker3 unavailable: " + "; ".join(preflight["reasons"]))
        # Official torch.hub local-source entry point; no sys.path mutation or RAFT fallback.
        model = torch.hub.load(str(self.repo), "cotracker3_offline", source="local", pretrained=False)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        model.model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval().requires_grad_(False)
        return self._model

    @torch.no_grad()
    def track(self, frames: torch.Tensor) -> IndependentTrackState:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must be [T,3,H,W]")
        model = self._load()
        video = ((frames.detach().float().clamp(-1, 1) + 1.0) * 127.5).round()
        video = video.unsqueeze(0).to(self.device)
        tracks, visibility = model(video, grid_size=self.grid_size)
        points, visible = _normalise_cotracker_outputs(tracks, visibility)
        points[~visible] = float("nan")
        affine, pair_valid = fit_affine_background(points, visible)
        labels, residual = _track_strata(points, visible, affine, frames.detach().cpu())
        valid = bool(visible.any()) and bool(pair_valid.any())
        return IndependentTrackState(
            points=points, visibility=visible, labels=labels, query_points=points[:, 0].clone(),
            affine_background=affine,
            diagnostics={
                "provider": "cotracker3_offline", "uses_future_gt": False,
                "query_protocol": f"official_first_frame_grid_size_{self.grid_size}",
                "query_count": int(points.shape[0]), "pair_affine_valid_count": int(pair_valid.sum()),
                "residual_motion_median_px": float(torch.nanmedian(residual)) if bool(torch.isfinite(residual).any()) else None,
            },
            valid=valid,
        )
