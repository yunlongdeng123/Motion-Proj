"""P1：P-UNC RGB/VAE counterfactual target 的只读合法性审计。

本模块不接管正式 cache/renderer。它从冻结 Base RGB 重建 P0 的 P-UNC point-track
tube，临时用现有 crop/resize/paste compositor 生成 ``X_dagger``，并比较：

* ``z_full = E(X_dagger)``；
* ``z_hybrid = z_base + M * (z_full-z_base)``；
* 固定半径 dilation 后的 hybrid；
* decode -> encode 回环。

这使得「即使 P0 轨迹有效，RGB/latent endpoint 是否仍是合法 counterfactual」成为
独立、可失败的问题。不会写入或替换 V5 cache。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..auditor.generated_tracks import RAFTChainGeneratedTrackProvider
from ..auditor.state import Track
from ..cache.dataset import ProjectionCacheDataset
from ..config import config_fingerprint, get_paths, load_config, save_resolved_config
from ..projector.mask import downsample_mask_to_latent
from ..projector.support import classify_support
from ..projector.warper import composite_objects
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video, write_video
from ..utils.viz import hstack_panels
from .projector_validity import PRIMARY_STRATA, build_candidate_tracks


PROTOCOL_VERSION = "autoresearch-p1-target-validity-v1"
REVIEW_VALUES = {"valid", "invalid", "uncertain"}


def _track_label(track: Track) -> str:
    return str(track.category).rsplit("/", 1)[-1]


def _clone_track_cpu(track: Track) -> Track:
    return Track(
        str(track.instance_token), str(track.category), track.xyxy.detach().cpu().clone(),
        track.depth.detach().cpu().clone(), track.present.detach().cpu().bool().clone(),
    )


def dilate_latent_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """固定形态 dilation，保留 [T,1,H,W] 和 frame-0 空掩码。"""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("latent mask 必须是 [T,1,H,W]")
    if radius < 0:
        raise ValueError("dilation radius 不得为负")
    if radius == 0:
        result = mask.clone()
    else:
        kernel = 2 * radius + 1
        result = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)
    result[0] = 0
    return result.clamp(0, 1)


def make_hybrid_latent(base: torch.Tensor, full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """构造 ``base + mask * (full-base)``，不隐藏 mask 外的 latent 变化。"""
    if (
        base.shape != full.shape
        or mask.ndim != 5
        or mask.shape[:2] != base.shape[:2]
        or mask.shape[2] != 1
        or mask.shape[-2:] != base.shape[-2:]
    ):
        raise ValueError("base/full/mask 形状不一致")
    output = base + mask.to(base) * (full - base)
    # 预注册 hard invariant：frame 0 不得借由 dilation 或 VAE mask 重新开启。
    output[:, 0] = base[:, 0]
    return output


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    expanded = mask.to(value).expand_as(value)
    denom = expanded.sum()
    if float(denom) <= 1.0e-8:
        return None
    return float(((value.square() * expanded).sum() / denom).sqrt())


def _masked_l1(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    expanded = mask.to(value).expand_as(value)
    denom = expanded.sum()
    if float(denom) <= 1.0e-8:
        return None
    return float((value.abs() * expanded).sum() / denom)


def _relative_rms(value: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> float | None:
    numerator = _masked_rms(value, mask)
    denominator = _masked_rms(reference, mask)
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1.0e-8)


def _box_to_slice(box: torch.Tensor, height: int, width: int) -> tuple[slice, slice] | None:
    if not bool(torch.isfinite(box).all()):
        return None
    u0, v0, u1, v1 = [int(round(float(value))) for value in box]
    u0, u1 = max(u0, 0), min(u1, width)
    v0, v1 = max(v0, 0), min(v1, height)
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    return slice(v0, v1), slice(u0, u1)


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return 0.0
    lx0, ly0, lx1, ly1 = [float(value) for value in left]
    rx0, ry0, rx1, ry1 = [float(value) for value in right]
    ix0, iy0, ix1, iy1 = max(lx0, rx0), max(ly0, ry0), min(lx1, rx1), min(ly1, ry1)
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = max((lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection, 1.0e-8)
    return intersection / union


def _quantized_box(box: torch.Tensor, height: int, width: int) -> tuple[int, int, int, int] | None:
    region = _box_to_slice(box, height, width)
    if region is None:
        return None
    ys, xs = region
    return xs.start, ys.start, xs.stop, ys.stop


def _mean_abs_region(value: torch.Tensor, region: tuple[slice, slice]) -> float:
    ys, xs = region
    piece = value[:, ys, xs]
    return float(piece.abs().mean()) if int(piece.numel()) else 0.0


def source_duplication_rows(
    base: torch.Tensor,
    target: torch.Tensor,
    original: list[Track],
    projected: list[Track],
    *,
    minimum_destination_change_l1: float,
    maximum_source_change_l1: float,
    maximum_overlap_iou: float,
) -> list[dict[str, Any]]:
    """检测 crop/paste 后 source 仍保留、destination 新出现的 duplication proxy。

    这是由 RGB 本身计算的保守 proxy，不把它伪装成 GT identity/occlusion 标签。
    """
    if base.shape != target.shape or base.ndim != 4:
        raise ValueError("base/target 必须是同形状 [T,3,H,W]")
    by_token = {track.instance_token: track for track in original}
    height, width = base.shape[-2:]
    rows = []
    for track in projected:
        source = by_token.get(track.instance_token)
        if source is None:
            continue
        for time in range(base.shape[0]):
            if not bool(source.present[time] and track.present[time]):
                continue
            source_box, destination_box = source.xyxy[time], track.xyxy[time]
            source_region = _box_to_slice(source_box, height, width)
            destination_region = _box_to_slice(destination_box, height, width)
            if source_region is None or destination_region is None:
                continue
            source_change = _mean_abs_region(target[time] - base[time], source_region)
            destination_change = _mean_abs_region(target[time] - base[time], destination_region)
            overlap = _box_iou(source_box, destination_box)
            # compositor 是整数 crop/resize/paste；以其实际量化后的盒子判断有无移动，
            # 不能把仅有亚像素 P-UNC correction 误记成 RGB target。
            moved = _quantized_box(source_box, height, width) != _quantized_box(destination_box, height, width)
            duplicated = bool(
                moved and overlap <= maximum_overlap_iou
                and destination_change >= minimum_destination_change_l1
                and source_change <= maximum_source_change_l1
            )
            rows.append({
                "track_token": track.instance_token,
                "stratum": _track_label(track),
                "time": time,
                "moved_after_quantization": moved,
                "source_destination_iou": overlap,
                "source_change_l1": source_change,
                "destination_change_l1": destination_change,
                "source_retained_duplication_proxy": duplicated,
                "texture_area_ratio": 1.0,
            })
    return rows


def _overlap_rows(
    projected: list[Track], height: int, width: int, *, iou_threshold: float,
    moved_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    """只审计与实际 integer-paste move 相邻的 overlap，忽略未移动 query 密度。"""
    rows = []
    for time in range(projected[0].present.shape[0] if projected else 0):
        for left_index, left in enumerate(projected):
            if not bool(left.present[time]):
                continue
            if _box_to_slice(left.xyxy[time], height, width) is None:
                continue
            for right in projected[left_index + 1:]:
                if not bool(right.present[time]):
                    continue
                if (left.instance_token, time) not in moved_keys and (right.instance_token, time) not in moved_keys:
                    continue
                overlap = _box_iou(left.xyxy[time], right.xyxy[time])
                if overlap >= iou_threshold:
                    rows.append({
                        "time": time, "left": left.instance_token, "right": right.instance_token,
                        "iou": overlap, "occlusion_order_known": False,
                    })
    return rows


def _model_fingerprint(pretrained: str) -> str:
    root = Path(pretrained)
    if not root.is_dir():
        return sha256_json({"pretrained": pretrained})
    files = [
        root / "model_index.json", root / "vae" / "config.json", root / "unet" / "config.json",
        root / "scheduler" / "scheduler_config.json", root / "image_encoder" / "config.json",
    ]
    return sha256_json([
        (str(path.relative_to(root)), file_fingerprint(str(path))) for path in files if path.is_file()
    ])


class _VAECodec:
    """P1 只加载 frozen VAE，不加载 UNet/adapter。"""

    def __init__(self, pretrained: str, device: str, dtype: torch.dtype):
        from diffusers import AutoencoderKLTemporalDecoder

        self.device = torch.device(device)
        self.dtype = dtype
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained, subfolder="vae").to(self.device, dtype)
        self.vae.eval().requires_grad_(False)
        self.scaling_factor = float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames 必须是 [B,T,3,H,W]")
        batch, time = frames.shape[:2]
        flat = frames.reshape(batch * time, *frames.shape[2:]).to(self.device, self.dtype)
        latent = self.vae.encode(flat).latent_dist.mode() * self.scaling_factor
        return latent.reshape(batch, time, *latent.shape[1:])

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError("latents 必须是 [B,T,C,H,W]")
        batch, time = latents.shape[:2]
        flat = latents.reshape(batch * time, *latents.shape[2:]).to(self.device, self.dtype) / self.scaling_factor
        frames = self.vae.decode(flat, num_frames=1).sample
        return frames.reshape(batch, time, *frames.shape[1:]).clamp(-1, 1)


class _LPIPS:
    def __init__(self, device: torch.device):
        import lpips

        self.model = lpips.LPIPS(net="alex").eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> float:
        if left.shape != right.shape or left.ndim != 4:
            raise ValueError("LPIPS inputs must be [T,3,H,W] with same shape")
        values = []
        # 逐小 batch 以避免 1024-wide frame 的 metric 峰值显存掩盖 P1 本身。
        for begin in range(0, left.shape[0], 4):
            value = self.model(left[begin:begin + 4].float().to(self.device), right[begin:begin + 4].float().to(self.device))
            values.append(value.detach().float().cpu().reshape(-1))
        return float(torch.cat(values).mean())


def _validate_replay_metadata(metadata: Mapping[str, Any], index: int) -> None:
    expected = {
        "source": "replay_v2", "parent_kind": "base", "adapter_loaded": False,
        "uses_future_gt_ego": False, "uses_future_gt_track": False,
    }
    mismatch = {key: {"expected": value, "actual": metadata.get(key)} for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"P1 index {index} is not leakage-free frozen-Base replay: {mismatch}")


def _validate_reconstruction(index: int, metadata: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> None:
    cached = metadata["projector_diagnostics"]["generated_tracks"]
    fields = ("provider", "uses_future_gt", "query_count", "stratum_query_count", "valid_track_count")
    mismatch = {key: {"cached": cached.get(key), "actual": diagnostics.get(key)} for key in fields if cached.get(key) != diagnostics.get(key)}
    if mismatch:
        raise RuntimeError(f"P1 generated-track reconstruction mismatch for index {index}: {mismatch}")


def _finite_quantiles(values: Iterable[float | None]) -> dict[str, float | None]:
    tensor = torch.tensor([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=torch.float64)
    if not int(tensor.numel()):
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(tensor.mean()), "median": float(tensor.median()),
        "p90": float(torch.quantile(tensor, 0.9)), "max": float(tensor.max()),
    }


def _variant_metrics(
    *,
    name: str,
    decoded: torch.Tensor,
    target: torch.Tensor,
    decoded_base: torch.Tensor,
    base_latent: torch.Tensor,
    full_latent: torch.Tensor,
    variant_latent: torch.Tensor,
    mask_rgb: torch.Tensor,
    mask_latent: torch.Tensor,
    codec: _VAECodec,
    lpips_metric: _LPIPS,
) -> tuple[dict[str, Any], torch.Tensor]:
    reencoded = codec.encode(decoded.unsqueeze(0))
    delta = decoded - target
    inside_rgb = mask_rgb
    outside_rgb = 1.0 - inside_rgb
    latent_delta = variant_latent - full_latent
    roundtrip = reencoded - variant_latent
    decoded_motion = decoded - decoded_base
    target_motion = target - decoded_base
    # 只在 target mask 内评估方向一致性；无 target RGB correction 时显式记为 None。
    weighted_target = target_motion * inside_rgb
    weighted_decoded = decoded_motion * inside_rgb
    numerator = float((weighted_target * weighted_decoded).sum())
    denominator = float(weighted_target.square().sum().sqrt() * weighted_decoded.square().sum().sqrt())
    return {
        "variant": name,
        "target_lpips": lpips_metric(decoded, target),
        "target_rgb_inside_l1": _masked_l1(delta, inside_rgb),
        "target_rgb_outside_l1": _masked_l1(delta, outside_rgb),
        "target_rgb_inside_rms": _masked_rms(delta, inside_rgb),
        "target_rgb_outside_rms": _masked_rms(delta, outside_rgb),
        "outside_latent_rms_over_base": _relative_rms(
            variant_latent - base_latent, base_latent, 1.0 - mask_latent
        ),
        "full_vs_variant_latent_rms": float((latent_delta.float().square().mean()).sqrt()),
        "decode_reencode_latent_rms": float((roundtrip.float().square().mean()).sqrt()),
        "decoded_target_motion_cosine_inside": numerator / max(denominator, 1.0e-8) if denominator > 1.0e-8 else None,
        "decoded_motion_rms_inside": _masked_rms(decoded_motion, inside_rgb),
    }, reencoded


def _label(image, text: str):
    import cv2

    result = image.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 260), 24), (0, 0, 0), -1)
    cv2.putText(result, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _review_video(base: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, hybrid: torch.Tensor, dilated: torch.Tensor) -> Any:
    import cv2
    import numpy as np

    base_u8, target_u8 = to_uint8_video(base), to_uint8_video(target)
    hybrid_u8, dilated_u8 = to_uint8_video(hybrid), to_uint8_video(dilated)
    frames = []
    for time in range(base.shape[0]):
        diff = (target[time] - base[time]).abs().mean(dim=0).detach().float().cpu()
        diff = (diff / max(float(diff.max()), 1.0e-6) * 255).to(torch.uint8).numpy()
        diff_u8 = np.repeat(diff[..., None], 3, axis=-1)
        mask_u8 = (mask[time].detach().float().cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
        mask_u8 = np.repeat(mask_u8[..., None], 3, axis=-1)
        columns = [
            _label(base_u8[time], "Base RGB"), _label(target_u8[time], "P-UNC X_dagger"),
            _label(diff_u8, "|target-base|"), _label(mask_u8, "P-UNC mask"),
            _label(hybrid_u8[time], "decode(hybrid)"), _label(dilated_u8[time], "decode(dilated)"),
        ]
        columns = [cv2.resize(column, (256, 144), interpolation=cv2.INTER_AREA) for column in columns]
        frames.append(hstack_panels(*columns))
    return np.stack(frames)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _write_reviews(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    template = run_dir / "reviews.template.jsonl"
    if not template.exists():
        atomic_write_text(str(template), "".join(json.dumps({
            "case_id": row["case_id"], "verdict": "pending", "reviewer": "human", "notes": "",
            "rubric": "是否存在 source duplication、ghosting、无 occlusion order 的错误覆盖、texture stretching，且 decoded target 是否实现预期方向？",
        }, ensure_ascii=False) + "\n" for row in rows))
    readme = run_dir / "REVIEW_README.md"
    if not readme.exists():
        atomic_write_text(
            str(readme),
            "# P1 P-UNC RGB/VAE target 人工复核\n\n"
            "每个 `panels/*.mp4` 依次显示 Base、P-UNC compositor `X_dagger`、差分、mask、"
            "decode(hybrid)、decode(dilated hybrid)。重点检查 source duplication、ghosting、无深度"
            "occlusion order 的覆盖、纹理拉伸和运动方向。\n\n"
            "复制 `reviews.template.jsonl` 为 `reviews.jsonl`，逐项填 `valid`/`invalid`/`uncertain`，"
            "然后以 `--aggregate-only` 重跑。未完成 8 个 review 不得将 P1 标记 pass。\n",
        )


def _review_summary(run_dir: Path, rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    by_id = {}
    path = run_dir / "reviews.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if str(value.get("verdict")) in REVIEW_VALUES:
                    by_id[str(value.get("case_id"))] = value
    reviews = [by_id[row["case_id"]] for row in rows if row["case_id"] in by_id]
    decisive = [row for row in reviews if row["verdict"] != "uncertain"]
    valid = sum(row["verdict"] == "valid" for row in decisive)
    rate = valid / len(decisive) if decisive else None
    required = int(settings["review"]["required_cases"])
    passed = bool(len(reviews) >= required and rate is not None and rate >= float(settings["review"]["minimum_valid_rate"]))
    return {
        "required": required, "completed": len(reviews), "decisive": len(decisive),
        "valid": valid, "invalid": sum(row["verdict"] == "invalid" for row in decisive),
        "valid_rate": rate, "minimum_valid_rate": float(settings["review"]["minimum_valid_rate"]),
        "human_pass": passed, "status": "pass" if passed else "awaiting_reviews",
    }


def _clean_markers(run_dir: Path) -> None:
    for name in ("COMPLETE", "FAILED", "awaiting_reviews"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def _update_reviews(run_dir: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    machine = json.loads((run_dir / "machine_summary.json").read_text(encoding="utf-8"))
    cases = json.loads((run_dir / "review_cases.json").read_text(encoding="utf-8"))
    review = _review_summary(run_dir, cases, settings)
    status = "pass" if machine["machine_pass"] and review["human_pass"] else ("awaiting_reviews" if machine["machine_pass"] else "fail")
    summary = {**machine, "status": status, "human_review": review}
    atomic_write_json(str(run_dir / "summary.json"), summary)
    _clean_markers(run_dir)
    atomic_write_text(str(run_dir / ("awaiting_reviews" if status == "awaiting_reviews" else "COMPLETE")), sha256_json(summary) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": "human_review" if machine["machine_pass"] else "machine_gate_fail"})
    atomic_write_json(str(manifest_path), manifest)
    return summary


def _machine_decision(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    threshold = settings["thresholds"]
    all_frame0 = all(float(row["target_frame0_rgb_max_error"]) <= float(threshold["frame0_max_error"]) and float(row["full_latent_frame0_max_error"]) <= float(threshold["frame0_max_error"]) and float(row["hybrid_latent_frame0_max_error"]) <= float(threshold["frame0_max_error"]) for row in rows)
    hybrid_lpips = [float(row["hybrid_target_lpips"]) for row in rows]
    hybrid_outside = [row["hybrid_outside_latent_rms_over_base"] for row in rows]
    no_duplication = all(float(row["source_duplication_fraction"]) <= float(threshold["maximum_source_duplication_fraction"]) for row in rows)
    correction_realized = all(int(row["renderable_move_count"]) > 0 and int(row["target_changed_pixel_count"]) > 0 for row in rows)
    overlap_ok = all(float(row["unresolved_occlusion_overlap_fraction"]) <= float(threshold["maximum_unresolved_overlap_fraction"]) for row in rows)
    checks = {
        "frame0_rgb_and_latent_exact": all_frame0,
        "hybrid_outside_latent_local": all(value is not None and float(value) <= float(threshold["maximum_outside_latent_rms_over_base"]) for value in hybrid_outside),
        "decode_hybrid_target_lpips": all(value <= float(threshold["maximum_hybrid_target_lpips"]) for value in hybrid_lpips),
        "projected_rgb_correction_realized": correction_realized,
        "no_systematic_source_duplication": no_duplication,
        "no_systematic_unresolved_occlusion": overlap_ok,
    }
    return {
        "checks": checks, "machine_pass": all(checks.values()),
        "hybrid_target_lpips": _finite_quantiles(hybrid_lpips),
        "hybrid_outside_latent_rms_over_base": _finite_quantiles(hybrid_outside),
    }


def run_target_validity(cfg: Any, *, aggregate_only: bool = False) -> dict[str, Any]:
    settings = OmegaConf.to_container(cfg.p1, resolve=True)
    assert isinstance(settings, dict)
    indices = [int(value) for value in settings["dataset_indices"]]
    if not 1 <= len(indices) <= 8 or len(indices) != len(set(indices)):
        raise ValueError("P1 needs 1-8 unique P0-machine-eligible indices")
    if str(settings["candidate"]) != "P-UNC":
        raise ValueError("this P1 preregistration accepts only P-UNC")
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("formal P1 refuses to run in a dirty worktree")
    run_dir = Path(str(cfg.work_dir))
    if aggregate_only:
        return _update_reviews(run_dir, settings)
    if run_dir.exists():
        raise RuntimeError(f"P1 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "panels").mkdir()
    paths = get_paths(cfg)
    dataset = ProjectionCacheDataset(str(paths.cache_dir), expected_fingerprint=str(settings["cache_fingerprint"]))
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("P1 replay index is outside cache")
    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=config_fp,
        cache_fingerprint=str(settings["cache_fingerprint"]), seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.data.split),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(settings["task_id"]), "protocol": PROTOCOL_VERSION,
        "dataset_indices": indices, "p0_machine_eligible_candidate": "P-UNC",
        "base_model_fingerprint": _model_fingerprint(str(cfg.model.pretrained)),
        "adapter_loaded": False, "uses_future_gt": False,
        "dilation_radius": int(settings["dilation_radius"]), "thresholds": settings["thresholds"],
    }
    atomic_write_json(str(run_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(run_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(run_dir / "metrics.jsonl"))
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed_everything(int(cfg.seed), deterministic=False)
        options = dict(OmegaConf.to_container(cfg.auditor.generated_tracks, resolve=True))
        options.pop("provider", None)
        provider = RAFTChainGeneratedTrackProvider(device=str(cfg.device), **options)
        dtype = torch.bfloat16 if str(cfg.dtype) == "bf16" else (torch.float16 if str(cfg.dtype) == "fp16" else torch.float32)
        codec = _VAECodec(str(cfg.model.pretrained), str(cfg.device), dtype)
        lpips_metric = _LPIPS(codec.device)
        rows: list[dict[str, Any]] = []
        duplication_rows: list[dict[str, Any]] = []
        overlap_rows: list[dict[str, Any]] = []
        review_cases: list[dict[str, Any]] = []
        for index in indices:
            item = dataset[index]
            metadata = item["metadata"]
            _validate_replay_metadata(metadata, index)
            if "base_rgb" not in item:
                raise RuntimeError(f"P1 index {index} misses base_rgb")
            state = provider.track(item["base_rgb"])
            if state.uses_future_gt:
                raise RuntimeError("P1 generated track provider reported future-GT use")
            _validate_reconstruction(index, metadata, state.diagnostics)
            original = [_clone_track_cpu(track) for track in state.tracks]
            height, width = item["base_rgb"].shape[-2:]
            candidates = build_candidate_tracks(original, state.confidence.detach().cpu(), (height, width), cfg.p0)
            punc = candidates["P-UNC"]
            primary_pairs = [
                (source, target) for source, target in zip(original, punc.tracks)
                if _track_label(source) in PRIMARY_STRATA
            ]
            primary_original = [value[0] for value in primary_pairs]
            primary_projected = [value[1] for value in primary_pairs]
            support_all = classify_support(original, (height, width))
            support = {track.instance_token: support_all[track.instance_token] for track in primary_original}
            base = item["base_rgb"].detach().cpu()
            raw_frame0_track_error = max((
                float((source.center[0] - target.center[0]).abs().max())
                for source, target in primary_pairs
            ), default=0.0)
            if raw_frame0_track_error > float(settings["thresholds"]["frame0_max_error"]):
                raise RuntimeError(f"P1 received a non-exact P-UNC frame0 track at index {index}: {raw_frame0_track_error}")
            target, object_mask = composite_objects(base, base, primary_original, primary_projected, support)
            target_frame0_error = float((target[0] - base[0]).abs().max())
            # 不用 post-hoc frame0 overwrite 掩盖 compositor 违规。
            mask = object_mask.unsqueeze(1).float()
            mask[0] = 0
            base_latent = codec.encode(base.unsqueeze(0))
            full_latent = codec.encode(target.unsqueeze(0))
            scale = int(cfg.model.vae_scale_factor)
            latent_mask = downsample_mask_to_latent(mask.to(codec.device), scale).unsqueeze(0)
            latent_mask[:, 0] = 0
            dilated_mask = dilate_latent_mask(latent_mask[0], int(settings["dilation_radius"])).unsqueeze(0)
            hybrid_latent = make_hybrid_latent(base_latent, full_latent, latent_mask)
            dilated_latent = make_hybrid_latent(base_latent, full_latent, dilated_mask)
            decoded_base = codec.decode(base_latent)[0].detach().cpu()
            decoded_full = codec.decode(full_latent)[0].detach().cpu()
            decoded_hybrid = codec.decode(hybrid_latent)[0].detach().cpu()
            decoded_dilated = codec.decode(dilated_latent)[0].detach().cpu()
            full_metrics, _ = _variant_metrics(
                name="full", decoded=decoded_full, target=target, decoded_base=decoded_base,
                base_latent=base_latent,
                full_latent=full_latent, variant_latent=full_latent, mask_rgb=mask,
                mask_latent=latent_mask, codec=codec, lpips_metric=lpips_metric,
            )
            hybrid_metrics, _ = _variant_metrics(
                name="hybrid", decoded=decoded_hybrid, target=target, decoded_base=decoded_base,
                base_latent=base_latent,
                full_latent=full_latent, variant_latent=hybrid_latent, mask_rgb=mask,
                mask_latent=latent_mask, codec=codec, lpips_metric=lpips_metric,
            )
            dilated_metrics, _ = _variant_metrics(
                name="dilated_hybrid", decoded=decoded_dilated, target=target, decoded_base=decoded_base,
                base_latent=base_latent,
                full_latent=full_latent, variant_latent=dilated_latent, mask_rgb=mask,
                mask_latent=dilated_mask, codec=codec, lpips_metric=lpips_metric,
            )
            source_rows = source_duplication_rows(
                base, target, primary_original, primary_projected,
                minimum_destination_change_l1=float(settings["render"]["minimum_destination_change_l1"]),
                maximum_source_change_l1=float(settings["render"]["maximum_source_change_l1"]),
                maximum_overlap_iou=float(settings["render"]["maximum_source_destination_iou"]),
            )
            moved_keys = {
                (str(value["track_token"]), int(value["time"]))
                for value in source_rows if bool(value["moved_after_quantization"])
            }
            overlaps = _overlap_rows(
                primary_projected, height, width,
                iou_threshold=float(settings["render"]["occlusion_overlap_iou"]), moved_keys=moved_keys,
            )
            changed = (target - base).abs().amax(dim=1) > float(settings["render"]["minimum_destination_change_l1"])
            moved = [row for row in source_rows if row["moved_after_quantization"]]
            duplicated = [row for row in source_rows if row["source_retained_duplication_proxy"]]
            row = {
                "dataset_index": index, "sample_id": str(metadata["sample_id"]),
                "candidate": "P-UNC", "primary_track_count": len(primary_original),
                "raw_punc_frame0_track_max_error": raw_frame0_track_error,
                "target_frame0_rgb_max_error": target_frame0_error,
                "full_latent_frame0_max_error": float((full_latent[:, 0] - base_latent[:, 0]).abs().max()),
                "hybrid_latent_frame0_max_error": float((hybrid_latent[:, 0] - base_latent[:, 0]).abs().max()),
                "dilated_hybrid_latent_frame0_max_error": float((dilated_latent[:, 0] - base_latent[:, 0]).abs().max()),
                "object_mask_fraction": float(mask.mean()), "latent_mask_fraction": float(latent_mask.mean()),
                "dilated_mask_fraction": float(dilated_mask.mean()), "target_changed_pixel_count": int(changed.sum()),
                "target_rgb_inside_rms": _masked_rms(target - base, mask),
                "target_rgb_outside_rms": _masked_rms(target - base, 1.0 - mask),
                "renderable_move_count": len(moved), "source_duplication_count": len(duplicated),
                "source_duplication_fraction": len(duplicated) / max(len(moved), 1),
                "unresolved_occlusion_overlap_count": len(overlaps),
                "unresolved_occlusion_overlap_fraction": len(overlaps) / max(len(moved), 1),
                "full_target_lpips": full_metrics["target_lpips"],
                "hybrid_target_lpips": hybrid_metrics["target_lpips"],
                "dilated_hybrid_target_lpips": dilated_metrics["target_lpips"],
                "hybrid_outside_latent_rms_over_base": hybrid_metrics["outside_latent_rms_over_base"],
                "dilated_outside_latent_rms_over_base": dilated_metrics["outside_latent_rms_over_base"],
                "hybrid_decode_reencode_latent_rms": hybrid_metrics["decode_reencode_latent_rms"],
                "dilated_decode_reencode_latent_rms": dilated_metrics["decode_reencode_latent_rms"],
                "hybrid_decoded_target_motion_cosine_inside": hybrid_metrics["decoded_target_motion_cosine_inside"],
                "dilated_decoded_target_motion_cosine_inside": dilated_metrics["decoded_target_motion_cosine_inside"],
            }
            rows.append(row)
            metrics.append(index, {"phase": "target_validity", **row, "full": full_metrics, "hybrid": hybrid_metrics, "dilated": dilated_metrics})
            duplication_rows.extend([{"dataset_index": index, "sample_id": row["sample_id"], **value} for value in source_rows])
            overlap_rows.extend([{"dataset_index": index, "sample_id": row["sample_id"], **value} for value in overlaps])
            case_id = f"p1-i{index:03d}"
            panel_path = run_dir / "panels" / f"{case_id}.mp4"
            write_video(_review_video(base, target, mask[:, 0], decoded_hybrid, decoded_dilated), str(panel_path), fps=int(settings["review"]["panel_fps"]))
            review_cases.append({"case_id": case_id, "dataset_index": index, "sample_id": row["sample_id"], "panel_path": str(panel_path), "candidate": "P-UNC"})
            del base_latent, full_latent, hybrid_latent, dilated_latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del provider, codec, lpips_metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        decision = _machine_decision(rows, settings)
        machine = {
            "task_id": str(settings["task_id"]), "protocol": PROTOCOL_VERSION,
            "dataset_indices": indices, "sample_count": len(indices), "candidate": "P-UNC",
            "uses_future_gt": False, "adapter_loaded": False, "rows": rows,
            "source_duplication": {
                "count": sum(int(row["source_duplication_count"]) for row in rows),
                "fraction": _finite_quantiles([row["source_duplication_fraction"] for row in rows]),
            },
            "unresolved_occlusion": {
                "count": sum(int(row["unresolved_occlusion_overlap_count"]) for row in rows),
                "fraction": _finite_quantiles([row["unresolved_occlusion_overlap_fraction"] for row in rows]),
            },
            "decision": decision, "machine_pass": bool(decision["machine_pass"]),
            "experiment_fingerprint": sha256_json({"config": config_fp, "rows": rows, "decision": decision}),
        }
        _write_csv(run_dir / "target_rows.csv", rows)
        _write_csv(run_dir / "source_duplication_rows.csv", duplication_rows or [{"dataset_index": None, "sample_id": None, "track_token": None, "stratum": None, "time": None, "moved_after_quantization": False, "source_destination_iou": None, "source_change_l1": None, "destination_change_l1": None, "source_retained_duplication_proxy": False, "texture_area_ratio": None}])
        _write_csv(run_dir / "occlusion_overlap_rows.csv", overlap_rows or [{"dataset_index": None, "sample_id": None, "time": None, "left": None, "right": None, "iou": None, "occlusion_order_known": False}])
        atomic_write_json(str(run_dir / "review_cases.json"), review_cases)
        _write_reviews(run_dir, review_cases)
        atomic_write_json(str(run_dir / "machine_summary.json"), machine)
        return _update_reviews(run_dir, settings)
    except Exception as exc:
        atomic_write_json(str(run_dir / "summary.json"), {"status": "failed", "error": repr(exc)})
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(run_dir / "manifest.json"), manifest_data)
        _clean_markers(run_dir)
        atomic_write_text(str(run_dir / "FAILED"), repr(exc) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    result = run_target_validity(load_config(args.config, list(args.overrides)), aggregate_only=bool(args.aggregate_only))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
