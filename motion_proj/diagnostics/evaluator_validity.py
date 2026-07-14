"""E0：CoTracker3 独立 rollout evaluator 的有效性审计。

E0 只能读取冻结生成的 ``base_rgb``，并只用 CoTracker3 自己的 first-frame grid 查询。
它绝不复用 RAFT-chain target tracks、P0/P1 outputs、future GT 或 source future metadata；官方
checkpoint 不可用时明确写 ``blocked``，绝不静默回退到 RAFT 或将无效轨迹记成 0。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..cache.dataset import ProjectionCacheDataset
from ..config import config_fingerprint, get_paths, load_config, save_resolved_config
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video, write_video
from ..eval.independent_tracks import (
    STRATA,
    CoTracker3IndependentEvaluator,
    IndependentTrackState,
    aggregate_dynamics,
    summarize_camera_compensated_dynamics,
)


PROTOCOL_VERSION = "autoresearch-e0-independent-cotracker3-v1"
REVIEW_VALUES = {"valid", "invalid", "uncertain"}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    rank = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        value = (start + end - 1) / 2.0
        for index in order[start:end]:
            rank[index] = value
        start = end
    return rank


def spearman_rank_correlation(left: Iterable[float | None], right: Iterable[float | None]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if _finite(a) is not None and _finite(b) is not None]
    if len(pairs) < 3:
        return None
    lx, rx = _rank([item[0] for item in pairs]), _rank([item[1] for item in pairs])
    mean_l, mean_r = sum(lx) / len(lx), sum(rx) / len(rx)
    numerator = sum((a - mean_l) * (b - mean_r) for a, b in zip(lx, rx))
    denominator = math.sqrt(sum((a - mean_l) ** 2 for a in lx) * sum((b - mean_r) ** 2 for b in rx))
    return numerator / denominator if denominator > 1.0e-12 else None


def relative_metric_delta(left: Mapping[str, float] | None, right: Mapping[str, float] | None) -> dict[str, float | None]:
    """不把 invalid/NaN 变成 0；只比较双方都有效的 aggregate metric。"""
    if left is None or right is None:
        return {"max_relative_delta": None}
    values = []
    result: dict[str, float | None] = {}
    for key in sorted(set(left) & set(right)):
        a, b = _finite(left[key]), _finite(right[key])
        if a is None or b is None:
            result[key] = None
            continue
        value = abs(a - b) / max(abs(a), 1.0e-6)
        result[key] = value
        values.append(value)
    result["max_relative_delta"] = max(values) if values else None
    return result


def _track_rerun_difference(left: IndependentTrackState, right: IndependentTrackState) -> dict[str, float | int | None]:
    common = left.visibility & right.visibility
    if not bool(common.any()):
        return {"point_max_error_px": None, "visibility_mismatch_count": int((left.visibility != right.visibility).sum())}
    difference = (left.points[common] - right.points[common]).abs()
    return {
        "point_max_error_px": float(difference.max()),
        "visibility_mismatch_count": int((left.visibility != right.visibility).sum()),
    }


def perturb_video(frames: torch.Tensor, mode: str) -> torch.Tensor:
    """语义不变的小扰动：亮度、8-bit codec-like quantization、resize round-trip。"""
    if mode == "photometric":
        return (frames + 0.015).clamp(-1, 1)
    if mode == "codec_quantization":
        return (((frames + 1.0) * 127.5).round() / 127.5 - 1.0).clamp(-1, 1)
    if mode == "resize_roundtrip":
        height, width = frames.shape[-2:]
        small = F.interpolate(frames, size=(max(32, round(height * 0.97)), max(32, round(width * 0.97))), mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False).clamp(-1, 1)
    raise ValueError(f"unknown perturbation: {mode}")


def _warp_translation(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """以 texture first frame 构造可控 synthetic image-plane motion。"""
    _, height, width = image.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid = torch.stack([
        2.0 * (xx.float() - dx) / max(width - 1, 1) - 1.0,
        2.0 * (yy.float() - dy) / max(height - 1, 1) - 1.0,
    ], dim=-1).unsqueeze(0)
    return F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)[0]


def synthetic_videos(seed: int, *, frames: int = 8, height: int = 144, width: int = 240) -> dict[str, torch.Tensor]:
    """有纹理、已知平移/加速/转弯/遮挡的 E0 sanity inputs；不依赖数据集或 GT。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    texture = torch.randn(3, height, width, generator=generator)
    texture = F.avg_pool2d(texture.unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].tanh()
    time = torch.arange(frames, dtype=torch.float32)
    paths = {
        "constant_velocity": (2.0 * time, torch.zeros_like(time)),
        "constant_acceleration": (0.28 * time.square(), torch.zeros_like(time)),
        "smooth_turn": (1.6 * time, 0.18 * time.square()),
        "occlusion": (1.4 * time, 0.12 * time.square()),
    }
    output = {}
    for name, (x, y) in paths.items():
        video = torch.stack([_warp_translation(texture, float(dx), float(dy)) for dx, dy in zip(x, y)])
        if name == "occlusion":
            video[frames // 2 - 1:frames // 2 + 1, :, height // 3:height * 2 // 3, width // 3:width * 2 // 3] = -1.0
        output[name] = video
    return output


def _draw_overlay(frames: torch.Tensor, state: IndependentTrackState) -> Any:
    import cv2
    import numpy as np

    palette = {"background": (80, 180, 255), "dynamic_residual": (255, 90, 90), "foreground_candidate": (80, 230, 100)}
    video = to_uint8_video(frames)
    output = video.copy()
    for time in range(video.shape[0]):
        for index, label in enumerate(state.labels):
            if not bool(state.visibility[index, time]):
                continue
            x, y = state.points[index, time].round().long().tolist()
            color = palette.get(label, (220, 220, 220))
            cv2.circle(output[time], (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
            if time and bool(state.visibility[index, time - 1]):
                px, py = state.points[index, time - 1].round().long().tolist()
                cv2.line(output[time], (int(px), int(py)), (int(x), int(y)), color, 1, cv2.LINE_AA)
    return output


def _review_template(run_dir: Path, cases: list[dict[str, Any]], settings: Mapping[str, Any]) -> None:
    template = run_dir / "reviews.template.jsonl"
    if not template.exists():
        atomic_write_text(str(template), "".join(json.dumps({
            "case_id": row["case_id"], "verdict": "pending", "reviewer": "human", "notes": "",
            "rubric": "overlay point 是否贴合可追踪局部；遮挡/低纹理失效是否被 visibility 标记，而非被伪造为稳定轨迹？",
        }, ensure_ascii=False) + "\n" for row in cases))
    readme = run_dir / "REVIEW_README.md"
    if not readme.exists():
        atomic_write_text(
            str(readme),
            "# E0 CoTracker3 独立 evaluator 人工复核\n\n"
            "仅查看 `track_overlay/*.mp4`。所有点来自 CoTracker3 first-frame grid，颜色为 evaluator-only"
            "camera-compensated strata，不是 RAFT/P0/P1 track。`valid`：大多数点贴合纹理且遮挡/低纹理失效"
            "被 visibility 正确标记；`invalid`：系统性漂移/伪连续；`uncertain`：无法判定。\n\n"
            "复制模板为 `reviews.jsonl` 后以 `--aggregate-only` 更新。\n",
        )


def _read_reviews(run_dir: Path, cases: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    reviews: dict[str, dict[str, Any]] = {}
    path = run_dir / "reviews.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if str(row.get("verdict")) in REVIEW_VALUES:
                    reviews[str(row.get("case_id"))] = row
    selected = [reviews[row["case_id"]] for row in cases if row["case_id"] in reviews]
    decisive = [row for row in selected if row["verdict"] != "uncertain"]
    valid = sum(row["verdict"] == "valid" for row in decisive)
    rate = valid / len(decisive) if decisive else None
    required = int(settings["review"]["required_cases"])
    passed = bool(len(selected) >= required and rate is not None and rate >= float(settings["review"]["minimum_valid_rate"]))
    return {
        "required": required, "completed": len(selected), "decisive": len(decisive), "valid": valid,
        "invalid": sum(row["verdict"] == "invalid" for row in decisive), "valid_rate": rate,
        "minimum_valid_rate": float(settings["review"]["minimum_valid_rate"]),
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
    review = _read_reviews(run_dir, cases, settings)
    if machine.get("status") == "blocked":
        status = "blocked"
    elif machine["machine_pass"]:
        status = "pass" if review["human_pass"] else "awaiting_reviews"
    else:
        status = "fail"
    summary = {**machine, "status": status, "human_review": review}
    atomic_write_json(str(run_dir / "summary.json"), summary)
    _clean_markers(run_dir)
    atomic_write_text(str(run_dir / ("awaiting_reviews" if status == "awaiting_reviews" else "COMPLETE")), sha256_json(summary) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": "review" if status != "blocked" else "official_checkpoint_unavailable"})
    atomic_write_json(str(manifest_path), manifest)
    return summary


def _machine_decision(real_rows: list[dict[str, Any]], synthetic: Mapping[str, Any], threshold: Mapping[str, Any]) -> dict[str, Any]:
    rerun = [row["rerun"]["aggregate_relative_delta"]["max_relative_delta"] for row in real_rows]
    rerun_ok = all(value is not None and float(value) <= float(threshold["maximum_rerun_relative_delta"]) for value in rerun)
    ranks = [value for value in synthetic.get("threshold_sweep_rank_correlations", []) if value is not None]
    synthetic_ok = bool(synthetic.get("acceleration_order_correct")) and bool(synthetic.get("jerk_order_correct"))
    invalid_recognized = bool(synthetic.get("occlusion_invalid_or_downweighted"))
    checks = {
        "identical_video_rerun": rerun_ok,
        "threshold_sweep_rank_correlation": bool(ranks) and min(float(value) for value in ranks) >= float(threshold["minimum_rank_correlation"]),
        "synthetic_acceleration_and_jerk_order": synthetic_ok,
        "occlusion_low_texture_invalidity_recognized": invalid_recognized,
        "all_real_clips_have_valid_tracks": all(bool(row["valid"]) for row in real_rows),
    }
    return {"checks": checks, "machine_pass": all(checks.values())}


def _validate_base_metadata(metadata: Mapping[str, Any], index: int) -> None:
    expected = {"source": "replay_v2", "parent_kind": "base", "adapter_loaded": False, "uses_future_gt_ego": False, "uses_future_gt_track": False}
    mismatch = {key: {"expected": value, "actual": metadata.get(key)} for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"E0 index {index} is not leakage-free frozen Base: {mismatch}")


def run_evaluator_validity(cfg: Any, *, aggregate_only: bool = False) -> dict[str, Any]:
    settings = OmegaConf.to_container(cfg.e0, resolve=True)
    assert isinstance(settings, dict)
    indices = [int(value) for value in settings["dataset_indices"]]
    if not 1 <= len(indices) <= 8 or len(indices) != len(set(indices)):
        raise ValueError("E0 requires 1-8 unique Base replay indices")
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("formal E0 refuses to run in a dirty worktree")
    run_dir = Path(str(cfg.work_dir))
    if aggregate_only:
        return _update_reviews(run_dir, settings)
    if run_dir.exists():
        raise RuntimeError(f"E0 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "track_overlay").mkdir()
    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=config_fp,
        cache_fingerprint=str(settings["cache_fingerprint"]), seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.data.split),
    )
    evaluator = CoTracker3IndependentEvaluator({**settings["provider"], "device": str(cfg.device)})
    preflight = evaluator.preflight()
    manifest_data = manifest.__dict__ | {
        "task_id": str(settings["task_id"]), "protocol": PROTOCOL_VERSION, "dataset_indices": indices,
        "input_provenance": "frozen generated base_rgb + CoTracker3 first-frame grid + evaluator weights only",
        "forbidden_inputs": ["cache generated_tracks", "projector outputs", "future GT", "source future metadata"],
        "provider_preflight": preflight,
    }
    atomic_write_json(str(run_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(run_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(run_dir / "metrics.jsonl"))
    # Checkpoint/resource absence is a scientific E0 block, not an invitation to fall back to RAFT.
    if not preflight["available"]:
        machine = {
            "task_id": str(settings["task_id"]), "protocol": PROTOCOL_VERSION, "status": "blocked",
            "machine_pass": False, "reason": "official_cotracker3_checkpoint_unavailable",
            "provider_preflight": preflight, "uses_future_gt": False, "fallback_used": False,
            "experiment_fingerprint": sha256_json({"config": config_fp, "preflight": preflight}),
        }
        metrics.append(-1, {"phase": "preflight", **machine})
        atomic_write_json(str(run_dir / "machine_summary.json"), machine)
        atomic_write_json(str(run_dir / "review_cases.json"), [])
        return _update_reviews(run_dir, settings)

    paths = get_paths(cfg)
    dataset = ProjectionCacheDataset(str(paths.cache_dir), expected_fingerprint=str(settings["cache_fingerprint"]))
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("E0 replay index is outside cache")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed_everything(int(cfg.seed), deterministic=True)
        real_rows: list[dict[str, Any]] = []
        review_cases: list[dict[str, Any]] = []
        for index in indices:
            item = dataset[index]
            _validate_base_metadata(item["metadata"], index)
            frames = item["base_rgb"]
            first, second = evaluator.track(frames), evaluator.track(frames)
            dynamics = summarize_camera_compensated_dynamics(first)
            first_aggregate, second_aggregate = aggregate_dynamics(dynamics), aggregate_dynamics(summarize_camera_compensated_dynamics(second))
            rerun = {
                "track": _track_rerun_difference(first, second),
                "aggregate_relative_delta": relative_metric_delta(first_aggregate, second_aggregate),
            }
            perturb = {}
            for mode in settings["perturbations"]:
                state = evaluator.track(perturb_video(frames, str(mode)))
                perturb[str(mode)] = {
                    "valid": state.valid,
                    "aggregate_relative_delta": relative_metric_delta(first_aggregate, aggregate_dynamics(summarize_camera_compensated_dynamics(state))),
                }
            row = {
                "dataset_index": index, "sample_id": str(item["metadata"]["sample_id"]), "valid": first.valid,
                "provider_diagnostics": first.diagnostics, "dynamics": dynamics, "aggregate": first_aggregate,
                "rerun": rerun, "perturbations": perturb,
            }
            real_rows.append(row)
            metrics.append(index, {"phase": "real_base", **row})
            case_id = f"e0-real-i{index:03d}"
            panel_path = run_dir / "track_overlay" / f"{case_id}.mp4"
            write_video(_draw_overlay(frames, first), str(panel_path), fps=int(settings["review"]["panel_fps"]))
            review_cases.append({"case_id": case_id, "kind": "real_base", "dataset_index": index, "panel_path": str(panel_path)})

        synthetic_rows = {}
        for name, video in synthetic_videos(int(cfg.seed)).items():
            state = evaluator.track(video)
            summary = summarize_camera_compensated_dynamics(state)
            aggregate = aggregate_dynamics(summary)
            synthetic_rows[name] = {"valid": state.valid, "summary": summary, "aggregate": aggregate}
            metrics.append(-1, {"phase": "synthetic", "name": name, **synthetic_rows[name]})
            case_id = f"e0-synthetic-{name}"
            panel_path = run_dir / "track_overlay" / f"{case_id}.mp4"
            write_video(_draw_overlay(video, state), str(panel_path), fps=int(settings["review"]["panel_fps"]))
            review_cases.append({"case_id": case_id, "kind": "synthetic", "name": name, "panel_path": str(panel_path)})

        baseline = [row["aggregate"].get("camera_compensated_image_plane_acceleration_rms_px") if row["aggregate"] else None for row in real_rows]
        sweep = []
        # CoTracker3 returns hard visibility after its official threshold; sweep is a deterministic
        # clip-level survival eligibility threshold, not a hidden tracker confidence modification.
        for threshold in settings["visibility_survival_thresholds"]:
            filtered = [
                (row["aggregate"].get("camera_compensated_image_plane_acceleration_rms_px") if row["aggregate"] and row["aggregate"].get("survival_rate", 0.0) >= float(threshold) else None)
                for row in real_rows
            ]
            sweep.append(spearman_rank_correlation(baseline, filtered))
        cv = synthetic_rows.get("constant_velocity", {}).get("aggregate") or {}
        ca = synthetic_rows.get("constant_acceleration", {}).get("aggregate") or {}
        turn = synthetic_rows.get("smooth_turn", {}).get("aggregate") or {}
        occlusion = synthetic_rows.get("occlusion", {}).get("aggregate")
        synthetic_summary = {
            "rows": synthetic_rows, "threshold_sweep_rank_correlations": sweep,
            "acceleration_order_correct": (
                _finite(ca.get("camera_compensated_image_plane_acceleration_rms_px")) is not None
                and _finite(cv.get("camera_compensated_image_plane_acceleration_rms_px")) is not None
                and float(ca["camera_compensated_image_plane_acceleration_rms_px"]) > float(cv["camera_compensated_image_plane_acceleration_rms_px"])
            ),
            "jerk_order_correct": (
                _finite(turn.get("camera_compensated_image_plane_jerk_rms_px")) is not None
                and _finite(cv.get("camera_compensated_image_plane_jerk_rms_px")) is not None
                and float(turn["camera_compensated_image_plane_jerk_rms_px"]) >= float(cv["camera_compensated_image_plane_jerk_rms_px"])
            ),
            "occlusion_invalid_or_downweighted": occlusion is None or not bool(synthetic_rows["occlusion"]["valid"]) or _finite(occlusion.get("survival_rate")) is None or float(occlusion["survival_rate"]) < float(cv.get("survival_rate", 1.0)),
        }
        decision = _machine_decision(real_rows, synthetic_summary, settings["thresholds"])
        machine = {
            "task_id": str(settings["task_id"]), "protocol": PROTOCOL_VERSION, "status": "completed",
            "machine_pass": bool(decision["machine_pass"]), "provider_preflight": preflight,
            "uses_future_gt": False, "fallback_used": False, "real_rows": real_rows,
            "synthetic": synthetic_summary, "decision": decision,
            "experiment_fingerprint": sha256_json({"config": config_fp, "real": real_rows, "synthetic": synthetic_summary, "decision": decision}),
        }
        atomic_write_json(str(run_dir / "machine_summary.json"), machine)
        atomic_write_json(str(run_dir / "review_cases.json"), review_cases)
        _review_template(run_dir, review_cases, settings)
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
    result = run_evaluator_validity(load_config(args.config, list(args.overrides)), aggregate_only=bool(args.aggregate_only))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
