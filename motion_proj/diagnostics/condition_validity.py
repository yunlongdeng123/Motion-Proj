"""P2-V2-COND-00：SVD generated rollout 的条件有效性诊断。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from ..auditor import (
    MotionAuditor,
    masked_flow_statistics,
    render_pairwise_background_correction,
)
from ..backbones import build_backbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import ExperimentRegistry, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..utils.geometry import se3_inverse
from ..utils.io import to_uint8_video, write_video
from ..utils.viz import flow_to_rgb, hstack_panels


PROTOCOL_VERSION = "p2-v2-condition-validity-v2"
AUDIT_MODES = ("gt_ego_debug", "identity_ego", "estimated_background_motion")
REVIEW_VALUES = {"yes", "no", "uncertain"}
REVIEW_TARGETS = {"static", "generated_tracks"}


def _model_fingerprint(pretrained: str) -> str:
    root = Path(pretrained)
    if not root.is_dir():
        return sha256_json({"pretrained": pretrained})
    candidates = [
        root / "model_index.json",
        root / "unet" / "config.json",
        root / "vae" / "config.json",
        root / "scheduler" / "scheduler_config.json",
        root / "image_encoder" / "config.json",
    ]
    rows = [(str(path.relative_to(root)), file_fingerprint(str(path))) for path in candidates if path.is_file()]
    return sha256_json(rows)


def _ego_motion_summary(poses: torch.Tensor) -> dict[str, float]:
    translations = []
    rotations = []
    for index in range(poses.shape[0] - 1):
        relative = se3_inverse(poses[index]) @ poses[index + 1]
        translations.append(float(relative[:3, 3].norm()))
        trace = float(torch.trace(relative[:3, :3]))
        cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
        rotations.append(math.degrees(math.acos(cosine)))
    return {
        "translation_mean_m": float(np.mean(translations)) if translations else 0.0,
        "translation_max_m": float(np.max(translations)) if translations else 0.0,
        "rotation_mean_deg": float(np.mean(rotations)) if rotations else 0.0,
        "rotation_max_deg": float(np.max(rotations)) if rotations else 0.0,
    }


def _generated_sample(frames: torch.Tensor, source: dict[str, Any], mode: str) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "frames": frames,
        "boxes": [[] for _ in range(frames.shape[0])],
        "intrinsics": source["intrinsics"],
        "cam2ego": source["cam2ego"],
        "sample_id": source["sample_id"] + "_base",
    }
    if mode == "gt_ego_debug":
        sample["ego2global"] = source["ego2global"]
    return sample


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    import cv2

    output = image.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 260), 28), (0, 0, 0), -1)
    cv2.putText(output, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _condition_panel(
    base: torch.Tensor,
    gt_target: torch.Tensor,
    estimated_target: torch.Tensor,
    observed_flow: torch.Tensor,
    estimated_mask: torch.Tensor,
    generated_tracks: list,
) -> np.ndarray:
    base_video = to_uint8_video(base)
    gt_video = to_uint8_video(gt_target)
    estimated_video = to_uint8_video(estimated_target)
    tracks_video = _draw_generated_tracks(base_video, generated_tracks)
    frames = []
    for index in range(base.shape[0]):
        pair_index = min(index, observed_flow.shape[0] - 1)
        flow_rgb = flow_to_rgb(observed_flow[pair_index])
        mask = estimated_mask[index, 0].detach().float().cpu().numpy()[..., None]
        overlay = (flow_rgb.astype(np.float32) * (0.35 + 0.65 * mask)).clip(0, 255).astype(np.uint8)
        frames.append(
            hstack_panels(
                _label_panel(base_video[index], "Base rollout"),
                _label_panel(tracks_video[index], "Generated point tracks (no GT)"),
                _label_panel(gt_video[index], "GT-ego correction [debug]"),
                _label_panel(estimated_video[index], "Self-estimated correction"),
                _label_panel(overlay, "Observed flow / self mask"),
            )
        )
    return np.stack(frames)


def _draw_generated_tracks(video: np.ndarray, tracks: list) -> np.ndarray:
    """在 Base rollout 上绘制 provider 的局部点轨迹，供人工检查而非训练输入。"""
    import cv2

    palette = {
        "background": (80, 180, 255),
        "dynamic_residual": (255, 90, 90),
        "foreground_candidate": (80, 230, 100),
    }
    rendered = video.copy()
    for time in range(rendered.shape[0]):
        for track in tracks:
            if not bool(track.present[time]):
                continue
            label = str(track.category).split("/")[-1]
            color = palette.get(label, (220, 220, 220))
            center = track.center[time].detach().float().cpu().round().to(torch.int64).tolist()
            point = (int(center[0]), int(center[1]))
            cv2.circle(rendered[time], point, 3, color, -1, lineType=cv2.LINE_AA)
            if time and bool(track.present[time - 1]):
                previous = track.center[time - 1].detach().float().cpu().round().to(torch.int64).tolist()
                cv2.line(rendered[time], (int(previous[0]), int(previous[1])), point, color, 1, cv2.LINE_AA)
    return rendered


def _correction_stats(base: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    delta = target.float() - base.float()
    active = mask.expand_as(delta) > 0
    values = delta[active]
    return {
        "rms": float(values.square().mean().sqrt()) if values.numel() else None,
        "mask_fraction": float((mask > 0).float().mean()),
        "finite": bool(torch.isfinite(target).all() and torch.isfinite(mask).all()),
        "first_frame_frozen": bool(torch.equal(base[0].cpu(), target[0].cpu())),
        "first_frame_mask_zero": bool(torch.count_nonzero(mask[0]) == 0),
    }


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "cases").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda row: int(row["case_index"]))


def _load_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_condition_validity(
    rows: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    *,
    required_reviews: int,
    minimum_reasonable_rate: float,
    review_target: str = "static",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有 condition validity case")
    if review_target not in REVIEW_TARGETS:
        raise ValueError(f"未知 review_target: {review_target}")
    review_map = {str(row.get("case_id")): row for row in reviews}
    review_field = "self_estimated_valid" if review_target == "static" else "point_track_valid"
    valid_reviews = []
    for row in rows:
        review = review_map.get(str(row["case_id"]))
        if review is None:
            continue
        if str(review.get(review_field, "")) in REVIEW_VALUES:
            valid_reviews.append(review)
    decisive = [row for row in valid_reviews if row[review_field] != "uncertain"]
    self_yes = sum(row[review_field] == "yes" for row in decisive)
    reasonable_rate = self_yes / len(decisive) if decisive else None
    if review_target == "static":
        automated_pass = all(
            row["modes"]["estimated_background_motion"]["correction"]["finite"]
            and row["modes"]["estimated_background_motion"]["correction"]["first_frame_frozen"]
            and row["modes"]["estimated_background_motion"]["correction"]["first_frame_mask_zero"]
            and not row["modes"]["estimated_background_motion"]["uses_future_gt_ego"]
            for row in rows
        )
    else:
        automated_pass = all(
            not row["modes"]["estimated_background_motion"]["uses_future_gt_track"]
            and int(row["modes"]["estimated_background_motion"]["track_diagnostics"]["valid_track_count"]) > 0
            and float(row["modes"]["estimated_background_motion"]["track_diagnostics"]["median_track_length"]) >= 3.0
            for row in rows
        )
    review_complete = len(valid_reviews) >= required_reviews
    promoted = bool(
        automated_pass
        and review_complete
        and reasonable_rate is not None
        and reasonable_rate >= minimum_reasonable_rate
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "status": "completed" if review_complete else "awaiting_reviews",
        "cases": len(rows),
        "uses_future_gt_ego": False,
        "base_generation_adapter_loaded": False,
        "review_target": review_target,
        "automated_checks_passed": automated_pass,
        "reviews": {
            "required": required_reviews,
            "completed": len(valid_reviews),
            "field": review_field,
            "decisive": len(decisive),
            "yes": self_yes,
            "reasonable_rate": reasonable_rate,
            "minimum_reasonable_rate": minimum_reasonable_rate,
        },
        "static_branch_decision": (
            "promote" if promoted else ("blocked" if review_complete else "pending_review")
        ) if review_target == "static" else "not_assessed",
        "generated_track_decision": (
            "promote" if promoted else ("blocked" if review_complete else "pending_review")
        ) if review_target == "generated_tracks" else "not_assessed",
        "mean_residual": {
            mode: float(np.mean([row["modes"][mode]["residual"]["mean"] for row in rows]))
            for mode in AUDIT_MODES
            if all(row["modes"][mode]["residual"]["mean"] is not None for row in rows)
        },
    }


@torch.no_grad()
def export_cases(cfg: Any, run_dir: Path) -> list[dict[str, Any]]:
    settings = OmegaConf.to_container(cfg.condition_validity, resolve=True)
    assert isinstance(settings, dict)
    indices = [int(value) for value in settings["clip_indices"]]
    completed = {int(row["case_index"]): row for row in _load_rows(run_dir)}
    missing = [case_index for case_index in range(len(indices)) if case_index not in completed]
    if not missing:
        return [completed[index] for index in range(len(indices))]

    dataset = NuScenesFutureVideoDataset(cfg.data)
    backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
    backbone.set_train_mode(False)
    fit_options = OmegaConf.to_container(cfg.auditor.background_fit, resolve=True)
    assert isinstance(fit_options, dict)
    auditor = MotionAuditor(
        device=str(cfg.device),
        enable_depth=True,
        background_fit_options=fit_options,
    )
    generation = settings["generation"]
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)
    (run_dir / "condition_validity_panel").mkdir(parents=True, exist_ok=True)

    for case_index in missing:
        clip_index = indices[case_index]
        source = dataset[clip_index]
        generation_seed = int(settings["generation_seed"]) + case_index
        generator = torch.Generator(device=str(cfg.device)).manual_seed(generation_seed)
        base = backbone.generation(
            source["cond_frame"].to(str(cfg.device)),
            num_frames=int(cfg.data.num_frames),
            num_inference_steps=int(generation["num_inference_steps"]),
            generator=generator,
            height=int(cfg.data.height),
            width=int(cfg.data.width),
            decode_chunk_size=int(generation["decode_chunk_size"]),
        )

        gt_sample = _generated_sample(base, source, "gt_ego_debug")
        gt_state = auditor.audit(gt_sample, generated_geometry_mode="gt_ego_debug")
        observed_flow = gt_state.u_static
        flow_confidence = gt_state.flow_conf
        depth = gt_state.depth
        states = {"gt_ego_debug": gt_state}
        for mode in ("identity_ego", "estimated_background_motion"):
            states[mode] = auditor.audit(
                _generated_sample(base, source, mode),
                generated_geometry_mode=mode,
                observed_flow=observed_flow,
                flow_confidence=flow_confidence,
                depth=depth,
            )

        targets = {}
        masks = {}
        mode_rows = {}
        for mode, state in states.items():
            target, mask = render_pairwise_background_correction(
                base,
                state.u_ego,
                state.static_mask,
                inverse_iterations=int(settings["inverse_iterations"]),
            )
            targets[mode] = target
            masks[mode] = mask
            mode_rows[mode] = {
                "uses_future_gt_ego": bool(state.meta["uses_future_gt_ego"]),
                "uses_future_gt_track": bool(state.meta["uses_future_gt_track"]),
                "residual": masked_flow_statistics(observed_flow, state.u_ego, state.static_mask),
                "static_mask_fraction": float((state.static_mask > 0).float().mean()),
                "expected_flow_mean": float(state.u_ego.float().norm(dim=-1).mean()),
                "geometry_diagnostics": state.meta["geometry_diagnostics"],
                "track_diagnostics": state.meta["track_diagnostics"],
                "correction": _correction_stats(base, target, mask),
            }

        case_id = f"cond-{case_index:02d}-clip-{clip_index:04d}-seed-{generation_seed}"
        panel_path = run_dir / "condition_validity_panel" / f"{case_id}.mp4"
        panel = _condition_panel(
            base,
            targets["gt_ego_debug"],
            targets["estimated_background_motion"],
            observed_flow,
            masks["estimated_background_motion"],
            states["estimated_background_motion"].tracks,
        )
        write_video(panel, str(panel_path), fps=int(settings["panel_fps"]))
        row = {
            "case_index": case_index,
            "case_id": case_id,
            "clip_index": clip_index,
            "clip_id": source["sample_id"],
            "generation_seed": generation_seed,
            "generation_steps": int(generation["num_inference_steps"]),
            "adapter_loaded": False,
            "dataset_ego": _ego_motion_summary(source["ego2global"]),
            "generated_global_flow_mean": mode_rows["estimated_background_motion"]["expected_flow_mean"],
            "gt_expected_flow_mean": mode_rows["gt_ego_debug"]["expected_flow_mean"],
            "modes": mode_rows,
            "panel_path": str(panel_path),
        }
        atomic_write_json(str(run_dir / "cases" / f"{case_id}.json"), row)
        completed[case_index] = row
        atomic_write_text(
            str(run_dir / "condition_validity.jsonl"),
            "".join(json.dumps(completed[index], ensure_ascii=False, allow_nan=False) + "\n" for index in sorted(completed)),
        )

    return [completed[index] for index in range(len(indices))]


def _write_review_package(
    run_dir: Path,
    rows: list[dict[str, Any]],
    required_reviews: int,
    review_target: str,
) -> None:
    template = run_dir / "reviews.template.jsonl"
    if not template.exists():
        lines = []
        for row in rows[:required_reviews]:
            review = {
                "case_id": row["case_id"],
                "failure_reason": "",
                "reviewer": "human",
            }
            if review_target == "static":
                review.update({"gt_ego_valid": "uncertain", "self_estimated_valid": "uncertain"})
            else:
                review["point_track_valid"] = "uncertain"
            lines.append(
                json.dumps(review, ensure_ascii=False)
            )
        atomic_write_text(str(template), "\n".join(lines) + "\n")
    readme = run_dir / "REVIEW_README.md"
    if not readme.exists():
        atomic_write_text(
            str(readme),
            "# P2-V2 人工复核\n\n"
            + (
                "每个视频依次为 `[Base | generated point tracks | GT-ego debug correction | self-estimated correction | observed flow/self mask]`。\n"
                "至少复核模板中的 case，分别填写 `yes/no/uncertain`；重点判断背景运动修正是否符合 "
                "Base 自身生成的相机运动，以及是否引入撕裂、冻结或主体破坏。\n"
                if review_target == "static" else
                "每个视频依次为 `[Base | generated point tracks | GT-ego debug correction | self-estimated correction | observed flow/self mask]`。\n"
                "只评第二栏 generated point tracks：点是否大多贴合可见、可追踪的图像局部，跨帧是否连续；"
                "若点系统性漂移到无关区域、跨帧跳变或明显把前景/背景混淆，填写 `no`，无法判断填写 `uncertain`。"
                "后面三栏是已阻断 static branch 的诊断上下文，不作为本轮 verdict 对象。\n"
            )
            + "复制 `reviews.template.jsonl` 为 `reviews.jsonl` 后运行同一命令并增加 `--aggregate-only`。\n",
        )


def run_experiment(cfg: Any, run_id: str | None, aggregate_only: bool) -> tuple[Path, dict[str, Any]]:
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 condition validity 诊断拒绝在 dirty worktree 上运行")
    settings = OmegaConf.to_container(cfg.condition_validity, resolve=True)
    assert isinstance(settings, dict)
    review_target = str(settings.get("review_target", "static"))
    if review_target not in REVIEW_TARGETS:
        raise ValueError("condition_validity.review_target 必须是 static 或 generated_tracks")
    cfg_fingerprint = config_fingerprint(cfg)
    fingerprint = sha256_json(
        {"protocol": PROTOCOL_VERSION, "config": cfg_fingerprint, "git_commit": git["commit"]}
    )
    if run_id is None:
        run_id = (
            f"p2-v2-cond16-s{int(settings['generation_seed'])}-"
            f"{str(git['commit'])[:8]}-{cfg_fingerprint[:8]}"
        )
    run_dir = Path(str(cfg.work_dir)) / run_id
    resolved_path = run_dir / "resolved.yaml"
    resolved_text = OmegaConf.to_yaml(cfg, resolve=True)
    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        raise RuntimeError(f"run 目录已有不同配置: {run_dir}")
    if not resolved_path.exists():
        save_resolved_config(cfg, str(resolved_path))

    manifest_path = run_dir / "manifest.json"
    if not aggregate_only:
        manifest = RunManifest(
            run_id=run_id,
            command=list(sys.argv),
            config_fingerprint=fingerprint,
            cache_fingerprint=f"not-applicable:{PROTOCOL_VERSION}",
            seed=int(settings["generation_seed"]),
            git=git,
            environment=environment_fingerprint(),
            data_split=f"{cfg.data.version}:{cfg.data.split}:CAM_FRONT:fixed-16",
        )
        manifest.save(str(manifest_path))
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_payload.update(
            {
                "task_id": "P2-V2-COND-00",
                "protocol": PROTOCOL_VERSION,
                "base_model_fingerprint": _model_fingerprint(str(cfg.model.pretrained)),
                "adapter_loaded": False,
                "uses_future_gt_ego_for_formal_candidate": False,
                "generation": settings["generation"],
            }
        )
        atomic_write_json(str(manifest_path), manifest_payload)
        rows = export_cases(cfg, run_dir)
        atomic_write_text(
            str(run_dir / "condition_validity.jsonl"),
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        )
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺少 manifest: {manifest_path}")
        rows = _load_rows(run_dir)

    required_reviews = int(settings["required_reviews"])
    _write_review_package(run_dir, rows, required_reviews, review_target)
    reviews = _load_reviews(run_dir / "reviews.jsonl")
    summary = summarize_condition_validity(
        rows,
        reviews,
        required_reviews=required_reviews,
        minimum_reasonable_rate=float(settings["minimum_reasonable_rate"]),
        review_target=review_target,
    )
    summary.update(
        {
            "run_id": run_id,
            "task_id": "P2-V2-COND-00",
            "git_commit": json.loads(manifest_path.read_text(encoding="utf-8"))["git"]["commit"],
            "config_fingerprint": cfg_fingerprint,
            "experiment_fingerprint": fingerprint,
            "review_fingerprint": (
                file_fingerprint(str(run_dir / "reviews.jsonl"))
                if (run_dir / "reviews.jsonl").is_file()
                else None
            ),
        }
    )
    atomic_write_json(str(run_dir / "condition_validity_summary.json"), summary)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "status": "completed",
            "ended_at": utc_now(),
            "exit_reason": summary["status"],
        }
    )
    atomic_write_json(str(manifest_path), manifest_payload)
    atomic_write_text(str(run_dir / "COMPLETE"), fingerprint + "\n")

    registry = ExperimentRegistry(str(Path(str(cfg.work_dir)) / "experiments.sqlite3"))
    known = {row["run_id"] for row in registry.list()}
    if run_id not in known:
        registry.register(run_id, "completed", fingerprint, str(run_dir))
    else:
        registry.update(run_id, "completed", exit_reason=str(summary["status"]), summary=summary)
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir, summary = run_experiment(cfg, args.run_id, args.aggregate_only)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
