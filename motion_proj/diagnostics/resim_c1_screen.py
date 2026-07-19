"""C1B-02：10-context ReSim E-vs-F action screen（单卡 L1）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from ..auditor.flow_raft import RAFTFlow
from ..auditor.generated_geometry import fit_affine_background_flow
from ..eval.independent_tracks import CoTracker3IndependentEvaluator
from ..preference.selective_order import video_quality_metrics
from ..runtime.atomic import atomic_write_json, atomic_write_text
from .resim_c1_proxy import (
    PROXY_PROTOCOL,
    affine_proxy_features,
    flow_with_confidence_chunked,
    predict_proxy,
)
from .resim_c1_smoke import (
    _environment_snapshot,
    _git_snapshot,
    _prepare_environment,
    _require_clean,
    _run_attempt,
    build_exact_manifest,
    build_resolved_config,
    decoded_repeat_difference,
    disk_budget,
    resolve_checkpoint,
    validate_video,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def decode_mp4(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"无法解码: {path}")
    return np.stack(frames)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    array = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0).mul(2.0).sub(1.0)
    return array


def mean_abs_uint8(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def load_source_history(
    nuscenes_root: Path, source_frames: Sequence[str], *, height: int, width: int, count: int
) -> np.ndarray:
    images = []
    for name in list(source_frames)[:count]:
        path = nuscenes_root / name
        image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
        images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def action_error(
    prediction: Mapping[str, Any],
    *,
    request_class: str,
    target_displacement_m: float,
    target_lateral_m: float,
) -> dict[str, float]:
    """预注册 action error；禁止看结果后改权。"""
    pred_class = str(prediction["predicted_class"])
    pred_disp = float(prediction["predicted_displacement_m"])
    pred_lat = float(prediction["predicted_lateral_m"])
    class_err = 0.0 if pred_class == request_class else 1.0
    if request_class in ("left", "right"):
        turn_err = 0.0 if pred_class == request_class else 1.0
    else:
        turn_err = 0.0 if pred_class not in ("left", "right") else 0.5
    disp_err = abs(pred_disp - float(target_displacement_m)) / max(float(target_displacement_m), 1.0)
    lat_err = abs(pred_lat - float(target_lateral_m)) / max(abs(float(target_lateral_m)), 1.0)
    total = class_err + 0.5 * turn_err + 0.25 * disp_err + 0.25 * lat_err
    return {
        "action_error": float(total),
        "class_err": float(class_err),
        "turn_err": float(turn_err),
        "disp_err": float(disp_err),
        "lat_err": float(lat_err),
    }


def compute_b00_null(smoke_run: Path) -> dict[str, Any]:
    first = next((smoke_run / "outputs" / "B00-L1" / "repeat_00").rglob("*.mp4"))
    second = next((smoke_run / "outputs" / "B00-L1" / "repeat_01").rglob("*.mp4"))
    delta = decoded_repeat_difference(first, second)
    a, b = decode_mp4(first), decode_mp4(second)
    future = mean_abs_uint8(a[9:], b[9:])
    history = mean_abs_uint8(a[:9], b[:9])
    return {
        "repeat_00": str(first),
        "repeat_01": str(second),
        "full_mean_abs": float(delta["mean_abs"]),
        "future_mean_abs": float(future),
        "history_mean_abs": float(history),
        "exact": bool(delta["exact"]),
    }


def score_generated_video(
    mp4_path: Path,
    *,
    context: Mapping[str, Any],
    proxy_model: Mapping[str, Any],
    nuscenes_root: Path,
    raft: RAFTFlow,
    proxy_cfg: Mapping[str, Any],
    history_frames: int,
    future_start: int,
    cotracker: CoTracker3IndependentEvaluator | None,
) -> dict[str, Any]:
    frames = decode_mp4(mp4_path)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    source_history = load_source_history(
        nuscenes_root, context["source_frames"], height=height, width=width, count=history_frames
    )
    history_mae = mean_abs_uint8(frames[:history_frames], source_history)
    proxy_frames = frames_to_tensor(frames[future_start:]).to(str(proxy_cfg["raft_device"]))
    observed, confidence = flow_with_confidence_chunked(
        raft, proxy_frames, pair_batch_size=int(proxy_cfg["pair_batch_size"])
    )
    estimate = fit_affine_background_flow(
        observed, confidence, **dict(proxy_cfg["affine_fit"])
    )
    features = affine_proxy_features(estimate.diagnostics, height=height, width=width)
    if not features["valid"]:
        raise RuntimeError(f"proxy 特征无效: {features.get('reason')}")
    prediction = predict_proxy(proxy_model, features["features"])
    errors = action_error(
        prediction,
        request_class=str(context["action_class"]),
        target_displacement_m=float(context["target_displacement_m"]),
        target_lateral_m=float(context["target_lateral_m"]),
    )
    quality = video_quality_metrics(frames_to_tensor(frames))
    tracker: dict[str, Any] = {"enabled": cotracker is not None}
    if cotracker is not None:
        state = cotracker.track(frames_to_tensor(frames).to(cotracker.device))
        tracker.update({
            "valid": bool(state.valid),
            "visible_fraction": float(state.visibility.float().mean()) if state.visibility.numel() else 0.0,
            "track_count": int(state.points.shape[0]) if state.points.ndim >= 2 else 0,
        })
    return {
        "mp4": str(mp4_path),
        "decoded_shape": list(frames.shape),
        "history_vs_source_mae": float(history_mae),
        "proxy_prediction": prediction,
        "action_error": errors,
        "quality": quality,
        "cotracker": tracker,
        "proxy_features": features["features"],
    }


def evaluate_gates(
    scores: Sequence[Mapping[str, Any]],
    *,
    null: Mapping[str, Any],
    gates: Mapping[str, Any],
    stationary_p95: float,
) -> dict[str, Any]:
    moving = [row for row in scores if row["action_class"] != "stationary"]
    stationary = [row for row in scores if row["action_class"] == "stationary"]
    if len(moving) != int(gates["minimum_moving_contexts"]):
        raise RuntimeError(f"moving contexts 数量异常: {len(moving)}")
    improvements = [float(row["paired_improvement"]) for row in moving]
    e_wins = sum(1 for value in improvements if value > 0)
    future_floor = max(float(null["future_mean_abs"]), float(gates["future_effect_min_abs"]))
    history_ceiling = max(
        float(gates["history_effect_null_multiplier"]) * float(null["full_mean_abs"]),
        float(gates["history_effect_null_multiplier"]),
    )
    future_ok = sum(1 for row in moving if float(row["future_effect_mae"]) > future_floor)
    history_ok = sum(1 for row in moving if float(row["history_effect_mae"]) <= history_ceiling)
    quality_ok = []
    for row in moving:
        sharp_e = float(row["E"]["quality"]["sharpness_median"])
        sharp_f = float(row["F"]["quality"]["sharpness_median"])
        flick_e = float(row["E"]["quality"]["temporal_l1_max"])
        flick_f = float(row["F"]["quality"]["temporal_l1_max"])
        ratio_ok = sharp_e >= float(gates["sharpness_ratio_min"]) * max(sharp_f, 1e-6)
        flicker_ok = flick_e <= float(gates["flicker_ratio_max"]) * max(flick_f, 1e-6)
        quality_ok.append(bool(ratio_ok and flicker_ok and row["E"]["quality"]["finite"]))
    stationary_ok = all(
        float(row["E"]["proxy_prediction"]["predicted_displacement_m"]) <= stationary_p95
        and float(row["F"]["proxy_prediction"]["predicted_displacement_m"]) <= stationary_p95
        for row in stationary
    )
    checks = {
        "e_wins": e_wins >= int(gates["minimum_moving_e_wins"]),
        "median_improvement_positive": (
            bool(np.median(improvements) > 0) if gates["require_median_improvement_positive"] else True
        ),
        "future_effect": future_ok >= int(gates["minimum_moving_e_wins"]),
        "history_effect": history_ok >= int(gates["minimum_moving_e_wins"]),
        "quality_safeguard": all(quality_ok),
        "stationary_false_motion": stationary_ok,
    }
    return {
        "moving_count": len(moving),
        "e_wins": e_wins,
        "median_paired_improvement": float(np.median(improvements)),
        "future_ok": future_ok,
        "history_ok": history_ok,
        "quality_ok_count": sum(quality_ok),
        "stationary_ok": stationary_ok,
        "future_floor": future_floor,
        "history_ceiling": history_ceiling,
        "stationary_p95_m": stationary_p95,
        "checks": checks,
        "passed": all(checks.values()),
    }


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
        "task_id": str(cfg.task_id), "run_id": str(cfg.run_id), "status": "running",
        "started_at": _utc_now(), "git": {"motion_proj": motion_git, "resim": resim_git},
        "known_deviation": str(cfg.known_deviation),
    }
    try:
        disk = disk_budget(
            Path(str(cfg.paths.disk_root)),
            int(cfg.disk.estimated_peak_bytes),
            int(cfg.disk.minimum_free_bytes),
        )
        summary["disk_before"] = disk
        if not disk["passed"]:
            raise RuntimeError("C1B-02 磁盘安全门禁失败")
        parent_proxy = Path(str(cfg.paths.parent_proxy_run)).resolve()
        parent_smoke = Path(str(cfg.paths.parent_smoke_run)).resolve()
        if not (parent_proxy / "PASSED").is_file():
            raise FileNotFoundError("parent proxy run 缺少 PASSED")
        proxy_model = _read_json(parent_proxy / "proxy_model.json")
        if str(proxy_model.get("protocol")) != PROXY_PROTOCOL:
            raise RuntimeError(f"proxy protocol 不匹配: {proxy_model.get('protocol')}")
        contexts = _read_jsonl(parent_proxy / "screen_contexts.jsonl")
        if len(contexts) != 10:
            raise RuntimeError(f"期望 10 个 screen contexts，实际 {len(contexts)}")
        null = compute_b00_null(parent_smoke)
        height, width = map(int, cfg.sampling.video_size)
        latent_h, latent_w = map(int, cfg.sampling.latent_size)
        h_interp, w_interp = map(float, cfg.sampling.position_interpolation)
        protocol = {
            "task_id": str(cfg.task_id),
            "parent_proxy_run": str(parent_proxy),
            "parent_smoke_run": str(parent_smoke),
            "proxy_protocol": PROXY_PROTOCOL,
            "sampling": OmegaConf.to_container(cfg.sampling, resolve=True),
            "gates": OmegaConf.to_container(cfg.gates, resolve=True),
            "action_error": "class + 0.5*turn + 0.25*disp + 0.25*lat",
            "null_band": null,
            "matched_sensitivity_executed": bool(cfg.sampling.matched_sensitivity_executed),
        }
        summary.update({
            "protocol_fingerprint": _sha256_json(protocol),
            "parent_proxy_selection": _read_json(parent_proxy / "summary.json").get("selection_fingerprint"),
            "null_band": null,
        })
        atomic_write_json(str(run_dir / "frozen_protocol.json"), protocol)
        atomic_write_json(str(run_dir / "manifest.json"), summary)

        checkpoint = resolve_checkpoint(Path(str(cfg.paths.checkpoint_root)), use_ema=True)
        env = _prepare_environment({"HF_HOME": str(cfg.paths.hf_home), "PYTHONHASHSEED": "0"})
        environment = _environment_snapshot(Path(str(cfg.paths.resim_python)), env)
        summary["checkpoint"] = checkpoint
        summary["environment"] = environment
        raft = RAFTFlow(device=str(cfg.proxy.raft_device))
        cotracker = None
        if bool(cfg.cotracker3.enabled):
            cotracker = CoTracker3IndependentEvaluator(
                {
                    "repository_path": str(cfg.cotracker3.repository_path),
                    "checkpoint_path": str(cfg.cotracker3.checkpoint_path),
                    "checkpoint_url": str(cfg.cotracker3.checkpoint_url),
                    "grid_size": int(cfg.cotracker3.grid_size),
                    "device": str(cfg.cotracker3.device),
                }
            )
            preflight = cotracker.preflight()
            if not preflight["available"]:
                raise RuntimeError("CoTracker3 unavailable: " + "; ".join(preflight["reasons"]))

        score_rows: list[dict[str, Any]] = []
        resim_output_root = resim_root / "outputs"
        stdout_path = run_dir / "combined_stdout.log"
        stdout_path.write_text("", encoding="utf-8")
        for context in contexts:
            context_id = str(context["context_id"])
            context_dir = run_dir / "contexts" / context_id
            context_dir.mkdir(parents=True)
            arm_scores: dict[str, Any] = {}
            for arm in list(cfg.sampling.arms):
                apply_traj = arm == "E"
                arm_dir = context_dir / str(arm)
                arm_dir.mkdir()
                manifest, provenance = build_exact_manifest(
                    Path(str(cfg.paths.source_json)),
                    Path(str(cfg.paths.nuscenes_root)),
                    int(context["clip_index"]),
                    required_rgb_frames=int(cfg.sampling.source_rgb_frames),
                )
                manifest_path = arm_dir / "manifest.json"
                atomic_write_json(str(manifest_path), manifest)
                atomic_write_json(str(arm_dir / "data_provenance.json"), provenance)
                resolved = build_resolved_config(
                    Path(str(cfg.paths.infer_template)),
                    data_manifest_path=manifest_path,
                    checkpoint_root=Path(str(cfg.paths.checkpoint_root)),
                    t5_root=Path(str(cfg.paths.t5_root)),
                    vae_path=Path(str(cfg.paths.vae_path)),
                    seed=int(context["seed"]),
                    source_rgb_frames=int(cfg.sampling.source_rgb_frames),
                    output_rgb_frames=int(cfg.sampling.expected_rgb_frames),
                    height=height,
                    width=width,
                    latent_height=latent_h,
                    latent_width=latent_w,
                    height_interpolation=h_interp,
                    width_interpolation=w_interp,
                    apply_traj=apply_traj,
                )
                config_path = arm_dir / "resolved.yaml"
                OmegaConf.save(resolved, str(config_path))
                command = [
                    str(Path(str(cfg.paths.resim_python)).resolve()),
                    str(Path(str(cfg.paths.entry_script)).resolve()),
                    "--resim-root", str(resim_root),
                    "--config", str(config_path),
                ]
                print(f"C1B-02 generate {context_id}/{arm}", flush=True)
                attempt = _run_attempt(
                    command=command,
                    cwd=motion_root,
                    env=env,
                    resim_output_root=resim_output_root,
                    attempt_dir=arm_dir / "attempt",
                    destination=arm_dir / "output",
                    expected_frames=int(cfg.sampling.expected_rgb_frames),
                    height=height,
                    width=width,
                    stdout_path=stdout_path,
                )
                if int(attempt["exit_code"]) != 0:
                    raise RuntimeError(f"{context_id}/{arm} 采样失败: exit={attempt['exit_code']}")
                mp4 = Path(attempt["video"]["path"])
                scored = score_generated_video(
                    mp4,
                    context=context,
                    proxy_model=proxy_model,
                    nuscenes_root=Path(str(cfg.paths.nuscenes_root)),
                    raft=raft,
                    proxy_cfg=OmegaConf.to_container(cfg.proxy, resolve=True),
                    history_frames=int(cfg.sampling.history_frames),
                    future_start=int(cfg.sampling.future_start_frame),
                    cotracker=cotracker,
                )
                scored["attempt"] = {
                    "exit_code": attempt["exit_code"],
                    "peak_vram_mib": attempt.get("peak_vram_mib"),
                    "duration_seconds": attempt.get("duration_seconds"),
                }
                arm_scores[str(arm)] = scored
                atomic_write_json(str(arm_dir / "score.json"), scored)
                torch.cuda.empty_cache()

            e_frames = decode_mp4(Path(arm_scores["E"]["mp4"]))
            f_frames = decode_mp4(Path(arm_scores["F"]["mp4"]))
            row = {
                "context_id": context_id,
                "action_class": context["action_class"],
                "scene_name": context["scene_name"],
                "clip_index": context["clip_index"],
                "seed": context["seed"],
                "E": arm_scores["E"],
                "F": arm_scores["F"],
                "paired_improvement": float(
                    arm_scores["F"]["action_error"]["action_error"]
                    - arm_scores["E"]["action_error"]["action_error"]
                ),
                "future_effect_mae": mean_abs_uint8(e_frames[9:], f_frames[9:]),
                "history_effect_mae": mean_abs_uint8(e_frames[:9], f_frames[:9]),
            }
            score_rows.append(row)
            _write_jsonl(run_dir / "scores.jsonl", score_rows)
            print(
                f"C1B-02 scored {context_id}: improvement={row['paired_improvement']:.4f} "
                f"future_mae={row['future_effect_mae']:.4f}",
                flush=True,
            )

        gate = evaluate_gates(
            score_rows,
            null=null,
            gates=OmegaConf.to_container(cfg.gates, resolve=True),
            stationary_p95=float(proxy_model["stationary_false_motion_p95_m"]),
        )
        atomic_write_json(str(run_dir / "gate.json"), gate)
        _write_jsonl(run_dir / "scores.jsonl", score_rows)
        summary.update({
            "status": "completed" if gate["passed"] else "rejected",
            "exit_reason": "c1b02_passed" if gate["passed"] else "h1_action_screen_failed",
            "gate": gate,
            "ended_at": _utc_now(),
            "context_count": len(score_rows),
        })
        marker = "PASSED" if gate["passed"] else "REJECTED"
        atomic_write_text(str(run_dir / marker), _sha256_json(summary) + "\n")
    except Exception as error:
        summary.update({
            "status": "failed", "ended_at": _utc_now(),
            "exit_reason": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        atomic_write_text(str(run_dir / "FAILED"), _sha256_json(summary) + "\n")
        atomic_write_json(str(run_dir / "summary.json"), summary)
        raise
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(
        str(run_dir / "RUN_PROVENANCE.md"),
        "\n".join([
            "# C1B-02 Run Provenance", "",
            f"- run_id: `{summary['run_id']}`",
            f"- status: `{summary['status']}`",
            f"- motion_proj_head: `{motion_git['head']}`",
            f"- resim_head: `{resim_git['head']}`",
            f"- protocol_fingerprint: `{summary.get('protocol_fingerprint')}`",
            f"- parent_proxy: `{cfg.paths.parent_proxy_run}`",
            f"- gate_passed: `{summary.get('gate', {}).get('passed')}`",
            f"- matched_sensitivity_executed: `{cfg.sampling.matched_sensitivity_executed}`",
            "",
        ]),
    )
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir, summary = _run(Path(args.config))
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
