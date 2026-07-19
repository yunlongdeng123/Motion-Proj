"""ReSim C1B-00 单卡采样、形状与确定性门禁。"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from omegaconf import OmegaConf

from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.fingerprint import file_fingerprint, sha256_json


GIB = 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        list(command), cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "head": _run_text(["git", "rev-parse", "HEAD"], cwd=root),
        "status": _run_text(["git", "status", "--short", "--branch"], cwd=root),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        ).hexdigest(),
    }


def _require_clean(snapshot: Mapping[str, Any], label: str) -> None:
    status_lines = str(snapshot["status"]).splitlines()[1:]
    if any(line.strip() for line in status_lines):
        raise RuntimeError(f"{label} 正式运行要求 clean worktree: {snapshot['status']}")


def _frame_paths(clip: Mapping[str, Any]) -> list[str]:
    if "img_seq" in clip:
        return [str(path) for path in clip["img_seq"]]
    return [str(path) for path in clip["img_seq_his"]] + [
        str(path) for path in clip["img_seq_fut"]
    ]


def _nuscenes_mapping(nuscenes_root: Path, filename: str) -> dict[str, str]:
    metadata_root = nuscenes_root / "v1.0-trainval"
    sample_data = {
        str(row["filename"]): row for row in _read_json(metadata_root / "sample_data.json")
    }
    if filename not in sample_data:
        raise KeyError(f"nuScenes sample_data 找不到文件: {filename}")
    samples = {str(row["token"]): row for row in _read_json(metadata_root / "sample.json")}
    scenes = {str(row["token"]): row for row in _read_json(metadata_root / "scene.json")}
    sd = sample_data[filename]
    sample = samples[str(sd["sample_token"])]
    scene = scenes[str(sample["scene_token"])]
    return {
        "filename": filename,
        "sample_data_token": str(sd["token"]),
        "sample_token": str(sample["token"]),
        "scene_token": str(scene["token"]),
        "scene_name": str(scene["name"]),
    }


def build_exact_manifest(
    source_json: Path,
    nuscenes_root: Path,
    clip_index: int,
    required_rgb_frames: int = 33,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """冻结一条 ReSim clip，并建立 filename→sample→scene 溯源。"""
    source = _read_json(source_json)
    clips = source.get("clips", [])
    if not 0 <= int(clip_index) < len(clips):
        raise IndexError(f"clip_index={clip_index} 超出 0..{len(clips) - 1}")
    clip = copy.deepcopy(clips[int(clip_index)])
    paths = _frame_paths(clip)
    if len(paths) < required_rgb_frames:
        raise ValueError(f"clip 只有 {len(paths)} 帧，少于 {required_rgb_frames}")
    missing = [path for path in paths[:required_rgb_frames] if not (nuscenes_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"前 {required_rgb_frames} 帧缺失，示例: {missing[:3]}")
    trajectory = np.asarray(clip.get("traj_fut", []), dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 8 or trajectory.shape[1] != 3:
        raise ValueError(f"traj_fut 形状非法: {trajectory.shape}")

    mapping = _nuscenes_mapping(nuscenes_root, paths[0])
    manifest = {
        "meta": {
            **copy.deepcopy(source.get("meta", {})),
            "data_root": str(nuscenes_root.resolve()),
            "num_clips": 1,
            "source_json": str(source_json.resolve()),
            "source_json_sha256": file_fingerprint(str(source_json)),
            "source_clip_index": int(clip_index),
        },
        "clips": [clip],
    }
    provenance = {
        **mapping,
        "clip_index": int(clip_index),
        "clip_token": str(clip.get("lidar_pc_token", clip.get("token"))),
        "command": str(clip.get("cmd", "")),
        "trajectory": trajectory[:8].tolist(),
        "trajectory_shape": list(trajectory[:8].shape),
        "source_frame_count": len(paths),
        "consumed_rgb_frame_count": int(required_rgb_frames),
        "first_frame": paths[0],
        "last_consumed_frame": paths[required_rgb_frames - 1],
    }
    return manifest, provenance


def resolve_checkpoint(checkpoint_root: Path, *, use_ema: bool) -> dict[str, Any]:
    latest_path = checkpoint_root / "latest"
    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)
    latest = latest_path.read_text(encoding="utf-8").strip()
    if not latest.isdigit() or int(latest) <= 0:
        raise ValueError(f"checkpoint latest 非正整数: {latest!r}")
    tag = f"{latest}-ema" if use_ema else latest
    state_path = checkpoint_root / tag / "mp_rank_00_model_states.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    return {
        "checkpoint_root": str(checkpoint_root.resolve()),
        "latest": int(latest),
        "use_ema": bool(use_ema),
        "resolved_tag": tag,
        "resolved_state": str(state_path.resolve()),
        "resolved_state_bytes": state_path.stat().st_size,
    }


def build_resolved_config(
    template_path: Path,
    *,
    data_manifest_path: Path,
    checkpoint_root: Path,
    t5_root: Path,
    vae_path: Path,
    seed: int,
    source_rgb_frames: int,
    height: int,
    width: int,
    latent_height: int,
    latent_width: int,
    height_interpolation: float,
    width_interpolation: float,
) -> Any:
    cfg = OmegaConf.load(str(template_path))
    cfg.args.load = str(checkpoint_root.resolve())
    cfg.args.use_ema = True
    cfg.args.seed = int(seed)
    cfg.args.sampling_num_frames = 9
    cfg.args.sampling_video_size = [int(height), int(width)]
    cfg.args.valid_data = [str(data_manifest_path.resolve())]
    cfg.args.apply_traj = True
    cfg.args.save_gt = False
    cfg.args.concat_gt_for_demo = False
    cfg.args.save_recon = False
    cfg.args.concat_recon_for_demo = False
    cfg.args.n_prediction_round = 1
    cfg.args.model_parallel_size = 1
    cfg.args.fp16 = True

    cfg.data.target = "data_nus.nuScenesDataset"
    cfg.data.params.video_size = [int(height), int(width)]
    cfg.data.params.max_num_frames = int(source_rgb_frames)
    cfg.data.params.fps = 10
    cfg.data.params.reshape_mode = "center"
    cfg.data.params.p_drop_action_caption = 0.0
    cfg.data.params.p_drop_traj = 0.0
    cfg.data.params.p_mask_out_heading = 0.0
    cfg.data.params.pop("n_subset", None)
    cfg.data.params.pop("ind_subset", None)

    network = cfg.model.network_config.params
    network.latent_height = int(latent_height)
    network.latent_width = int(latent_width)
    pos = network.modules.pos_embed_config.params
    pos.height_interpolation = float(height_interpolation)
    pos.width_interpolation = float(width_interpolation)
    embedders = cfg.model.conditioner_config.params.emb_models
    embedders[0].params.model_dir = str(t5_root.resolve())
    cfg.model.first_stage_config.params.ckpt_path = str(vae_path.resolve())
    return cfg


def _asset_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_fingerprint(str(path)),
            }
        )
    return rows


def disk_budget(path: Path, estimated_peak_bytes: int, minimum_free_bytes: int) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    projected_free = int(usage.free) - int(estimated_peak_bytes)
    return {
        "path": str(path.resolve()),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "minimum_free_bytes": int(minimum_free_bytes),
        "projected_free_bytes": projected_free,
        "passed": projected_free >= int(minimum_free_bytes),
    }


def new_output_directories(output_root: Path, before: set[str]) -> list[Path]:
    after = {item.name for item in output_root.iterdir() if item.is_dir()} if output_root.is_dir() else set()
    return [output_root / name for name in sorted(after - before)]


def validate_video(path: Path, *, expected_frames: int, height: int, width: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"视频无法解码: {path}")
    array = np.stack(frames)
    checks = {
        "frame_count": int(array.shape[0]) == int(expected_frames),
        "height": int(array.shape[1]) == int(height),
        "width": int(array.shape[2]) == int(width),
        "channels": int(array.shape[3]) == 3,
        "finite": bool(np.isfinite(array).all()),
    }
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "mp4_sha256": file_fingerprint(str(path)),
        "decoded_shape": list(array.shape),
        "decoded_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "pixel_min": int(array.min()),
        "pixel_max": int(array.max()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def decoded_repeat_difference(first: Path, second: Path) -> dict[str, Any]:
    def decode(path: Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return np.stack(frames)

    a, b = decode(first), decode(second)
    if a.shape != b.shape:
        return {"same_shape": False, "first_shape": list(a.shape), "second_shape": list(b.shape), "exact": False}
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "same_shape": True,
        "first_shape": list(a.shape),
        "second_shape": list(b.shape),
        "exact": bool(np.array_equal(a, b)),
        "max_abs": int(delta.max()),
        "mean_abs": float(delta.mean()),
        "nonzero_fraction": float(np.count_nonzero(delta) / delta.size),
    }


def is_cuda_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered


class GpuMemoryMonitor:
    def __init__(self, output_path: Path, interval_seconds: float = 1.0):
        self.output_path = output_path
        self.interval_seconds = float(interval_seconds)
        self.stop_event = threading.Event()
        self.rows: list[tuple[str, int, int]] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                raw = _run_text(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ]
                )
                used, total = [int(value.strip()) for value in raw.splitlines()[0].split(",")]
                self.rows.append((_utc_now(), used, total))
            except Exception:
                pass
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "GpuMemoryMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "memory_used_mib", "memory_total_mib"])
            writer.writerows(self.rows)

    @property
    def peak_used_mib(self) -> int | None:
        return max((row[1] for row in self.rows), default=None)


def _prepare_environment(base: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in base.items()})
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "OMP_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": env.get("PYTHONHASHSEED", "0"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _environment_snapshot(resim_python: Path, env: Mapping[str, str]) -> dict[str, Any]:
    script = (
        "import json, platform, torch; "
        "print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, "
        "'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    python_info = json.loads(
        subprocess.check_output(
            [str(resim_python), "-c", script], env=dict(env), text=True, stderr=subprocess.STDOUT
        )
    )
    driver = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    name, driver_version, memory_total = [value.strip() for value in driver.splitlines()[0].split(",")]
    return {
        **python_info,
        "nvidia_smi_gpu": name,
        "driver_version": driver_version,
        "memory_total_mib": int(memory_total),
    }


def _run_attempt(
    *,
    command: list[str],
    cwd: Path,
    env: Mapping[str, str],
    resim_output_root: Path,
    attempt_dir: Path,
    destination: Path,
    expected_frames: int,
    height: int,
    width: int,
    stdout_path: Path,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    before = {item.name for item in resim_output_root.iterdir() if item.is_dir()} if resim_output_root.is_dir() else set()
    started = time.monotonic()
    attempt_log_path = attempt_dir / "stdout.log"
    with attempt_log_path.open("w", encoding="utf-8") as stdout:
        with GpuMemoryMonitor(attempt_dir / "gpu_memory.csv") as monitor:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                env=dict(env),
                stdout=stdout,
                stderr=subprocess.STDOUT,
                check=False,
            )
        peak = monitor.peak_used_mib
    duration = time.monotonic() - started
    new_dirs = new_output_directories(resim_output_root, before)
    if len(new_dirs) == 1:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(new_dirs[0]), str(destination))
    elif new_dirs:
        partial_root = attempt_dir / "ambiguous_outputs"
        partial_root.mkdir()
        for path in new_dirs:
            shutil.move(str(path), str(partial_root / path.name))

    log_text = attempt_log_path.read_text(encoding="utf-8", errors="replace")
    with stdout_path.open("a", encoding="utf-8") as combined:
        combined.write(f"\n===== {' '.join(command)} =====\n")
        combined.write(log_text)
    result: dict[str, Any] = {
        "command": command,
        "exit_code": int(proc.returncode),
        "duration_seconds": float(duration),
        "peak_vram_mib": peak,
        "new_output_count": len(new_dirs),
        "new_output_names": [path.name for path in new_dirs],
        "cuda_oom": is_cuda_oom(log_text),
        "output_dir": str(destination) if destination.is_dir() else None,
    }
    if proc.returncode == 0:
        if len(new_dirs) != 1:
            raise RuntimeError(f"ReSim 退出 0，但新增输出目录数量为 {len(new_dirs)}")
        mp4s = sorted(destination.rglob("*.mp4"))
        if len(mp4s) != 1:
            raise RuntimeError(f"期望 1 个 MP4，实际 {len(mp4s)}: {mp4s}")
        video = validate_video(mp4s[0], expected_frames=expected_frames, height=height, width=width)
        if not video["passed"]:
            raise RuntimeError(f"视频 shape/finite 检查失败: {video}")
        result["video"] = video
    atomic_write_json(str(attempt_dir / "result.json"), result)
    return result


def _provenance_markdown(summary: Mapping[str, Any]) -> str:
    attempts = list(summary.get("attempts", []))
    peak_vram = max(
        (int(row["peak_vram_mib"]) for row in attempts if row.get("peak_vram_mib") is not None),
        default=None,
    )
    elapsed = sum(float(row.get("duration_seconds", 0.0)) for row in attempts)
    environment = dict(summary.get("environment", {}))
    return "\n".join(
        [
            "# C1B-00 Run Provenance",
            "",
            f"- task_id: `{summary['task_id']}`",
            f"- run_id: `{summary['run_id']}`",
            f"- started_at: `{summary['started_at']}`",
            f"- ended_at: `{summary.get('ended_at')}`",
            f"- status: `{summary.get('status')}`",
            f"- motion_proj_head: `{summary['git']['motion_proj']['head']}`",
            f"- resim_head: `{summary['git']['resim']['head']}`",
            f"- config_fingerprint: `{summary['config_fingerprint']}`",
            f"- protocol_fingerprint: `{summary['protocol_fingerprint']}`",
            f"- data_fingerprint: `{summary['data_fingerprint']}`",
            f"- scene: `{summary['data_provenance']['scene_name']}`",
            f"- sample_token: `{summary['data_provenance']['sample_token']}`",
            f"- seed: `{summary['seed']}`",
            f"- resolved_checkpoint: `{summary['checkpoint']['resolved_state']}`",
            f"- python: `{environment.get('python')}`",
            f"- torch/cuda: `{environment.get('torch')} / {environment.get('torch_cuda')}`",
            f"- gpu/driver: `{environment.get('gpu')} / {environment.get('driver_version')}`",
            f"- disk_free_before_bytes: `{summary['disk_before']['free_bytes']}`",
            f"- estimated_peak_bytes: `{summary['disk_before']['estimated_peak_bytes']}`",
            f"- selected_level: `{summary.get('selected_level')}`",
            f"- peak_vram_mib: `{peak_vram}`",
            f"- sampling_elapsed_seconds: `{elapsed}`",
            f"- attempt_exit_codes: `{[row.get('exit_code') for row in attempts]}`",
            f"- deterministic_exact: `{summary.get('determinism', {}).get('exact')}`",
            f"- known_deviation: `{summary.get('known_deviation', 'none')}`",
            "",
        ]
    )


def run(config_path: Path, *, run_id_override: str | None = None, prepare_only: bool = False) -> tuple[Path, dict[str, Any]]:
    cfg = OmegaConf.load(str(config_path))
    task_id = str(cfg.task_id)
    run_id = str(run_id_override or cfg.run_id)
    paths = cfg.paths
    motion_root = Path(str(paths.motion_proj_root)).resolve()
    resim_root = Path(str(paths.resim_root)).resolve()
    run_dir = Path(str(paths.output_root)).resolve() / task_id / run_id
    if run_dir.exists():
        raise FileExistsError(f"正式 run ID 不可复用: {run_dir}")

    motion_git = _git_snapshot(motion_root)
    resim_git = _git_snapshot(resim_root)
    _require_clean(motion_git, "motion_proj")
    _require_clean(resim_git, "ReSim")

    run_dir.mkdir(parents=True)
    started_at = _utc_now()
    summary: dict[str, Any] = {
        "task_id": task_id,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "seed": int(cfg.seed),
        "git": {"motion_proj": motion_git, "resim": resim_git},
        "known_deviation": str(cfg.known_deviation),
    }
    atomic_write_json(str(run_dir / "manifest.json"), summary)

    try:
        source_json = Path(str(paths.source_json)).resolve()
        nuscenes_root = Path(str(paths.nuscenes_root)).resolve()
        checkpoint_root = Path(str(paths.checkpoint_root)).resolve()
        t5_root = Path(str(paths.t5_root)).resolve()
        vae_path = Path(str(paths.vae_path)).resolve()
        infer_template = Path(str(paths.infer_template)).resolve()
        resim_python = Path(str(paths.resim_python)).resolve()
        entry_script = Path(str(paths.entry_script)).resolve()
        required_paths = [source_json, vae_path, infer_template, resim_python, entry_script]
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        if not t5_root.is_dir():
            raise FileNotFoundError(t5_root)

        data_manifest, data_provenance = build_exact_manifest(
            source_json,
            nuscenes_root,
            int(cfg.source.clip_index),
            int(cfg.sampling.source_rgb_frames),
        )
        data_manifest_path = run_dir / "data_manifest.json"
        atomic_write_json(str(data_manifest_path), data_manifest)
        atomic_write_json(str(run_dir / "data_provenance.json"), data_provenance)
        summary["data_fingerprint"] = sha256_json(data_manifest)
        summary["data_provenance"] = data_provenance

        checkpoint = resolve_checkpoint(checkpoint_root, use_ema=True)
        summary["checkpoint"] = checkpoint
        asset_paths = [
            Path(checkpoint["resolved_state"]),
            vae_path,
            t5_root / "model-00001-of-00002.safetensors",
            t5_root / "model-00002-of-00002.safetensors",
            t5_root / "model.safetensors.index.json",
            source_json,
        ]
        assets = _asset_rows(asset_paths)
        summary["assets_fingerprint"] = sha256_json(assets)
        atomic_write_text(
            str(run_dir / "asset_manifest.sha256"),
            "".join(f"{row['sha256']}  {row['path']}\n" for row in assets),
        )
        atomic_write_json(str(run_dir / "asset_manifest.json"), assets)

        estimated_peak = int(cfg.disk.estimated_peak_bytes)
        minimum_free = int(cfg.disk.minimum_free_bytes)
        disk_before = disk_budget(Path(str(paths.disk_root)), estimated_peak, minimum_free)
        summary["disk_before"] = disk_before
        if not disk_before["passed"]:
            raise RuntimeError(f"磁盘峰值门禁失败: {disk_before}")

        protocol = OmegaConf.to_container(cfg, resolve=True)
        protocol_fingerprint = sha256_json(protocol)
        summary["protocol_fingerprint"] = protocol_fingerprint
        atomic_write_json(str(run_dir / "frozen_protocol.json"), protocol)
        atomic_write_text(str(run_dir / "frozen_protocol.sha256"), protocol_fingerprint + "\n")
        summary["config_fingerprint"] = file_fingerprint(str(config_path))

        git_text = json.dumps(summary["git"], ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(str(run_dir / "git_state.txt"), git_text)
        atomic_write_json(str(run_dir / "seeds.json"), {"sampling_seed": int(cfg.seed), "repeat_count": int(cfg.sampling.repeats)})
        env_freeze = _run_text([str(resim_python), "-m", "pip", "freeze"])
        atomic_write_text(str(run_dir / "env_freeze.txt"), env_freeze + "\n")

        levels = list(cfg.sampling.levels)
        if len(levels) != 2 or str(levels[0].name) != "B00-L0" or str(levels[1].name) != "B00-L1":
            raise ValueError("只允许固定 B00-L0→B00-L1 两级阶梯")

        resolved_configs: dict[str, Path] = {}
        for level in levels:
            size = [int(v) for v in level.video_size]
            latent = [int(v) for v in level.latent_size]
            interpolation = [float(v) for v in level.position_interpolation]
            level_cfg = build_resolved_config(
                infer_template,
                data_manifest_path=data_manifest_path,
                checkpoint_root=checkpoint_root,
                t5_root=t5_root,
                vae_path=vae_path,
                seed=int(cfg.seed),
                source_rgb_frames=int(cfg.sampling.source_rgb_frames),
                height=size[0],
                width=size[1],
                latent_height=latent[0],
                latent_width=latent[1],
                height_interpolation=interpolation[0],
                width_interpolation=interpolation[1],
            )
            resolved_path = run_dir / f"resolved_config.{level.name}.yaml"
            OmegaConf.save(level_cfg, str(resolved_path), resolve=True)
            resolved_configs[str(level.name)] = resolved_path
        shutil.copy2(resolved_configs["B00-L0"], run_dir / "resolved_config.yaml")

        env = _prepare_environment({"HF_HOME": str(paths.hf_home)})
        summary["environment"] = _environment_snapshot(resim_python, env)
        preflight_command = [
            str(resim_python),
            str(entry_script),
            "--resim-root",
            str(resim_root),
            "--config",
            str(resolved_configs["B00-L0"]),
            "--preflight-output",
            str(run_dir / "dataset_preflight.json"),
        ]
        preflight = subprocess.run(
            preflight_command,
            cwd=str(motion_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        atomic_write_text(str(run_dir / "preflight.log"), preflight.stdout)
        if preflight.returncode != 0:
            raise RuntimeError(f"dataset/shape preflight 失败，exit={preflight.returncode}")

        commands = [" ".join(preflight_command)]
        atomic_write_text(str(run_dir / "command.sh"), "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
        if prepare_only:
            summary.update({"status": "prepared", "ended_at": _utc_now()})
            atomic_write_json(str(run_dir / "summary.json"), summary)
            atomic_write_text(str(run_dir / "PREPARED"), sha256_json(summary) + "\n")
            atomic_write_text(str(run_dir / "RUN_PROVENANCE.md"), _provenance_markdown(summary))
            return run_dir, summary

        attempts: list[dict[str, Any]] = []
        selected_level: str | None = None
        stdout_path = run_dir / "stdout.log"
        resim_output_root = resim_root / "outputs"
        for level_index, level in enumerate(levels):
            level_name = str(level.name)
            size = [int(v) for v in level.video_size]
            level_results: list[dict[str, Any]] = []
            for repeat in range(int(cfg.sampling.repeats)):
                current_budget = disk_budget(Path(str(paths.disk_root)), estimated_peak, minimum_free)
                if not current_budget["passed"]:
                    raise RuntimeError(f"运行前磁盘峰值门禁失败: {current_budget}")
                repeat_cfg = run_dir / f"resim_c1b00_{level_name}_repeat{repeat:02d}.yaml"
                shutil.copy2(resolved_configs[level_name], repeat_cfg)
                command = [
                    str(resim_python),
                    str(entry_script),
                    "--resim-root",
                    str(resim_root),
                    "--config",
                    str(repeat_cfg),
                ]
                commands.append(" ".join(command))
                result = _run_attempt(
                    command=command,
                    cwd=motion_root,
                    env=env,
                    resim_output_root=resim_output_root,
                    attempt_dir=run_dir / "attempts" / level_name / f"repeat_{repeat:02d}",
                    destination=run_dir / "outputs" / level_name / f"repeat_{repeat:02d}",
                    expected_frames=int(cfg.sampling.expected_rgb_frames),
                    height=size[0],
                    width=size[1],
                    stdout_path=stdout_path,
                )
                result.update({"level": level_name, "repeat": repeat})
                attempts.append(result)
                level_results.append(result)
                if result["exit_code"] != 0:
                    if level_index == 0 and repeat == 0 and result["cuda_oom"]:
                        break
                    raise RuntimeError(f"{level_name}/repeat-{repeat} 采样失败: {result}")
            if level_results and level_results[0]["exit_code"] != 0:
                if level_index == 0 and level_results[0]["cuda_oom"]:
                    continue
                raise RuntimeError(f"{level_name} 失败")
            if len(level_results) != int(cfg.sampling.repeats):
                raise RuntimeError(f"{level_name} 未完成固定重复次数")
            selected_level = level_name
            break

        atomic_write_text(str(run_dir / "command.sh"), "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
        if selected_level is None:
            summary["attempts"] = attempts
            raise RuntimeError("B00-L0 与 B00-L1 均未通过")

        shutil.copy2(resolved_configs[selected_level], run_dir / "resolved_config.yaml")

        successful = [row for row in attempts if row["level"] == selected_level and row["exit_code"] == 0]
        first_mp4 = Path(successful[0]["video"]["path"])
        second_mp4 = Path(successful[1]["video"]["path"])
        determinism = decoded_repeat_difference(first_mp4, second_mp4)
        summary.update(
            {
                "attempts": attempts,
                "selected_level": selected_level,
                "determinism": determinism,
                "disk_after": disk_budget(Path(str(paths.disk_root)), 0, minimum_free),
            }
        )
        if not determinism["exact"]:
            summary.update({"status": "blocked", "ended_at": _utc_now(), "exit_reason": "repeat_not_bit_exact"})
            atomic_write_json(str(run_dir / "metrics.json"), summary)
            atomic_write_json(str(run_dir / "summary.json"), summary)
            atomic_write_text(str(run_dir / "BLOCKED"), sha256_json(summary) + "\n")
            atomic_write_text(str(run_dir / "RUN_PROVENANCE.md"), _provenance_markdown(summary))
            return run_dir, summary

        summary.update({"status": "completed", "ended_at": _utc_now(), "exit_reason": "c1b00_passed"})
        atomic_write_json(str(run_dir / "metrics.json"), summary)
        atomic_write_json(str(run_dir / "summary.json"), summary)
        atomic_write_text(str(run_dir / "COMPLETE"), sha256_json(summary) + "\n")
        atomic_write_text(str(run_dir / "RUN_PROVENANCE.md"), _provenance_markdown(summary))
        return run_dir, summary
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "exit_reason": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(str(run_dir / "summary.json"), summary)
        atomic_write_text(str(run_dir / "FAILED"), sha256_json(summary) + "\n")
        if "config_fingerprint" in summary:
            atomic_write_text(str(run_dir / "RUN_PROVENANCE.md"), _provenance_markdown(summary))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run_dir, summary = run(Path(args.config), run_id_override=args.run_id, prepare_only=args.prepare_only)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
