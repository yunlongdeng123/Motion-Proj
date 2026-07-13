"""配置组合、schema 校验和不可变快照。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

CURRENT_SCHEMA_VERSION = 2
EXPERIMENT_TYPES = {
    "base", "real-only", "flow", "synthetic", "replay", "full",
    "full-no-anchor", "full-no-tube",
}
GENERATED_GEOMETRY_MODES = {
    "gt_ego_debug",
    "identity_ego",
    "estimated_background_motion",
    "controlled_ego",
}
LORA_SCOPES = {"temporal_only", "spatial_only", "all_attention"}


class ConfigError(ValueError):
    """配置在启动前不满足可复现实验约束。"""


def _resolve_relative(base_file: str, rel: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(base_file), rel))


def load_config(path: str, overrides: list[str] | None = None, *, validate: bool = True) -> DictConfig:
    """组合 ``defaults_chain``，应用 dotlist 覆盖并返回只读配置。"""
    path = os.path.abspath(path)
    leaf = OmegaConf.load(path)
    merged = OmegaConf.create({})
    for rel in leaf.get("defaults_chain", []) or []:
        parent = load_config(_resolve_relative(path, rel), validate=False)
        merged = OmegaConf.merge(merged, parent)
    leaf = OmegaConf.create({k: v for k, v in leaf.items() if k != "defaults_chain"})
    merged = OmegaConf.merge(merged, leaf)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(merged)
    if validate:
        validate_config(merged)
        OmegaConf.set_readonly(merged, True)
    return merged  # type: ignore[return-value]


def validate_config(cfg: DictConfig) -> None:
    """尽早拒绝旧 schema、未知关键枚举和相互矛盾的形状。"""
    errors: list[str] = []
    version = cfg.get("schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {CURRENT_SCHEMA_VERSION}，实际为 {version!r}")
    for key in ("seed", "device", "dtype", "work_dir", "paths", "data", "model", "train", "cache"):
        if key not in cfg:
            errors.append(f"缺少必需配置: {key}")
    if cfg.get("dtype") not in {"bf16", "fp16", "fp32"}:
        errors.append("dtype 必须是 bf16/fp16/fp32")
    if "cache" in cfg and cfg.cache.get("store") not in {"latent", "rgb"}:
        errors.append("cache.store 必须是 latent 或 rgb")
    if "cache" in cfg and cfg.cache.get("source", "synthetic") not in {"clean", "synthetic", "replay", "replay_v2"}:
        errors.append("cache.source 必须是 clean、synthetic、replay 或 replay_v2")
    if "auditor" in cfg:
        mode = str(cfg.auditor.get("generated_geometry_mode", "gt_ego_debug"))
        if mode not in GENERATED_GEOMETRY_MODES:
            allowed = ", ".join(sorted(GENERATED_GEOMETRY_MODES))
            errors.append(f"auditor.generated_geometry_mode 必须是 {allowed}")
        generated_tracks = cfg.auditor.get("generated_tracks")
        if generated_tracks is not None:
            provider = str(generated_tracks.get("provider", "raft_chain"))
            if provider not in {"raft_chain", "cotracker3"}:
                errors.append("auditor.generated_tracks.provider 必须是 raft_chain 或 cotracker3")
            for name in ("queries_per_stratum", "min_track_length"):
                if generated_tracks.get(name) is not None and int(generated_tracks[name]) <= 0:
                    errors.append(f"auditor.generated_tracks.{name} 必须大于 0")
            for name in ("point_box_size", "min_distance", "fb_alpha", "fb_beta", "min_confidence"):
                if generated_tracks.get(name) is not None and float(generated_tracks[name]) <= 0:
                    errors.append(f"auditor.generated_tracks.{name} 必须大于 0")
    if "data" in cfg and "model" in cfg:
        data_frames = cfg.data.get("num_frames")
        model_frames = cfg.model.get("num_frames")
        if data_frames is not None and model_frames is not None and int(data_frames) != int(model_frames):
            errors.append(f"data.num_frames={data_frames} 与 model.num_frames={model_frames} 不一致")
        sigma_floor = cfg.model.get("sigma_floor", 1.0e-3)
        if (
            not isinstance(sigma_floor, (int, float))
            or not math.isfinite(float(sigma_floor))
            or float(sigma_floor) <= 0
        ):
            errors.append("model.sigma_floor 必须大于 0")
        lora = cfg.model.get("lora")
        if lora is not None:
            scope = str(lora.get("scope", "all_attention"))
            if scope not in LORA_SCOPES:
                errors.append(f"model.lora.scope 必须是 {', '.join(sorted(LORA_SCOPES))}")
            if int(lora.get("rank", 0)) <= 0:
                errors.append("model.lora.rank 必须大于 0")
        if cfg.cache.get("source") == "replay_v2":
            if bool(lora.get("enable", False)):
                errors.append("正式 replay_v2 必须 model.lora.enable=false")
            if str(cfg.auditor.get("generated_geometry_mode")) != "estimated_background_motion":
                errors.append("正式 replay_v2 必须使用 estimated_background_motion")
    if "train" in cfg:
        experiment_type = str(cfg.train.get("experiment_type", "synthetic"))
        if experiment_type not in EXPERIMENT_TYPES:
            errors.append(f"train.experiment_type 未知: {experiment_type}")
        resume = str(cfg.train.get("resume", "auto"))
        if resume not in {"auto", "none"} and not resume:
            errors.append("train.resume 必须是 auto/none/有效 checkpoint 路径")
        for name in ("max_steps", "micro_batch_size", "grad_accum"):
            if cfg.train.get(name) is not None and int(cfg.train[name]) <= 0:
                errors.append(f"train.{name} 必须大于 0")
        if bool(cfg.train.get("deterministic", True)) and int(cfg.train.get("num_workers", 0)) != 0:
            errors.append("确定性精确恢复要求 train.num_workers=0")
        if experiment_type in {"full", "full-no-anchor", "full-no-tube"}:
            ratios = cfg.train.get("cache_mix") or {}
            actual = {str(key): int(value) for key, value in ratios.items()}
            if actual != {"synthetic": 3, "replay": 1}:
                errors.append("full 系列实验的 train.cache_mix 必须固定为 synthetic:3,replay:1")
        if experiment_type == "flow" and float(cfg.train.get("lambda_flow", 0.0)) <= 0:
            errors.append("flow 实验要求 train.lambda_flow > 0")
        parent = cfg.get("parent_run_id")
        if parent is not None and not str(parent).strip():
            errors.append("parent_run_id 不得为空字符串")
    if errors:
        raise ConfigError("配置校验失败:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class ResolvedPaths:
    data_root: str
    cache_dir: str
    ckpt_dir: str
    log_dir: str


def get_paths(cfg: DictConfig) -> ResolvedPaths:
    p = cfg.paths
    paths = ResolvedPaths(p.data_root, p.cache_dir, p.ckpt_dir, p.log_dir)
    for directory in (paths.cache_dir, paths.ckpt_dir, paths.log_dir):
        os.makedirs(directory, exist_ok=True)
    return paths


def to_container(cfg: Any) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def config_fingerprint(cfg: Any, *, resume_compatible: bool = False) -> str:
    """生成稳定指纹；续跑指纹排除只影响运行时长/日志频率的字段。"""
    data = to_container(cfg)
    if resume_compatible:
        data = json.loads(json.dumps(data))
        train = data.get("train", {})
        for key in ("max_steps", "resume", "log_every", "ckpt_every", "sample_every"):
            train.pop(key, None)
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_config_fingerprint(cfg: Any) -> str:
    """cache 只由数据、骨干编码和投影配置决定，不受训练步数影响。"""
    data = to_container(cfg)
    relevant = {
        "schema_version": data.get("schema_version"), "data": data.get("data"),
        "model": data.get("model"), "projector": data.get("projector", {}),
        "store": data.get("cache", {}).get("store"),
        "source": data.get("cache", {}).get("source"),
    }
    raw = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_stage_fingerprint(cfg: Any) -> str:
    """stage 完整性还依赖本次要求覆盖的样本范围与填充策略。"""
    payload = {
        "sample_fingerprint": cache_config_fingerprint(cfg),
        "max_samples": to_container(cfg).get("cache", {}).get("max_samples"),
        "fill_policy": "skip-empty-tracks-until-max",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_resolved_config(cfg: Any, path: str) -> None:
    """原子保存完全解析后的配置。"""
    from .runtime.atomic import atomic_write_text

    atomic_write_text(path, OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    import sys

    loaded = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/train/motionproj_v1.yaml")
    print(OmegaConf.to_yaml(loaded, resolve=True))
