"""与具体骨干/日志后端解耦、支持精确断点的 projection-distillation trainer。"""
from __future__ import annotations

import math
import os
import sys

import torch
from torch.utils.data import DataLoader

from ..backbones import Conditioning, build_backbone
from ..cache.dataset import MixedProjectionCacheDataset, ProjectionCacheDataset, cache_collate
from ..config import cache_config_fingerprint, config_fingerprint, get_paths, save_resolved_config
from ..losses import anchor_loss, flow_warp_charbonnier_loss, projection_loss, real_loss
from ..runtime.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..runtime.experiment import ExperimentRegistry, JsonlMetrics, RunManifest
from ..runtime.atomic import atomic_write_json
from ..runtime.fingerprint import directory_manifest_fingerprint, environment_fingerprint, git_state, sha256_json
from ..runtime.sampler import ResumableRandomSampler
from ..utils.logging import get_logger
from .callbacks import EarlyStopCallback

log = get_logger(__name__)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """在模型/LoRA 构造前统一播种，并约束 CUDA 算法选择。"""
    import random

    import numpy as np

    if deterministic:
        # 必须在首次 cuBLAS 调用前设置，否则确定性算法会拒绝矩阵乘法。
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, cfg, callbacks: list | None = None):
        from accelerate import Accelerator

        self.cfg = cfg
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.get("deterministic", True)))
        self.paths = get_paths(cfg)
        self.work_dir = str(cfg.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.experiment_type = str(cfg.train.get("experiment_type", "synthetic"))
        self.config_fingerprint = config_fingerprint(cfg, resume_compatible=True)
        self.cache_specs = self._cache_specs()
        self.cache_fingerprint = (
            "not-applicable:frozen-base" if self.experiment_type == "base" else
            sha256_json({name: directory_manifest_fingerprint(spec["path"])
                         for name, spec in self.cache_specs.items()})
        )
        self.run_id = str(cfg.get("run_id") or os.path.basename(os.path.normpath(self.work_dir)))
        self.metrics = JsonlMetrics(os.path.join(self.work_dir, "metrics.jsonl"))
        self.registry = ExperimentRegistry(os.path.join(os.path.dirname(self.work_dir), "experiments.sqlite3"))
        self._init_manifest()
        save_resolved_config(cfg, os.path.join(self.work_dir, "resolved.yaml"))

        tcfg = cfg.train
        self.accelerator = Accelerator(
            mixed_precision="bf16" if cfg.dtype == "bf16" else ("fp16" if cfg.dtype == "fp16" else "no"),
            gradient_accumulation_steps=int(tcfg.grad_accum),
        )
        self.device = self.accelerator.device
        self.backbone = build_backbone(cfg.model, load=True, device=str(self.device))
        self.writer = self._make_tensorboard()
        self.step = 0
        self.should_stop = False
        self.stop_reason = "max_steps"
        patience = tcfg.get("early_stop_patience")
        self.callbacks = list(callbacks or []) + [EarlyStopCallback(None if patience is None else int(patience))]
        if self.experiment_type == "base":
            self.backbone.training_module().requires_grad_(False)
            self.sampler = None
            self.loader = None
            self.optimizer = None
            return
        dataset = self._build_dataset()
        self.sampler = ResumableRandomSampler(dataset, seed=int(cfg.seed))
        self.loader = DataLoader(
            dataset, batch_size=int(tcfg.micro_batch_size), sampler=self.sampler,
            collate_fn=cache_collate, num_workers=int(tcfg.get("num_workers", 0)), drop_last=True,
        )
        params = self.backbone.trainable_parameters()
        if not params:
            raise RuntimeError("no trainable params; enable LoRA in the model config")
        self.optimizer = torch.optim.AdamW(params, lr=float(tcfg.lr), weight_decay=float(tcfg.weight_decay))
        self.optimizer, self.loader = self.accelerator.prepare(self.optimizer, self.loader)
        self._resume(str(tcfg.get("resume", "auto")))

    def _cache_specs(self) -> dict[str, dict]:
        raw = self.cfg.train.get("cache_sources")
        if raw:
            specs = {}
            for name, value in raw.items():
                if isinstance(value, str):
                    specs[str(name)] = {"path": value, "fingerprint": None}
                else:
                    specs[str(name)] = {
                        "path": str(value["path"]),
                        "fingerprint": value.get("fingerprint"),
                    }
            return specs
        source = self.experiment_type if self.experiment_type in {"synthetic", "replay"} else "default"
        return {source: {"path": self.paths.cache_dir,
                         "fingerprint": cache_config_fingerprint(self.cfg)}}

    def _build_dataset(self):
        datasets = {
            name: ProjectionCacheDataset(spec["path"], expected_fingerprint=spec.get("fingerprint"))
            for name, spec in self.cache_specs.items()
        }
        if self.experiment_type in {"full", "full-no-anchor", "full-no-tube"}:
            required = {"synthetic", "replay"}
            if set(datasets) != required:
                raise ValueError(f"full 系列 cache_sources 必须恰好为 {sorted(required)}")
            ratios = {str(key): int(value) for key, value in self.cfg.train.cache_mix.items()}
            return MixedProjectionCacheDataset(
                datasets, ratios, epoch_size=self.cfg.train.get("cache_epoch_size")
            )
        preferred = self.experiment_type if self.experiment_type in datasets else next(iter(datasets))
        return datasets[preferred]

    def _init_manifest(self) -> None:
        manifest = RunManifest(
            run_id=self.run_id, command=sys.argv, config_fingerprint=self.config_fingerprint,
            cache_fingerprint=self.cache_fingerprint, seed=int(self.cfg.seed), git=git_state(),
            environment=environment_fingerprint(), data_split=str(self.cfg.data.get("split", "unspecified")),
            parent_run_id=self.cfg.get("parent_run_id"),
        )
        path = os.path.join(self.work_dir, "manifest.json")
        if os.path.isfile(path):
            import json

            with open(path, encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous.get("config_fingerprint") != self.config_fingerprint:
                raise RuntimeError(f"run 目录不可复用且 fingerprint 不匹配: {self.work_dir}")
            if previous.get("status") == "completed":
                raise RuntimeError(f"已完成 run 目录禁止复用: {self.work_dir}")
            previous.update({"status": "running", "resumed_at": manifest.started_at,
                             "command": manifest.command, "exit_reason": None, "ended_at": None})
            atomic_write_json(path, previous)
        else:
            manifest.save(path)
        try:
            self.registry.register(self.run_id, "running", self.config_fingerprint, self.work_dir,
                                   manifest.parent_run_id)
        except Exception:
            self.registry.update(self.run_id, "running")

    def _finish_manifest(self, status: str, reason: str) -> None:
        import json
        from datetime import datetime, timezone

        path = os.path.join(self.work_dir, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        value.update({"status": status, "exit_reason": reason,
                      "ended_at": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(path, value)

    def _make_tensorboard(self):
        if self.cfg.train.logger == "tensorboard" and self.accelerator.is_main_process:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(self.paths.log_dir)
        return None

    def _resume(self, resume: str) -> None:
        if resume == "none":
            return
        path = (find_latest_checkpoint(self.paths.ckpt_dir, self.config_fingerprint, self.cache_fingerprint)
                if resume == "auto" else resume)
        if path:
            self.step = load_checkpoint(
                path, self.backbone, self.optimizer, self.sampler,
                expected_config=self.config_fingerprint, expected_cache=self.cache_fingerprint,
            )
            log.info("Resumed exact checkpoint %s at global step %d", path, self.step)
        elif resume != "auto":
            raise FileNotFoundError(resume)

    def _to_device(self, batch):
        clean = batch["clean"].to(self.device)
        y = batch["y"].to(self.device)
        target = batch["x_dagger"].to(self.device)
        mask = batch["mask"].to(self.device)
        cond = Conditioning(data={key: value.to(self.device) for key, value in batch["context"].items()})
        flow = batch.get("latent_flow")
        confidence = batch.get("flow_confidence")
        return (clean, y, target, mask, cond,
                flow.to(self.device) if flow is not None else None,
                confidence.to(self.device) if confidence is not None else None)

    def train(self):
        if self.experiment_type == "base":
            summary = {"experiment_type": "base", "frozen": True,
                       "checkpoint": "pretrained", "trained_steps": 0}
            atomic_write_json(os.path.join(self.work_dir, "summary.json"), summary)
            self.registry.update(self.run_id, "completed", exit_reason="frozen_base", summary=summary)
            self._finish_manifest("completed", "frozen_base")
            if self.writer:
                self.writer.close()
            return
        max_steps = int(self.cfg.train.max_steps)
        if self.step >= max_steps:
            log.info("Checkpoint 已达到目标总步数 %d，无需继续训练", max_steps)
            self.registry.update(self.run_id, "completed", exit_reason="already_at_target")
            self._finish_manifest("completed", "already_at_target")
            return
        self.backbone.set_train_mode(True)
        for callback in self.callbacks:
            callback.on_train_start(self)
        data_iter = _cycle(self.loader)
        try:
            while self.step < max_steps and not self.should_stop:
                logs = self._train_step(next(data_iter))
                if not all(math.isfinite(float(value)) for value in logs.values()):
                    self.stop_reason = "nan"
                    raise FloatingPointError(f"non-finite metrics at step {self.step}: {logs}")
                if self.accelerator.sync_gradients:
                    self.step += 1
                    logs["step"] = self.step
                    for callback in self.callbacks:
                        callback.on_step_end(self, logs)
                    if self.step % int(self.cfg.train.log_every) == 0:
                        self._log(logs)
                    if self.step % int(self.cfg.train.ckpt_every) == 0:
                        self.save_ckpt()
            self.save_ckpt(final=True)
            self.registry.update(self.run_id, "completed", exit_reason=self.stop_reason)
            self._finish_manifest("completed", self.stop_reason)
        except Exception as exc:
            self.registry.update(self.run_id, "failed", exit_reason=repr(exc))
            self._finish_manifest("failed", repr(exc))
            raise
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self, self.stop_reason)
            if self.writer:
                self.writer.close()

    def _train_step(self, batch) -> dict:
        tcfg = self.cfg.train
        with self.accelerator.accumulate(self.backbone.training_module()):
            with self.accelerator.autocast():
                clean, y, target, mask, cond, flow, confidence = self._to_device(batch)
                proj = None
                logs = {}
                if self.experiment_type in {"real-only", "flow"}:
                    clean_result = real_loss(self.backbone, clean, cond)
                    loss = clean_result["loss"]
                    logs["loss/real"] = float(clean_result["loss"])
                    if self.experiment_type == "flow":
                        if flow is None or confidence is None:
                            raise RuntimeError("flow 实验要求 cache 提供 latent_flow 与 flow_confidence")
                        flow_value = flow_warp_charbonnier_loss(
                            clean_result["x0_hat"], flow, confidence
                        )
                        loss = loss + float(tcfg.lambda_flow) * flow_value
                        logs["loss/flow"] = float(flow_value)
                else:
                    proj = projection_loss(
                        self.backbone, y, target, mask, cond, tcfg.tube,
                        use_tube=self.experiment_type != "full-no-tube",
                    )
                    loss = float(tcfg.lambda_proj) * proj["loss"]
                    logs.update({"loss/proj": float(proj["loss"]),
                                 "tube/gate_frac": float(proj["gate_frac"])})
                    if bool(tcfg.use_real_loss):
                        clean_result = real_loss(self.backbone, clean, cond)
                        loss = loss + clean_result["loss"]
                        logs["loss/real"] = float(clean_result["loss"])
                beta_anchor = (0.0 if self.experiment_type == "full-no-anchor"
                               else float(tcfg.beta_anchor))
                if proj is not None and beta_anchor > 0:
                    anchor = anchor_loss(self.backbone, proj["z"], proj["sigma"], cond, proj["x0_hat"])
                    loss = loss + beta_anchor * anchor
                    logs["loss/anchor"] = float(anchor)
            self._set_learning_rate()
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.backbone.trainable_parameters(), float(tcfg.max_grad_norm))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        logs["loss/total"] = float(loss)
        return logs

    def _set_learning_rate(self) -> None:
        warmup = int(self.cfg.train.get("warmup_steps", 0))
        scale = min(1.0, (self.step + 1) / max(warmup, 1)) if warmup > 0 else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.cfg.train.lr) * scale

    def _log(self, logs: dict) -> None:
        values = {key: value for key, value in logs.items() if key != "step"}
        log.info("step %d | %s", self.step, " | ".join(f"{key}={value:.4f}" for key, value in values.items()))
        self.metrics.append(self.step, values)
        if self.writer:
            for key, value in values.items():
                self.writer.add_scalar(key, value, self.step)

    def save_ckpt(self, final: bool = False) -> str:
        if not self.accelerator.is_main_process:
            return ""
        path = save_checkpoint(
            self.paths.ckpt_dir, self.step, self.backbone,
            self.optimizer, self.sampler, self.cfg, self.config_fingerprint, self.cache_fingerprint, final=final,
        )
        for callback in self.callbacks:
            callback.on_checkpoint(self, path)
        return path


def _cycle(loader):
    while True:
        yield from loader
