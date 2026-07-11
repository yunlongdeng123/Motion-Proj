"""与具体骨干/日志后端解耦、支持精确断点的 projection-distillation trainer。"""
from __future__ import annotations

import math
import os
import sys

import torch
from torch.utils.data import DataLoader

from ..backbones import Conditioning, build_backbone
from ..cache.dataset import ProjectionCacheDataset, cache_collate
from ..config import cache_config_fingerprint, config_fingerprint, get_paths, save_resolved_config
from ..losses import anchor_loss, projection_loss, real_loss
from ..runtime.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..runtime.experiment import ExperimentRegistry, JsonlMetrics, RunManifest
from ..runtime.atomic import atomic_write_json
from ..runtime.fingerprint import directory_manifest_fingerprint, environment_fingerprint, git_state
from ..runtime.sampler import ResumableRandomSampler
from ..utils.logging import get_logger
from .callbacks import EarlyStopCallback

log = get_logger(__name__)


class Trainer:
    def __init__(self, cfg, callbacks: list | None = None):
        from accelerate import Accelerator

        self.cfg = cfg
        self.paths = get_paths(cfg)
        self.work_dir = str(cfg.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.config_fingerprint = config_fingerprint(cfg, resume_compatible=True)
        self.cache_fingerprint = directory_manifest_fingerprint(self.paths.cache_dir)
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
        dataset = ProjectionCacheDataset(
            self.paths.cache_dir,
            expected_fingerprint=cache_config_fingerprint(cfg),
        )
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
        self.writer = self._make_tensorboard()
        self.step = 0
        self.should_stop = False
        self.stop_reason = "max_steps"
        patience = tcfg.get("early_stop_patience")
        self.callbacks = list(callbacks or []) + [EarlyStopCallback(None if patience is None else int(patience))]
        self._resume(str(tcfg.get("resume", "auto")))

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
        return clean, y, target, mask, cond

    def train(self):
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
                clean, y, target, mask, cond = self._to_device(batch)
                proj = projection_loss(self.backbone, y, target, mask, cond, tcfg.tube)
                loss = float(tcfg.lambda_proj) * proj["loss"]
                logs = {"loss/proj": float(proj["loss"]), "tube/gate_frac": float(proj["gate_frac"])}
                if tcfg.use_real_loss:
                    clean_result = real_loss(self.backbone, clean, cond)
                    loss = loss + clean_result["loss"]
                    logs["loss/real"] = float(clean_result["loss"])
                if float(tcfg.beta_anchor) > 0:
                    anchor = anchor_loss(self.backbone, proj["z"], proj["sigma"], cond, proj["x0_hat"])
                    loss = loss + float(tcfg.beta_anchor) * anchor
                    logs["loss/anchor"] = float(anchor)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.backbone.trainable_parameters(), float(tcfg.max_grad_norm))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        logs["loss/total"] = float(loss)
        return logs

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
