"""投影蒸馏训练器（方案第 9 节，阶段 3）。

总目标函数::

    L = L_real + lambda_proj * L_proj + beta * L_anchor

仅在（冻结的）SVD UNet 上训练 LoRA adapter。使用 Accelerate 实现 bf16
混合精度 + 梯度累积。消费的是 *latent* 投影缓存，因此笨重的
auditor/projector 从不进入训练计算图（方案 8.4）。
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from ..backbones import Conditioning, build_backbone
from ..cache.dataset import ProjectionCacheDataset, cache_collate
from ..config import get_paths
from ..losses import anchor_loss, projection_loss, real_loss
from ..utils.logging import get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(self, cfg):
        from accelerate import Accelerator

        self.cfg = cfg
        self.paths = get_paths(cfg)
        tcfg = cfg.train
        self.accelerator = Accelerator(
            mixed_precision="bf16" if cfg.dtype == "bf16" else "no",
            gradient_accumulation_steps=int(tcfg.grad_accum),
        )
        self.device = self.accelerator.device

        self.backbone = build_backbone(cfg.model, load=True, device=str(self.device))
        ds = ProjectionCacheDataset(self.paths.cache_dir)
        self.loader = DataLoader(
            ds,
            batch_size=int(tcfg.micro_batch_size),
            shuffle=True,
            collate_fn=cache_collate,
            num_workers=2,
            drop_last=True,
        )
        params = self.backbone.trainable_parameters()
        if not params:
            raise RuntimeError("no trainable params; enable LoRA in the model config")
        self.optimizer = torch.optim.AdamW(params, lr=float(tcfg.lr), weight_decay=float(tcfg.weight_decay))
        self.optimizer, self.loader = self.accelerator.prepare(self.optimizer, self.loader)
        self.writer = self._make_logger()
        self.step = 0

    def _make_logger(self):
        if self.cfg.train.logger == "tensorboard" and self.accelerator.is_main_process:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(self.paths.log_dir)
        return None

    def _to_device(self, batch):
        y = batch["y"].to(self.device)
        xd = batch["x_dagger"].to(self.device)
        mask = batch["mask"].to(self.device)
        cond = Conditioning(data={k: v.to(self.device) for k, v in batch["context"].items()})
        return y, xd, mask, cond

    def train(self):
        tcfg = self.cfg.train
        max_steps = int(tcfg.max_steps)
        log.info("Start training for %d steps", max_steps)
        self.backbone.unet.train()
        data_iter = _cycle(self.loader)
        while self.step < max_steps:
            batch = next(data_iter)
            with self.accelerator.accumulate(self.backbone.unet):
                with self.accelerator.autocast():
                    y, xd, mask, cond = self._to_device(batch)
                    proj = projection_loss(self.backbone, y, xd, mask, cond, tcfg.tube)
                    loss = float(tcfg.lambda_proj) * proj["loss"]
                    logs = {"loss/proj": float(proj["loss"]), "tube/gate_frac": proj["gate_frac"]}
                    if tcfg.use_real_loss:
                        rl = real_loss(self.backbone, y, cond)
                        loss = loss + rl["loss"]
                        logs["loss/real"] = float(rl["loss"])
                    if float(tcfg.beta_anchor) > 0:
                        al = anchor_loss(self.backbone, proj["z"], proj["sigma"], cond, proj["x0_hat"])
                        loss = loss + float(tcfg.beta_anchor) * al
                        logs["loss/anchor"] = float(al)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.backbone.trainable_parameters(), float(tcfg.max_grad_norm))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.accelerator.sync_gradients:
                self.step += 1
                logs["loss/total"] = float(loss)
                if self.step % int(tcfg.log_every) == 0:
                    self._log(logs)
                if self.step % int(tcfg.ckpt_every) == 0:
                    self.save_ckpt()
        self.save_ckpt(final=True)
        log.info("Training complete.")

    def _log(self, logs: dict):
        msg = " | ".join(f"{k}={v:.4f}" for k, v in logs.items())
        log.info("step %d | %s", self.step, msg)
        if self.writer:
            for k, v in logs.items():
                self.writer.add_scalar(k, v, self.step)

    def save_ckpt(self, final: bool = False):
        if not self.accelerator.is_main_process:
            return
        name = "adapter_final.safetensors" if final else f"adapter_step{self.step}.safetensors"
        path = os.path.join(self.paths.ckpt_dir, name)
        self.backbone.save_adapter(path)


def _cycle(loader):
    while True:
        for b in loader:
            yield b
