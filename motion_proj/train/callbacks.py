"""训练日志、checkpoint 与 early-stop hooks。"""
from __future__ import annotations

from typing import Protocol


class TrainerCallback(Protocol):
    def on_train_start(self, trainer) -> None: ...
    def on_step_end(self, trainer, logs: dict) -> None: ...
    def on_checkpoint(self, trainer, path: str) -> None: ...
    def on_train_end(self, trainer, reason: str) -> None: ...


class Callback:
    def on_train_start(self, trainer) -> None: pass
    def on_step_end(self, trainer, logs: dict) -> None: pass
    def on_checkpoint(self, trainer, path: str) -> None: pass
    def on_train_end(self, trainer, reason: str) -> None: pass


class EarlyStopCallback(Callback):
    """监视指定指标；patience 为 None 时完全禁用。"""

    def __init__(self, patience: int | None, metric: str = "loss/total", min_delta: float = 0.0):
        self.patience = patience
        self.metric = metric
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad_steps = 0

    def on_step_end(self, trainer, logs: dict) -> None:
        if self.patience is None or self.metric not in logs:
            return
        value = float(logs[self.metric])
        if value < self.best - self.min_delta:
            self.best, self.bad_steps = value, 0
        else:
            self.bad_steps += 1
        if self.bad_steps >= self.patience:
            trainer.should_stop = True
            trainer.stop_reason = f"early_stop:{self.metric}"
