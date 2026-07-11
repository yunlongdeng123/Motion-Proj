"""能精确保存 epoch、shuffle seed 和当前位置的随机 sampler。"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    def __init__(self, data_source, seed: int = 0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0
        self._order: list[int] | None = None

    def _make_order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self):
        if self._order is None or len(self._order) != len(self.data_source):
            self._order = self._make_order()
        while self.position < len(self._order):
            index = self._order[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self.position = 0
        self._order = None

    def __len__(self) -> int:
        return len(self.data_source) - self.position

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "position": self.position,
                "dataset_size": len(self.data_source), "order": self._order}

    def load_state_dict(self, state: dict) -> None:
        if int(state["dataset_size"]) != len(self.data_source):
            raise ValueError("sampler dataset_size 与当前数据集不匹配")
        if int(state["seed"]) != self.seed:
            raise ValueError("sampler seed 与当前配置不匹配")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        order = state.get("order")
        self._order = list(order) if order is not None else self._make_order()
        if not 0 <= self.position <= len(self._order):
            raise ValueError("sampler position 越界")
