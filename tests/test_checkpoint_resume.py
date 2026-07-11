import random

import numpy as np
import torch
from omegaconf import OmegaConf

from motion_proj.runtime.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from motion_proj.runtime.sampler import ResumableRandomSampler


class DummyBackbone:
    def __init__(self):
        self.value = torch.tensor([1.0])

    def save_adapter(self, path):
        torch.save(self.value, path)

    def load_adapter(self, path):
        self.value = torch.load(path, weights_only=True)


def test_checkpoint_roundtrip_restores_step_sampler_optimizer_and_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = ResumableRandomSampler(list(range(8)), seed=5)
    iterator = iter(sampler)
    next(iterator)
    next(iterator)
    cfg = OmegaConf.create({"schema_version": 2, "train": {"max_steps": 10}})
    backbone = DummyBackbone()

    path = save_checkpoint(str(tmp_path), 2, backbone, optimizer, sampler, cfg, "cfg", "cache")
    expected_py = random.random()
    expected_np = np.random.rand()
    expected_torch = torch.rand(1)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_sampler = ResumableRandomSampler(list(range(8)), seed=5)
    step = load_checkpoint(path, backbone, optimizer, resumed_sampler,
                           expected_config="cfg", expected_cache="cache")
    assert step == 2
    assert random.random() == expected_py
    assert np.random.rand() == expected_np
    assert torch.equal(torch.rand(1), expected_torch)
    assert find_latest_checkpoint(str(tmp_path), "cfg", "cache") == path


def test_incomplete_checkpoint_is_ignored(tmp_path):
    bad = tmp_path / "step_000000999"
    bad.mkdir()
    (bad / "metadata.json").write_text('{"global_step":999,"config_fingerprint":"cfg","cache_fingerprint":"cache"}')
    assert find_latest_checkpoint(str(tmp_path), "cfg", "cache") is None
