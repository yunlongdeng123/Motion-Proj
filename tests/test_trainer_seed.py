import os
import random

import numpy as np
import torch

from motion_proj.train.trainer import seed_everything


def test_seed_everything_reproduces_python_numpy_and_torch(monkeypatch):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        seed_everything(123)
        first = (random.random(), np.random.rand(), torch.rand(3))
        seed_everything(123)
        second = (random.random(), np.random.rand(), torch.rand(3))
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
