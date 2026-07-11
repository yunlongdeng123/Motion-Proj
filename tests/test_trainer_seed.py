import random

import numpy as np
import torch

from motion_proj.train.trainer import seed_everything


def test_seed_everything_reproduces_python_numpy_and_torch():
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
