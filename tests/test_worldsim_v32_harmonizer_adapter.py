from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from motion_proj.worldsim_v32.harmonizer_adapter import (
    prepare_image,
    register_tex_ts_rmsnorm_fallback,
    restore_image,
    rmsnorm_reference,
)


def test_rmsnorm_reference_matches_manual_formula() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    weight = torch.tensor([0.5, 1.0, 1.5])
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight
    actual = rmsnorm_reference(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected)


def test_registered_operator_matches_reference() -> None:
    register_tex_ts_rmsnorm_fallback()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 7.0
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4])
    actual = torch.ops.tex_ts.rmsnorm_fwd_inf_ts(x, weight, 1e-5, 0, True)
    expected = rmsnorm_reference(x, weight, 1e-5, True)
    torch.testing.assert_close(actual, expected)


def test_prepare_image_freezes_shape_layout_and_range() -> None:
    image = Image.fromarray(np.full((45, 80, 3), 128, dtype=np.uint8))
    tensor = prepare_image(
        image,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert tensor.shape == (1, 3, 1, 576, 1024)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_restore_image_recovers_original_size() -> None:
    output = torch.zeros((1, 3, 1, 576, 1024), dtype=torch.float32)
    image = restore_image(output, (800, 450))
    assert image.size == (800, 450)
    array = np.asarray(image)
    assert array.shape == (450, 800, 3)
    assert int(array.min()) == 128
    assert int(array.max()) == 128
