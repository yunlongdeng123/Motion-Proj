import numpy as np
import torch
from .road_asset import build_observation_priority_atlas
from .road_texture_training import LocalWorldTextureCorrection


def test_true_bilinear_color_gradient_only_reaches_eligible_texels():
    keys = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    world = np.c_[keys * .02, np.zeros(4)]
    rgb = np.full((4, 3), .3, np.float32)
    atlas, _ = build_observation_priority_atlas(world, keys, rgb, np.ones(4, bool), np.ones(4, bool))
    eligible = np.array([False, True, False, False])
    model = LocalWorldTextureCorrection(atlas, eligible, device="cpu")
    result = model.sample(torch.tensor([[.5, .5]], dtype=torch.float32))
    result.sum().backward()
    gradient = model.delta_raw.grad
    assert torch.isfinite(gradient).all()
    assert (gradient[0, :, 0, 1] > 0).all()
    assert gradient[0, :, 0, 0].count_nonzero() == 0
    assert gradient[0, :, 1].count_nonzero() == 0
    with torch.no_grad():
        model.delta_raw.fill_(100)
    delta = model.delta()
    assert float(delta.max()) <= .08000001
    assert delta[0, :, 0, 0].count_nonzero() == 0
