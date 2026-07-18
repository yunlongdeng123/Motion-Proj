import torch

from motion_proj.data.motion_feature_records import fit_ridge, relative_improvement, vector_epe


def test_linear_probe_recovers_held_out_residual_signal():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((240, 6), generator=generator)
    weight = torch.tensor(
        [[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2], [0.0, 0.1], [0.2, 0.0], [-0.1, -0.2]]
    )
    targets = features @ weight
    model = fit_ridge(features[:180], targets[:180], regularization=1.0e-3)
    prediction = model.predict(features[180:])
    error = vector_epe(prediction, targets[180:])
    zero = vector_epe(torch.zeros_like(targets[180:]), targets[180:])
    assert error < 0.01
    assert relative_improvement(error, zero) > 0.98

