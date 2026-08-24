import torch

from motion_proj.worldsim_v62.projection import project_feasible_tristate


def test_hard_evidence_projection_contract_and_gradient() -> None:
    logits = torch.tensor(
        [
            [-2.0, 5.0, -1.0],
            [5.0, -2.0, -1.0],
            [0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2],
            [0.1, 4.0, -1.0],
            [0.5, -0.2, 0.1],
        ],
        requires_grad=True,
    )
    output = project_feasible_tristate(
        logits,
        observed_free=torch.tensor([1, 0, 1, 0, 0, 0]),
        observed_occupied=torch.tensor([0, 1, 1, 0, 0, 0]),
        contradiction=torch.tensor([0, 0, 0, 1, 0, 0]),
        outside_lifecycle=torch.tensor([0, 0, 0, 0, 1, 0]),
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(output.probabilities[:5], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.probabilities.sum(dim=-1), torch.ones(6), rtol=0, atol=1e-7
    )

    weights = torch.tensor([0.2, 0.7, 1.3])
    (output.probabilities * weights).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:5]) == 0
    assert torch.count_nonzero(logits.grad[5]) > 0
