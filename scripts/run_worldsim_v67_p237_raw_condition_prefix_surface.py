"""Compile the surface directly from eight score/clearance conditions."""

import torch
from torch import nn
import torch.nn.functional as functional
import scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface as base


class RawConditionPrefixSurface(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers = []
        width = 8
        for hidden in widths:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 29))
        self.network = nn.Sequential(*layers)

    def forward(self, feature):
        raw = self.network(feature[:, :8])
        base_curve = functional.softmax(raw[:, :8], dim=1).cumsum(dim=1)[:, :7]
        transition = raw[:, 8:].reshape(-1, 3, 7)
        increments = functional.softplus(transition[:, :, 1:])
        logits = transition[:, :, :1] + torch.cat(
            (torch.zeros_like(transition[:, :, :1]), torch.cumsum(increments, dim=2)), dim=2
        )
        retention = torch.sigmoid(logits)
        curves = [base_curve]
        for index in range(3):
            curves.append(curves[-1] * retention[:, index])
        return torch.stack(curves, dim=1)


if __name__ == "__main__":
    base.MonotonePrefixSurface = RawConditionPrefixSurface
    base.main()
