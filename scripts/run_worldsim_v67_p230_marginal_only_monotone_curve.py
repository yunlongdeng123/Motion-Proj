"""Distill the P203 curve using only the 28 marginal-CDF inputs."""

import torch
from torch import nn
import torch.nn.functional as functional
import scripts.run_worldsim_v67_p227_monotone_reliability_curve_distillation as base


class MarginalOnlyMonotoneCurveStudent(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers = []
        width = 28
        for hidden in widths:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 8))
        self.network = nn.Sequential(*layers)

    def forward(self, feature):
        masses = functional.softmax(self.network(feature[:, 8:]), dim=1)
        return masses.cumsum(dim=1)[:, :7]


if __name__ == "__main__":
    base.MonotoneCurveStudent = MarginalOnlyMonotoneCurveStudent
    base.main()
