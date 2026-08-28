"""Run a temporal-token then Actor-set occupancy-flip risk model."""

from __future__ import annotations

from typing import Sequence

import torch

import scripts.run_worldsim_v67_p90_plain_trajectory_max_error as runner
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import DeepSetRisk


class HierarchicalTemporalInteractionRisk(torch.nn.Module):
    """Encode ordered clearance samples per Actor before invariant Actor pooling."""

    def __init__(
        self, feature_count: int, element_dimensions: Sequence[int], decoder_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        self.actor_only = feature_count <= 24
        if self.actor_only:
            self.base = DeepSetRisk(feature_count, element_dimensions, decoder_dimensions)
            return
        if feature_count != 42:
            raise ValueError(f"expected 42 query features, got {feature_count}")
        temporal_width = 64
        actor_width = int(element_dimensions[-1])
        self.actor_encoder = torch.nn.Sequential(
            torch.nn.Linear(24, int(element_dimensions[0])), torch.nn.SiLU(),
            torch.nn.Linear(int(element_dimensions[0]), actor_width), torch.nn.SiLU(),
        )
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, temporal_width), torch.nn.SiLU(),
            torch.nn.Linear(temporal_width, temporal_width), torch.nn.SiLU(),
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(actor_width + 2 * temporal_width, actor_width), torch.nn.SiLU(),
        )
        decoder_layers: list[torch.nn.Module] = []
        width = actor_width * 2
        for hidden in decoder_dimensions:
            decoder_layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.event_head = torch.nn.Linear(width, 1)
        self.error_head = torch.nn.Linear(width, 1)
        self.register_buffer("fractions", torch.linspace(0.0, 1.0, 9).view(1, 1, 9, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.actor_only:
            return self.base(features, mask)
        actor = self.actor_encoder(features[..., :24])
        signed = features[..., 24:33].unsqueeze(-1)
        boundary = features[..., 33:42].unsqueeze(-1)
        fractions = self.fractions.expand(features.shape[0], features.shape[1], -1, -1)
        temporal = self.temporal_encoder(torch.cat((signed, boundary, fractions), dim=-1))
        temporal_summary = torch.cat((temporal.mean(dim=2), temporal.max(dim=2).values), dim=-1)
        encoded = self.fuse(torch.cat((actor, temporal_summary), dim=-1))
        expanded_mask = mask.unsqueeze(-1)
        mean = (encoded * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1)
        maximum = encoded.masked_fill(~expanded_mask, -torch.inf).max(dim=1).values
        decoded = self.decoder(torch.cat((mean, maximum), dim=-1))
        return self.event_head(decoded).squeeze(-1), torch.nn.functional.softplus(
            self.error_head(decoded).squeeze(-1),
        )


if __name__ == "__main__":
    runner.DeepSetRisk = HierarchicalTemporalInteractionRisk
    runner.main()
