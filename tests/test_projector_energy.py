import torch

from motion_proj.auditor.state import MotionState
from motion_proj.projector.projector import DynamicsProjector


class _IdentityBackground:
    def project(self, frames, state, anchor):
        return frames


class _EmptyObjects:
    def smooth(self, tracks):
        return tracks

    def project(self, background, frames, tracks, projected_tracks, support):
        mask = torch.zeros(frames.shape[0], 1, frames.shape[2], frames.shape[3])
        return background, mask


class _EmptySupport:
    def classify(self, tracks, hw):
        return {}


class _OnesReliability:
    def build(self, state, object_mask, k):
        return torch.ones(k, 1, object_mask.shape[-2], object_mask.shape[-1])


def test_projector_rejects_vacuous_empty_track_energy_decrease():
    frames = torch.zeros(2, 3, 8, 8)
    state = MotionState(
        tracks=[],
        u_static=torch.zeros(1, 8, 8, 2),
        u_ego=torch.zeros(1, 8, 8, 2),
        static_mask=torch.ones(2, 8, 8),
        flow_conf=torch.ones(1, 8, 8),
        depth=torch.ones(2, 8, 8),
        meta={"sample_id": "empty"},
    )
    projector = DynamicsProjector(
        background=_IdentityBackground(),
        objects=_EmptyObjects(),
        support=_EmptySupport(),
        reliability=_OnesReliability(),
    )
    result = projector.project(frames, state)
    assert float(result.energy_before["total"]) == float(result.energy_after["total"])
    assert result.diagnostics["energy_decreased"] is False
    assert result.diagnostics["num_tracks"] == 0
