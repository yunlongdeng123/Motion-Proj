import torch
import pytest

from motion_proj.auditor import MotionAuditor, RAFTChainGeneratedTrackProvider
from motion_proj.auditor.generated_tracks import CoTracker3GeneratedTrackProvider
from motion_proj.projector import DynamicsProjector
from motion_proj.config import ConfigError, load_config
from motion_proj.replay.mine import _generated_track_provider


def _translation_flow(frame_count=4, height=24, width=32, dx=1.0):
    flow = torch.zeros(frame_count - 1, height, width, 2)
    flow[..., 0] = dx
    backward = -flow
    confidence = torch.ones(frame_count - 1, height, width)
    return flow, backward, confidence


def test_raft_chain_tracks_are_no_gt_and_chain_forward_with_fb_consistency():
    torch.manual_seed(0)
    frames = torch.rand(4, 3, 24, 32) * 2 - 1
    flow, backward, confidence = _translation_flow()
    provider = RAFTChainGeneratedTrackProvider(device="cpu", queries_per_stratum=4, min_track_length=3)
    state = provider.track(frames, observed_flow=flow, backward_flow=backward,
                           flow_confidence=confidence, background_flow=torch.zeros_like(flow))
    assert not state.uses_future_gt
    assert state.diagnostics["provider"] == "raft_chain_v1"
    assert state.diagnostics["stratum_query_count"].keys() == {
        "background", "dynamic_residual", "foreground_candidate"
    }
    assert state.diagnostics["valid_track_count"] > 0
    for track in state.tracks:
        present = track.present
        assert int(present.sum()) == 4
        delta = track.center[1:, 0] - track.center[:-1, 0]
        assert torch.allclose(delta, torch.ones_like(delta), atol=1e-4)


def test_raft_chain_drops_fb_inconsistent_points():
    frames = torch.zeros(4, 3, 24, 32)
    flow, _, confidence = _translation_flow()
    provider = RAFTChainGeneratedTrackProvider(device="cpu", queries_per_stratum=4, min_track_length=3)
    state = provider.track(frames, observed_flow=flow, backward_flow=torch.zeros_like(flow),
                           flow_confidence=confidence, background_flow=torch.zeros_like(flow))
    assert state.diagnostics["valid_track_count"] == 0
    assert state.diagnostics["survival_rate"] == 0.0


class _Flow:
    def estimate_bidirectional(self, frames):
        flow, backward, confidence = _translation_flow(frames.shape[0], frames.shape[-2], frames.shape[-1])
        return flow, backward, confidence


class _Depth:
    def estimate(self, frames, sample):
        return torch.ones(frames.shape[0], frames.shape[-2], frames.shape[-1])


def test_generated_auditor_ignores_source_boxes_and_projector_uses_estimated_background():
    frames = torch.rand(4, 3, 24, 32) * 2 - 1
    provider = RAFTChainGeneratedTrackProvider(device="cpu", queries_per_stratum=4, min_track_length=3)
    auditor = MotionAuditor(device="cpu", flow_provider=_Flow(), depth_provider=_Depth(),
                            generated_track_provider=provider,
                            generated_geometry_mode="estimated_background_motion",
                            background_fit_options={"min_points": 32})
    sample = {
        "frames": frames,
        # 值故意很大：generated path 若读取它会造成失败或产生 GT token。
        "boxes": [[{"instance_token": "future_gt"}]] * 4,
        "intrinsics": torch.eye(3), "cam2ego": torch.eye(4), "sample_id": "generated",
    }
    state = auditor.audit(sample)
    assert state.meta["uses_future_gt_track"] is False
    assert state.meta["track_diagnostics"]["valid_track_count"] > 0
    assert all(track.instance_token.startswith("generated_") for track in state.tracks)
    result = DynamicsProjector().project(frames, state)
    assert result.diagnostics["generated_tracks"]["uses_future_gt"] is False
    assert result.diagnostics["object_correction_coverage"] > 0


def test_cotracker_adapter_requires_explicit_independent_predictor():
    with torch.no_grad():
        provider = CoTracker3GeneratedTrackProvider()
        try:
            provider.track(torch.zeros(3, 3, 8, 8))
        except RuntimeError as exc:
            assert "不能静默回退" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("missing predictor must fail closed")


def test_generated_track_provider_config_fails_closed_for_unknown_training_backend():
    with pytest.raises(ConfigError, match="generated_tracks.provider"):
        load_config("configs/default.yaml", ["auditor.generated_tracks.provider=unknown"])


def test_replay_uses_configured_raft_provider_and_rejects_cotracker_training_path():
    cfg = load_config("configs/default.yaml", ["device=cpu", "auditor.generated_tracks.queries_per_stratum=3"])
    provider = _generated_track_provider(cfg)
    assert provider.queries_per_stratum == 3
    cotracker_cfg = load_config("configs/default.yaml", ["auditor.generated_tracks.provider=cotracker3"])
    with pytest.raises(ValueError, match="只允许 raft_chain"):
        _generated_track_provider(cotracker_cfg)
