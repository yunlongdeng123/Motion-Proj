from motion_proj.auditor.state import Track
from motion_proj.tune.policy import prune_reason
from motion_proj.tune.trial_eval import warp_tracks_by_flow
from motion_proj.tune.trial_runner import trial_overrides
from motion_proj.tune.trial_summary import (
    build_trial_metrics,
    merge_trial_summary,
    normalized_improvement,
)

import torch


def test_normalized_improvement_and_trial_metrics():
    base = {"static_drift": 10.0, "track_acceleration": 4.0, "lpips": 0.20}
    adapter = {
        "static_drift": 6.0,
        "track_acceleration": 3.0,
        "lpips": 0.19,
        "projection_eligible_fraction": 0.80,
    }
    metrics = build_trial_metrics(adapter, base)
    assert metrics["normalized_static_drift_improvement"] == normalized_improvement(10.0, 6.0)
    assert metrics["normalized_track_acceleration_improvement"] == 0.25
    assert prune_reason(metrics, base_lpips=0.20) is None
    summary = merge_trial_summary({"trained_steps": 100, "checkpoint": "x"}, metrics)
    assert summary["trained_steps"] == 100
    assert summary["lpips"] == 0.19


def test_warp_tracks_by_flow_moves_centers():
    track = Track(
        instance_token="a",
        category="car",
        xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]]),
        depth=torch.tensor([5.0, 5.0]),
        present=torch.tensor([True, True]),
    )
    flow = torch.zeros(1, 64, 64, 2)
    flow[0, 30, 20] = torch.tensor([3.0, -2.0])
    warped = warp_tracks_by_flow([track], flow)[0]
    assert torch.allclose(warped.xyxy[1], torch.tensor([13.0, 18.0, 33.0, 38.0]))


def test_trial_overrides_map_search_space():
    overrides = trial_overrides(
        {
            "lr": 2e-5,
            "lambda_proj": 0.1,
            "beta_anchor": 0.5,
            "bound_B": 4,
            "tube_upper": 0.35,
            "lora_rank": 16,
        },
        steps=100,
        run_dir="/tmp/tune-t0",
        parent_run_id=None,
    )
    assert "train.max_steps=100" in overrides
    assert "train.tube.bound_B=4.0" in overrides
    assert "train.tube.sigma_quantile_range=[0.0,0.35]" in overrides
