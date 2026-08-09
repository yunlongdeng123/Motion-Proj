from __future__ import annotations

import numpy as np
import scripts.eval_worldsim_v3_a3_r1_heldout as heldout_eval

from scripts.eval_worldsim_v3_a3_r1_heldout import (
    aggregate_metric_rows,
    build_resource_audit,
    build_variant_aggregate,
    classify_exact_pareto,
    load_model_checkpoint_read_only,
    squared_rgb_error,
)


def metric_row(variant: str, role: str, edit: str, *, better: bool) -> dict:
    return {
        "role": role,
        "edit": edit,
        "frame": 10,
        "camera": 0,
        "variant": variant,
        "affected_pixels": 2,
        "s_b_t0_pixels": 2,
        "t1_valid_pixels": 2 if better else 1,
        "depth_order_violations": 0 if better else 1,
        "common_t1_valid_pixels": 1,
        "t0_abs_error_sum_m": 0.1 if better else 0.2,
        "non_target_squared_uint8_error": 10 if better else 20,
        "non_target_channel_elements": 30,
    }


def test_uint8_rgb_error_uses_only_frozen_mask() -> None:
    target = np.zeros((2, 2, 3), dtype=np.uint8)
    predicted = target.copy()
    predicted[0, 0] = [1, 2, 3]
    predicted[1, 1] = [10, 10, 10]
    mask = np.array([[True, False], [False, False]])
    squared, elements = squared_rgb_error(predicted, target, mask)
    assert squared == 1 + 4 + 9
    assert elements == 3


def test_metric_aggregation_counts_invalid_first_hit_in_frozen_denominator() -> None:
    aggregate = aggregate_metric_rows(
        [metric_row("r0", "high-support", "lateral", better=False)]
    )
    assert aggregate["s_b_first_hit_valid_coverage"] == 0.5
    assert aggregate["s_b_depth_order_violation_rate"] == 0.5
    assert aggregate["s_b_t0_first_hit_mae_m"] == 0.2


def test_variant_aggregate_and_exact_pareto_are_result_independent() -> None:
    groups = [
        "high-support::lateral",
        "high-support::delete",
        "boundary-support::lateral",
        "boundary-support::delete",
    ]
    rows = []
    for group in groups:
        role, edit = group.split("::")
        rows.append(metric_row("r0", role, edit, better=False))
        rows.append(metric_row("r1", role, edit, better=True))
    global_rows = [
        {
            "variant": "r0",
            "frame": 10,
            "camera": 0,
            "squared_uint8_error": 20,
            "channel_elements": 30,
        },
        {
            "variant": "r1",
            "frame": 10,
            "camera": 0,
            "squared_uint8_error": 10,
            "channel_elements": 30,
        },
    ]
    r0 = build_variant_aggregate(rows, global_rows, variant="r0", group_order=groups)
    r1 = build_variant_aggregate(rows, global_rows, variant="r1", group_order=groups)
    result = classify_exact_pareto(
        r0["primary_axes"],
        r1["primary_axes"],
        {
            "s_b_first_hit_valid_coverage": "higher",
            "s_b_depth_order_violation_rate": "lower",
            "non_target_observed_rgb_mse": "lower",
            "original_global_observed_rgb_mse": "lower",
        },
    )
    assert result["classification"] == "r1_dominates_r0_pass"
    assert result["r1_non_worse"] is True


def test_exact_pareto_reports_tradeoff_and_missing_evidence() -> None:
    tradeoff = classify_exact_pareto(
        {"a": 1.0, "b": 1.0},
        {"a": 0.5, "b": 1.5},
        {"a": "lower", "b": "lower"},
    )
    assert tradeoff["classification"] == "tradeoff_non_dominated"
    missing = classify_exact_pareto(
        {"a": None}, {"a": 1.0}, {"a": "lower"}
    )
    assert missing["classification"] == "insufficient_evidence"


def test_resource_audit_persists_exact_failed_dimension() -> None:
    audit = build_resource_audit(
        duration=10.0,
        peak_gpu_mib=11.0,
        cgroup_samples=[100, None, 201],
        run_bytes=12,
        oom_delta=0,
        oom_kill_delta=0,
        ceilings={
            "wall_time_seconds": 10,
            "peak_gpu_memory_mib": 12,
            "peak_cgroup_memory_bytes": 200,
            "run_bytes": 12,
            "oom_events_delta": 0,
            "oom_kill_events_delta": 0,
        },
    )
    assert audit["status"] == "failed"
    assert audit["measured"]["cgroup_memory_samples_bytes"] == [100, 201]
    assert audit["violations"] == {
        "wall_time_seconds": False,
        "peak_gpu_memory_mib": False,
        "peak_cgroup_memory_bytes": True,
        "run_bytes": False,
        "oom_events_delta": False,
        "oom_kill_events_delta": False,
    }


def test_checkpoint_loader_stages_tensors_on_cpu(monkeypatch, tmp_path) -> None:
    rigid_state = {"value": 2}
    state_dict = {
        "step": 1,
        "models": {"Background": {"value": 1}, "RigidNodes": rigid_state},
    }
    observed = {}

    def fake_load(path, *, map_location):
        observed["path"] = path
        observed["map_location"] = map_location
        return state_dict

    def fake_to_device(value, device):
        observed["rigid_state"] = value
        observed["rigid_device"] = device
        return {"staged": value}

    class FakeTrainer:
        def load_state_dict(self, value, *, load_only_model, strict):
            observed["state_dict"] = value
            observed["load_only_model"] = load_only_model
            observed["strict"] = strict

    monkeypatch.setattr(heldout_eval.torch, "load", fake_load)
    monkeypatch.setattr(heldout_eval, "to_device", fake_to_device)
    checkpoint = tmp_path / "checkpoint.pth"
    device = heldout_eval.torch.device("cuda:0")
    load_model_checkpoint_read_only(FakeTrainer(), checkpoint, device)
    assert observed == {
        "path": checkpoint,
        "map_location": "cpu",
        "rigid_state": rigid_state,
        "rigid_device": device,
        "state_dict": state_dict,
        "load_only_model": True,
        "strict": True,
    }
    assert state_dict["models"]["Background"] == {"value": 1}
    assert state_dict["models"]["RigidNodes"] == {"staged": rigid_state}
