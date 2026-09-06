from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from motion_proj.worldsim_v71.evaluate_surface import (
    evaluate_actor_surface,
    summarize_surface_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "run_worldsim_v71_m43_m39_av2_zero_shot",
    SCRIPTS_ROOT / "run_worldsim_v71_m43_m39_av2_zero_shot.py",
)
assert SPEC is not None and SPEC.loader is not None
M43_RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M43_RUNNER)


def test_literal_and_categorical_counts_have_independent_summaries() -> None:
    row = {
        "hazardous": False,
        "target_ray_count": 10,
        "ray_count": 10,
        "baseline_early_count": 1,
        "output_early_count": 2,
        "baseline_hit_count": 3,
        "output_hit_count": 4,
        "categorical_baseline_early_count": 5,
        "categorical_baseline_hit_count": 6,
        "m39_early_count": 7,
        "m39_hit_count": 8,
        "m39_observable_count": 10,
        "baseline_chamfer_m": 0.1,
        "output_chamfer_m": 0.2,
        "actor_state_retention": 1.0,
        "hazard_state_retention": 1.0,
    }

    geometry = summarize_surface_rows([row])
    returns = M43_RUNNER._surface_return_summary([row])

    assert geometry["all"]["baseline_early_rate"] == 0.1
    assert geometry["all"]["output_early_rate"] == 0.2
    assert geometry["baseline_hit_recall"] == 0.3
    assert geometry["output_hit_recall"] == 0.4
    assert returns["all"]["baseline_early_rate"] == 0.5
    assert returns["all"]["m39_early_rate"] == 0.7
    assert returns["all"]["baseline_hit_rate"] == 0.6
    assert returns["all"]["m39_hit_rate"] == 0.8


def test_actor_retention_is_measured_from_state_fields() -> None:
    state = {
        "track_id": "actor-1",
        "trajectory_xyz_m": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "size_lwh_m": np.asarray([4.0, 2.0, 1.5], dtype=np.float32),
        "hazardous": True,
    }
    common = {
        "baseline_surface": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "output_surface": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "target": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "origins": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "hazardous": True,
        "device": torch.device("cpu"),
        "lateral_tolerance_m": 0.2,
        "depth_tolerance_m": 0.2,
        "distance_chunk_size": 8,
        "actor_state_before": state,
    }

    retained = evaluate_actor_surface(actor_state_after=dict(state), **common)
    changed = dict(state)
    changed["size_lwh_m"] = np.asarray([4.2, 2.0, 1.5], dtype=np.float32)
    rejected = evaluate_actor_surface(actor_state_after=changed, **common)

    assert retained["retention_measurement"] == "measured_input_output"
    assert retained["actor_state_retention"] == 1.0
    assert retained["hazard_state_retention"] == 1.0
    assert rejected["actor_state_retention"] == 0.0
    assert rejected["hazard_state_retention"] == 0.0
    assert rejected["retention_fields_equal"]["size_equal"] is False
