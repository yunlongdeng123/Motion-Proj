from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_worldsim_v51_v5_unary.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v51_a0_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_bit_mismatch_count_is_dtype_and_bit_exact() -> None:
    left = np.asarray([0.0, -0.0, 1.0], dtype=np.float32)
    right = left.copy()
    assert RUNNER._bit_mismatch_count(left, right) == 0

    right[1] = 0.0
    assert RUNNER._bit_mismatch_count(left, right) == 1
    assert RUNNER._bit_mismatch_count(left, right.astype(np.float64)) == 3


def test_gaussian_metrics_match_frozen_definitions() -> None:
    posterior = np.asarray([0.1, 0.9], dtype=np.float32)
    target = np.asarray([0.0, 1.0], dtype=np.float32)

    metrics = RUNNER._gaussian_metrics(
        posterior, target, threshold=0.5, ece_bins=2
    )

    assert metrics["iou_at_frozen_threshold"] == 1.0
    assert metrics["false_positive_semantic_mass"] == pytest.approx(0.1)
    assert metrics["false_negative_semantic_mass"] == pytest.approx(0.1)
    assert metrics["nll"] > 0.0


def test_replay_contract_is_historical_only() -> None:
    config = RUNNER.load_yaml(
        ROOT / "configs/worldsim_v51/m1_unary_baselines_v1.yaml"
    )
    assert list(config["canonical_runs"]) == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert config["failure_ledger_delta"] == "pending"
