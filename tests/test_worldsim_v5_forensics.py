from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_worldsim_v5_m1_forensics import (  # noqa: E402
    missing_collection_fields,
    summarize_state,
)
from run_worldsim_v5_m2_forensics import (  # noqa: E402
    extract_requests,
    geometry_diagnostics,
)


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_m1_state_summary_keeps_o1_proxy_and_observation_denominators(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    np.savez_compressed(
        path,
        actor_instance_id=np.asarray(7, dtype=np.int32),
        actor_token=np.asarray("actor", dtype="U64"),
        gaussian_id=np.arange(4, dtype=np.int64),
        hard_instance_id=np.asarray([7, 7, 2, 2], dtype=np.int32),
        posterior=np.asarray([0.9, 0.1, 0.995, 0.005], dtype=np.float32),
        uncertainty=np.asarray([0.0, 0.002, 0.0, 0.002], dtype=np.float32),
        positive_count=np.asarray([1, 0, 1, 0], dtype=np.float32),
        negative_count=np.asarray([1, 0, 0, 2], dtype=np.float32),
    )
    required = [
        "actor_instance_id",
        "actor_token",
        "gaussian_id",
        "hard_instance_id",
        "posterior",
        "uncertainty",
        "positive_count",
        "negative_count",
    ]
    result = summarize_state(path, required)
    assert result["gaussian_count"] == 4
    assert result["o1_proxy_target_count"] == 2
    assert result["o1_proxy_target_recalled_count"] == 1
    assert result["o1_proxy_target_recall"] == pytest.approx(0.5)
    assert result["posterior_extreme_count"] == 2
    assert result["uncertainty_le_1e3_count"] == 2
    assert result["unobserved_count"] == 1
    assert result["mixed_positive_negative_count"] == 1


def test_m1_missing_collection_contract_distinguishes_virtual_identity() -> None:
    result = missing_collection_fields(
        {"gaussian_id", "base_index"},
        {"scene": "scene-0001", "role": "high_support"},
        {
            "per_gaussian": ["scene", "role", "gaussian_id", "center"],
            "per_observation": ["view_id", "sam_probability"],
        },
    )
    assert result == {
        "per_gaussian": ["center"],
        "per_observation": ["sam_probability", "view_id"],
    }


def _candidate(arm: str, candidate_id: str, mae: float, risk: float) -> dict:
    return {
        "arm": arm,
        "candidate": {"candidate_id": candidate_id, "geometry_risk": risk},
        "metrics": {"hole_geometry_mae_m": mae},
    }


def _decision(
    request_id: str,
    *,
    accepted: bool,
    candidate_id: str | None,
    mae: float,
) -> dict:
    return {
        "request_id": request_id,
        "accepted": accepted,
        "selected_candidate_id": candidate_id,
        "metrics": {"hole_geometry_mae_m": mae},
    }


def test_m2_geometry_diagnostics_preserve_abstain_denominator() -> None:
    scene_payloads = {
        "scene-a": {
            "requests": [
                {
                    "request_id": "accepted",
                    "candidates": [
                        _candidate("OBSERVED", "accepted-observed", 0.25, 0.5),
                        _candidate("TELEA", "accepted-telea", 1.0, 1.0),
                    ],
                    "matched_arms": [],
                },
                {
                    "request_id": "risk-abstain",
                    "candidates": [_candidate("TELEA", "risk-telea", 2.0, 1.0)],
                    "matched_arms": [],
                },
            ],
            "blocked_requests": [
                {
                    "request_id": "blocked",
                    "candidates": [],
                    "matched_arms": [
                        {
                            "arm": "ABSTAIN",
                            "metrics": {"hole_geometry_mae_m": 0.7},
                        }
                    ],
                }
            ],
        }
    }
    requests, candidates = extract_requests(scene_payloads)
    result = geometry_diagnostics(
        requests,
        candidates,
        [
            _decision(
                "accepted", accepted=True, candidate_id="accepted-observed", mae=0.25
            ),
            _decision("risk-abstain", accepted=False, candidate_id="risk-telea", mae=5.0),
            _decision("blocked", accepted=False, candidate_id=None, mae=0.7),
        ],
        saturation_scale_m=0.5,
    )
    assert result["request_count"] == 3
    assert result["measured_request_count"] == 2
    assert result["candidate_count"] == 3
    assert result["router"] == {"accepted_count": 1, "abstain_count": 2}
    assert result["geometry_oracle"]["accepted_exact_oracle_count"] == 1
    assert result["denominator_decomposition"]["accepted"]["count"] == 1
    assert result["denominator_decomposition"]["risk_abstain"]["count"] == 1
    assert result["denominator_decomposition"]["role_asset_blocked"]["count"] == 1
    assert result["denominator_decomposition"]["full_denominator"][
        "request_mean_delta_m"
    ] == pytest.approx(0.75)


def test_v5_forensic_configs_freeze_bindings_and_missing_evidence() -> None:
    m1 = _yaml(ROOT / "configs/worldsim_v5/m1_forensics_v1.yaml")
    m2 = _yaml(ROOT / "configs/worldsim_v5/m2_forensics_v1.yaml")
    assert m1["task_id"] == "WS-V5-M1-D0-BAYES-FORENSICS-01"
    assert len(m1["historical_binding"]["state_files"]) == 4
    assert "topology_disagreement" in m1["required_blocked_evidence"]
    assert m2["task_id"] == "WS-V5-M2-D0-GEOMETRY-FORENSICS-01"
    assert len(m2["historical_binding"]["scene_summary_files"]) == 6
    assert m2["frozen_v4_mapping"]["refit_allowed"] is False
    assert "reference_confidence" in m2["required_blocked_evidence"]
    assert "pre_gaussianization_geometry_error" in m2["required_blocked_evidence"]
