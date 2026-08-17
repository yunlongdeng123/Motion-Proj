from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_a0_baselines_bind_only_historical_scenes() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/m1_unary_baselines_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert list(payload["canonical_runs"]) == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert payload["phase"] == "a0_exact_replay"
    assert payload["replay_contract"]["unary_arrays_bit_exact"] is True
    assert payload["failure_ledger_delta"] == "pending"
