from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(scene: str) -> dict:
    path = ROOT / f"configs/worldsim_v5/m1_graph_diagnostic_{scene}_v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_replication_graph_configs_preserve_scene0471_method_contract() -> None:
    reference = _load("scene0471")
    for scene in ("scene1087", "scene0379"):
        config = _load(scene)
        assert config["graph"] == reference["graph"]
        for key in (
            "probability_threshold",
            "boundary_tolerance_px",
            "ece_bins",
            "comparator",
            "direction_only_mechanism_diagnostic",
            "automatic_validation_unlock",
            "automatic_semantic_split_unlock",
        ):
            assert config["evaluation"][key] == reference["evaluation"][key]
        assert config["resources"] == reference["resources"]
        assert config["runtime"] == reference["runtime"]


def test_replication_graph_configs_bind_frozen_unary_artifacts() -> None:
    expected = {
        "scene1087": {
            "scene": "scene-1087",
            "index": 827,
            "summary": "d19cabd9a2bb48ded6e73ef6bf83a8073d3704198dd69ec628f39ddd98e47f8b",
            "diagnostics": "e97e8d46fdff8a80792aec68280d08ca28d08ac854de860c8a0025215cbeb2db",
            "target": "frozen_r042_evaluation_npz",
        },
        "scene0379": {
            "scene": "scene-0379",
            "index": 296,
            "summary": "1beff3d934c2628cb2b7a262d5e7ab75c4cf214db3bdf39cc51c59690140803d",
            "diagnostics": "f539be1ce74bdd5eb10dc3d7525f4225299b1d98c7ff4e649144187caf876008",
            "target": "frozen_r043_evaluation_npz",
        },
    }
    for scene, binding in expected.items():
        config = _load(scene)
        assert config["scene"]["name"] == binding["scene"]
        assert config["scene"]["index"] == binding["index"]
        assert config["inputs"]["unary_summary"]["sha256"] == binding["summary"]
        assert (
            config["inputs"]["unary_diagnostics"]["sha256"]
            == binding["diagnostics"]
        )
        assert config["evaluation"]["target_source"] == binding["target"]
        assert config["graph"]["base_model_consumed_by_graph"] is False
