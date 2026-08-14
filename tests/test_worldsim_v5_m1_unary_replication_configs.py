from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(scene: str) -> dict:
    path = ROOT / f"configs/worldsim_v5/m1_unary_diagnostic_{scene}_v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_replication_unary_configs_preserve_scene0471_method_contract() -> None:
    reference = _load("scene0471")
    for scene in ("scene1087", "scene0379"):
        config = _load(scene)
        assert config["unary"] == reference["unary"]
        assert config["evidence"] == reference["evidence"]
        assert config["evaluation"] == reference["evaluation"]
        assert config["resources"] == reference["resources"]
        assert config["runtime"] == reference["runtime"]


def test_replication_unary_configs_bind_frozen_sam_artifacts() -> None:
    expected = {
        "scene1087": {
            "scene": "scene-1087",
            "index": 827,
            "summary": "4b4f8d2b85809926adf02827203cfb9816fdabb49a9180617baee6fe06345f68",
            "masks": "da1bd44e0955fd04c29e33ed2ee71d4ba30fdf8e7f1002948d6e0dbaf13b334c",
        },
        "scene0379": {
            "scene": "scene-0379",
            "index": 296,
            "summary": "47f33cb4baa49251a7ffdc4ff7133273504934cc84656c3a405e90c9c7cddbb6",
            "masks": "a4f7e6955e2cf6b1580d76f00ecd7ae662778a267944758a3f4b5bdf921fc61d",
        },
    }
    for scene, binding in expected.items():
        config = _load(scene)
        assert config["scene"]["name"] == binding["scene"]
        assert config["scene"]["index"] == binding["index"]
        assert config["inputs"]["sam_summary"]["sha256"] == binding["summary"]
        assert config["inputs"]["sam_mask_manifest"]["sha256"] == binding["masks"]
        assert config["evaluation"]["automatic_graph_unlock"] is False
