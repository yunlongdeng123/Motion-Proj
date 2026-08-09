from __future__ import annotations

from pathlib import Path

import yaml
import pytest
from omegaconf import OmegaConf

from scripts.materialize_worldsim_v3_a2_d2_config import (
    materialize_config,
    normalized_pair_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def source_config() -> object:
    return OmegaConf.create(
        {
            "trainer": {
                "type": "models.trainers.SingleTrainer",
                "optim": {"num_iters": 30000},
                "gaussian_ctrl_general_cfg": {
                    "densify_grad_thresh": 0.0005
                },
            },
            "logging": {"saveckpt_freq": 5000},
            "render": {
                "render_full": True,
                "render_test": True,
                "render_novel": {"enabled": True},
            },
            "model": {
                "Affine": {"type": "native"},
                "CamPose": {"type": "native"},
                "RigidNodes": {"ctrl": {}},
            },
        }
    )


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/worldsim_v3/a2_d2_protocol_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_d1_d2_materialized_pair_only_differs_in_frozen_fields() -> None:
    d1 = materialize_config(
        source_config(), "d1-actor-quota", 1000, protocol()
    )
    d2 = materialize_config(
        source_config(), "d2-boundary-residual", 1000, protocol()
    )

    assert normalized_pair_payload(d1) == normalized_pair_payload(d2)
    assert d1.model.RigidNodes.ctrl.a2_actor_quota.enabled is True
    assert d2.model.RigidNodes.ctrl.a2_actor_quota.enabled is True
    assert d1.model.RigidNodes.ctrl.a2_boundary_residual.enabled is False
    assert d2.model.RigidNodes.ctrl.a2_boundary_residual.enabled is True
    assert d1.worldsim_v3.d1_inherited_exactly is True
    assert "Affine" not in d1.model and "CamPose" not in d1.model


def test_d2_materializer_rejects_budget_drift() -> None:
    try:
        materialize_config(
            source_config(), "d2-boundary-residual", 999, protocol()
        )
    except ValueError as error:
        assert "iteration budget drift" in str(error)
    else:
        raise AssertionError("D2 materializer accepted a changed smoke budget")


def test_d2_formal_materialization_freezes_30k_grid() -> None:
    d1 = materialize_config(
        source_config(),
        "d1-actor-quota",
        30_000,
        protocol(),
        stage="formal",
        checkpoint_interval=5_000,
    )
    d2 = materialize_config(
        source_config(),
        "d2-boundary-residual",
        30_000,
        protocol(),
        stage="formal",
        checkpoint_interval=5_000,
    )
    assert d1.trainer.optim.num_iters == d2.trainer.optim.num_iters == 30_000
    assert d1.logging.saveckpt_freq == d2.logging.saveckpt_freq == 5_000
    assert d1.worldsim_v3.formal is True
    assert d2.model.RigidNodes.ctrl.a2_boundary_residual.enabled is True
    assert normalized_pair_payload(d1) == normalized_pair_payload(d2)


@pytest.mark.parametrize(
    ("num_iters", "interval"), [(29_999, 5_000), (30_000, 10_000)]
)
def test_d2_formal_materialization_rejects_budget_drift(
    num_iters: int, interval: int
) -> None:
    with pytest.raises(ValueError, match="formal requires"):
        materialize_config(
            source_config(),
            "d2-boundary-residual",
            num_iters,
            protocol(),
            stage="formal",
            checkpoint_interval=interval,
        )
