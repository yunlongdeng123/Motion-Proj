from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.materialize_worldsim_v3_a2_d1_config import materialize_config


def _source() -> object:
    return OmegaConf.create(
        {
            "trainer": {
                "type": "models.trainers.MultiTrainer",
                "optim": {"num_iters": 30000},
                "gaussian_ctrl_general_cfg": {"densify_grad_thresh": 0.0005},
            },
            "model": {
                "Background": {"type": "background"},
                "RigidNodes": {"ctrl": {"cull_out_of_bound": True}},
                "Affine": {"type": "affine"},
                "CamPose": {"type": "pose"},
            },
            "logging": {"saveckpt_freq": 30000},
            "render": {
                "render_full": True,
                "render_test": True,
                "render_novel": "trajectory",
            },
        }
    )


def _quota() -> object:
    return OmegaConf.create(
        {
            "densify_grad_threshold": 0.00025,
            "minimum_initial_multiplier": 0.5,
            "minimum_absolute_floor": 1,
            "maximum_initial_multiplier": 2.4,
            "maximum_absolute_cap": 12000,
            "ranking": "screen_grad_desc_then_gaussian_index",
            "below_threshold_policy": "only_to_recover_minimum",
            "budget_policy": "gradient_ranked_prefix",
        }
    )


def test_formal_materialization_freezes_30k_and_5k_grid() -> None:
    d0 = materialize_config(
        _source(),
        "d0-native",
        30_000,
        _quota(),
        stage="formal",
        checkpoint_interval=5_000,
    )
    d1 = materialize_config(
        _source(),
        "d1-actor-quota",
        30_000,
        _quota(),
        stage="formal",
        checkpoint_interval=5_000,
    )
    assert d0.trainer.optim.num_iters == d1.trainer.optim.num_iters == 30_000
    assert d0.logging.saveckpt_freq == d1.logging.saveckpt_freq == 5_000
    assert d0.worldsim_v3.formal is True
    assert d0.worldsim_v3.checkpoint_interval == 5_000
    assert d0.model.RigidNodes.ctrl.a2_actor_quota.enabled is False
    assert d1.model.RigidNodes.ctrl.a2_actor_quota.enabled is True


@pytest.mark.parametrize(
    ("num_iters", "interval"), [(29_999, 5_000), (30_000, 10_000)]
)
def test_formal_materialization_rejects_budget_drift(
    num_iters: int, interval: int
) -> None:
    with pytest.raises(ValueError):
        materialize_config(
            _source(),
            "d1-actor-quota",
            num_iters,
            _quota(),
            stage="formal",
            checkpoint_interval=interval,
        )
