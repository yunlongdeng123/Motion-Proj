from __future__ import annotations

from omegaconf import OmegaConf

from scripts.materialize_worldsim_v3_a2_d1_config import (
    materialize_config,
)


def source_config() -> object:
    return OmegaConf.create(
        {
            "trainer": {
                "type": "models.trainers.MultiTrainer",
                "optim": {"num_iters": 30000},
                "gaussian_ctrl_general_cfg": {
                    "densify_grad_thresh": 0.0005
                },
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


def quota() -> object:
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


def test_materializes_matched_d0_and_d1_configs() -> None:
    d0 = materialize_config(
        source_config(), "d0-native", 1000, quota()
    )
    d1 = materialize_config(
        source_config(), "d1-actor-quota", 1000, quota()
    )

    assert d0.trainer.optim.num_iters == d1.trainer.optim.num_iters == 1000
    assert d0.logging.saveckpt_freq == d1.logging.saveckpt_freq == 1000
    assert d0.model.RigidNodes.ctrl.a2_actor_quota.enabled is False
    assert d1.model.RigidNodes.ctrl.a2_actor_quota.enabled is True
    assert d1.model.RigidNodes.ctrl.a2_actor_quota.maximum_absolute_cap == 12000
    assert (
        d1.trainer.gaussian_ctrl_general_cfg.densify_grad_thresh
        == 0.0005
    )
    assert d1.trainer.gaussian_ctrl_general_cfg.a2_ancestry.enabled is True
    assert "Affine" not in d1.model and "CamPose" not in d1.model
    assert d1.render.render_full is False


def test_rejects_budget_that_cannot_exercise_refinement() -> None:
    try:
        materialize_config(source_config(), "d1-actor-quota", 999, quota())
    except ValueError as error:
        assert "1000" in str(error)
    else:
        raise AssertionError("short A2-D1 smoke budget was accepted")
