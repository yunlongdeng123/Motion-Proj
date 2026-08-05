from omegaconf import OmegaConf

from scripts.materialize_worldsim_v3_a1_config import materialize_config


def source_config():
    return OmegaConf.create(
        {
            "trainer": {"type": "native", "optim": {"num_iters": 30000}},
            "logging": {"saveckpt_freq": 30000},
            "render": {"render_full": True, "render_test": True, "render_novel": {}},
            "model": {
                "Affine": {
                    "type": "native.Affine",
                    "params": {"embedding_dim": 4},
                    "optim": {"all": {"lr": 1e-5}},
                },
                "CamPose": {
                    "type": "native.CamPose",
                    "optim": {"all": {"lr": 1e-5}},
                },
                "Background": {"type": "background"},
            },
        }
    )


def test_c0_removes_both_native_calibration_modules() -> None:
    config = materialize_config(source_config(), "c0-off", 100)
    assert "Affine" not in config.model
    assert "CamPose" not in config.model
    assert config.trainer.optim.num_iters == 100
    assert config.worldsim_v3.rolling_shutter == "not_supported"


def test_c1_keeps_both_native_calibration_modules() -> None:
    config = materialize_config(source_config(), "c1-native", 150)
    assert config.model.Affine.type == "native.Affine"
    assert config.model.CamPose.type == "native.CamPose"
    assert config.trainer.optim.num_iters == 150


def test_c2_changes_only_affine_parameterization() -> None:
    config = materialize_config(source_config(), "c2-factorized-isp", 200)
    assert config.model.Affine.type.endswith("FactorizedAffineTransform")
    assert config.model.CamPose.type == "native.CamPose"
    assert config.model.Affine.optim.all.lr == 1e-5


def test_c3_adds_bounded_pose() -> None:
    config = materialize_config(source_config(), "c3-bounded-pose", 300)
    assert config.model.Affine.type.endswith("FactorizedAffineTransform")
    assert config.model.CamPose.type.endswith("BoundedCameraOptModule")
    assert config.model.CamPose.params.max_translation_m == 0.15
    assert config.model.CamPose.params.max_rotation_deg == 2.0
