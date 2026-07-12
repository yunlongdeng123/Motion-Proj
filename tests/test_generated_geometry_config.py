import pytest

from motion_proj.config import ConfigError, load_config


def test_generated_geometry_mode_is_validated() -> None:
    with pytest.raises(ConfigError, match="auditor.generated_geometry_mode"):
        load_config(
            "configs/train/motionproj_v1.yaml",
            ["auditor.generated_geometry_mode=not_a_geometry_mode"],
        )
