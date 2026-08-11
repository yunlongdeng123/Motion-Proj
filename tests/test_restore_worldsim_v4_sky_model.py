from __future__ import annotations

import pytest

from scripts.restore_worldsim_v4_sky_model import SkyModelRestoreError, load_config


def test_restore_config_requires_official_exact_revision_policy(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  restore:\n    endpoint: https://huggingface.co\n    policy: official_exact_revision_if_missing\n")
    assert load_config(path)["model"]["restore"]["endpoint"] == "https://huggingface.co"


def test_restore_config_rejects_mirror_or_floating_policy(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  restore:\n    endpoint: https://example.com\n    policy: latest\n")
    with pytest.raises(SkyModelRestoreError, match="endpoint/policy"):
        load_config(path)
