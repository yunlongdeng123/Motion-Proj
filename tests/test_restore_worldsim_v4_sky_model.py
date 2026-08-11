from __future__ import annotations

import pytest

from scripts.restore_worldsim_v4_sky_model import SkyModelRestoreError, load_config, restore_from_staging, sha256_file


def test_restore_config_requires_official_exact_revision_policy(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  restore:\n    endpoint: https://huggingface.co\n    policy: official_exact_revision_if_missing\n    remote_staging_dir: /staging\n")
    assert load_config(path)["model"]["restore"]["endpoint"] == "https://huggingface.co"


def test_restore_config_rejects_mirror_or_floating_policy(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  restore:\n    endpoint: https://example.com\n    policy: latest\n    remote_staging_dir: /staging\n")
    with pytest.raises(SkyModelRestoreError, match="endpoint/policy"):
        load_config(path)


def test_restore_from_staging_atomically_publishes_exact_files(tmp_path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    source = staging / "model.bin"
    source.write_bytes(b"exact-model")
    required = {"model.bin": {"bytes": source.stat().st_size, "sha256": sha256_file(source)}}
    target = tmp_path / "cache" / "snapshots" / "revision"

    restore_from_staging(staging, target, required)

    assert (target / "model.bin").read_bytes() == b"exact-model"
    assert not list(target.parent.glob(".revision.partial-*"))


def test_restore_from_staging_rejects_drift_without_creating_target(tmp_path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "model.bin").write_bytes(b"drift")
    target = tmp_path / "cache" / "snapshots" / "revision"

    with pytest.raises(SkyModelRestoreError, match="staging file 漂移"):
        restore_from_staging(staging, target, {"model.bin": {"bytes": 5, "sha256": "0" * 64}})

    assert not target.exists()
