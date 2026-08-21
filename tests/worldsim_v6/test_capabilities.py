from pathlib import Path

import pytest

from motion_proj.worldsim_v6.capabilities import (
    CapabilityError,
    LogicalURIResolver,
    SCHEMA_VERSION,
)


def _manifest(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    python = tmp_path / "env/bin/python"
    checkpoint = tmp_path / "weights/model.ckpt"
    for directory in (repo, runs, data, python.parent, checkpoint.parent):
        directory.mkdir(parents=True, exist_ok=True)
    python.write_text("python", encoding="utf-8")
    checkpoint.write_text("weight", encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "roots": {"repo": str(repo), "runs": str(runs), "cache": str(tmp_path / "cache"), "asset": str(tmp_path / "asset")},
        "datasets": {"nuscenes": {"path": str(data)}},
        "envs": {"primary": {"path": str(python)}},
        "third_party": {"demo": {"path": str(repo)}},
        "checkpoints": {"demo": {"path": str(checkpoint)}},
    }


def test_logical_uri_resolves_known_roots(tmp_path: Path) -> None:
    resolver = LogicalURIResolver(_manifest(tmp_path))
    assert resolver.resolve("repo://configs/demo.yaml") == (tmp_path / "repo/configs/demo.yaml").resolve()
    assert resolver.resolve("dataset://nuscenes/samples") == (tmp_path / "data/samples").resolve()
    assert resolver.resolve("env://primary") == (tmp_path / "env/bin/python").resolve()
    assert resolver.resolve("checkpoint://demo") == (tmp_path / "weights/model.ckpt").resolve()


@pytest.mark.parametrize(
    "uri",
    [
        "/root/private",
        "repo:///etc/passwd",
        "repo://../secret",
        "repo://a/../../secret",
        "dataset://missing/file",
        "unknown://item",
        "repo://C:\\host\\file",
        "env://primary/child",
    ],
)
def test_logical_uri_rejects_unsafe_or_unknown_values(tmp_path: Path, uri: str) -> None:
    resolver = LogicalURIResolver(_manifest(tmp_path))
    with pytest.raises(CapabilityError):
        resolver.resolve(uri)
