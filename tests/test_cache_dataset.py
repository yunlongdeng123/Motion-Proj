import pytest
import torch

from motion_proj.cache import ProjectionCacheDataset, ProjectionCacheWriter


def _tensors():
    y = torch.zeros(2, 3, 4, 4)
    target = torch.ones_like(y)
    mask = torch.ones(2, 1, 4, 4)
    return y, target, mask


def test_cache_dataset_skips_stale_revisions(tmp_path):
    y, target, mask = _tensors()
    clean = torch.full_like(y, 2.0)
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="old").write(
        "sample-0",
        y,
        target,
        mask,
        {},
        clean=clean,
    )
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="current").write(
        "sample-0",
        y,
        target,
        mask,
        {},
    )

    dataset = ProjectionCacheDataset(
        str(tmp_path),
        expected_fingerprint="current",
    )

    assert len(dataset) == 1
    assert dataset[0]["metadata"]["cache_fingerprint"] == "current"
    assert torch.equal(dataset[0]["clean"], y)


def test_cache_dataset_keeps_clean_latent_separate(tmp_path):
    y, target, mask = _tensors()
    clean = torch.full_like(y, 2.0)
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="current").write(
        "sample-0",
        y,
        target,
        mask,
        {},
        clean=clean,
    )

    item = ProjectionCacheDataset(
        str(tmp_path),
        expected_fingerprint="current",
    )[0]

    assert torch.equal(item["clean"], clean)
    assert not torch.equal(item["clean"], item["y"])


def test_cache_dataset_rejects_mixed_fingerprints(tmp_path):
    y, target, mask = _tensors()
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="expected").write(
        "sample-0",
        y,
        target,
        mask,
        {},
    )
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="other").write(
        "sample-1",
        y,
        target,
        mask,
        {},
    )

    with pytest.raises(RuntimeError, match="fingerprint 不匹配"):
        ProjectionCacheDataset(
            str(tmp_path),
            expected_fingerprint="expected",
        )
