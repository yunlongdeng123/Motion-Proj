import pytest
import torch

from motion_proj.cache import MixedProjectionCacheDataset, ProjectionCacheDataset, ProjectionCacheWriter
from motion_proj.cache.writer import CACHE_SCHEMA_VERSION


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


def test_schema_v4_roundtrip_optional_flow_and_provenance(tmp_path):
    y, target, mask = _tensors()
    flow = torch.zeros(1, 4, 4, 2)
    confidence = torch.ones(1, 1, 4, 4)
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="v4").write(
        "sample-0", y, target, mask, {}, latent_flow=flow,
        flow_confidence=confidence, source="replay", generation_seed=7,
        parent_checkpoint="synthetic/step_100", source_fingerprint="source-fp",
    )

    item = ProjectionCacheDataset(str(tmp_path), expected_fingerprint="v4")[0]
    assert item["metadata"]["cache_schema_version"] == CACHE_SCHEMA_VERSION == 4
    assert item["metadata"]["source"] == "replay"
    assert item["metadata"]["generation_seed"] == 7
    assert item["metadata"]["parent_checkpoint"] == "synthetic/step_100"
    assert torch.equal(item["latent_flow"], flow)
    assert torch.equal(item["flow_confidence"], confidence)


def test_mixed_cache_has_exact_three_to_one_schedule(tmp_path):
    roots = {}
    y, target, mask = _tensors()
    for source in ("synthetic", "replay"):
        root = tmp_path / source
        ProjectionCacheWriter(str(root), store="rgb", fingerprint=source).write(
            f"{source}-0", y, target, mask, {"source": source}
        )
        roots[source] = ProjectionCacheDataset(str(root), expected_fingerprint=source)
    mixed = MixedProjectionCacheDataset(roots, {"synthetic": 3, "replay": 1}, epoch_size=8)
    sources = [mixed[index]["cache_source"] for index in range(len(mixed))]
    assert sources.count("synthetic") == 6
    assert sources.count("replay") == 2
