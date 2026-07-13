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


def test_schema_v5_roundtrip_optional_flow_and_provenance(tmp_path):
    y, target, mask = _tensors()
    flow = torch.zeros(1, 4, 4, 2)
    confidence = torch.ones(1, 1, 4, 4)
    ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="v4").write(
        "sample-0", y, target, mask, {}, latent_flow=flow,
        flow_confidence=confidence, source="replay", generation_seed=7,
        parent_checkpoint="synthetic/step_100", source_fingerprint="source-fp",
    )

    item = ProjectionCacheDataset(str(tmp_path), expected_fingerprint="v4")[0]
    assert item["metadata"]["cache_schema_version"] == CACHE_SCHEMA_VERSION == 5
    assert item["metadata"]["source"] == "replay"
    assert item["metadata"]["generation_seed"] == 7
    assert item["metadata"]["parent_checkpoint"] == "synthetic/step_100"
    assert torch.equal(item["latent_flow"], flow)
    assert torch.equal(item["flow_confidence"], confidence)


def _formal_metadata(sample_id="formal-0"):
    return {
        "sample_id": sample_id, "source": "replay_v2", "parent_kind": "base",
        "base_model_fingerprint": "base-fp", "adapter_loaded": False,
        "condition_id": "condition", "condition_frame": 0, "generation_seed": 7,
        "generation_sampler": "torch.Generator", "generation_steps": 25,
        "generation_settings": {"decode_chunk_size": 4}, "first_frame_frozen": True,
        "auditor_version": "generated-point-track-v1", "projector_version": "v5",
        "geometry_mode": "estimated_background_motion", "uses_future_gt_ego": False,
        "uses_future_gt_track": False,
        "energy_before_by_component": {"static": 1.0, "object": 1.0},
        "energy_after_by_component": {"static": 0.9, "object": 0.9},
        "projector_diagnostics": {}, "base_vae_fingerprint": "vae-fp",
        "projected_vae_fingerprint": "vae-fp",
    }


def test_formal_v2_requires_base_provenance_components_and_frozen_first_frame(tmp_path):
    y, target, mask = _tensors()
    target[0] = y[0]
    mask[0] = 0
    static_mask = mask.clone()
    object_mask = torch.zeros_like(mask)
    static_confidence = static_mask.clone()
    object_confidence = object_mask.clone()
    base_rgb = y.clone()
    projected_rgb = target.clone()
    projected_rgb[0] = base_rgb[0]
    writer = ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="formal", formal_v2=True)
    writer.write(
        "formal-0", y, target, mask, _formal_metadata(), static_mask=static_mask,
        object_mask=object_mask, static_confidence=static_confidence,
        object_confidence=object_confidence, base_rgb=base_rgb, projected_rgb=projected_rgb,
    )
    item = ProjectionCacheDataset(str(tmp_path), expected_fingerprint="formal")[0]
    assert item["metadata"]["formal_v2"] is True
    assert (tmp_path / "formal-0" / "static_mask.pt").is_file()


def test_formal_v2_rejects_future_gt_and_unfrozen_first_frame(tmp_path):
    y, target, mask = _tensors()
    static_mask = mask.clone()
    object_mask = torch.zeros_like(mask)
    metadata = _formal_metadata()
    metadata["uses_future_gt_track"] = True
    writer = ProjectionCacheWriter(str(tmp_path), store="rgb", fingerprint="formal", formal_v2=True)
    with pytest.raises(ValueError, match="future GT"):
        writer.write(
            "formal-0", y, target, mask, metadata, static_mask=static_mask,
            object_mask=object_mask, static_confidence=static_mask,
            object_confidence=object_mask, base_rgb=y, projected_rgb=target,
        )


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
