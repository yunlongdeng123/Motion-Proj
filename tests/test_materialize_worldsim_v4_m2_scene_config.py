from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.materialize_worldsim_v4_m2_scene_config import (
    materialize_scene_config,
    sha256_file,
)


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, abstain: bool = False) -> tuple[Path, Path]:
    scene = "scene-a"
    run = tmp_path / "m1-run"
    source_path = run / "source_snapshot/scene-a.yaml"
    summary = {
        "status": "done",
        "phase": "six_scene_development",
        "heldout_content_read": False,
        "test_quality_read": False,
        "scenes": [scene],
        "scene_records": [{"scene": scene, "status": "abstain", "reason": "ABSTAIN_NO_ACTOR"}],
    }
    _json(run / "summary.json", summary)
    _json(run / "manifest.json", {"status": "done"})
    if abstain:
        source = {
            "scene": scene,
            "partition": "development",
            "status": "abstain",
            "reason": "ABSTAIN_NO_ACTOR",
            "heldout_content_read": False,
            "test_quality_read": False,
        }
    else:
        checkpoint, ds_cfg, field = [tmp_path / name for name in ("c.pth", "d.yaml", "f.npz")]
        state = run / "artifacts/states/scene-a/high_support.npz"
        mask, image = tmp_path / "mask.npz", tmp_path / "image.jpg"
        for path in (checkpoint, ds_cfg, field, state, mask, image):
            _write(path, path.name.encode())
        mask_manifest = tmp_path / "masks.json"
        _json(mask_manifest, {
            "evaluation_partition": "development",
            "optimization_forbidden": True,
            "accepted_mask_count": 1,
            "masks": [{
                "accepted": True, "role": "high_support", "frame": 7, "camera_id": 0,
                "mask": str(mask), "mask_sha256": sha256_file(mask),
                "source_image": str(image), "source_image_sha256": sha256_file(image),
                "positive_pixels": 4,
            }],
        })
        package_root = tmp_path / "spatial/package/artifacts/worldsim_asset"
        erase = package_root / "deltas/delete_actor_8/erase_indices.npz"
        _write(erase, b"erase")
        package = {
            "actor": {"dataset_instance_id": 8},
            "inventory": [{"path": "deltas/delete_actor_8/erase_indices.npz", "sha256": sha256_file(erase), "bytes": erase.stat().st_size}],
        }
        _json(package_root / "package_manifest.json", package)
        chain = tmp_path / "chain.json"
        _json(chain, {"stages": {"spatial_delta": {"status": "done", "run": str(tmp_path / "spatial")}}})
        _write(state, b"state")
        source = {
            "scene": scene, "partition": "development", "status": "ready",
            "heldout_content_read": False, "test_quality_read": False,
            "actors": {"high_support": {"instance_token": "token", "dataset_instance_id": 8, "rigid_model_index": 0}},
            "inputs": {
                "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                "drivestudio_source_config": {"path": str(ds_cfg), "sha256": sha256_file(ds_cfg)},
                "v33_o1_instance_field": {"path": str(field), "sha256": sha256_file(field)},
                "development_evaluation_masks": {"path": str(mask_manifest), "sha256": sha256_file(mask_manifest)},
                "processed_scene_dir": str(tmp_path / "processed"),
            },
            "v33_scene_chain": {"path": str(chain), "sha256": sha256_file(chain)},
            "runtime": {"drivestudio_python": "python", "drivestudio_checkout": "/code"},
        }
    _yaml(source_path, source)
    config = tmp_path / "m2.yaml"
    _yaml(config, {
        "schema_version": "worldsim_v4_m2_router_v1", "task_id": "WS-V4-M2-REPAIR-ROUTER-01",
        "execution_gate": {"dataset": "nuScenes", "m1_task_status": "rejected", "m1_validation_status": "done", "fallback_scope": "evidence_routed_delta_compiler", "test_quality_read": False},
        "protocol": {"development_scenes": [scene], "partition_contract": "sample_index_mod_5", "target_partition": "development", "target_remainder": 2, "heldout_remainder": 4, "support_offsets": [-1, 1], "support_partition": "train_only", "retain_all_accepted_masks": True, "heldout_content_read": False, "test_quality_read": False},
        "inputs": {"m1_development_run": {"path": str(run), "summary_sha256": sha256_file(run / "summary.json"), "manifest_sha256": sha256_file(run / "manifest.json")}},
        "asset_build": {"builders": {}},
    })
    return config, source_path


def test_materialize_ready_scene_keeps_all_masks_and_train_only_support(tmp_path: Path) -> None:
    config, source = _fixture(tmp_path)
    result = materialize_scene_config(config_path=config, scene="scene-a", m1_scene_path=source)
    assert result["status"] == "ready"
    assert result["request_count"] == 1
    assert result["requests"][0]["support_views"] == [[6, 0], [8, 0]]
    assert result["all_accepted_masks_retained"] is True
    assert result["candidate_availability"]["GENERATED"] == "abstain_no_frozen_model"
    assert result["test_quality_read"] is False


def test_materialize_abstain_scene_remains_in_denominator(tmp_path: Path) -> None:
    config, source = _fixture(tmp_path, abstain=True)
    result = materialize_scene_config(config_path=config, scene="scene-a", m1_scene_path=source)
    assert result["status"] == "abstain"
    assert result["reason"] == "ABSTAIN_NO_ACTOR"
    assert result["requests"] == []
    assert result["retained_in_denominator"] is True
