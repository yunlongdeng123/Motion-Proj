from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from scripts.audit_worldsim_v3_a4_p0_resume import build_resume_audit, sha256_file
from scripts.run_worldsim_v3_a4_p0_profile import (
    build_aggregate,
    build_inventory,
    nearest_rank,
    rgb_sha256,
)


def test_nearest_rank_is_frozen() -> None:
    values = [9, 1, 2, 3, 4, 5, 6, 7, 8]
    assert nearest_rank(values, 0.50) == 5
    assert nearest_rank(values, 0.95) == 9
    with pytest.raises(ValueError):
        nearest_rank([], 0.5)


def test_rgb_hash_includes_dtype_and_shape() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    expected = hashlib.sha256()
    expected.update(b"uint8")
    expected.update(json.dumps([2, 3, 3]).encode())
    expected.update(rgb.tobytes())
    assert rgb_sha256(rgb) == expected.hexdigest()
    assert rgb_sha256(np.zeros((3, 2, 3), dtype=np.uint8)) != rgb_sha256(rgb)


def test_aggregate_keeps_missing_recovery_explicit() -> None:
    inventory = {
        "conversion_status": "inventory_only_no_parameter_conversion",
        "checkpoint_bytes": 8,
        "source_config_bytes": 2,
        "checkpoint_and_registry_inventory_bytes": 10,
        "historical_stages": {
            name: {
                "status": "done",
                "duration_seconds": 1.0,
                "peak_gpu_memory_mib_sampled": 2,
                "peak_gpu_memory_mib_torch_log": None,
                "peak_cgroup_memory_bytes": 3,
                "source_path": name,
                "source_sha256": name,
                "input_bytes": None,
                "input_bytes_missing_reason": "historical",
                "output_bytes": 4,
                "filesystem_cache": None,
                "filesystem_cache_missing_reason": "historical",
            }
            for name in ("train", "render_eval", "actor_metrics", "registry")
        },
    }
    runtime = {
        "prepare_dataset_seconds": 1.0,
        "process_cold_checkpoint_load_seconds": 2.0,
        "process_warm_checkpoint_reload_seconds": 1.0,
        "filesystem_cache": "uncontrolled_report_explicitly",
        "runtime_render_sample_seconds": [0.1] * 9,
        "runtime_render_p50_seconds": 0.1,
        "runtime_render_p95_seconds": 0.1,
        "runtime_render_fps": 10.0,
        "runtime_rgb_bytes_in_memory": 11,
    }
    aggregate = build_aggregate(inventory, runtime)
    recovery = aggregate["stage_ledger"]["failure_recovery"]
    assert recovery["status"] is None
    assert recovery["missing_reason"]


def test_inventory_counts_checkpoint_and_registry_without_config(tmp_path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "actor_count": 2,
                "available_actor_count": 1,
                "empty_checkpoint_actor_count": 1,
            }
        ),
        encoding="utf-8",
    )
    stages = {}
    for name in ("train", "render_eval", "actor_metrics", "registry"):
        path = tmp_path / f"stage_{name}.json"
        path.write_text(
            json.dumps({"status": "done", "duration_seconds": 1.0}),
            encoding="utf-8",
        )
        stages[name] = {"path": str(path), "sha256": name}
    protocol = {
        "selected_asset": {
            "role": "r0",
            "checkpoint": {"bytes": 10},
            "source_config": {"bytes": 2},
            "actor_registry": {"path": str(registry), "bytes": 3},
        },
        "historical_evidence": {"stages": stages},
    }
    inventory = build_inventory(protocol, {})
    assert inventory["checkpoint_and_registry_inventory_bytes"] == 13


def test_resume_audit_reuses_exact_completed_stages(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True)
    hashes = {}
    for name in ("inventory", "runtime_probe", "aggregate"):
        path = stages / f"{name}.json"
        path.write_text(json.dumps({"stage": name}), encoding="utf-8")
        hashes[name] = sha256_file(path)
    (run_dir / "manifest.json").write_text(
        json.dumps({"stage_hashes": hashes}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "scripts.audit_worldsim_v3_a4_p0_resume.gpu_rows", lambda: []
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    result = build_resume_audit(run_dir)
    assert result["gpu_launch_observed"] is False
    assert result["torch_imported"] is False
    assert all(
        row["action"] == "reuse_completed_stage"
        for row in result["actions"].values()
    )


def test_resume_audit_rejects_corrupted_stage(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True)
    hashes = {}
    for name in ("inventory", "runtime_probe", "aggregate"):
        path = stages / f"{name}.json"
        path.write_text(json.dumps({"stage": name}), encoding="utf-8")
        hashes[name] = sha256_file(path)
    (run_dir / "manifest.json").write_text(
        json.dumps({"stage_hashes": hashes}), encoding="utf-8"
    )
    (stages / "runtime_probe.json").write_text("corrupted", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.audit_worldsim_v3_a4_p0_resume.gpu_rows", lambda: []
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(RuntimeError, match="hash drift"):
        build_resume_audit(run_dir)
