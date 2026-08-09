from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from scripts.materialize_worldsim_v3_a3_config import (
    materialize_r1_config,
    r0_alias_manifest,
)
from scripts.run_worldsim_v3_a3_synthetic_smoke import run_smoke


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "a" * 64


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/worldsim_v3/a3_local_refine_protocol_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def source_config() -> object:
    return OmegaConf.create(
        {
            "seed": 0,
            "trainer": {
                "type": "models.trainers.MultiTrainer",
                "optim": {"num_iters": 30_000, "use_grad_scaler": False},
            },
            "model": {
                "Background": {"optim": {"opacity": {"lr": 0.05}, "scaling": {"lr": 0.005}}},
                "RigidNodes": {},
                "Sky": {},
            },
            "logging": {"saveckpt_freq": 5_000},
            "data": {
                "scene_idx": 179,
                "pixel_source": {"cameras": [0, 1, 2], "test_image_stride": 10},
            },
            "worldsim_v3": {"variant": "d2-boundary-residual", "formal": True},
        }
    )


def sidecar() -> dict:
    return {
        "schema_version": 1,
        "task_id": "WS-V3-A3-LOCAL-REFINE-01",
        "audit_version": "A3-R1-SIDECAR-v1",
        "variant": "r1-reactivate",
        "formal_training_authorized": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": protocol()["depends_on"]["selected_checkpoint_sha256"],
        "background_point_count": 4,
        "arrays": {
            "path": "rows.npz",
            "sha256": "b" * 64,
            "affected_rows_key": "affected_background_rows",
            "support_strata_key": "support_strata_codes",
        },
        "evidence": {
            "support_provenance_complete": True,
            "heldout_frames": list(range(10, 200, 10)),
            "heldout_excluded_from_support": True,
            "typed_depth_truth_tiers": {
                "depth_render_expected": "diagnostic",
                "depth_surface_first_hit": "T1",
                "depth_lidar_measured": "T0",
            },
        },
    }


def test_r0_is_metadata_only_exact_checkpoint_alias() -> None:
    contract = protocol()
    manifest = r0_alias_manifest(
        source_config(),
        contract,
        protocol_sha256=PROTOCOL_SHA256,
        source_config_sha256=contract["depends_on"]["selected_checkpoint_config_sha256"],
        checkpoint_sha256=contract["depends_on"]["selected_checkpoint_sha256"],
    )
    assert manifest["optimizer_steps"] == 0
    assert manifest["new_checkpoint_keys"] is False
    assert manifest["checkpoint_sha256"] == contract["depends_on"]["selected_checkpoint_sha256"]
    assert "config" not in manifest


def test_r1_materializer_binds_sidecar_and_remains_engineering_only() -> None:
    contract = protocol()
    config = materialize_r1_config(
        source_config(),
        contract,
        sidecar(),
        protocol_sha256=PROTOCOL_SHA256,
        source_config_sha256=contract["depends_on"]["selected_checkpoint_config_sha256"],
        checkpoint_sha256=contract["depends_on"]["selected_checkpoint_sha256"],
        sidecar_manifest_path="/tmp/a3-sidecar.json",
        optimizer_steps=2,
    )
    assert config.trainer.optim.num_iters == 30_002
    assert config.trainer.a3_local_refinement.enabled is True
    assert config.trainer.a3_local_refinement.require_external_paired_loss_masks is True
    assert config.worldsim_v3.formal is False
    assert config.worldsim_v3.numeric_protocol_frozen is False
    assert config.worldsim_v3.sidecar_arrays_sha256 == "b" * 64


def test_r1_rejects_formal_or_source_hash_drift() -> None:
    contract = protocol()
    kwargs = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_config_sha256": contract["depends_on"]["selected_checkpoint_config_sha256"],
        "checkpoint_sha256": contract["depends_on"]["selected_checkpoint_sha256"],
        "sidecar_manifest_path": "/tmp/a3-sidecar.json",
        "optimizer_steps": 1,
    }
    with pytest.raises(ValueError, match="formal materialization"):
        materialize_r1_config(
            source_config(), contract, sidecar(), stage="formal", **kwargs
        )
    kwargs["source_config_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="config SHA drift"):
        materialize_r1_config(source_config(), contract, sidecar(), **kwargs)


def test_r0_r1_synthetic_smoke_is_contract_only_and_exact() -> None:
    summary = run_smoke()
    assert summary["pass"] is True
    assert summary["evidence_tier"] == "synthetic_contract_only_not_quality_evidence"
    assert summary["r0"]["new_checkpoint_created"] is False
    assert all(summary["r1"]["outside_exact"].values())
    assert all(summary["r1"]["forbidden_exact"].values())
    assert summary["formal_training_authorized"] is False
