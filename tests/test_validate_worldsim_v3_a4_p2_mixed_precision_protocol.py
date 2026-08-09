from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from scripts.validate_worldsim_v3_a4_p2_mixed_precision_protocol import (
    PROTOCOL,
    directory_digest,
    validate_checkpoint_state,
    validate_inputs,
    validate_schema,
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(PROTOCOL), resolve=True)


def test_frozen_protocol_schema_is_valid(protocol: dict) -> None:
    validate_schema(protocol)


def test_protocol_keeps_training_renderer_fp16_and_p3_unauthorized(protocol: dict) -> None:
    authorization = protocol["authorization"]
    assert not authorization["training_authorized"]
    assert not authorization["optimizer_authorized"]
    assert not authorization["fp16_renderer_compute_authorized"]
    assert not authorization["p3_chunk_authorized"]


def test_protocol_rejects_result_dependent_extra_arm(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["precision_contract"]["arms"].append(
        {
            "id": "p2-post-hoc",
            "checkpoint_storage": "forbidden",
            "persistent_gaussian_parameter_dtype": "float16",
            "renderer_input_dtype": "float16",
        }
    )
    with pytest.raises(RuntimeError, match="arm grid"):
        validate_schema(changed)


def test_protocol_rejects_fp16_means_or_renderer_claim(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["precision_contract"]["converted_fields"].insert(0, "_means")
    with pytest.raises(RuntimeError, match="converted field"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["precision_contract"]["runtime_adapter"]["fp16_renderer_kernel_claim_allowed"] = True
    with pytest.raises(RuntimeError, match="renderer claim"):
        validate_schema(changed)


def test_protocol_rejects_quality_or_resource_relaxation(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["quality_contract"]["global_endpoints"]["metrics"][0]["maximum_regression"] = 0.5
    with pytest.raises(RuntimeError, match="global thresholds"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["resource_ceilings"]["wall_time_seconds"] = 1800
    with pytest.raises(RuntimeError, match="resource ceilings"):
        validate_schema(changed)


def test_protocol_rejects_recovery_or_fallback_drift(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["recovery_contract"]["stage_order"].reverse()
    with pytest.raises(RuntimeError, match="stage order"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["selection_contract"]["candidate_fail"]["selected_arm"] = "p2-gs-param-fp16"
    with pytest.raises(RuntimeError, match="source fallback"):
        validate_schema(changed)


def test_frozen_inputs_and_p1_selection_are_exact(protocol: dict) -> None:
    audits = validate_inputs(protocol)
    assert len(audits) == 10
    assert audits["baseline_quality.actor_masks"]["file_count"] == 33


def test_directory_digest_binds_names_bytes_and_content(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"bb")
    first = directory_digest(tmp_path, "*.png")
    assert first["file_count"] == 2
    assert first["total_bytes"] == 3
    (tmp_path / "b.png").write_bytes(b"bc")
    assert directory_digest(tmp_path, "*.png")["sha256"] != first["sha256"]


def test_checkpoint_state_audit_locks_source_and_candidate_dtypes(
    protocol: dict, tmp_path: Path
) -> None:
    fields = protocol["precision_contract"]["converted_fields"]
    models = {}
    for name, count in (("Background", 3), ("RigidNodes", 2)):
        state = {"_means": torch.tensor([[1.1, 2.2, 3.3]]).repeat(count, 1)}
        for field in fields:
            width = 4 if field == "_quats" else 3
            state[field] = torch.linspace(-1.0, 1.0, count * width).reshape(count, width)
        models[name] = state
    path = tmp_path / "checkpoint.pth"
    torch.save({"models": models}, path)
    changed = deepcopy(protocol)
    changed["selected_asset"]["inventory"]["background_gaussians"] = 3
    changed["selected_asset"]["inventory"]["rigid_gaussians"] = 2
    for name in ("Background", "RigidNodes"):
        means = models[name]["_means"]
        changed["precision_contract"]["means_fp16_exclusion"][
            "background_fp16_roundtrip_max_absolute_error"
            if name == "Background"
            else "rigid_fp16_roundtrip_max_absolute_error"
        ] = float((means.half().float() - means).abs().max())
    audit = validate_checkpoint_state(changed, path)
    assert audit["converted_field_count"] == 10
    assert all(
        row["source_dtype"] == "float32" and row["candidate_dtype"] == "float16"
        for row in audit["converted_fields"].values()
    )
