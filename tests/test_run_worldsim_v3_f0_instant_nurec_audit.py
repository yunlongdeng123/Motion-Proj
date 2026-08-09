from __future__ import annotations

from copy import deepcopy

from omegaconf import OmegaConf
import pytest

from scripts.run_worldsim_v3_f0_instant_nurec_audit import (
    PROTOCOL,
    audit_source,
    build_capability_matrix,
    build_f1_decision,
    evaluate_smoke_prerequisites,
    parse_nvidia_smi_gpu,
    validate_schema,
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(PROTOCOL), resolve=True)


def passing_environment() -> dict:
    return {
        "python_3_11": "/usr/bin/python3.11",
        "uv": "/usr/bin/uv",
        "gpu": {
            "available": True,
            "gpus": [{"memory_total_mib": 81_920, "compute_capability": 9.0}],
        },
        "system_memory_bytes": 128_000_000_000,
        "disk": {"free_bytes": 200_000_000_000},
        "weight": {"exact_supported_weight_present": True},
        "ncore_input": {"exists": True, "suffix": ".json"},
        "dataset_terms_acceptance_recorded": True,
        "source_checkout_all_exact": True,
        "cli_help": {"exit_code": 0},
    }


def test_frozen_protocol_schema_is_valid(protocol: dict) -> None:
    validate_schema(protocol)


def test_protocol_rejects_download_or_gpu_fail_open(protocol: dict) -> None:
    changed = deepcopy(protocol)
    changed["authorization"]["weight_download_authorized"] = True
    with pytest.raises(RuntimeError, match="weight_download"):
        validate_schema(changed)
    changed = deepcopy(protocol)
    changed["authorization"]["gpu_launch_when_any_prerequisite_fails_authorized"] = True
    with pytest.raises(RuntimeError, match="gpu_launch"):
        validate_schema(changed)


def test_protocol_rejects_cli_capability_inflation(protocol: dict) -> None:
    for capability in ("dynamic_layer", "sky_cubemap", "isp_affine", "actor_registry"):
        changed = deepcopy(protocol)
        changed["standalone_cli_contract"]["exports"][capability] = True
        with pytest.raises(RuntimeError, match="CLI export boundary"):
            validate_schema(changed)


def test_all_prerequisites_are_jointly_required(protocol: dict) -> None:
    result = evaluate_smoke_prerequisites(protocol, passing_environment())
    assert result["all_passed"]
    assert result["inference_smoke_authorized"]
    assert not result["inference_command_constructed"]

    for mutation, expected in (
        (("gpu", "gpus"), "inference_vram_minimum_mib"),
        (("weight", "exact_supported_weight_present"), "exact_supported_weight_present"),
        (("ncore_input", "exists"), "licensed_ncore_v4_input_present"),
    ):
        environment = passing_environment()
        first, second = mutation
        environment[first][second] = [] if second == "gpus" else False
        failed = evaluate_smoke_prerequisites(protocol, environment)
        assert not failed["all_passed"]
        assert expected in failed["failed"]
        assert failed["inference_smoke_status"] == "not_run_prerequisites_failed"


def test_nvidia_smi_parser_keeps_exact_hardware_facts() -> None:
    parsed = parse_nvidia_smi_gpu({
        "exit_code": 0,
        "stdout": "NVIDIA GeForce RTX 3090, 24576, 580.105.08, 8.6\n",
        "stderr": "",
    })
    assert parsed["available"]
    assert parsed["gpus"] == [{
        "name": "NVIDIA GeForce RTX 3090",
        "memory_total_mib": 24_576,
        "driver_version": "580.105.08",
        "compute_capability": 8.6,
    }]


def test_exact_official_checkout_and_code_signatures_match(protocol: dict) -> None:
    audit = audit_source(protocol)
    assert audit["all_exact"]
    assert audit["head"] == protocol["official_source_checkout"]["repository_revision"]
    assert audit["tree"] == protocol["official_source_checkout"]["repository_tree"]
    assert len(audit["files"]) == 16
    assert audit["lidar_code_matches"] == []


def test_capability_matrix_separates_paper_from_cli(protocol: dict) -> None:
    matrix = build_capability_matrix(protocol)
    assert matrix["research_model"]["output"]["layered_dynamic_3dgs"]
    assert not matrix["standalone_cli"]["exports"]["dynamic_layer"]
    assert not matrix["standalone_cli"]["reads"]["lidar"]


def test_f1_remains_conditional_when_local_gate_fails(protocol: dict) -> None:
    environment = passing_environment()
    environment["disk"]["free_bytes"] = 42_000_000_000
    prerequisites = evaluate_smoke_prerequisites(protocol, environment)
    decision = build_f1_decision(protocol, prerequisites)
    assert decision["decision"] == "conditional_not_unlocked"
    assert not decision["f1_authorized"]
    assert "local_prerequisite_failed:disk_free_minimum_bytes" in decision["reasons"]
